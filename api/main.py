"""
FastAPI application for the Diabetic Retinopathy Early Detection system.

Endpoints:
  GET  /health          — Check API and model health
  POST /predict/biomarker — Predict DR from clinical biomarkers only
  POST /predict/image     — Predict DR from a retinal fundus image only
  POST /predict/unified   — Predict DR from both image + biomarkers (late fusion)
"""

import io
import os
import sys
import base64
import logging
import importlib
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from httpx import request
import numpy as np
import joblib
import cv2
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.models.biomarker_rf import load_biomarker_model, predict_biomarker_proba
from src.models.retinal_cnn import load_cnn_model
from src.models.late_fusion import unified_prediction
from src.explainability.grad_cam import explain_prediction
from src.data_prep.image_loader import apply_clahe, crop_to_circle
from api.schemas import (
    BiomarkerInput,
    PredictionRequest,
    PredictionResponse,
    GradeProbability,
    HealthResponse,
    BatchItemResult,
    BatchPredictionResponse,
)
from api.prioritization import (
    prioritize,
    clinical_recommendation_from_score,
    compute_clinical_rule_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = load_settings()

_models = {
    "biomarker": None,
    "scaler": None,
    "cnn": None,
}

DR_LABELS = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
FEATURE_ORDER = settings["tabular"]["features"]


_model_mtimes = {
    "biomarker": 0.0,
    "scaler": 0.0,
    "cnn": 0.0,
}


def _ensure_models_loaded():
    """Ensure all models are loaded in memory and reload if updated on disk."""
    model_dir = settings["paths"]["saved_models"]
    bio_path = os.path.join(model_dir, "biomarker_model.pkl")
    scaler_path = os.path.join(model_dir, "biomarker_scaler.pkl")

    if os.path.exists(bio_path):
        mtime = os.path.getmtime(bio_path)
        if _models["biomarker"] is None or mtime > _model_mtimes["biomarker"]:
            try:
                _models["biomarker"] = load_biomarker_model(bio_path)
                _model_mtimes["biomarker"] = mtime
                logger.info("Biomarker model loaded.")
            except Exception as e:
                logger.warning(f"Failed to load biomarker model: {e}")

    if os.path.exists(scaler_path):
        mtime = os.path.getmtime(scaler_path)
        if _models["scaler"] is None or mtime > _model_mtimes["scaler"]:
            try:
                _models["scaler"] = joblib.load(scaler_path)
                _model_mtimes["scaler"] = mtime
                logger.info("Scaler loaded.")
            except Exception as e:
                logger.warning(f"Failed to load scaler: {e}")

    if _models["cnn"] is None:
        try:
            _models["cnn"] = load_cnn_model()
            logger.info("CNN model loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load CNN: {e}")



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup."""
    _ensure_models_loaded()
    yield
    _models.clear()



app = FastAPI(
    title="Diabetic Retinopathy Early Detection API",
    description="Unified AI system for DR screening using biomarkers and retinal images.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _biomarker_to_array(bio: BiomarkerInput) -> np.ndarray:
    """Convert BiomarkerInput to a numpy array in the correct feature order (with engineered features)."""
    base_vals = [float(getattr(bio, f)) for f in FEATURE_ORDER]
    hba1c_duration = bio.hba1c * bio.diabetes_duration_years
    bp_pulse = bio.blood_pressure_systolic - bio.blood_pressure_diastolic
    chol_ratio = bio.cholesterol_total / (bio.cholesterol_hdl + 0.001)
    high_risk_combo = 1.0 if (bio.hba1c > 8.0 and bio.diabetes_duration_years > 10) else 0.0
    metabolic_score = bio.bmi * 0.3 + bio.triglycerides * 0.01 + bio.hba1c * 0.5
    all_vals = base_vals + [hba1c_duration, bp_pulse, chol_ratio, high_risk_combo, metabolic_score]
    return np.array([all_vals])


def _build_response(
    predicted_grade: int,
    risk_score: float,
    proba: np.ndarray,
    model_used: str,
    baseline_clinical_score: Optional[float] = None,
    baseline_recommendation: Optional[str] = None,
    baseline_factor_breakdown: Optional[dict] = None,
    grad_cam_available: bool = False,
    grad_cam_heatmap: Optional[str] = None,
    grad_cam_overlay: Optional[str] = None,
) -> PredictionResponse:
    """Build a standardized PredictionResponse."""
    tier, _, grade_label = prioritize(risk_score, predicted_grade)

    grade_probs = [
        GradeProbability(grade=i, label=DR_LABELS[i], probability=round(float(proba[i]), 4))
        for i in range(len(proba))
    ]

    return PredictionResponse(
        predicted_grade=predicted_grade,
        predicted_label=grade_label,
        risk_score=round(float(risk_score), 4),
        screening_tier=tier,
        grade_probabilities=grade_probs,
        model_used=model_used,
        baseline_clinical_score=(
            round(float(baseline_clinical_score), 4)
            if baseline_clinical_score is not None else None
        ),
        baseline_recommendation=baseline_recommendation,
        baseline_factor_breakdown=baseline_factor_breakdown,
        grad_cam_available=grad_cam_available,
        grad_cam_heatmap=grad_cam_heatmap,
        grad_cam_overlay=grad_cam_overlay,
    )


def _encode_image_to_data_url(image: np.ndarray, mode: str) -> str:
    """Encode a numpy image array to a PNG base64 data URL."""
    image_pil = Image.fromarray(image, mode=mode)
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _generate_gradcam_payload(
    image_for_model: np.ndarray,
    predicted_grade: int,
) -> tuple[str, str]:
    """Return encoded Grad-CAM heatmap and overlay images."""
    overlay, heatmap, _, _ = explain_prediction(
        _models["cnn"],
        image_for_model,
        target_class=predicted_grade,
    )
    heatmap_img = np.uint8(np.clip(heatmap, 0.0, 1.0) * 255)
    heatmap_b64 = _encode_image_to_data_url(heatmap_img, mode="L")
    overlay_b64 = _encode_image_to_data_url(overlay, mode="RGB")
    return heatmap_b64, overlay_b64


def _extract_biomarkers_from_form(
    age: float,
    bmi: float,
    hba1c: float,
    blood_pressure_systolic: float,
    blood_pressure_diastolic: float,
    cholesterol_total: float,
    cholesterol_hdl: float,
    cholesterol_ldl: float,
    triglycerides: float,
    diabetes_duration_years: float,
    smoking_status: int,
    family_history_dr: int,
) -> BiomarkerInput:
    return BiomarkerInput(
        age=age,
        bmi=bmi,
        hba1c=hba1c,
        blood_pressure_systolic=blood_pressure_systolic,
        blood_pressure_diastolic=blood_pressure_diastolic,
        cholesterol_total=cholesterol_total,
        cholesterol_hdl=cholesterol_hdl,
        cholesterol_ldl=cholesterol_ldl,
        triglycerides=triglycerides,
        diabetes_duration_years=diabetes_duration_years,
        smoking_status=smoking_status,
        family_history_dr=family_history_dr,
    )


def _preprocess_image_for_inference(image_bytes: bytes) -> np.ndarray:
    """Apply Stage-2 preprocessing: crop + CLAHE + resize + normalization."""
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image file is empty.")
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")
    image_np = np.array(image, dtype=np.uint8)
    image_np = crop_to_circle(image_np)
    image_np = apply_clahe(image_np)
    target = tuple(settings["image"]["target_size"])
    image_np = cv2.resize(image_np, target, interpolation=cv2.INTER_AREA)
    return image_np.astype(np.float32) / 255.0


def _predict_unified_from_inputs(
    image_bytes: bytes,
    biomarkers: BiomarkerInput,
    include_gradcam: bool = True,
) -> PredictionResponse:
    _ensure_models_loaded()
    if _models["cnn"] is None or _models["biomarker"] is None:
        raise HTTPException(status_code=503, detail="Both models must be loaded for unified prediction.")

    img_resized = _preprocess_image_for_inference(image_bytes)
    cnn_proba = _models["cnn"].predict(np.expand_dims(img_resized, axis=0), verbose=0)

    baseline_score, factors = compute_clinical_rule_score(biomarkers)
    baseline_recommendation = clinical_recommendation_from_score(baseline_score)

    X = _biomarker_to_array(biomarkers)
    if _models["scaler"] is not None:
        X = _models["scaler"].transform(X)
    bio_proba = predict_biomarker_proba(_models["biomarker"], X)

    grades, risk_scores, fused_proba = unified_prediction(cnn_proba, bio_proba)
    grade = int(grades[0])
    risk_score = float(risk_scores[0])

    heatmap_b64 = None
    overlay_b64 = None
    if include_gradcam:
        heatmap_b64, overlay_b64 = _generate_gradcam_payload(img_resized, grade)

    return _build_response(
        grade,
        risk_score,
        fused_proba[0],
        model_used="unified (CNN + Biomarker + Rule-based Stage-1)",
        baseline_clinical_score=baseline_score,
        baseline_recommendation=baseline_recommendation,
        baseline_factor_breakdown=factors,
        grad_cam_available=include_gradcam,
        grad_cam_heatmap=heatmap_b64,
        grad_cam_overlay=overlay_b64,
    )


def _decode_data_url_to_bytes(data_url: str) -> bytes:
    """Decode a data URL (base64 PNG/JPG) into raw bytes."""
    if "," not in data_url:
        raise ValueError("Invalid data URL")
    return base64.b64decode(data_url.split(",", 1)[1])


def _build_clinical_report_pdf(response: PredictionResponse, biomarkers: BiomarkerInput) -> bytes:
    """Create a concise PDF with Stage-1, Stage-2, and Stage-3 outputs."""
    try:
        pagesizes_module = importlib.import_module("reportlab.lib.pagesizes")
        pdfgen_canvas_module = importlib.import_module("reportlab.pdfgen.canvas")
        utils_module = importlib.import_module("reportlab.lib.utils")
    except ModuleNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="PDF report dependency missing. Install reportlab to enable report export.",
        ) from exc

    A4 = pagesizes_module.A4
    canvas = pdfgen_canvas_module
    ImageReader = utils_module.ImageReader

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 40
    pdf.setTitle("DR Clinical Screening Report")
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, y, "Diabetic Retinopathy Clinical Screening Report")

    y -= 24
    pdf.setFont("Helvetica", 10)
    pdf.drawString(40, y, "Stage 1: Rule-Based Baseline Clinical Risk")
    y -= 16
    baseline_score = response.baseline_clinical_score if response.baseline_clinical_score is not None else 0.0
    pdf.drawString(50, y, f"Baseline clinical score: {baseline_score:.3f}")
    y -= 14
    recommendation = response.baseline_recommendation or "Not available"
    pdf.drawString(50, y, f"Recommendation: {recommendation}")

    y -= 22
    pdf.drawString(40, y, "Stage 2: AI DR Grading")
    y -= 16
    pdf.drawString(50, y, f"Predicted grade: {response.predicted_grade} ({response.predicted_label})")
    y -= 14
    pdf.drawString(50, y, f"Unified risk score: {response.risk_score:.3f}")
    y -= 14
    pdf.drawString(50, y, f"Screening tier: {response.screening_tier}")

    y -= 22
    pdf.drawString(40, y, "Biomarker Snapshot")
    y -= 16
    pdf.drawString(50, y, f"HbA1c: {biomarkers.hba1c:.2f}%, SBP: {biomarkers.blood_pressure_systolic:.0f} mmHg, BMI: {biomarkers.bmi:.1f}")
    y -= 14
    pdf.drawString(50, y, f"LDL: {biomarkers.cholesterol_ldl:.0f} mg/dL, Triglycerides: {biomarkers.triglycerides:.0f} mg/dL")

    y -= 22
    pdf.drawString(40, y, "Grade Probabilities")
    y -= 16
    for prob in response.grade_probabilities:
        pdf.drawString(50, y, f"Grade {prob.grade} ({prob.label}): {prob.probability:.3f}")
        y -= 13
        if y < 120:
            pdf.showPage()
            y = height - 40
            pdf.setFont("Helvetica", 10)

    if response.grad_cam_overlay:
        try:
            overlay_bytes = _decode_data_url_to_bytes(response.grad_cam_overlay)
            image_reader = ImageReader(io.BytesIO(overlay_bytes))
            img_w = width - 120
            img_h = 220
            if y < img_h + 80:
                pdf.showPage()
                y = height - 40
            pdf.drawString(40, y - 6, "Stage 3: Grad-CAM Overlay")
            y -= img_h + 16
            pdf.drawImage(image_reader, 60, y, width=img_w, height=img_h, preserveAspectRatio=True, mask="auto")
        except Exception:
            pass

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()



@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API status and which models are loaded."""
    _ensure_models_loaded()
    return HealthResponse(
        status="healthy",
        models_loaded={
            "biomarker": _models["biomarker"] is not None,
            "scaler": _models["scaler"] is not None,
            "cnn": _models["cnn"] is not None,
        },
    )


@app.post("/predict/biomarker", response_model=PredictionResponse)
async def predict_biomarker(request: PredictionRequest):
    """Predict DR from clinical biomarkers only."""
    _ensure_models_loaded()
    if _models["biomarker"] is None:
        raise HTTPException(status_code=503, detail="Biomarker model not loaded.")

    import pandas as pd
    b = request.biomarkers
    df = pd.DataFrame([b.model_dump()])
    df["hba1c_duration"]  = df["hba1c"] * df["diabetes_duration_years"]
    df["bp_pulse"]        = df["blood_pressure_systolic"] - df["blood_pressure_diastolic"]
    df["chol_ratio"]      = df["cholesterol_total"] / (df["cholesterol_hdl"] + 0.001)
    df["high_risk_combo"] = ((df["hba1c"] > 8.0) & (df["diabetes_duration_years"] > 10)).astype(int)
    df["metabolic_score"] = df["bmi"] * 0.3 + df["triglycerides"] * 0.01 + df["hba1c"] * 0.5
    FEATURE_COLS = ["age","bmi","hba1c","blood_pressure_systolic","blood_pressure_diastolic",
                    "cholesterol_total","cholesterol_hdl","cholesterol_ldl","triglycerides",
                    "diabetes_duration_years","smoking_status","family_history_dr",
                    "hba1c_duration","bp_pulse","chol_ratio","high_risk_combo","metabolic_score"]
    X = df[FEATURE_COLS].values
    if _models["scaler"] is not None:
        X = _models["scaler"].transform(X)

    proba = predict_biomarker_proba(_models["biomarker"], X)[0]
    grade = int(np.argmax(proba))
    severity_weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    risk_score = float(proba @ severity_weights)
    baseline_score, factors = compute_clinical_rule_score(request.biomarkers)
    baseline_recommendation = clinical_recommendation_from_score(baseline_score)

    return _build_response(
        grade,
        risk_score,
        proba,
        model_used="biomarker + rule-based stage-1",
        baseline_clinical_score=baseline_score,
        baseline_recommendation=baseline_recommendation,
        baseline_factor_breakdown=factors,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """Process a batch of patient records from a real CSV file."""
    _ensure_models_loaded()
    if _models["biomarker"] is None:
        raise HTTPException(status_code=503, detail="Biomarker model not loaded.")

    import pandas as pd
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file format: {e}")

    # Standardize column names (lowercase, stripped)
    df.columns = [c.strip().lower() for c in df.columns]

    # Required base columns with defaults if optional
    field_aliases = {
        "age": ["age", "patient_age"],
        "bmi": ["bmi", "body_mass_index"],
        "hba1c": ["hba1c", "a1c", "glycated_hemoglobin"],
        "blood_pressure_systolic": ["blood_pressure_systolic", "systolic_bp", "sbp", "bp_sys"],
        "blood_pressure_diastolic": ["blood_pressure_diastolic", "diastolic_bp", "dbp", "bp_dia"],
        "cholesterol_total": ["cholesterol_total", "total_cholesterol", "chol_tot", "cholesterol"],
        "cholesterol_hdl": ["cholesterol_hdl", "hdl_cholesterol", "hdl"],
        "cholesterol_ldl": ["cholesterol_ldl", "ldl_cholesterol", "ldl"],
        "triglycerides": ["triglycerides", "triglyceride", "tg"],
        "diabetes_duration_years": ["diabetes_duration_years", "diabetes_duration", "duration_years", "duration"],
        "smoking_status": ["smoking_status", "smoking", "smoker"],
        "family_history_dr": ["family_history_dr", "family_history", "fam_hist"],
    }

    # Map columns
    mapped_cols = {}
    for target, aliases in field_aliases.items():
        matched = False
        for a in aliases:
            if a in df.columns:
                mapped_cols[target] = a
                matched = True
                break
        if not matched:
            # Fallback defaults for missing columns
            defaults = {
                "age": 55.0, "bmi": 27.0, "hba1c": 6.5, "blood_pressure_systolic": 130.0,
                "blood_pressure_diastolic": 80.0, "cholesterol_total": 200.0, "cholesterol_hdl": 50.0,
                "cholesterol_ldl": 120.0, "triglycerides": 150.0, "diabetes_duration_years": 5.0,
                "smoking_status": 0, "family_history_dr": 0,
            }
            df[target] = defaults[target]
            mapped_cols[target] = target

    # Rename to canonical
    rename_dict = {orig: target for target, orig in mapped_cols.items() if orig != target and orig in df.columns}
    df = df.rename(columns=rename_dict)

    # Compute engineered features
    df["hba1c_duration"]  = df["hba1c"] * df["diabetes_duration_years"]
    df["bp_pulse"]        = df["blood_pressure_systolic"] - df["blood_pressure_diastolic"]
    df["chol_ratio"]      = df["cholesterol_total"] / (df["cholesterol_hdl"] + 0.001)
    df["high_risk_combo"] = ((df["hba1c"] > 8.0) & (df["diabetes_duration_years"] > 10)).astype(int)
    df["metabolic_score"] = df["bmi"] * 0.3 + df["triglycerides"] * 0.01 + df["hba1c"] * 0.5

    FEATURE_COLS = ["age","bmi","hba1c","blood_pressure_systolic","blood_pressure_diastolic",
                    "cholesterol_total","cholesterol_hdl","cholesterol_ldl","triglycerides",
                    "diabetes_duration_years","smoking_status","family_history_dr",
                    "hba1c_duration","bp_pulse","chol_ratio","high_risk_combo","metabolic_score"]

    X = df[FEATURE_COLS].values
    if _models["scaler"] is not None:
        X = _models["scaler"].transform(X)

    probas = predict_biomarker_proba(_models["biomarker"], X)
    severity_weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    risk_scores = probas @ severity_weights

    results: List[BatchItemResult] = []
    urgent_count = 0
    moderate_count = 0
    low_risk_count = 0

    id_col = "patient_id" if "patient_id" in df.columns else ("id" if "id" in df.columns else None)

    for i, row in df.iterrows():
        p_id = str(row[id_col]) if id_col else f"PAT-{i+1:04d}"
        r_score = float(risk_scores[i])
        grade = int(np.argmax(probas[i]))
        tier, _, grade_label = prioritize(r_score, grade)

        if tier == "Urgent":
            urgent_count += 1
        elif tier == "Moderate":
            moderate_count += 1
        else:
            low_risk_count += 1

        b_inp = BiomarkerInput(
            age=float(row["age"]),
            bmi=float(row["bmi"]),
            hba1c=float(row["hba1c"]),
            blood_pressure_systolic=float(row["blood_pressure_systolic"]),
            blood_pressure_diastolic=float(row["blood_pressure_diastolic"]),
            cholesterol_total=float(row["cholesterol_total"]),
            cholesterol_hdl=float(row["cholesterol_hdl"]),
            cholesterol_ldl=float(row["cholesterol_ldl"]),
            triglycerides=float(row["triglycerides"]),
            diabetes_duration_years=float(row["diabetes_duration_years"]),
            smoking_status=int(row["smoking_status"]),
            family_history_dr=int(row["family_history_dr"]),
        )
        b_score, _ = compute_clinical_rule_score(b_inp)
        rec = clinical_recommendation_from_score(b_score)

        results.append(BatchItemResult(
            patient_id=p_id,
            age=float(row["age"]),
            hba1c=float(row["hba1c"]),
            blood_pressure_systolic=float(row["blood_pressure_systolic"]),
            blood_pressure_diastolic=float(row["blood_pressure_diastolic"]),
            bmi=float(row["bmi"]),
            diabetes_duration_years=float(row["diabetes_duration_years"]),
            risk_score=round(r_score, 4),
            screening_tier=tier,
            predicted_grade=grade,
            predicted_label=grade_label,
            baseline_recommendation=rec,
        ))

    # Sort results by risk score descending
    results.sort(key=lambda x: x.risk_score, reverse=True)

    avg_score = float(np.mean(risk_scores)) if len(risk_scores) > 0 else 0.0

    return BatchPredictionResponse(
        total_patients=len(results),
        urgent_count=urgent_count,
        moderate_count=moderate_count,
        low_risk_count=low_risk_count,
        avg_risk_score=round(avg_score, 4),
        results=results,
    )


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Predict DR from a retinal fundus image only."""
    _ensure_models_loaded()
    if _models["cnn"] is None:
        raise HTTPException(status_code=503, detail="CNN model not loaded.")

    contents = await file.read()
    img_resized = _preprocess_image_for_inference(contents)

    pred = _models["cnn"].predict(np.expand_dims(img_resized, axis=0), verbose=0)[0]
    grade = int(np.argmax(pred))
    severity_weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    risk_score = float(pred @ severity_weights)
    heatmap_b64, overlay_b64 = _generate_gradcam_payload(img_resized, grade)

    return _build_response(
        grade,
        risk_score,
        pred,
        model_used="cnn",
        grad_cam_available=True,
        grad_cam_heatmap=heatmap_b64,
        grad_cam_overlay=overlay_b64,
    )


@app.post("/predict/unified", response_model=PredictionResponse)
async def predict_unified(
    file: UploadFile = File(...),
    age: float = Form(...),
    bmi: float = Form(...),
    hba1c: float = Form(...),
    blood_pressure_systolic: float = Form(...),
    blood_pressure_diastolic: float = Form(...),
    cholesterol_total: float = Form(...),
    cholesterol_hdl: float = Form(...),
    cholesterol_ldl: float = Form(...),
    triglycerides: float = Form(...),
    diabetes_duration_years: float = Form(...),
    smoking_status: int = Form(...),
    family_history_dr: int = Form(...),
):
    """Predict DR using both image + biomarkers (late fusion)."""
    contents = await file.read()
    bio = _extract_biomarkers_from_form(
        age,
        bmi,
        hba1c,
        blood_pressure_systolic,
        blood_pressure_diastolic,
        cholesterol_total,
        cholesterol_hdl,
        cholesterol_ldl,
        triglycerides,
        diabetes_duration_years,
        smoking_status,
        family_history_dr,
    )
    return _predict_unified_from_inputs(contents, bio, include_gradcam=True)


@app.post("/predict/unified/report")
async def predict_unified_report(
    file: UploadFile = File(...),
    age: float = Form(...),
    bmi: float = Form(...),
    hba1c: float = Form(...),
    blood_pressure_systolic: float = Form(...),
    blood_pressure_diastolic: float = Form(...),
    cholesterol_total: float = Form(...),
    cholesterol_hdl: float = Form(...),
    cholesterol_ldl: float = Form(...),
    triglycerides: float = Form(...),
    diabetes_duration_years: float = Form(...),
    smoking_status: int = Form(...),
    family_history_dr: int = Form(...),
):
    """Run unified prediction and return a downloadable clinical PDF report."""
    contents = await file.read()
    bio = _extract_biomarkers_from_form(
        age,
        bmi,
        hba1c,
        blood_pressure_systolic,
        blood_pressure_diastolic,
        cholesterol_total,
        cholesterol_hdl,
        cholesterol_ldl,
        triglycerides,
        diabetes_duration_years,
        smoking_status,
        family_history_dr,
    )
    result = _predict_unified_from_inputs(contents, bio, include_gradcam=True)
    pdf_bytes = _build_clinical_report_pdf(result, bio)

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": "attachment; filename=dr_clinical_report.pdf"
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

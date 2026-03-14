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
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import joblib
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.models.biomarker_rf import load_biomarker_model, predict_biomarker_proba
from src.models.retinal_cnn import load_cnn_model
from src.models.late_fusion import unified_prediction
from src.explainability.grad_cam import explain_prediction
from api.schemas import (
    BiomarkerInput,
    PredictionRequest,
    PredictionResponse,
    GradeProbability,
    HealthResponse,
)
from api.prioritization import prioritize, get_grade_label

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = load_settings()

# Global model references
_models = {
    "biomarker": None,
    "scaler": None,
    "cnn": None,
}

DR_LABELS = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
FEATURE_ORDER = settings["tabular"]["features"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup."""
    model_dir = settings["paths"]["saved_models"]

    bio_path = os.path.join(model_dir, "biomarker_model.pkl")
    scaler_path = os.path.join(model_dir, "biomarker_scaler.pkl")
    cnn_path = os.path.join(model_dir, "cnn_weights.h5")

    if os.path.exists(bio_path):
        _models["biomarker"] = load_biomarker_model(bio_path)
        logger.info("Biomarker model loaded.")
    else:
        logger.warning(f"Biomarker model not found at {bio_path}")

    if os.path.exists(scaler_path):
        _models["scaler"] = joblib.load(scaler_path)
        logger.info("Scaler loaded.")

    if os.path.exists(cnn_path):
        try:
            _models["cnn"] = load_cnn_model(cnn_path)
            logger.info("CNN model loaded.")
        except Exception as e:
            logger.warning(f"Failed to load CNN: {e}")

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
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _biomarker_to_array(bio: BiomarkerInput) -> np.ndarray:
    """Convert BiomarkerInput to a numpy array in the correct feature order."""
    return np.array([[getattr(bio, f) for f in FEATURE_ORDER]])


def _build_response(
    predicted_grade: int,
    risk_score: float,
    proba: np.ndarray,
    model_used: str,
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


# ─── Endpoints ──────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API status and which models are loaded."""
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
    if _models["biomarker"] is None:
        raise HTTPException(status_code=503, detail="Biomarker model not loaded.")

    X = _biomarker_to_array(request.biomarkers)

    if _models["scaler"] is not None:
        X = _models["scaler"].transform(X)

    proba = predict_biomarker_proba(_models["biomarker"], X)[0]
    grade = int(np.argmax(proba))
    severity_weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    risk_score = float(proba @ severity_weights)

    return _build_response(grade, risk_score, proba, model_used="biomarker")


@app.post("/predict/image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Predict DR from a retinal fundus image only."""
    if _models["cnn"] is None:
        raise HTTPException(status_code=503, detail="CNN model not loaded.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image).astype(np.float32) / 255.0

    # Resize
    import cv2
    target = tuple(settings["image"]["target_size"])
    img_resized = cv2.resize(img_array, target)

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
    if _models["cnn"] is None or _models["biomarker"] is None:
        raise HTTPException(status_code=503, detail="Both models must be loaded for unified prediction.")

    # Process image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image).astype(np.float32) / 255.0
    import cv2
    target = tuple(settings["image"]["target_size"])
    img_resized = cv2.resize(img_array, target)
    cnn_proba = _models["cnn"].predict(np.expand_dims(img_resized, axis=0), verbose=0)

    # Process biomarkers
    bio = BiomarkerInput(
        age=age, bmi=bmi, hba1c=hba1c,
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
    X = _biomarker_to_array(bio)
    if _models["scaler"] is not None:
        X = _models["scaler"].transform(X)
    bio_proba = predict_biomarker_proba(_models["biomarker"], X)

    # Late fusion
    grades, risk_scores, fused_proba = unified_prediction(cnn_proba, bio_proba)
    grade = int(grades[0])
    risk_score = float(risk_scores[0])
    heatmap_b64, overlay_b64 = _generate_gradcam_payload(img_resized, grade)

    return _build_response(
        grade, risk_score, fused_proba[0],
        model_used="unified (CNN + Biomarker)",
        grad_cam_available=True,
        grad_cam_heatmap=heatmap_b64,
        grad_cam_overlay=overlay_b64,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

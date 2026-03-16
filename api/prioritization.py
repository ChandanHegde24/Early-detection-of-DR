"""
Screening prioritization logic — maps the AI risk score to clinical tiers.

Tiers:
  - Urgent:   Score >= 0.75 → Immediate ophthalmologist referral
  - Moderate: Score >= 0.45 → Schedule follow-up within weeks
  - Low Risk: Score <  0.45 → Routine annual screening
"""

from typing import Tuple, Dict, Any
from src.config import load_settings

settings = load_settings()

DR_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

TIER_DESCRIPTIONS = {
    "Urgent": "Immediate referral to ophthalmologist recommended. "
              "High probability of sight-threatening retinopathy.",
    "Moderate": "Follow-up appointment within 2-4 weeks recommended. "
                "Signs of non-proliferative diabetic retinopathy detected.",
    "Low Risk": "Routine annual screening sufficient. "
                "No significant signs of diabetic retinopathy detected.",
}


CLINICAL_RULE_WEIGHTS = {
    "sbp_moderate": 0.12,
    "sbp_high": 0.20,
    "hba1c_moderate": 0.15,
    "hba1c_high": 0.25,
    "bmi_moderate": 0.08,
    "bmi_high": 0.12,
    "ldl_moderate": 0.08,
    "ldl_high": 0.12,
    "triglycerides_high": 0.08,
    "duration_moderate": 0.07,
    "duration_high": 0.12,
    "smoking_current": 0.06,
    "family_history": 0.07,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def compute_clinical_rule_score(biomarkers: Any) -> Tuple[float, Dict[str, float]]:
    """Compute baseline risk from expert clinical thresholds.

    This is a transparent rules engine used before deep image analysis.

    Returns:
        (normalized_score, factor_breakdown)
    """
    w = CLINICAL_RULE_WEIGHTS

    factors: Dict[str, float] = {}

    sbp = float(getattr(biomarkers, "blood_pressure_systolic"))
    if sbp >= 160:
        factors["sbp"] = w["sbp_high"]
    elif sbp >= 140:
        factors["sbp"] = w["sbp_moderate"]

    hba1c = float(getattr(biomarkers, "hba1c"))
    if hba1c >= 9.0:
        factors["hba1c"] = w["hba1c_high"]
    elif hba1c >= 7.0:
        factors["hba1c"] = w["hba1c_moderate"]

    bmi = float(getattr(biomarkers, "bmi"))
    if bmi >= 35:
        factors["bmi"] = w["bmi_high"]
    elif bmi >= 30:
        factors["bmi"] = w["bmi_moderate"]

    ldl = float(getattr(biomarkers, "cholesterol_ldl"))
    if ldl >= 160:
        factors["cholesterol_ldl"] = w["ldl_high"]
    elif ldl >= 130:
        factors["cholesterol_ldl"] = w["ldl_moderate"]

    triglycerides = float(getattr(biomarkers, "triglycerides"))
    if triglycerides >= 200:
        factors["triglycerides"] = w["triglycerides_high"]

    duration = float(getattr(biomarkers, "diabetes_duration_years"))
    if duration >= 20:
        factors["diabetes_duration_years"] = w["duration_high"]
    elif duration >= 10:
        factors["diabetes_duration_years"] = w["duration_moderate"]

    smoking_status = int(getattr(biomarkers, "smoking_status"))
    if smoking_status == 2:
        factors["smoking_status"] = w["smoking_current"]

    family_history = int(getattr(biomarkers, "family_history_dr"))
    if family_history == 1:
        factors["family_history_dr"] = w["family_history"]

    raw_score = sum(factors.values())
    normalized = _clamp01(raw_score)
    return normalized, factors


def clinical_recommendation_from_score(score: float) -> str:
    """Return Stage-1 recommendation based on baseline clinical risk."""
    threshold = settings.get("clinical_rules", {}).get("recommend_fundus_threshold", 0.55)
    if score >= float(threshold):
        return "High baseline risk detected. Proceed to retinal fundus scan for DR grading."
    return "Baseline risk below high-risk cutoff. Continue routine monitoring and scheduled screening."


def classify_tier(risk_score: float) -> str:
    """Map a continuous risk score to a screening tier.

    Args:
        risk_score: Value in [0, 1] from the late fusion model.

    Returns:
        One of 'Urgent', 'Moderate', or 'Low Risk'.
    """
    thresholds = settings["prioritization"]["thresholds"]

    if risk_score >= thresholds["urgent"]:
        return "Urgent"
    elif risk_score >= thresholds["moderate"]:
        return "Moderate"
    else:
        return "Low Risk"


def get_tier_description(tier: str) -> str:
    """Return a clinical description for the given tier."""
    return TIER_DESCRIPTIONS.get(tier, "Unknown tier.")


def get_grade_label(grade: int) -> str:
    """Return a human-readable DR grade label."""
    return DR_LABELS.get(grade, f"Unknown (grade={grade})")


def prioritize(risk_score: float, predicted_grade: int) -> Tuple[str, str, str]:
    """Full prioritization pipeline.

    Returns:
        (tier, tier_description, grade_label)
    """
    tier = classify_tier(risk_score)
    description = get_tier_description(tier)
    grade_label = get_grade_label(predicted_grade)
    return tier, description, grade_label

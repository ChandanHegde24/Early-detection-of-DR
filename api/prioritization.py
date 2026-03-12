"""
Screening prioritization logic — maps the AI risk score to clinical tiers.

Tiers:
  - Urgent:   Score >= 0.75 → Immediate ophthalmologist referral
  - Moderate: Score >= 0.45 → Schedule follow-up within weeks
  - Low Risk: Score <  0.45 → Routine annual screening
"""

from typing import Tuple
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

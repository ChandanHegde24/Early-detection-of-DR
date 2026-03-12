"""
Pydantic schemas defining the API request/response data formats.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ─── Request Schemas ────────────────────────────────────────────

class BiomarkerInput(BaseModel):
    """Clinical biomarker data for a single patient."""
    age: float = Field(..., ge=0, le=120, description="Patient age in years")
    bmi: float = Field(..., ge=10, le=70, description="Body Mass Index")
    hba1c: float = Field(..., ge=3.0, le=20.0, description="HbA1c level (%)")
    blood_pressure_systolic: float = Field(..., ge=60, le=250, description="Systolic BP (mmHg)")
    blood_pressure_diastolic: float = Field(..., ge=30, le=150, description="Diastolic BP (mmHg)")
    cholesterol_total: float = Field(..., ge=50, le=500, description="Total cholesterol (mg/dL)")
    cholesterol_hdl: float = Field(..., ge=10, le=150, description="HDL cholesterol (mg/dL)")
    cholesterol_ldl: float = Field(..., ge=10, le=400, description="LDL cholesterol (mg/dL)")
    triglycerides: float = Field(..., ge=30, le=1000, description="Triglycerides (mg/dL)")
    diabetes_duration_years: float = Field(..., ge=0, le=80, description="Years since diabetes diagnosis")
    smoking_status: int = Field(..., ge=0, le=2, description="0=Never, 1=Former, 2=Current")
    family_history_dr: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")


class PredictionRequest(BaseModel):
    """Request schema when only biomarkers are provided (no image)."""
    biomarkers: BiomarkerInput


# ─── Response Schemas ───────────────────────────────────────────

class GradeProbability(BaseModel):
    """Probability for a single DR grade."""
    grade: int = Field(..., description="DR grade (0-4)")
    label: str = Field(..., description="Human-readable grade label")
    probability: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Unified prediction response from the AI system."""
    predicted_grade: int = Field(..., description="Predicted DR grade (0-4)")
    predicted_label: str = Field(..., description="Human-readable prediction")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Continuous risk score")
    screening_tier: str = Field(..., description="Screening priority: Urgent / Moderate / Low Risk")
    grade_probabilities: List[GradeProbability] = Field(
        ..., description="Per-class probability breakdown"
    )
    model_used: str = Field(..., description="Which model(s) contributed to the prediction")
    grad_cam_available: bool = Field(
        default=False, description="Whether a Grad-CAM heatmap was generated"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: dict

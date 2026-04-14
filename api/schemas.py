"""
Pydantic schemas defining the API request/response data formats.
Enhanced with stricter validation.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator


class BiomarkerInput(BaseModel):
    """Clinical biomarker data for a single patient with realistic ranges."""
    age: float = Field(..., ge=18, le=100, description="Patient age in years (18-100)")
    bmi: float = Field(..., ge=12, le=65, description="Body Mass Index (12-65)")
    hba1c: float = Field(..., ge=4.0, le=15.0, description="HbA1c level (%) (4-15)")
    blood_pressure_systolic: float = Field(..., ge=70, le=240, description="Systolic BP (mmHg) (70-240)")
    blood_pressure_diastolic: float = Field(..., ge=40, le=150, description="Diastolic BP (mmHg) (40-150)")
    cholesterol_total: float = Field(..., ge=100, le=400, description="Total cholesterol (mg/dL) (100-400)")
    cholesterol_hdl: float = Field(..., ge=20, le=120, description="HDL cholesterol (mg/dL) (20-120)")
    cholesterol_ldl: float = Field(..., ge=20, le=350, description="LDL cholesterol (mg/dL) (20-350)")
    triglycerides: float = Field(..., ge=30, le=800, description="Triglycerides (mg/dL) (30-800)")
    diabetes_duration_years: float = Field(..., ge=0, le=70, description="Years since diabetes diagnosis (0-70)")
    smoking_status: int = Field(..., ge=0, le=2, description="0=Never, 1=Former, 2=Current")
    family_history_dr: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")

    @field_validator('cholesterol_total')
    @classmethod
    def validate_cholesterol(cls, v, info):
        """Ensure total cholesterol > HDL + LDL minimum."""
        if v < 50:
            raise ValueError('Total cholesterol must be at least 50 mg/dL')
        return v


class PredictionRequest(BaseModel):
    """Request schema when only biomarkers are provided (no image)."""
    biomarkers: BiomarkerInput



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
    baseline_clinical_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Rule-based baseline clinical risk score before image analysis",
    )
    baseline_recommendation: Optional[str] = Field(
        default=None,
        description="Stage-1 recommendation from the clinical rules engine",
    )
    baseline_factor_breakdown: Optional[Dict[str, float]] = Field(
        default=None,
        description="Contribution of each clinical risk factor to baseline score",
    )
    grad_cam_available: bool = Field(
        default=False, description="Whether a Grad-CAM heatmap was generated"
    )
    grad_cam_heatmap: Optional[str] = Field(
        default=None,
        description="Optional base64 data URL for Grad-CAM heatmap image",
    )
    grad_cam_overlay: Optional[str] = Field(
        default=None,
        description="Optional base64 data URL for overlay of heatmap on fundus image",
    )

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: dict
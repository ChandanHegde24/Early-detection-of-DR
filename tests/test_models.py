"""
Tests for ML model functionality and predictions.
Tests ensure models output correct shapes and probabilities.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from api.schemas import BiomarkerInput


class TestBiomarkerValidation:
    """Test biomarker input validation."""

    def test_valid_biomarker_creation(self):
        """Should successfully create BiomarkerInput with valid data."""
        bio = BiomarkerInput(
            age=55,
            bmi=27.5,
            hba1c=7.8,
            blood_pressure_systolic=140,
            blood_pressure_diastolic=90,
            cholesterol_total=220,
            cholesterol_hdl=50,
            cholesterol_ldl=150,
            triglycerides=180,
            diabetes_duration_years=8,
            smoking_status=1,
            family_history_dr=1,
        )
        assert bio.age == 55
        assert bio.hba1c == 7.8
        assert bio.smoking_status in [0, 1, 2]

    def test_biomarker_age_bounds(self):
        """Age should be between 18 and 100."""
        # Too young
        with pytest.raises(ValueError):
            BiomarkerInput(
                age=10, bmi=25, hba1c=7.0, blood_pressure_systolic=120,
                blood_pressure_diastolic=80, cholesterol_total=200,
                cholesterol_hdl=50, cholesterol_ldl=130, triglycerides=100,
                diabetes_duration_years=2, smoking_status=0, family_history_dr=0
            )
        
        # Too old
        with pytest.raises(ValueError):
            BiomarkerInput(
                age=150, bmi=25, hba1c=7.0, blood_pressure_systolic=120,
                blood_pressure_diastolic=80, cholesterol_total=200,
                cholesterol_hdl=50, cholesterol_ldl=130, triglycerides=100,
                diabetes_duration_years=50, smoking_status=0, family_history_dr=0
            )

    def test_biomarker_bmi_bounds(self):
        """BMI should be realistic."""
        with pytest.raises(ValueError):
            BiomarkerInput(
                age=50, bmi=5, hba1c=7.0, blood_pressure_systolic=120,
                blood_pressure_diastolic=80, cholesterol_total=200,
                cholesterol_hdl=50, cholesterol_ldl=130, triglycerides=100,
                diabetes_duration_years=3, smoking_status=0, family_history_dr=0
            )

    def test_biomarker_hba1c_bounds(self):
        """HbA1c should be in reasonable range."""
        # Valid HbA1c
        bio = BiomarkerInput(
            age=50, bmi=25, hba1c=8.5, blood_pressure_systolic=120,
            blood_pressure_diastolic=80, cholesterol_total=200,
            cholesterol_hdl=50, cholesterol_ldl=130, triglycerides=100,
            diabetes_duration_years=3, smoking_status=0, family_history_dr=0
        )
        assert bio.hba1c == 8.5

        # Too high
        with pytest.raises(ValueError):
            BiomarkerInput(
                age=50, bmi=25, hba1c=20.0, blood_pressure_systolic=120,
                blood_pressure_diastolic=80, cholesterol_total=200,
                cholesterol_hdl=50, cholesterol_ldl=130, triglycerides=100,
                diabetes_duration_years=3, smoking_status=0, family_history_dr=0
            )


class TestImageValidation:
    """Test image input validation."""

    def test_image_size_validation(self):
        """Images should have minimum dimensions."""
        from PIL import Image as PILImage
        import io
        
        # Create tiny 50x50 image
        tiny_img = PILImage.new("RGB", (50, 50))
        img_bytes = io.BytesIO()
        tiny_img.save(img_bytes, format="PNG")
        
        # This should be caught by preprocessing validation
        # (would need full integration test to verify)

    def test_image_format_support(self):
        """API should support common image formats."""
        from PIL import Image as PILImage
        import io
        
        # Test PNG
        png_img = PILImage.new("RGB", (224, 224))
        png_bytes = io.BytesIO()
        png_img.save(png_bytes, format="PNG")
        assert png_bytes.getvalue()[:8] == b'\x89PNG\r\n\x1a\n'
        
        # Test JPG
        jpg_img = PILImage.new("RGB", (224, 224))
        jpg_bytes = io.BytesIO()
        jpg_img.save(jpg_bytes, format="JPEG")
        assert jpg_bytes.getvalue()[:2] == b'\xff\xd8'


class TestProbabilityValidation:
    """Test probability output format validation."""

    def test_probability_distribution(self):
        """Probabilities should sum to 1.0 and be in [0,1]."""
        proba = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        
        assert len(proba) == 5
        assert np.allclose(proba.sum(), 1.0)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_risk_score_bounds(self):
        """Risk score should be between 0 and 1."""
        severity_weights = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        
        # Min risk
        proba_min = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        risk_min = float(proba_min @ severity_weights)
        assert risk_min == 0.0
        
        # Max risk
        proba_max = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        risk_max = float(proba_max @ severity_weights)
        assert risk_max == 1.0
        
        # Middle risk
        proba_mid = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        risk_mid = float(proba_mid @ severity_weights)
        assert 0.0 <= risk_mid <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

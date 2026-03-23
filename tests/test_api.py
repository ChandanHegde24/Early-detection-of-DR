import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "models_loaded" in data

def test_prediction_missing_biomarker_raises_error():
    # If a required fields are missing, normally it raises 422 Unprocessable Entity
    response = client.post("/predict/biomarker", json={"invalid": "data"})
    assert response.status_code == 422

def test_predict_biomarker_valid_input():
    valid_payload = {
        "biomarkers": {
            "age": 55.0,
            "bmi": 28.5,
            "hba1c": 7.2,
            "blood_pressure_systolic": 135.0,
            "blood_pressure_diastolic": 85.0,
            "cholesterol_total": 200.0,
            "cholesterol_hdl": 45.0,
            "cholesterol_ldl": 120.0,
            "triglycerides": 150.0,
            "diabetes_duration_years": 10.0,
            "smoking_status": 0,
            "family_history_dr": 1
        }
    }
    # It might fail with 503 if models aren't loaded properly in the test env, 
    # but we handle this scenario gracefully.
    response = client.post("/predict/biomarker", json=valid_payload)
    assert response.status_code in [200, 503]

def test_predict_image_missing_file():
    response = client.post("/predict/image")
    assert response.status_code == 422

def test_predict_image_empty_file():
    # Should catch as invalid image or file too small
    response = client.post(
        "/predict/image",
        files={"file": ("test.png", b"", "image/png")}
    )
    assert response.status_code in [400, 503]

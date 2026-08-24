#!/usr/bin/env python
"""Test image-based prediction endpoint."""
import requests
import json
from pathlib import Path

API_URL = "http://127.0.0.1:8000/predict/image"
IMAGE_PATH = r"C:\Users\chandan hegde\Downloads\Early-detection-of-DR\raw_combined\raw_combined\3\99_right.jpeg"

def test_image_prediction():
    """Test the /predict/image endpoint."""
    image_file = Path(IMAGE_PATH)
    
    if not image_file.exists():
        print(f"❌ Image not found: {IMAGE_PATH}")
        return
    
    print(f"📷 Testing image prediction with: {image_file.name}")
    print(f"📤 Sending to: {API_URL}\n")
    
    try:
        with open(image_file, "rb") as f:
            files = {"file": (image_file.name, f, "image/jpeg")}
            response = requests.post(API_URL, files=files, timeout=120)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Prediction successful!\n")
            print(json.dumps(result, indent=2))
            
            # Extract key results
            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"Predicted Grade: {result['predicted_label']} (Grade {result['predicted_grade']})")
            print(f"Risk Score: {result['risk_score']:.4f}")
            print(f"Screening Tier: {result['screening_tier']}")
            print(f"Model Used: {result['model_used']}")
            print(f"Grad-CAM Available: {result['grad_cam_available']}")
            print("="*60)
        else:
            print(f"\n❌ Error {response.status_code}")
            print(f"Response:\n{response.text}")
    
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_image_prediction()

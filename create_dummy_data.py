import os
import pandas as pd
import numpy as np
from PIL import Image

def generate_dummy_data():
    os.makedirs("data/raw/images", exist_ok=True)
    os.makedirs("data/processed/images", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    # 1. Generate tabular data
    n_samples = 50
    features = [
        "age", "bmi", "hba1c", "blood_pressure_systolic", 
        "blood_pressure_diastolic", "cholesterol_total", "cholesterol_hdl",
        "cholesterol_ldl", "triglycerides", "diabetes_duration_years",
        "smoking_status", "family_history_dr"
    ]
    df = pd.DataFrame(np.random.rand(n_samples, len(features)), columns=features)
    
    df["age"] = np.random.randint(20, 80, n_samples)
    df["bmi"] = np.random.uniform(18, 40, n_samples)
    df["hba1c"] = np.random.uniform(4.5, 12.0, n_samples)
    df["blood_pressure_systolic"] = np.random.randint(100, 180, n_samples)
    df["blood_pressure_diastolic"] = np.random.randint(60, 110, n_samples)
    df["cholesterol_total"] = np.random.randint(120, 300, n_samples)
    df["cholesterol_hdl"] = np.random.randint(30, 80, n_samples)
    df["cholesterol_ldl"] = np.random.randint(50, 200, n_samples)
    df["triglycerides"] = np.random.randint(50, 300, n_samples)
    df["diabetes_duration_years"] = np.random.randint(0, 30, n_samples)
    df["smoking_status"] = np.random.randint(0, 2, n_samples)
    df["family_history_dr"] = np.random.randint(0, 2, n_samples)
    df["dr_grade"] = np.random.randint(0, 5, n_samples)

    df.to_csv("data/raw/clinical_data.csv", index=False)

    # 2. Generate Images
    filenames = []
    labels = []
    for i in range(10):
        fname = f"dummy_{i}.jpg"
        filenames.append(fname)
        labels.append(np.random.randint(0, 5))
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(f"data/raw/images/{fname}")

    # 3. Generate image labels
    labels_df = pd.DataFrame({"filename": filenames, "label": labels})
    labels_df.to_csv("data/raw/image_labels.csv", index=False)
    print("Dummy data generated successfully.")

if __name__ == "__main__":
    generate_dummy_data()
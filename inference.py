#!/usr/bin/env python3
"""Quick inference script: loads saved CNN and predicts on one image."""
import sys
import numpy as np
from pathlib import Path

from src.models.retinal_cnn import load_cnn_model
from src.data_prep.image_loader import preprocess_single_image

DEFAULT_IMAGE = "raw_combined/raw_combined/0/3447_right.jpeg"

def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    p = Path(img_path)
    if not p.exists():
        print(f"Image not found: {p}")
        return

    print(f"Loading model and predicting on: {p}")
    model = load_cnn_model()
    img = preprocess_single_image(str(p))
    proba = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
    labels = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]
    pred = int(np.argmax(proba))
    print(f"Predicted grade: {pred} ({labels[pred]})")
    print("Probabilities:")
    for i, p_val in enumerate(proba):
        print(f"  Grade {i}: {p_val:.4f}")

if __name__ == '__main__':
    main()

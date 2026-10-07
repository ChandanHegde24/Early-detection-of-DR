"""
retinal_cnn_tf.py — Loads the trained CNN model for DR grading.
Loads cnn_model.keras (full model) or falls back to cnn_weights.weights.h5.
Place at: src/models/retinal_cnn_tf.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from pathlib import Path
import keras

IMG_SIZE     = 224
NUM_CLASSES  = 5
GRADE_LABELS = ["No DR","Mild NPDR","Moderate NPDR","Severe NPDR","Proliferative DR"]


def load_cnn_model(path: str = None):
    """
    Load the CNN model. Tries cnn_model.keras first (full model),
    then falls back to cnn_weights.weights.h5.
    Returns a Keras model with .predict() method.
    """
    # Try full model first (.keras file — easiest)
    keras_path = "saved_models/cnn_model.keras"
    h5_path    = "saved_models/cnn_weights.weights.h5"

    if Path(keras_path).exists():
        print(f"[retinal_cnn_tf] Loading full model from {keras_path}...")
        model = keras.models.load_model(keras_path)
        print(f"[retinal_cnn_tf] Loaded ✅ — {model.count_params():,} params")
        return model

    if Path(h5_path).exists():
        print(f"[retinal_cnn_tf] .keras not found, loading weights from {h5_path}...")
        # Build MobileNetV2 architecture (matches friend's training)
        base    = keras.applications.MobileNetV2(
            include_top=False, weights=None,
            input_shape=(IMG_SIZE, IMG_SIZE, 3)
        )
        inp  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
        x    = base(inp, training=False)
        x    = keras.layers.GlobalAveragePooling2D()(x)
        x    = keras.layers.Dense(256, activation="relu")(x)
        x    = keras.layers.BatchNormalization()(x)
        x    = keras.layers.Dropout(0.3)(x)
        out  = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
        model = keras.Model(inp, out)
        _ = model(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))
        model.load_weights(h5_path)
        print(f"[retinal_cnn_tf] Loaded weights ✅ — {model.count_params():,} params")
        return model

    raise FileNotFoundError(
        f"No CNN model found. Checked:\n  {keras_path}\n  {h5_path}"
    )


def predict_fundus(model, image_bytes: bytes) -> dict:
    """Standalone predict — returns dict compatible with PredictionResponse."""
    from PIL import Image
    import io
    pil  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img  = np.array(pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    pred = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    grade = int(np.argmax(pred))
    risk  = float(pred @ [0.0, 0.25, 0.5, 0.75, 1.0])
    tier  = "Urgent" if risk >= 0.72 else "Moderate" if risk >= 0.42 else "Low Risk"
    recs  = [
        "No DR detected. Continue annual screening.",
        "Mild NPDR. Monitor every 12 months.",
        "Moderate NPDR. Referral within 3 months.",
        "Severe NPDR. Urgent referral within 1 month.",
        "Proliferative DR. Immediate referral required.",
    ]
    return {
        "predicted_grade":           grade,
        "predicted_label":           GRADE_LABELS[grade],
        "risk_score":                risk,
        "screening_tier":            tier,
        "grade_probabilities":       [
            {"grade": i, "label": GRADE_LABELS[i], "probability": float(pred[i])}
            for i in range(NUM_CLASSES)
        ],
        "model_used":                "CNN (Keras)",
        "grad_cam_available":        False,
        "grad_cam_heatmap":          None,
        "grad_cam_overlay":          None,
        "baseline_clinical_score":   None,
        "baseline_recommendation":   recs[grade],
        "baseline_factor_breakdown": None,
    }
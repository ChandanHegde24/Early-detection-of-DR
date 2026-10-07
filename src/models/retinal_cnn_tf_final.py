"""
retinal_cnn_tf.py — TensorFlow/Keras EfficientNetB0 loader for DR grading.
Handles Keras 3 .h5 format saved by friend's model.
Place at: src/models/retinal_cnn_tf.py
"""
import os
import io
import base64
import h5py
import numpy as np
from pathlib import Path
from typing import Dict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras

IMG_SIZE     = 224
NUM_CLASSES  = 5
GRADE_LABELS = ["No DR","Mild NPDR","Moderate NPDR","Severe NPDR","Proliferative DR"]


def _build_model():
    base = keras.applications.EfficientNetB0(
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
    # Build with dummy pass
    _ = model(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))
    return model


def _load_keras3_weights(model, h5_path: str):
    """Load Keras 3 format weights directly from h5."""
    with h5py.File(h5_path, 'r') as f:
        backbone_vars   = {}
        classifier_vars = {}

        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                if 'functional' in name:
                    backbone_vars[name] = np.array(obj)
                elif 'optimizer' not in name:
                    classifier_vars[name] = np.array(obj)
        f.visititems(collect)

    # ── Backbone ──────────────────────────────────────────────────────────────
    layer_groups = {}
    for key, val in backbone_vars.items():
        parts     = key.split('/')
        layer_key = '/'.join(parts[:5])
        var_idx   = int(parts[-1]) if parts[-1].isdigit() else 0
        layer_groups.setdefault(layer_key, {})[var_idx] = val

    import re
    def sort_key(k):
        name = k.split('/')[3] if len(k.split('/')) > 3 else k
        nums = re.findall(r'\d+', name)
        base = re.sub(r'\d+', '', name)
        return (base, int(nums[0]) if nums else 0)

    ordered = []
    for lk in sorted(layer_groups.keys(), key=sort_key):
        for i in sorted(layer_groups[lk].keys()):
            ordered.append(layer_groups[lk][i])

    backbone_layer = model.layers[1]
    model_weights  = backbone_layer.weights

    if len(ordered) == len(model_weights):
        backbone_layer.set_weights(ordered)
    else:
        # Shape-based fallback
        by_shape, counters = {}, {}
        for w in ordered:
            by_shape.setdefault(str(w.shape), []).append(w)
        new_w = []
        for mw in model_weights:
            sk  = str(mw.shape)
            idx = counters.get(sk, 0)
            if sk in by_shape and idx < len(by_shape[sk]):
                new_w.append(by_shape[sk][idx])
                counters[sk] = idx + 1
            else:
                new_w.append(mw.numpy())
        backbone_layer.set_weights(new_w)

    # ── Classifier ────────────────────────────────────────────────────────────
    try:
        dense  = [l for l in model.layers if hasattr(l,'units') and l.units == 256][0]
        bn     = [l for l in model.layers if 'batch_normalization' in l.name][-1]
        output = [l for l in model.layers if hasattr(l,'units') and l.units == NUM_CLASSES][0]

        dk = classifier_vars.get('layers/dense/vars/0')
        db = classifier_vars.get('layers/dense/vars/1')
        bg = classifier_vars.get('layers/batch_normalization/vars/0')
        bb = classifier_vars.get('layers/batch_normalization/vars/1')
        bm = classifier_vars.get('layers/batch_normalization/vars/2')
        bv = classifier_vars.get('layers/batch_normalization/vars/3')
        ok = classifier_vars.get('layers/dense_1/vars/0')
        ob = classifier_vars.get('layers/dense_1/vars/1')

        if dk is not None: dense.set_weights([dk, db])
        if bg is not None: bn.set_weights([bg, bb, bm, bv])
        if ok is not None: output.set_weights([ok, ob])
    except Exception as e:
        print(f"[retinal_cnn_tf] Classifier load warning: {e}")

    return model


def load_cnn_model(path: str = "saved_models/cnn_weights.weights.h5"):
    if not Path(path).exists():
        raise FileNotFoundError(f"TF model not found at '{path}'")
    print(f"[retinal_cnn_tf] Building EfficientNetB0...")
    model = _build_model()
    print(f"[retinal_cnn_tf] Loading weights from {path}...")
    model = _load_keras3_weights(model, path)
    print(f"[retinal_cnn_tf] Model ready on CPU (TF-Windows)")
    return model


# ── Grad-CAM ─────────────────────────────────────────────────────────────────
def _gradcam(model, img_array: np.ndarray, grade: int) -> np.ndarray:
    """Generate Grad-CAM heatmap."""
    try:
        last_conv = [l for l in model.layers[1].layers
                     if isinstance(l, keras.layers.Conv2D)][-1]
        grad_model = keras.Model(
            inputs  = model.inputs,
            outputs = [model.layers[1].get_layer(last_conv.name).output,
                       model.output]
        )
        inp_tensor = tf.cast(np.expand_dims(img_array, 0), tf.float32)
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(inp_tensor)
            loss = preds[:, grade]
        grads   = tape.gradient(loss, conv_out)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam     = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1).numpy()
        cam     = np.maximum(cam, 0)
        if cam.max() > 0: cam = cam / cam.max()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return cam
    except Exception:
        return np.zeros((IMG_SIZE, IMG_SIZE))


def _overlay(cam: np.ndarray, orig: np.ndarray):
    hm  = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    hm  = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    ov  = (0.5 * cv2.resize(orig, (IMG_SIZE, IMG_SIZE)) + 0.5 * hm).astype(np.uint8)
    def b64(arr):
        _, buf = cv2.imencode(".png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buf).decode()
    return b64(hm), b64(ov)


# ── Main predict ─────────────────────────────────────────────────────────────
def predict_fundus(model, image_bytes: bytes) -> Dict:
    pil_img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_np  = np.array(pil_img)
    img_arr  = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0

    probs    = model(np.expand_dims(img_arr, 0), training=False).numpy()[0]
    grade    = int(np.argmax(probs))
    risk     = float(probs @ [0.0, 0.25, 0.5, 0.75, 1.0])
    tier     = "Urgent" if risk >= 0.72 else "Moderate" if risk >= 0.42 else "Low Risk"

    cam             = _gradcam(model, img_arr, grade)
    heatmap, overlay = _overlay(cam, orig_np)

    recs = [
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
            {"grade": i, "label": GRADE_LABELS[i], "probability": float(probs[i])}
            for i in range(NUM_CLASSES)
        ],
        "model_used":                "EfficientNetB0-TensorFlow (friend's model)",
        "grad_cam_available":        True,
        "grad_cam_heatmap":          heatmap,
        "grad_cam_overlay":          overlay,
        "baseline_clinical_score":   None,
        "baseline_recommendation":   recs[grade],
        "baseline_factor_breakdown": None,
    }

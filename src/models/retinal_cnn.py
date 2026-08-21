"""
CNN-based retinal image classification using Transfer Learning.

Supports EfficientNetV2B0 (EfficientNetB0), MobileNetV2, and ResNet50 backbones
for classifying fundus images into DR severity grades (0–4).
"""

import os
from typing import Optional, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    ResNet50,
    EfficientNetV2B0,
    EfficientNetV2S,
    MobileNetV2,
)

from src.config import load_settings

settings = load_settings()

BACKBONE_MAP = {
    "ResNet50": ResNet50,
    "EfficientNetB0": EfficientNetV2B0,
    "EfficientNetV2B0": EfficientNetV2B0,
    "EfficientNetB3": EfficientNetV2S,
    "MobileNetV2": MobileNetV2,
}


def build_cnn_model(
    backbone_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    input_shape: Optional[Tuple[int, int, int]] = None,
    freeze_base: Optional[bool] = None,
    dropout_rate: Optional[float] = None,
    learning_rate: Optional[float] = None,
) -> Model:
    """Build a transfer-learning CNN for retinal image classification.

    Architecture:
        Pretrained Backbone → GlobalAveragePooling → Dense(256) → BatchNorm → Dropout → Softmax(5)
    """
    cfg = settings.get("cnn", {})
    backbone_name = backbone_name or cfg.get("backbone", "MobileNetV2")
    num_classes = num_classes or cfg.get("num_classes", 5)
    dropout_rate = dropout_rate if dropout_rate is not None else cfg.get("dropout_rate", 0.3)
    freeze_base = freeze_base if freeze_base is not None else cfg.get("freeze_base", True)
    learning_rate = learning_rate or cfg.get("learning_rate", 1e-4)

    target_size = settings.get("image", {}).get("target_size", [224, 224])
    input_shape = input_shape or (target_size[1], target_size[0], settings.get("image", {}).get("channels", 3))

    if backbone_name not in BACKBONE_MAP:
        raise ValueError(
            f"Unsupported backbone: {backbone_name}. "
            f"Choose from: {list(BACKBONE_MAP.keys())}"
        )

    base_model = BACKBONE_MAP[backbone_name](
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base_model.trainable = not freeze_base

    inputs = keras.Input(shape=input_shape, name="fundus_input")
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="rescaling")(inputs)
    x = base_model(x, training=False if freeze_base else True)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="dense_features")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="dr_grade_output")(x)

    model = Model(inputs, outputs, name=f"DR_{backbone_name}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def unfreeze_and_fine_tune(
    model: Model,
    fine_tune_at: Optional[int] = None,
    learning_rate: float = 1e-5,
) -> Model:
    """Unfreeze top layers in the base model for fine-tuning."""
    fine_tune_at = fine_tune_at or settings.get("cnn", {}).get("fine_tune_at", 100)

    base_model = None
    for layer in model.layers:
        if isinstance(layer, Model):
            base_model = layer
            break

    if base_model is None:
        base_model = model.layers[1]

    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_callbacks(checkpoint_path: Optional[str] = None) -> list:
    """Return standard Keras callbacks for training."""
    cfg = settings.get("cnn", {})
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=cfg.get("early_stopping_patience", 5),
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        ckpt_file = checkpoint_path
        if not ckpt_file.endswith(".weights.h5"):
            if ckpt_file.endswith(".h5"):
                ckpt_file = ckpt_file[:-3] + ".weights.h5"
            else:
                ckpt_file = ckpt_file + ".weights.h5"
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=ckpt_file,
                monitor="val_accuracy",
                save_best_only=True,
                save_weights_only=True,
                mode="max",
                verbose=1,
            )
        )
    return callbacks


def save_cnn_model(model: Model, path: Optional[str] = None) -> str:
    """Save CNN weights and full model to disk."""
    saved_dir = settings.get("paths", {}).get("saved_models", "saved_models")
    os.makedirs(saved_dir, exist_ok=True)
    
    weights_path = path or os.path.join(saved_dir, "cnn_weights.weights.h5")
    if not weights_path.endswith(".weights.h5"):
        if weights_path.endswith(".h5"):
            weights_path = weights_path[:-3] + ".weights.h5"
        else:
            weights_path = weights_path + ".weights.h5"
    
    model.save_weights(weights_path)
    print(f"[retinal_cnn] Saved weights to {weights_path}")

    try:
        keras_path = os.path.join(saved_dir, "cnn_model.keras")
        model.save(keras_path)
        print(f"[retinal_cnn] Saved full model to {keras_path}")
    except Exception as e:
        print(f"[retinal_cnn] Notice: could not save .keras format: {e}")
        
    return weights_path


def load_cnn_model(path: Optional[str] = None,
                   backbone_name: Optional[str] = None) -> Model:
    """Rebuild the model architecture and load saved weights."""
    import json
    saved_dir = settings.get("paths", {}).get("saved_models", "saved_models")
    keras_path = os.path.join(saved_dir, "cnn_model.keras")
    
    # Try loading from .keras if available
    if path and path.endswith(".keras") and os.path.exists(path):
        try:
            return keras.models.load_model(path)
        except Exception:
            pass
    if os.path.exists(keras_path):
        try:
            return keras.models.load_model(keras_path)
        except Exception:
            pass

    # Read backbone from training summary if available
    if backbone_name is None:
        summary_path = os.path.join(saved_dir, "cnn_training_summary.json")
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r") as f:
                    meta = json.load(f)
                    if "backbone" in meta:
                        backbone_name = meta["backbone"]
            except Exception:
                pass

    candidates = [
        path,
        os.path.join(saved_dir, "cnn_weights.weights.h5"),
        os.path.join(saved_dir, "cnn_weights.h5"),
    ]
    
    weights_path = None
    for cand in candidates:
        if cand and os.path.exists(cand):
            weights_path = cand
            break

    if not weights_path:
        raise FileNotFoundError("Model weights not found in saved_models directory.")
        
    model = build_cnn_model(backbone_name=backbone_name)
    model.load_weights(weights_path)
    return model


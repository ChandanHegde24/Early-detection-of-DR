"""
CNN-based retinal image classification using Transfer Learning.

Supports ResNet50, EfficientNetB0, and EfficientNetB3 backbones
for classifying fundus images into DR severity grades (0–4).
"""

from typing import Optional, Tuple

import tensorflow as tf
import keras
from keras import layers, Model
from keras.applications import (
    ResNet50,
    EfficientNetB0,
    EfficientNetB3,
)

from src.config import load_settings

settings = load_settings()

BACKBONE_MAP = {
    "ResNet50": ResNet50,
    "EfficientNetB0": EfficientNetB0,
    "EfficientNetB3": EfficientNetB3,
}


def build_cnn_model(
    backbone_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    input_shape: Optional[Tuple[int, int, int]] = None,
    freeze_base: Optional[bool] = None,
    dropout_rate: Optional[float] = None,
) -> Model:
    """Build a transfer-learning CNN for retinal image classification.

    Architecture:
        Pretrained Backbone → GlobalAveragePooling → Dense(256) → Dropout → Softmax

    Args:
        backbone_name: One of 'ResNet50', 'EfficientNetB0', 'EfficientNetB3'.
        num_classes: Number of output classes (default: 5 for DR grading).
        input_shape: (height, width, channels). Defaults to settings value.
        freeze_base: Whether to freeze backbone weights.
        dropout_rate: Dropout probability before the output layer.

    Returns:
        Compiled Keras Model.
    """
    cfg = settings["cnn"]
    backbone_name = backbone_name or cfg["backbone"]
    num_classes = num_classes or cfg["num_classes"]
    dropout_rate = dropout_rate or cfg["dropout_rate"]
    freeze_base = freeze_base if freeze_base is not None else cfg["freeze_base"]

    target_size = settings["image"]["target_size"]
    input_shape = input_shape or (target_size[1], target_size[0], settings["image"]["channels"])

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

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=not freeze_base)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name=f"DR_{backbone_name}")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def unfreeze_and_fine_tune(
    model: Model,
    fine_tune_at: Optional[int] = None,
    learning_rate: float = 1e-5,
) -> Model:
    """Unfreeze layers in the base model for fine-tuning.

    Unfreezes all layers from index `fine_tune_at` onwards and recompiles
    the model with a lower learning rate.
    """
    fine_tune_at = fine_tune_at or settings["cnn"]["fine_tune_at"]

    base_model = model.layers[1]  # The backbone is the second layer
    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_callbacks() -> list:
    """Return standard Keras callbacks for training."""
    cfg = settings["cnn"]
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg["early_stopping_patience"],
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
        ),
    ]


def save_cnn_model(model: Model, path: Optional[str] = None) -> str:
    """Save CNN weights to disk."""
    path = path or f"{settings['paths']['saved_models']}/cnn_weights.h5"
    model.save_weights(path)
    return path


def load_cnn_model(path: Optional[str] = None,
                   backbone_name: Optional[str] = None) -> Model:
    """Rebuild the model architecture and load saved weights."""
    path = path or f"{settings['paths']['saved_models']}/cnn_weights.h5"
    model = build_cnn_model(backbone_name=backbone_name)
    model.load_weights(path)
    return model

"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN interpretability.

Generates heatmaps highlighting the regions of retinal fundus images that
the CNN considers most important for its DR classification decision.
This is critical for clinical trust and model transparency.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf
import keras
from keras import Model

from src.config import load_settings

settings = load_settings()


def find_target_layer(model: Model) -> str:
    """Automatically find the last convolutional layer in the model.

    Searches through the model (and any nested sub-models) to find the
    last Conv2D layer, which typically produces the best Grad-CAM results.

    Returns:
        Name of the last convolutional layer.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, keras.layers.Conv2D):
                    return sub_layer.name
        if isinstance(layer, keras.layers.Conv2D):
            return layer.name

    raise ValueError("No Conv2D layer found in the model.")


def _get_nested_layer(model: Model, layer_name: str):
    """Retrieve a layer by name, searching nested sub-models."""
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
        if isinstance(layer, Model):
            try:
                return layer.get_layer(layer_name)
            except ValueError:
                continue
    raise ValueError(f"Layer '{layer_name}' not found in model.")


def generate_grad_cam(
    model: Model,
    image: np.ndarray,
    target_class: Optional[int] = None,
    layer_name: Optional[str] = None,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for a given image and model.

    Args:
        model: Trained Keras model.
        image: Preprocessed image array of shape (H, W, 3), values in [0, 1].
        target_class: Class index to explain. If None, uses the predicted class.
        layer_name: Name of the Conv2D layer to use. Auto-detected if None.

    Returns:
        Heatmap array of shape (H, W) with values in [0, 1].
    """
    layer_name = layer_name or find_target_layer(model)

    img_tensor = tf.expand_dims(tf.cast(image, tf.float32), axis=0)

    base_model = None
    for layer in model.layers:
        if isinstance(layer, Model):
            try:
                layer.get_layer(layer_name)
                base_model = layer
                break
            except ValueError:
                continue

    if base_model is not None:
        conv_layer = base_model.get_layer(layer_name)
        grad_model = Model(
            inputs=model.input,
            outputs=[base_model.get_layer(layer_name).output, model.output],
        )
    else:
        grad_model = Model(
            inputs=model.input,
            outputs=[model.get_layer(layer_name).output, model.output],
        )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if target_class is None:
            target_class = tf.argmax(predictions[0])
        class_output = predictions[:, target_class]

    grads = tape.gradient(class_output, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on the original image.

    Args:
        image: Original image of shape (H, W, 3), values in [0, 1] or [0, 255].
        heatmap: Grad-CAM heatmap of shape (h, w), values in [0, 1].
        alpha: Blending factor (0 = only image, 1 = only heatmap).
        colormap: OpenCV colormap for the heatmap visualization.

    Returns:
        Blended image of shape (H, W, 3) as uint8 in [0, 255].
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay

def explain_prediction(
    model: Model,
    image: np.ndarray,
    target_class: Optional[int] = None,
    layer_name: Optional[str] = None,
    alpha: float = 0.4,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Full Grad-CAM explanation pipeline.

    Returns:
        (overlay_image, heatmap, predicted_class, predicted_confidence)
    """
    img_tensor = tf.expand_dims(tf.cast(image, tf.float32), axis=0)
    predictions = model.predict(img_tensor, verbose=0)[0]
    predicted_class = int(np.argmax(predictions))
    confidence = float(predictions[predicted_class])

    explain_class = target_class if target_class is not None else predicted_class
    heatmap = generate_grad_cam(model, image, target_class=explain_class, layer_name=layer_name)
    overlay = overlay_heatmap(image, heatmap, alpha=alpha)

    return overlay, heatmap, predicted_class, confidence

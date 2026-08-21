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
from tensorflow import keras
from tensorflow.keras import Model

from src.config import load_settings

settings = load_settings()


def find_base_model(model: Model) -> Optional[Model]:
    """Find nested backbone Model in the outer Model."""
    for l in model.layers:
        if isinstance(l, Model):
            return l
    return None


def find_target_layer(model: Model) -> str:
    """Automatically find the last convolutional layer in the model."""
    base_model = find_base_model(model)
    if base_model is not None:
        for layer in reversed(base_model.layers):
            if "conv" in layer.name or isinstance(layer, keras.layers.Conv2D):
                return layer.name

    for layer in reversed(model.layers):
        if "conv" in layer.name or isinstance(layer, keras.layers.Conv2D):
            return layer.name

    return "top_conv"


def generate_grad_cam(
    model: Model,
    image: np.ndarray,
    target_class: Optional[int] = None,
    layer_name: Optional[str] = None,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for a given image and model."""
    img_tensor = tf.expand_dims(tf.cast(image, tf.float32), axis=0)

    base_model = find_base_model(model)
    if base_model is not None:
        base_idx = model.layers.index(base_model)
        layer_name = layer_name or find_target_layer(model)
        
        try:
            last_conv_layer = base_model.get_layer(layer_name)
        except Exception:
            last_conv_layer = [l for l in base_model.layers if "conv" in l.name][-1]

        # Pass image through layers before base_model (e.g. Rescaling)
        intermediate_tensor = img_tensor
        for l in model.layers[1:base_idx]:
            intermediate_tensor = l(intermediate_tensor)

        last_conv_model = keras.Model(base_model.inputs, last_conv_layer.output)

        classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        for layer in model.layers[base_idx + 1:]:
            x = layer(x)
        classifier_model = keras.Model(classifier_input, x)

        with tf.GradientTape() as tape:
            conv_outputs = last_conv_model(intermediate_tensor)
            tape.watch(conv_outputs)
            predictions = classifier_model(conv_outputs)
            if target_class is None:
                target_class = int(tf.argmax(predictions[0]))
            class_output = predictions[:, target_class]

        grads = tape.gradient(class_output, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    # Fallback for flat models
    layer_name = layer_name or find_target_layer(model)
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if target_class is None:
            target_class = int(tf.argmax(predictions[0]))
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
    """Overlay a Grad-CAM heatmap on the original image."""
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
    """Full Grad-CAM explanation pipeline."""
    img_tensor = tf.expand_dims(tf.cast(image, tf.float32), axis=0)
    predictions = model.predict(img_tensor, verbose=0)[0]
    predicted_class = int(np.argmax(predictions))
    confidence = float(predictions[predicted_class])

    explain_class = target_class if target_class is not None else predicted_class
    heatmap = generate_grad_cam(model, image, target_class=explain_class, layer_name=layer_name)
    overlay = overlay_heatmap(image, heatmap, alpha=alpha)

    return overlay, heatmap, predicted_class, confidence

"""
Image loading, preprocessing, and augmentation pipelines for retinal fundus images.

Handles:
- Loading images from disk
- CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement
- Resizing to model input dimensions
- Data augmentation using Albumentations
- Creating TensorFlow datasets for training/validation
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import cv2
import numpy as np
import tensorflow as tf
import albumentations as A

from src.config import load_settings

settings = load_settings()


def apply_clahe(image: np.ndarray,
                clip_limit: float = None,
                tile_grid_size: Tuple[int, int] = None) -> np.ndarray:
    """Apply CLAHE to the green channel of a fundus image for contrast enhancement.

    The green channel of retinal images contains the most diagnostically
    relevant information (vessel visibility, microaneurysms, exudates).
    """
    clip_limit = clip_limit or settings["image"]["clahe_clip_limit"]
    tile_grid_size = tuple(tile_grid_size or settings["image"]["clahe_tile_grid"])

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def resize_image(image: np.ndarray,
                 target_size: Tuple[int, int] = None) -> np.ndarray:
    """Resize image to the target dimensions specified in settings."""
    target_size = tuple(target_size or settings["image"]["target_size"])
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def crop_to_circle(image: np.ndarray) -> np.ndarray:
    """Crop circular fundus region from a black-bordered retinal image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return image[y:y + h, x:x + w]


def get_augmentation_pipeline(is_training: bool = True) -> A.Compose:
    """Build an Albumentations augmentation pipeline based on settings."""
    aug_cfg = settings["augmentation"]
    target_size = tuple(settings["image"]["target_size"])

    if is_training:
        return A.Compose([
            A.Resize(height=target_size[1], width=target_size[0]),
            A.HorizontalFlip(p=0.5 if aug_cfg["horizontal_flip"] else 0.0),
            A.VerticalFlip(p=0.5 if aug_cfg["vertical_flip"] else 0.0),
            A.Rotate(limit=aug_cfg["rotation_limit"], p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=aug_cfg["brightness_limit"],
                contrast_limit=aug_cfg["contrast_limit"],
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ])
    else:
        return A.Compose([
            A.Resize(height=target_size[1], width=target_size[0]),
        ])


def preprocess_single_image(image_path: str,
                            apply_clahe_flag: bool = True) -> np.ndarray:
    """Load and preprocess a single retinal image.

    Returns a float32 array normalized to [0, 1].
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = crop_to_circle(image)

    if apply_clahe_flag:
        image = apply_clahe(image)

    image = resize_image(image)
    return image.astype(np.float32) / 255.0


def load_image_dataset(
    image_dir: str,
    labels: dict,
    is_training: bool = True,
    batch_size: Optional[int] = None,
) -> tf.data.Dataset:
    """Create a tf.data.Dataset from a directory of images and a label mapping.

    Args:
        image_dir: Path to the directory containing images.
        labels: Dict mapping filename (str) -> label (int).
        is_training: Whether to apply training augmentations.
        batch_size: Batch size (defaults to settings value).

    Returns:
        A batched, prefetched tf.data.Dataset yielding (image, label) pairs.
    """
    batch_size = batch_size or settings["cnn"]["batch_size"]
    aug_pipeline = get_augmentation_pipeline(is_training)
    target_size = tuple(settings["image"]["target_size"])

    filepaths = []
    file_labels = []
    for fname, label in labels.items():
        full_path = os.path.join(image_dir, fname)
        if os.path.exists(full_path):
            filepaths.append(full_path)
            file_labels.append(label)

    def generator():
        indices = np.arange(len(filepaths))
        if is_training:
            np.random.shuffle(indices)
        for idx in indices:
            img = preprocess_single_image(filepaths[idx])
            augmented = aug_pipeline(image=(img * 255).astype(np.uint8))
            img_aug = augmented["image"].astype(np.float32) / 255.0
            yield img_aug, file_labels[idx]

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(target_size[1], target_size[0], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )

    if is_training:
        dataset = dataset.shuffle(buffer_size=512)

    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

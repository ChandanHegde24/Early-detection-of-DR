"""
Image loading, preprocessing, and augmentation pipelines for retinal fundus images.

Handles:
- Scanning structured datasets (e.g. raw_combined/0, 1, 2, 3, 4)
- Stratified train / validation / test splits
- Class weight calculation for imbalanced classes
- CLAHE (Contrast Limited Adaptive Histogram Equalization) enhancement
- Circular fundus ROI cropping
- High-speed OpenCV/NumPy data augmentation
- High-performance parallel tf.data.Dataset generation
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import random

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.config import load_settings

settings = load_settings()


def apply_clahe(image: np.ndarray,
                clip_limit: float = None,
                tile_grid_size: Tuple[int, int] = None) -> np.ndarray:
    """Apply CLAHE to the green channel of a fundus image for contrast enhancement."""
    clip_limit = clip_limit or settings.get("image", {}).get("clahe_clip_limit", 2.0)
    tile_grid_size = tuple(tile_grid_size or settings.get("image", {}).get("clahe_tile_grid", [8, 8]))

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    merged = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def resize_image(image: np.ndarray,
                 target_size: Tuple[int, int] = None) -> np.ndarray:
    """Resize image to target dimensions."""
    target_size = tuple(target_size or settings.get("image", {}).get("target_size", [224, 224]))
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def crop_to_circle(image: np.ndarray) -> np.ndarray:
    """Crop circular fundus region from a black-bordered retinal image."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        if w > 20 and h > 20:
            return image[y:y + h, x:x + w]
    except Exception:
        pass
    return image


def augment_image_cv2(image: np.ndarray, target_size: Tuple[int, int] = (224, 224), is_training: bool = True) -> np.ndarray:
    """Fast, dependency-free image augmentation with OpenCV."""
    if not is_training:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    h, w = image.shape[:2]
    
    # 1. Random horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # 2. Random vertical flip
    if random.random() > 0.5:
        image = cv2.flip(image, 0)

    # 3. Random rotation (-25 to +25 degrees)
    if random.random() > 0.5:
        angle = random.uniform(-25, 25)
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # 4. Resize to target size
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # 5. Random brightness & contrast
    if random.random() > 0.5:
        alpha = random.uniform(0.85, 1.15)  # Contrast
        beta = random.uniform(-20, 20)      # Brightness
        image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)

    return image


def preprocess_single_image(image_path: str,
                            apply_clahe_flag: bool = True,
                            target_size: Tuple[int, int] = None) -> np.ndarray:
    """Load and preprocess a single retinal image for model inference.

    Returns float32 RGB array normalized to [0, 1].
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_to_circle(image)
    if apply_clahe_flag:
        image = apply_clahe(image)
    image = resize_image(image, target_size=target_size)
    return image.astype(np.float32) / 255.0


def scan_dataset_directory(data_dir: str) -> Tuple[List[str], List[int], Dict[int, int]]:
    """Recursively search for class folders 0, 1, 2, 3, 4 in data_dir.

    Returns:
        filepaths: List of absolute or relative image file paths
        labels: List of integer class labels (0..4)
        class_counts: Dict mapping class_id -> count
    """
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    
    candidates = [
        data_dir,
        os.path.join(data_dir, "raw_combined"),
        os.path.join(data_dir, "raw"),
        os.path.join(data_dir, "images"),
    ]

    target_root = None
    for cand in candidates:
        if os.path.exists(cand):
            subdirs = [d for d in os.listdir(cand) if os.path.isdir(os.path.join(cand, d))]
            if all(str(c) in subdirs for c in range(5)):
                target_root = cand
                break

    if target_root is None:
        target_root = data_dir

    filepaths = []
    labels = []
    class_counts = {i: 0 for i in range(5)}

    for class_id in range(5):
        class_folder = os.path.join(target_root, str(class_id))
        if not os.path.isdir(class_folder):
            for root, dirs, _ in os.walk(target_root):
                if os.path.basename(root) == str(class_id):
                    class_folder = root
                    break

        if os.path.isdir(class_folder):
            for fname in os.listdir(class_folder):
                ext = os.path.splitext(fname)[1].lower()
                if ext in valid_exts:
                    full_path = os.path.join(class_folder, fname)
                    filepaths.append(full_path)
                    labels.append(class_id)
                    class_counts[class_id] += 1

    return filepaths, labels, class_counts


def get_stratified_split(
    filepaths: List[str],
    labels: List[int],
    val_size: float = 0.1,
    test_size: float = 0.1,
    subset_per_class: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """Create stratified train/val/test splits with optional class subsampling."""
    filepaths = np.array(filepaths)
    labels = np.array(labels)

    if subset_per_class is not None and subset_per_class > 0:
        selected_paths = []
        selected_labels = []
        rng = np.random.default_rng(random_state)
        for c in range(5):
            idx = np.where(labels == c)[0]
            if len(idx) > 0:
                sampled = rng.choice(idx, size=min(len(idx), subset_per_class), replace=False)
                selected_paths.extend(filepaths[sampled])
                selected_labels.extend(labels[sampled])
        filepaths = np.array(selected_paths)
        labels = np.array(selected_labels)

    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        filepaths, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    adjusted_val_size = val_size / (1.0 - test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=adjusted_val_size,
        stratify=train_val_labels,
        random_state=random_state,
    )

    return (
        list(train_paths), list(train_labels),
        list(val_paths), list(val_labels),
        list(test_paths), list(test_labels),
    )


def compute_balanced_class_weights(labels: List[int]) -> Dict[int, float]:
    """Compute class weights to balance gradient updates during training."""
    unique_classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=labels,
    )
    class_weights_dict = {int(cls): float(w) for cls, w in zip(unique_classes, weights)}
    for c in range(5):
        if c not in class_weights_dict:
            class_weights_dict[c] = 1.0
    return class_weights_dict


def create_tf_dataset(
    filepaths: List[str],
    labels: List[int],
    is_training: bool = True,
    batch_size: int = 32,
    target_size: Tuple[int, int] = (224, 224),
    apply_clahe_flag: bool = True,
) -> tf.data.Dataset:
    """Create an optimized tf.data.Dataset using parallel mapping."""

    def _process_path(path_bytes, label_val):
        path_str = path_bytes.numpy().decode("utf-8")
        try:
            img = cv2.imread(path_str)
            if img is None:
                img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = crop_to_circle(img)
                if apply_clahe_flag:
                    img = apply_clahe(img)
            
            img_aug = augment_image_cv2(img, target_size=target_size, is_training=is_training)
            img_processed = img_aug.astype(np.float32) / 255.0
        except Exception:
            img_processed = np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)

        return img_processed, np.int32(label_val)

    def _tf_map(path_tensor, label_tensor):
        img_out, lbl_out = tf.py_function(
            func=_process_path,
            inp=[path_tensor, label_tensor],
            Tout=[tf.float32, tf.int32],
        )
        img_out.set_shape((target_size[1], target_size[0], 3))
        lbl_out.set_shape(())
        return img_out, lbl_out

    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if is_training:
        ds = ds.shuffle(buffer_size=min(len(filepaths), 2048), reshuffle_each_iteration=True)
    
    ds = ds.map(_tf_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

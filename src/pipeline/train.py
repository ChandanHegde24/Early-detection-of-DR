"""
train.py — Diabetic Retinopathy Unified AI Training Pipeline

Trains:
  1. CNN Image Model on raw_combined dataset (EfficientNetB0 / MobileNetV2 / ResNet50)
  2. Clinical Biomarker Stacking Ensemble (XGBoost + GradientBoosting + HistGB + RF)

Usage:
  python -m src.pipeline.train --model all
  python -m src.pipeline.train --model cnn --subset 200
  python -m src.pipeline.train --model biomarker
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, Dict

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.data_prep.image_loader import (
    scan_dataset_directory,
    get_stratified_split,
    compute_balanced_class_weights,
    create_tf_dataset,
    preprocess_single_image,
)
from src.data_prep.tabular_prep import prepare_tabular_data
from src.models.retinal_cnn import (
    build_cnn_model,
    unfreeze_and_fine_tune,
    get_callbacks,
    save_cnn_model,
)
from src.models.biomarker_rf import (
    train_biomarker_model,
    evaluate_biomarker_model,
    save_biomarker_model,
)

settings = load_settings()

CLASS_NAMES = ["No DR (0)", "Mild NPDR (1)", "Moderate NPDR (2)", "Severe NPDR (3)", "Proliferative DR (4)"]


def train_cnn_pipeline(
    data_dir: Optional[str] = None,
    backbone: Optional[str] = None,
    epochs: int = 10,
    fine_tune_epochs: int = 3,
    batch_size: int = 32,
    subset_per_class: Optional[int] = None,
    learning_rate: float = 1e-4,
    save_weights_path: Optional[str] = None,
) -> Dict:
    """Train transfer learning CNN on the retinal fundus dataset."""
    saved_dir = settings.get("paths", {}).get("saved_models", "saved_models")
    os.makedirs(saved_dir, exist_ok=True)
    weights_path = save_weights_path or os.path.join(saved_dir, "cnn_weights.weights.h5")

    data_dir = data_dir or settings.get("paths", {}).get("raw_images", "raw_combined")
    backbone = backbone or settings.get("cnn", {}).get("backbone", "EfficientNetB0")

    print("\n" + "=" * 60)
    print(f"[*] TRAINING RETINAL CNN MODEL [{backbone}]")
    print("=" * 60)
    print(f"[cnn_train] Scanning dataset in '{data_dir}'...")

    filepaths, labels, class_counts = scan_dataset_directory(data_dir)
    total_imgs = len(filepaths)
    print(f"[cnn_train] Found {total_imgs} total images across 5 classes:")
    for c_id in range(5):
        print(f"  * Class {c_id} ({CLASS_NAMES[c_id]}): {class_counts[c_id]} images")

    if total_imgs == 0:
        raise FileNotFoundError(f"No images found in '{data_dir}'. Please verify dataset path.")

    if subset_per_class:
        print(f"[cnn_train] Subset mode active: max {subset_per_class} images per class.")

    # Split dataset
    train_p, train_l, val_p, val_l, test_p, test_l = get_stratified_split(
        filepaths=filepaths,
        labels=labels,
        val_size=0.1,
        test_size=0.1,
        subset_per_class=subset_per_class,
        random_state=42,
    )
    print(f"[cnn_train] Split sizes -> Train: {len(train_p)} | Val: {len(val_p)} | Test: {len(test_p)}")

    # Class weights for imbalanced classes
    class_weights = compute_balanced_class_weights(train_l)
    print(f"[cnn_train] Computed class weights:")
    for c_id, w in class_weights.items():
        print(f"  * Class {c_id}: {w:.3f}")

    target_size = tuple(settings.get("image", {}).get("target_size", [224, 224]))

    # Build tf.data datasets
    train_ds = create_tf_dataset(train_p, train_l, is_training=True, batch_size=batch_size, target_size=target_size)
    val_ds = create_tf_dataset(val_p, val_l, is_training=False, batch_size=batch_size, target_size=target_size)
    test_ds = create_tf_dataset(test_p, test_l, is_training=False, batch_size=batch_size, target_size=target_size)

    # Build CNN
    print(f"\n[cnn_train] Building {backbone} model (pretrained on ImageNet)...")
    model = build_cnn_model(
        backbone_name=backbone,
        num_classes=5,
        learning_rate=learning_rate,
        freeze_base=True,
    )

    callbacks = get_callbacks(checkpoint_path=weights_path)

    # Phase 1: Feature extraction (head training)
    print(f"\n[cnn_train] --- Phase 1: Training Classification Head ({epochs} epochs) ---")
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Phase 2: Fine-tuning top layers
    if fine_tune_epochs > 0:
        print(f"\n[cnn_train] --- Phase 2: Fine-tuning Top Layers ({fine_tune_epochs} epochs) ---")
        model = unfreeze_and_fine_tune(model, fine_tune_at=100, learning_rate=learning_rate * 0.1)
        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs + fine_tune_epochs,
            initial_epoch=epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )

    # Save final model weights
    save_cnn_model(model, path=weights_path)

    # Evaluate on test set
    print(f"\n[cnn_train] --- Evaluating on Test Set ({len(test_p)} samples) ---")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"[cnn_train] Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}%")

    # Predict on test set for classification report
    y_preds = []
    y_trues = []
    for imgs, lbls in test_ds:
        preds = model.predict(imgs, verbose=0)
        y_preds.extend(np.argmax(preds, axis=1))
        y_trues.extend(lbls.numpy())

    report = classification_report(y_trues, y_preds, target_names=CLASS_NAMES, zero_division=0)
    print("\n[cnn_train] Classification Report:")
    print(report)

    # Save summary
    summary = {
        "backbone": backbone,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "train_samples": len(train_p),
        "val_samples": len(val_p),
        "test_samples": len(test_p),
        "class_weights": class_weights,
    }
    
    summary_path = os.path.join(saved_dir, "cnn_training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[cnn_train] Saved summary to {summary_path}")

    return summary


def train_biomarker_pipeline(force_retrain: bool = False) -> str:
    """Train Stage 1 clinical biomarker model."""
    saved_dir = settings.get("paths", {}).get("saved_models", "saved_models")
    os.makedirs(saved_dir, exist_ok=True)

    model_path = os.path.join(saved_dir, "biomarker_model.pkl")
    scaler_path = os.path.join(saved_dir, "biomarker_scaler.pkl")
    features_path = os.path.join(saved_dir, "features.json")

    all_exist = all(os.path.exists(p) for p in [model_path, scaler_path, features_path])
    if all_exist and not force_retrain:
        print("[train_bio] Biomarker model already exists. Skipping. Use --force to retrain.")
        return model_path

    print("\n" + "=" * 60)
    print("[*] TRAINING CLINICAL BIOMARKER ENSEMBLE MODEL")
    print("=" * 60)
    print("[train_bio] Loading and preparing tabular data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_tabular_data(force_regenerate=force_retrain)
    print(f"[train_bio] Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"[train_bio] Features ({len(feature_names)}): {feature_names}")
    for c_id in range(5):
        print(f"  * Class {c_id}: {(y_train == c_id).sum()} train samples")

    print("[train_bio] Training multi-class stacking ensemble (XGBoost + HistGB + ExtraTrees + RF)...")
    model = train_biomarker_model(X_train, y_train, X_test, y_test)

    print("[train_bio] Evaluating...")
    acc, report = evaluate_biomarker_model(model, X_test, y_test)
    print(f"[train_bio] Accuracy: {acc * 100:.2f}%")
    print(report)

    print(f"[train_bio] Saving model    -> {model_path}")
    save_biomarker_model(model, model_path)

    print(f"[train_bio] Saving scaler   -> {scaler_path}")
    joblib.dump(scaler, scaler_path)

    print(f"[train_bio] Saving features -> {features_path}")
    with open(features_path, "w") as f:
        json.dump({"features": list(feature_names)}, f, indent=2)

    print("[train_bio] Complete.")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RetinaGuard DR AI models.")
    parser.add_argument("--model", type=str, default="all", choices=["all", "cnn", "biomarker"],
                        help="Which model to train: cnn, biomarker, or all")
    parser.add_argument("--data_dir", type=str, default="raw_combined",
                        help="Path to folder containing fundus images (classes 0..4)")
    parser.add_argument("--backbone", type=str, default="EfficientNetB0",
                        choices=["EfficientNetB0", "EfficientNetV2B0", "EfficientNetB3", "MobileNetV2", "ResNet50"],
                        help="CNN backbone architecture")
    parser.add_argument("--epochs", type=int, default=10, help="Number of head training epochs")
    parser.add_argument("--fine_tune_epochs", type=int, default=3, help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--subset", type=int, default=None,
                        help="Max samples per class for fast CPU training/testing (e.g. 200)")
    parser.add_argument("--force", action="store_true", help="Force retrain biomarker model")

    args = parser.parse_args()

    if args.model in ("biomarker", "all"):
        train_biomarker_pipeline(force_retrain=args.force)

    if args.model in ("cnn", "all"):
        train_cnn_pipeline(
            data_dir=args.data_dir,
            backbone=args.backbone,
            epochs=args.epochs,
            fine_tune_epochs=args.fine_tune_epochs,
            batch_size=args.batch_size,
            subset_per_class=args.subset,
        )

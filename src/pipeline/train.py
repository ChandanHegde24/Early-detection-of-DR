"""
Master training script — orchestrates the full training pipeline.

Steps:
  1. Prepare tabular clinical data (impute, scale, split)
  2. Train the biomarker model (XGBoost / RandomForest)
  3. Prepare image datasets (load, preprocess, augment)
  4. Train the CNN with transfer learning
  5. Fine-tune the CNN backbone
  6. Save all model artifacts
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.data_prep.tabular_prep import prepare_tabular_data
from src.data_prep.image_loader import load_image_dataset
from src.models.biomarker_rf import (
    train_biomarker_model,
    evaluate_biomarker_model,
    save_biomarker_model,
)
from src.models.retinal_cnn import (
    build_cnn_model,
    unfreeze_and_fine_tune,
    get_callbacks,
    save_cnn_model,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

settings = load_settings()


def ensure_directories():
    """Create output directories if they don't exist."""
    os.makedirs(settings["paths"]["saved_models"], exist_ok=True)
    os.makedirs(settings["paths"]["processed_images"], exist_ok=True)


def train_biomarker_pipeline(csv_path: Optional[str] = None) -> dict:
    """Train and evaluate the biomarker model.

    Returns:
        Dict with accuracy and classification report.
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: Training Biomarker Model")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_tabular_data(csv_path)
    logger.info(f"Tabular data: {X_train.shape[0]} train / {X_test.shape[0]} test samples")
    logger.info(f"Features: {feature_names}")

    model = train_biomarker_model(X_train, y_train, X_test, y_test)

    acc, report = evaluate_biomarker_model(model, X_test, y_test)
    logger.info(f"Biomarker Model Accuracy: {acc:.4f}")
    logger.info(f"\n{report}")

    save_path = save_biomarker_model(model)
    logger.info(f"Biomarker model saved to: {save_path}")

    # Also save the scaler for inference
    import joblib
    scaler_path = f"{settings['paths']['saved_models']}/biomarker_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to: {scaler_path}")

    return {"accuracy": acc, "report": report}


def train_cnn_pipeline(
    image_dir: Optional[str] = None,
    labels_csv: Optional[str] = None,
) -> dict:
    """Train the CNN model with transfer learning and fine-tuning.

    Expects a labels CSV with columns: 'filename', 'label' (0–4).

    Returns:
        Dict with training history metrics.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: Training CNN Model")
    logger.info("=" * 60)

    image_dir = image_dir or settings["paths"]["raw_images"]
    labels_csv = labels_csv or os.path.join(
        os.path.dirname(settings["paths"]["raw_tabular"]),
        "image_labels.csv",
    )

    # Load labels
    labels_df = pd.read_csv(labels_csv)
    all_labels = dict(zip(labels_df["filename"], labels_df["label"]))

    # Split labels for train / val
    filenames = list(all_labels.keys())
    np.random.seed(settings["tabular"]["random_state"])
    np.random.shuffle(filenames)

    split_idx = int(len(filenames) * (1 - settings["tabular"]["test_size"]))
    train_labels = {f: all_labels[f] for f in filenames[:split_idx]}
    val_labels = {f: all_labels[f] for f in filenames[split_idx:]}

    logger.info(f"Image dataset: {len(train_labels)} train / {len(val_labels)} val images")

    train_ds = load_image_dataset(image_dir, train_labels, is_training=True)
    val_ds = load_image_dataset(image_dir, val_labels, is_training=False)

    # Phase 1: Train with frozen backbone
    logger.info("Phase 1: Training with frozen backbone...")
    model = build_cnn_model()
    model.summary(print_fn=logger.info)

    callbacks = get_callbacks()
    cfg = settings["cnn"]

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"] // 2,
        callbacks=callbacks,
    )

    # Phase 2: Fine-tune
    logger.info("Phase 2: Fine-tuning backbone layers...")
    model = unfreeze_and_fine_tune(model)

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["epochs"],
        initial_epoch=len(history1.history["loss"]),
        callbacks=callbacks,
    )

    # Save model
    save_path = save_cnn_model(model)
    logger.info(f"CNN model saved to: {save_path}")

    # Combine histories
    combined_history = {}
    for key in history1.history:
        combined_history[key] = history1.history[key] + history2.history[key]

    return combined_history


def run_full_training(
    csv_path: Optional[str] = None,
    image_dir: Optional[str] = None,
    labels_csv: Optional[str] = None,
):
    """Execute the complete training pipeline for both models."""
    ensure_directories()

    # Train biomarker model
    bio_results = train_biomarker_pipeline(csv_path)

    # Train CNN model
    cnn_history = train_cnn_pipeline(image_dir, labels_csv)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Biomarker accuracy: {bio_results['accuracy']:.4f}")
    logger.info(f"  CNN final val_accuracy: {cnn_history.get('val_accuracy', [0])[-1]:.4f}")
    logger.info("=" * 60)

    # Save training summary
    summary = {
        "biomarker_accuracy": bio_results["accuracy"],
        "cnn_final_val_accuracy": cnn_history.get("val_accuracy", [0])[-1],
        "cnn_final_val_loss": cnn_history.get("val_loss", [0])[-1],
    }
    summary_path = os.path.join(settings["paths"]["saved_models"], "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    run_full_training()

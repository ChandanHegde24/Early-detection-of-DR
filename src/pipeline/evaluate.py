"""
Evaluation script — generates metrics for trained models.

Computes:
  - Per-model metrics: Accuracy, Precision, Recall, F1-Score
  - Fused model metrics with late fusion
  - ROC-AUC (One-vs-Rest for multiclass)
  - Confusion matrices
  - Saves all results and plots
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from sklearn.preprocessing import label_binarize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.data_prep.tabular_prep import prepare_tabular_data
from src.data_prep.image_loader import load_image_dataset, preprocess_single_image
from src.models.biomarker_rf import load_biomarker_model, predict_biomarker_proba
from src.models.retinal_cnn import load_cnn_model
from src.models.late_fusion import unified_prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

settings = load_settings()
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def evaluate_biomarker(
    csv_path: Optional[str] = None,
    model_path: Optional[str] = None,
) -> dict:
    """Evaluate the biomarker model on the test set."""
    logger.info("Evaluating Biomarker Model...")

    _, X_test, _, y_test, _, feature_names = prepare_tabular_data(csv_path)
    model = load_biomarker_model(model_path)

    y_pred = model.predict(X_test)
    y_proba = predict_biomarker_proba(model, X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    y_test_bin = label_binarize(y_test, classes=list(range(5)))
    try:
        roc_auc = roc_auc_score(y_test_bin, y_proba, multi_class="ovr", average="weighted")
    except ValueError:
        roc_auc = 0.0

    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0)

    logger.info(f"Biomarker — Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    logger.info(f"\n{report}")

    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "roc_auc": roc_auc,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def evaluate_cnn(
    image_dir: Optional[str] = None,
    labels_csv: Optional[str] = None,
    model_path: Optional[str] = None,
) -> dict:
    """Evaluate the CNN model on the test image set."""
    logger.info("Evaluating CNN Model...")

    image_dir = image_dir or settings["paths"]["raw_images"]
    labels_csv = labels_csv or os.path.join(
        os.path.dirname(settings["paths"]["raw_tabular"]),
        "image_labels.csv",
    )

    labels_df = pd.read_csv(labels_csv)
    all_labels = dict(zip(labels_df["filename"], labels_df["label"]))

    filenames = list(all_labels.keys())
    np.random.seed(settings["tabular"]["random_state"])
    np.random.shuffle(filenames)
    split_idx = int(len(filenames) * (1 - settings["tabular"]["test_size"]))
    test_labels = {f: all_labels[f] for f in filenames[split_idx:]}

    model = load_cnn_model(model_path)

    y_true = []
    y_proba_list = []

    for fname, label in test_labels.items():
        fpath = os.path.join(image_dir, fname)
        if not os.path.exists(fpath):
            continue
        img = preprocess_single_image(fpath)
        pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0]
        y_true.append(label)
        y_proba_list.append(pred)

    y_true = np.array(y_true)
    y_proba = np.array(y_proba_list)
    y_pred = np.argmax(y_proba, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    y_true_bin = label_binarize(y_true, classes=list(range(5)))
    try:
        roc_auc = roc_auc_score(y_true_bin, y_proba, multi_class="ovr", average="weighted")
    except ValueError:
        roc_auc = 0.0

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)

    logger.info(f"CNN — Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    logger.info(f"\n{report}")

    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "roc_auc": roc_auc,
        "y_test": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def evaluate_fused(bio_results: dict, cnn_results: dict) -> dict:
    """Evaluate the late-fused unified model."""
    logger.info("Evaluating Fused (Unified) Model...")

    n = min(len(bio_results["y_test"]), len(cnn_results["y_test"]))
    y_test = bio_results["y_test"][:n]
    bio_proba = bio_results["y_proba"][:n]
    cnn_proba = cnn_results["y_proba"][:n]

    grades, risk_scores, fused_proba = unified_prediction(cnn_proba, bio_proba)

    acc = accuracy_score(y_test, grades)
    f1 = f1_score(y_test, grades, average="weighted", zero_division=0)

    y_test_bin = label_binarize(y_test, classes=list(range(5)))
    try:
        roc_auc = roc_auc_score(y_test_bin, fused_proba, multi_class="ovr", average="weighted")
    except ValueError:
        roc_auc = 0.0

    report = classification_report(y_test, grades, target_names=CLASS_NAMES, zero_division=0)

    logger.info(f"Fused — Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    logger.info(f"\n{report}")

    return {
        "accuracy": acc,
        "f1_weighted": f1,
        "roc_auc": roc_auc,
        "y_test": y_test,
        "y_pred": grades,
        "risk_scores": risk_scores,
        "fused_proba": fused_proba,
    }


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: str):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved: {save_path}")


def run_full_evaluation(
    csv_path: Optional[str] = None,
    image_dir: Optional[str] = None,
    labels_csv: Optional[str] = None,
):
    """Execute the complete evaluation pipeline."""
    output_dir = os.path.join(settings["paths"]["saved_models"], "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    bio_results = evaluate_biomarker(csv_path)
    cnn_results = evaluate_cnn(image_dir, labels_csv)

    fused_results = evaluate_fused(bio_results, cnn_results)

    plot_confusion_matrix(
        bio_results["y_test"], bio_results["y_pred"],
        "Biomarker Model — Confusion Matrix",
        os.path.join(output_dir, "cm_biomarker.png"),
    )
    plot_confusion_matrix(
        cnn_results["y_test"], cnn_results["y_pred"],
        "CNN Model — Confusion Matrix",
        os.path.join(output_dir, "cm_cnn.png"),
    )
    plot_confusion_matrix(
        fused_results["y_test"], fused_results["y_pred"],
        "Fused Model — Confusion Matrix",
        os.path.join(output_dir, "cm_fused.png"),
    )

    summary = {
        "biomarker": {
            "accuracy": bio_results["accuracy"],
            "f1_weighted": bio_results["f1_weighted"],
            "roc_auc": bio_results["roc_auc"],
        },
        "cnn": {
            "accuracy": cnn_results["accuracy"],
            "f1_weighted": cnn_results["f1_weighted"],
            "roc_auc": cnn_results["roc_auc"],
        },
        "fused": {
            "accuracy": fused_results["accuracy"],
            "f1_weighted": fused_results["f1_weighted"],
            "roc_auc": fused_results["roc_auc"],
        },
    }
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Evaluation summary saved: {summary_path}")

    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info(f"{'Model':<15} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10}")
    logger.info("-" * 45)
    for name, res in [("Biomarker", bio_results), ("CNN", cnn_results), ("Fused", fused_results)]:
        logger.info(f"{name:<15} {res['accuracy']:>10.4f} {res['f1_weighted']:>10.4f} {res['roc_auc']:>10.4f}")

    return summary


if __name__ == "__main__":
    run_full_evaluation()

"""
Late Fusion module — combines CNN image predictions with biomarker model
predictions into a single unified risk score.

Fusion Strategy:
    final_score = w_cnn * cnn_proba + w_bio * biomarker_proba

The fused probability vector is then used to derive:
    - A predicted DR grade (argmax)
    - A continuous risk score (weighted sum toward severe classes)
"""

from typing import Tuple, Optional

import numpy as np

from src.config import load_settings

settings = load_settings()


# Severity weights: higher grades contribute more to the risk score
SEVERITY_WEIGHTS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


def fuse_predictions(
    cnn_proba: np.ndarray,
    biomarker_proba: np.ndarray,
    cnn_weight: Optional[float] = None,
    biomarker_weight: Optional[float] = None,
) -> np.ndarray:
    """Combine CNN and biomarker class probabilities via weighted average.

    Args:
        cnn_proba: Array of shape (n_samples, n_classes) from the CNN.
        biomarker_proba: Array of shape (n_samples, n_classes) from the biomarker model.
        cnn_weight: Weight for the CNN component. Defaults to settings value.
        biomarker_weight: Weight for the biomarker component. Defaults to settings value.

    Returns:
        Fused probability array of shape (n_samples, n_classes).
    """
    cfg = settings["fusion"]
    cnn_weight = cnn_weight if cnn_weight is not None else cfg["cnn_weight"]
    biomarker_weight = biomarker_weight if biomarker_weight is not None else cfg["biomarker_weight"]

    total = cnn_weight + biomarker_weight
    cnn_weight /= total
    biomarker_weight /= total

    return cnn_weight * cnn_proba + biomarker_weight * biomarker_proba


def get_predicted_grade(fused_proba: np.ndarray) -> np.ndarray:
    """Return the predicted DR grade (0–4) from the fused probabilities."""
    return np.argmax(fused_proba, axis=1)


def compute_risk_score(fused_proba: np.ndarray) -> np.ndarray:
    """Compute a continuous risk score in [0, 1] from the fused probabilities.

    The risk score is a dot product of class probabilities with severity weights,
    giving higher weight to more severe DR grades:
        score = sum(proba_i * severity_weight_i)

    Returns:
        Array of shape (n_samples,) with risk scores in [0, 1].
    """
    return fused_proba @ SEVERITY_WEIGHTS


def unified_prediction(
    cnn_proba: np.ndarray,
    biomarker_proba: np.ndarray,
    cnn_weight: Optional[float] = None,
    biomarker_weight: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full late fusion pipeline: fuse → predict grade → compute risk score.

    Returns:
        (predicted_grades, risk_scores, fused_probabilities)
    """
    fused = fuse_predictions(cnn_proba, biomarker_proba, cnn_weight, biomarker_weight)
    grades = get_predicted_grade(fused)
    scores = compute_risk_score(fused)
    return grades, scores, fused

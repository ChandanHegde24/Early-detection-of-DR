"""
Late Fusion module for RetinaGuard.

CNN:
    5-class DR severity prediction
    [No DR, Mild NPDR, Moderate NPDR, Severe NPDR, Proliferative DR]

Biomarker:
    Binary DR prediction
    [No DR, DR]

Because the models have different output spaces, the binary biomarker
probability is converted into a 5-class distribution using the CNN's
conditional severity distribution.

Fusion:
    fused = cnn_weight * CNN + biomarker_weight * converted_biomarker

The output remains a valid 5-class probability distribution.
"""

from typing import Tuple, Optional

import numpy as np

from src.config import load_settings

settings = load_settings()


SEVERITY_WEIGHTS = np.array(
    [0.0, 0.25, 0.50, 0.75, 1.0],
    dtype=np.float32,
)


def _validate_probability_array(
    probabilities: np.ndarray,
    expected_classes: int,
    name: str,
) -> np.ndarray:
    """Validate and normalize a probability array."""
    probabilities = np.asarray(probabilities, dtype=np.float32)

    if probabilities.ndim != 2:
        raise ValueError(
            f"{name} must have shape (n_samples, n_classes), "
            f"got {probabilities.shape}"
        )

    if probabilities.shape[1] != expected_classes:
        raise ValueError(
            f"{name} must have {expected_classes} classes, "
            f"got shape {probabilities.shape}"
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    if np.any(probabilities < 0):
        raise ValueError(f"{name} contains negative probabilities.")

    row_sums = probabilities.sum(axis=1, keepdims=True)

    if np.any(row_sums <= 0):
        raise ValueError(f"{name} contains a row whose probability sum is <= 0.")

    # Normalize defensively so floating-point differences do not break fusion.
    probabilities = probabilities / row_sums

    return probabilities


def convert_binary_biomarker_to_five_class(
    cnn_proba: np.ndarray,
    biomarker_proba: np.ndarray,
) -> np.ndarray:
    """
    Convert binary biomarker probabilities into a 5-class distribution.

    Biomarker classes:
        [P(No DR), P(DR)]

    CNN classes:
        [P(No DR), P(Mild), P(Moderate), P(Severe), P(Proliferative)]

    The biomarker determines the overall No-DR vs DR probability.
    The CNN determines the relative severity distribution among DR classes.

    This preserves the fact that the biomarker model does NOT predict
    severity grades 1-4 itself.
    """
    cnn_proba = _validate_probability_array(
        cnn_proba,
        expected_classes=5,
        name="cnn_proba",
    )

    biomarker_proba = _validate_probability_array(
        biomarker_proba,
        expected_classes=2,
        name="biomarker_proba",
    )

    if cnn_proba.shape[0] != biomarker_proba.shape[0]:
        raise ValueError(
            "CNN and biomarker predictions must contain the same number "
            f"of samples. Got {cnn_proba.shape[0]} and "
            f"{biomarker_proba.shape[0]}."
        )

    n_samples = cnn_proba.shape[0]

    converted = np.zeros((n_samples, 5), dtype=np.float32)

    # Binary biomarker probabilities.
    p_no_dr = biomarker_proba[:, 0]
    p_dr = biomarker_proba[:, 1]

    # CNN probability assigned to any DR grade.
    cnn_dr_mass = cnn_proba[:, 1:].sum(axis=1)

    # Handle the theoretically unlikely case where CNN assigns zero
    # probability to all DR grades.
    for i in range(n_samples):
        if cnn_dr_mass[i] > 0:
            severity_distribution = cnn_proba[i, 1:] / cnn_dr_mass[i]
        else:
            # No severity information from CNN:
            # distribute DR probability uniformly across grades 1-4.
            severity_distribution = np.full(4, 0.25, dtype=np.float32)

        converted[i, 0] = p_no_dr
        converted[i, 1:] = p_dr[i] * severity_distribution

    return converted


def fuse_predictions(
    cnn_proba: np.ndarray,
    biomarker_proba: np.ndarray,
    cnn_weight: Optional[float] = None,
    biomarker_weight: Optional[float] = None,
) -> np.ndarray:
    """
    Fuse a 5-class CNN with a binary biomarker model.

    Returns:
        Fused probability array of shape (n_samples, 5).
    """
    cfg = settings["fusion"]

    cnn_weight = (
        cnn_weight
        if cnn_weight is not None
        else cfg["cnn_weight"]
    )

    biomarker_weight = (
        biomarker_weight
        if biomarker_weight is not None
        else cfg["biomarker_weight"]
    )

    cnn_weight = float(cnn_weight)
    biomarker_weight = float(biomarker_weight)

    if cnn_weight < 0 or biomarker_weight < 0:
        raise ValueError("Fusion weights must be non-negative.")

    total_weight = cnn_weight + biomarker_weight

    if total_weight <= 0:
        raise ValueError("At least one fusion weight must be greater than zero.")

    cnn_weight /= total_weight
    biomarker_weight /= total_weight

    cnn_proba = _validate_probability_array(
        cnn_proba,
        expected_classes=5,
        name="cnn_proba",
    )

    converted_biomarker = convert_binary_biomarker_to_five_class(
        cnn_proba,
        biomarker_proba,
    )

    fused = (
        cnn_weight * cnn_proba
        + biomarker_weight * converted_biomarker
    )

    # Final normalization for numerical stability.
    fused /= fused.sum(axis=1, keepdims=True)

    return fused


def get_predicted_grade(fused_proba: np.ndarray) -> np.ndarray:
    """Return predicted DR grade 0-4."""
    fused_proba = _validate_probability_array(
        fused_proba,
        expected_classes=5,
        name="fused_proba",
    )

    return np.argmax(fused_proba, axis=1)


def compute_risk_score(fused_proba: np.ndarray) -> np.ndarray:
    """
    Compute continuous risk score in [0, 1].

    Grade severity weights:
        0 -> 0.00
        1 -> 0.25
        2 -> 0.50
        3 -> 0.75
        4 -> 1.00
    """
    fused_proba = _validate_probability_array(
        fused_proba,
        expected_classes=5,
        name="fused_proba",
    )

    scores = fused_proba @ SEVERITY_WEIGHTS

    return np.clip(scores, 0.0, 1.0)


def unified_prediction(
    cnn_proba: np.ndarray,
    biomarker_proba: np.ndarray,
    cnn_weight: Optional[float] = None,
    biomarker_weight: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete multimodal prediction pipeline.

    Returns:
        predicted_grades
        risk_scores
        fused_probabilities
    """
    fused = fuse_predictions(
        cnn_proba=cnn_proba,
        biomarker_proba=biomarker_proba,
        cnn_weight=cnn_weight,
        biomarker_weight=biomarker_weight,
    )

    grades = get_predicted_grade(fused)
    scores = compute_risk_score(fused)

    # Safety escalation based only on the CNN's severity prediction,
    # because the biomarker model is binary and cannot identify Grade 3/4.
    for i in range(len(grades)):
        cnn_grade = int(np.argmax(cnn_proba[i]))

        if cnn_grade >= 3 and cnn_proba[i, cnn_grade] > 0.35:
            grades[i] = cnn_grade

            minimum_score = 0.72 if cnn_grade == 3 else 0.88

            if scores[i] < minimum_score:
                scores[i] = minimum_score

    return grades, scores, fused
"""
biomarker_runtime.py

Production inference wrapper for the existing trained biomarker model.

The saved biomarker_model.pkl is:

    StackingClassifier
        XGBoost
        LightGBM
        CatBoost
        RandomForest
        LogisticRegression meta-model

The artifact was created under a newer runtime and the serialized
LightGBM sklearn wrapper crashes under the current environment.

Therefore this module:

1. Loads the original trained StackingClassifier.
2. Extracts the already-trained LightGBM Booster.
3. Reconstructs that Booster from its native model string.
4. Does NOT call StackingClassifier.predict_proba().
5. Reconstructs the same 4-feature meta-input used by the trained
   LogisticRegression.
6. Runs the original trained LogisticRegression meta-model.

NO MODEL RETRAINING IS PERFORMED.

Current artifact output:
    class 0 = No DR
    class 1 = DR
"""

from __future__ import annotations

import os
from typing import Optional

import joblib
import numpy as np
import lightgbm as lgb


DEFAULT_MODEL_PATH = "saved_models/biomarker_model.pkl"

EXPECTED_CLASSES = np.array([0, 1], dtype=np.int64)
EXPECTED_FEATURES = 17


def _positive_probability(
    estimator,
    X: np.ndarray,
) -> np.ndarray:
    """
    Return P(class=1) from a binary classifier.
    """

    proba = np.asarray(
        estimator.predict_proba(X),
        dtype=np.float32,
    )

    classes = np.asarray(
        estimator.classes_
    )

    if proba.ndim != 2:
        raise ValueError(
            f"Expected 2D probability output, got {proba.shape}"
        )

    positive_indices = np.where(classes == 1)[0]

    if len(positive_indices) != 1:
        raise ValueError(
            f"Could not identify class 1 in classes={classes}"
        )

    return proba[:, positive_indices[0]]


def _recover_lightgbm_booster(model):
    """
    Recover the trained native LightGBM Booster from the pickled
    LGBMClassifier without retraining.
    """

    named_estimators = model.named_estimators_

    if "lgbm" not in named_estimators:
        raise ValueError(
            "Saved biomarker model does not contain "
            "the expected 'lgbm' estimator."
        )

    lgbm_wrapper = named_estimators["lgbm"]

    original_booster = getattr(
        lgbm_wrapper,
        "booster_",
        None,
    )

    if original_booster is None:
        raise ValueError(
            "Saved LightGBM estimator has no booster_."
        )

    # Extract the already-trained model.
    model_string = original_booster.model_to_string()

    if not model_string:
        raise ValueError(
            "Could not extract the trained LightGBM model."
        )

    # Reconstruct native Booster.
    recovered_booster = lgb.Booster(
        model_str=model_string
    )

    print(
        "[biomarker_runtime] LightGBM Booster recovered."
    )

    print(
        "[biomarker_runtime] LightGBM model size:",
        f"{len(model_string):,}",
        "characters",
    )

    return recovered_booster


def load_biomarker_model(
    path: Optional[str] = None,
):
    """
    Load the original trained biomarker model and recover
    its LightGBM component.

    The original pickle is never overwritten.
    """

    path = path or DEFAULT_MODEL_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Biomarker model not found at: {path}"
        )

    print(
        f"[biomarker_runtime] Loading model: {path}"
    )

    model = joblib.load(path)

    classes = np.asarray(
        getattr(model, "classes_", [])
    )

    n_features = getattr(
        model,
        "n_features_in_",
        None,
    )

    print(
        "[biomarker_runtime] Model type:",
        type(model).__name__,
    )

    print(
        "[biomarker_runtime] Classes:",
        classes,
    )

    print(
        "[biomarker_runtime] Features:",
        n_features,
    )

    if not np.array_equal(
        classes,
        EXPECTED_CLASSES,
    ):
        raise ValueError(
            "Unexpected biomarker classes. "
            f"Expected [0, 1], got {classes}"
        )

    if n_features != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} input features, "
            f"got {n_features}"
        )

    # Recover LightGBM once when loading.
    recovered_booster = _recover_lightgbm_booster(
        model
    )

    # Store the recovered Booster on the model object.
    #
    # This is only runtime state. The original pickle
    # on disk is NOT modified.
    model._retinaguard_recovered_lgbm = recovered_booster

    # Restore compatibility for the serialized sklearn
    # LogisticRegression object.
    final_estimator = model.final_estimator_

    if not hasattr(
        final_estimator,
        "multi_class",
    ):
        final_estimator.multi_class = "ovr"

        print(
            "[biomarker_runtime] LogisticRegression "
            "compatibility restored."
        )

    print(
        "[biomarker_runtime] Biomarker model ready."
    )

    return model


def predict_biomarker_proba(
    model,
    X: np.ndarray,
) -> np.ndarray:
    """
    Perform production inference using the already-trained
    stacking components.

    Returns:

        [[P(No DR), P(DR)]]

    Shape:

        (n_samples, 2)
    """

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    if X.ndim != 2:
        raise ValueError(
            f"Expected 2D feature matrix, got {X.shape}"
        )

    if X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features, "
            f"got {X.shape[1]}"
        )

    if not np.all(
        np.isfinite(X)
    ):
        raise ValueError(
            "Input contains NaN or infinite values."
        )

    # Recover Booster if this model was loaded by some other caller.
    recovered_booster = getattr(
        model,
        "_retinaguard_recovered_lgbm",
        None,
    )

    if recovered_booster is None:
        recovered_booster = _recover_lightgbm_booster(
            model
        )

        model._retinaguard_recovered_lgbm = (
            recovered_booster
        )

    named_estimators = model.named_estimators_

    # ---------------------------------------------------------------
    # 1. XGBoost P(DR)
    # ---------------------------------------------------------------

    xgb_p1 = _positive_probability(
        named_estimators["xgb"],
        X,
    )

    # ---------------------------------------------------------------
    # 2. Recovered LightGBM P(DR)
    # ---------------------------------------------------------------

    lgbm_p1 = recovered_booster.predict(
        X
    )

    lgbm_p1 = np.asarray(
        lgbm_p1,
        dtype=np.float32,
    ).reshape(-1)

    # ---------------------------------------------------------------
    # 3. CatBoost P(DR)
    # ---------------------------------------------------------------

    cat_p1 = _positive_probability(
        named_estimators["cat"],
        X,
    )

    # ---------------------------------------------------------------
    # 4. RandomForest P(DR)
    # ---------------------------------------------------------------

    rf_p1 = _positive_probability(
        named_estimators["rf"],
        X,
    )

    print(
        "[biomarker_runtime] Base probabilities:",
        f"XGB={float(xgb_p1[0]):.6f}",
        f"LGBM={float(lgbm_p1[0]):.6f}",
        f"CAT={float(cat_p1[0]):.6f}",
        f"RF={float(rf_p1[0]):.6f}",
    )

    # ---------------------------------------------------------------
    # Recreate the exact stacking meta features
    # ---------------------------------------------------------------

    meta_features = np.column_stack(
        [
            xgb_p1,
            lgbm_p1,
            cat_p1,
            rf_p1,
        ]
    ).astype(
        np.float32
    )

    if meta_features.shape[1] != 4:
        raise ValueError(
            f"Expected 4 stacking meta features, "
            f"got {meta_features.shape}"
        )

    # ---------------------------------------------------------------
    # Original trained LogisticRegression
    # ---------------------------------------------------------------

    final_estimator = model.final_estimator_

    if not hasattr(
        final_estimator,
        "multi_class",
    ):
        final_estimator.multi_class = "ovr"

    final_proba = final_estimator.predict_proba(
        meta_features
    )

    final_proba = np.asarray(
        final_proba,
        dtype=np.float32,
    )

    if final_proba.shape != (
        X.shape[0],
        2,
    ):
        raise ValueError(
            "Unexpected biomarker output shape: "
            f"{final_proba.shape}"
        )

    if not np.all(
        np.isfinite(final_proba)
    ):
        raise ValueError(
            "Biomarker output contains NaN or infinite values."
        )

    if np.any(
        final_proba < 0
    ):
        raise ValueError(
            "Biomarker output contains negative probabilities."
        )

    row_sums = final_proba.sum(
        axis=1,
        keepdims=True,
    )

    if np.any(
        row_sums <= 0
    ):
        raise ValueError(
            "Biomarker probability sum is invalid."
        )

    final_proba /= row_sums

    return final_proba
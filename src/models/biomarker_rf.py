"""
biomarker_rf.py

Biomarker model training utilities and production inference wrapper.

IMPORTANT
---------
The currently supplied trained artifact:

    saved_models/biomarker_model.pkl

has been verified to be:

    sklearn StackingClassifier
    classes = [0, 1]
    class 0 = No DR
    class 1 = DR
    17 input features

The saved artifact was serialized under a newer scikit-learn/runtime
environment. Its embedded LightGBM sklearn wrapper causes a native
access violation under the current runtime.

Therefore, during loading we:
    1. Load the original trained StackingClassifier.
    2. Extract its already-trained native LightGBM Booster.
    3. Reconstruct the Booster from its model string.
    4. Replace only the broken LightGBM predict_proba method in memory.
    5. Restore the missing LogisticRegression compatibility attribute.
    6. Keep XGBoost, CatBoost, RandomForest and LogisticRegression
       learned parameters unchanged.

NO RETRAINING IS PERFORMED.
"""

import os
from types import MethodType
from typing import Tuple, Optional

import joblib
import numpy as np

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    LGBMClassifier = None
    HAS_LIGHTGBM = False

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


# ---------------------------------------------------------------------------
# Project configuration
# ---------------------------------------------------------------------------

try:
    from src.config import load_settings

    settings = load_settings()

except Exception:
    settings = {
        "biomarker_model": {
            "type": "stacking",
            "random_state": 42,
        },
        "paths": {
            "saved_models": "saved_models",
        },
    }


DEFAULT_MODEL_PATH = "saved_models/biomarker_model.pkl"


# ---------------------------------------------------------------------------
# CURRENT SAVED ARTIFACT
# ---------------------------------------------------------------------------

BIOMARKER_CLASS_NAMES = [
    "No DR",
    "DR",
]

BIOMARKER_CLASSES = np.array(
    [0, 1],
    dtype=np.int64,
)


# ---------------------------------------------------------------------------
# Intended 5-class DR labels retained for compatibility with older code
# ---------------------------------------------------------------------------

CLASS_NAMES = [
    "No DR (0)",
    "Mild NPDR (1)",
    "Moderate NPDR (2)",
    "Severe NPDR (3)",
    "Proliferative DR (4)",
]


# ---------------------------------------------------------------------------
# Training utility
# ---------------------------------------------------------------------------

def build_stacking_ensemble(
    random_state: int = 42,
) -> StackingClassifier:
    """
    Build the original stacking architecture.

    This is retained for compatibility with the project's training pipeline.
    It does NOT modify the already-trained biomarker_model.pkl.
    """

    estimators = []

    # XGBoost
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.05,
        verbosity=0,
        eval_metric="logloss",
        random_state=random_state,
    )

    estimators.append(("xgb", xgb))

    # LightGBM
    if not HAS_LIGHTGBM:
        raise ImportError(
            "LightGBM is required to build the biomarker stacking model."
        )

    lgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=random_state,
        verbosity=-1,
    )

    estimators.append(("lgbm", lgbm))

    # CatBoost
    if CatBoostClassifier is None:
        raise ImportError(
            "CatBoost is required to build the biomarker stacking model."
        )

    cat = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        verbose=False,
        random_seed=random_state,
    )

    estimators.append(("cat", cat))

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    estimators.append(("rf", rf))

    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            C=1.0,
            max_iter=1000,
        ),
        cv=3,
        stack_method="predict_proba",
        n_jobs=-1,
    )


def train_biomarker_model(
    X_train,
    y_train,
    X_test=None,
    y_test=None,
):
    """
    Train a biomarker stacking model.

    This function is NOT called by the production API.
    """

    random_state = settings.get(
        "biomarker_model",
        {},
    ).get(
        "random_state",
        42,
    )

    model = build_stacking_ensemble(
        random_state=random_state
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    print(
        f"5-Fold CV Accuracy: "
        f"{scores.mean() * 100:.2f}% ± "
        f"{scores.std() * 100:.2f}%"
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def evaluate_biomarker_model(
    model,
    X_test,
    y_test,
) -> Tuple[float, str]:
    """
    Evaluate the supplied biomarker model using its actual classes.
    """

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    classes = getattr(
        model,
        "classes_",
        None,
    )

    if classes is not None and len(classes) == 2:
        labels = [
            "No DR",
            "DR",
        ]
    elif classes is not None:
        labels = CLASS_NAMES[: len(classes)]
    else:
        labels = None

    report = classification_report(
        y_test,
        y_pred,
        target_names=labels,
        digits=4,
    )

    return accuracy, report


# ---------------------------------------------------------------------------
# LIGHTGBM COMPATIBILITY RECOVERY
# ---------------------------------------------------------------------------

def _patch_legacy_lightgbm(
    model: StackingClassifier,
) -> StackingClassifier:
    """
    Repair the embedded LightGBM estimator in memory.

    The original LightGBM Booster is extracted from the pickle and
    reconstructed using LightGBM's native model-string format.

    The learned trees are NOT changed.
    """

    if not HAS_LIGHTGBM:
        raise ImportError(
            "LightGBM is required to load the trained biomarker model."
        )

    named_estimators = getattr(
        model,
        "named_estimators_",
        None,
    )

    if named_estimators is None:
        raise ValueError(
            "Loaded object does not contain named_estimators_."
        )

    if "lgbm" not in named_estimators:
        raise ValueError(
            "The trained biomarker model does not contain the expected "
            "'lgbm' estimator."
        )

    lgbm_model = named_estimators["lgbm"]

    original_booster = getattr(
        lgbm_model,
        "booster_",
        None,
    )

    if original_booster is None:
        raise ValueError(
            "The saved LightGBM estimator does not contain booster_."
        )

    # Extract the existing trained LightGBM model.
    model_string = original_booster.model_to_string()

    if not model_string:
        raise ValueError(
            "Could not extract the trained LightGBM model."
        )

    # Reconstruct a native LightGBM Booster.
    recovered_booster = lgb.Booster(
        model_str=model_string
    )

    print(
        "[biomarker_rf] LightGBM native Booster recovered successfully."
    )

    print(
        f"[biomarker_rf] LightGBM model size: "
        f"{len(model_string):,} characters"
    )

    def recovered_predict_proba(
        self,
        X,
    ):
        """
        Replacement predict_proba using the recovered native Booster.

        The current biomarker model is binary:
            0 = No DR
            1 = DR
        """

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        p1 = recovered_booster.predict(
            X
        )

        p1 = np.asarray(
            p1,
            dtype=np.float32,
        ).reshape(-1)

        # Convert P(DR) to the standard sklearn binary layout:
        #
        # [P(No DR), P(DR)]
        return np.column_stack(
            [
                1.0 - p1,
                p1,
            ]
        )

    # Replace only the LightGBM sklearn wrapper's prediction method
    # in memory.
    lgbm_model.predict_proba = MethodType(
        recovered_predict_proba,
        lgbm_model,
    )

    print(
        "[biomarker_rf] LightGBM predict_proba compatibility patch applied."
    )

    return model


# ---------------------------------------------------------------------------
# SCIKIT-LEARN COMPATIBILITY RECOVERY
# ---------------------------------------------------------------------------

def _patch_logistic_regression(
    model: StackingClassifier,
) -> StackingClassifier:
    """
    Restore compatibility for the serialized LogisticRegression meta-model.

    The artifact was serialized under a newer scikit-learn version.
    The current runtime expects the multi_class attribute during
    predict_proba().
    """

    final_estimator = getattr(
        model,
        "final_estimator_",
        None,
    )

    if final_estimator is None:
        raise ValueError(
            "Loaded stacking model has no final_estimator_."
        )

    if isinstance(
        final_estimator,
        LogisticRegression,
    ):
        if not hasattr(
            final_estimator,
            "multi_class",
        ):
            final_estimator.multi_class = "ovr"

            print(
                "[biomarker_rf] LogisticRegression "
                "compatibility patch applied."
            )

    return model


# ---------------------------------------------------------------------------
# COMPLETE MODEL COMPATIBILITY REPAIR
# ---------------------------------------------------------------------------

def _repair_loaded_biomarker_model(
    model,
):
    """
    Apply all required runtime compatibility repairs.

    The original trained parameters remain untouched.
    """

    if not isinstance(
        model,
        StackingClassifier,
    ):
        raise TypeError(
            "Expected saved biomarker model to be a "
            f"StackingClassifier, got {type(model).__name__}."
        )

    classes = np.asarray(
        getattr(
            model,
            "classes_",
            [],
        )
    )

    if not np.array_equal(
        classes,
        BIOMARKER_CLASSES,
    ):
        raise ValueError(
            "Unexpected biomarker classes. "
            f"Expected [0, 1], got {classes}."
        )

    n_features = getattr(
        model,
        "n_features_in_",
        None,
    )

    if n_features != 17:
        raise ValueError(
            "Unexpected biomarker feature count. "
            f"Expected 17, got {n_features}."
        )

    model = _patch_legacy_lightgbm(
        model
    )

    model = _patch_logistic_regression(
        model
    )

    return model


# ---------------------------------------------------------------------------
# PROBABILITY PREDICTION
# ---------------------------------------------------------------------------

def predict_biomarker_proba(
    model,
    X: np.ndarray,
) -> np.ndarray:
    """
    Predict binary biomarker probabilities.

    Output:

        column 0 = P(No DR)
        column 1 = P(DR)

    Shape:

        (n_samples, 2)
    """

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    if X.ndim != 2:
        raise ValueError(
            f"Biomarker input must be 2D, got {X.shape}"
        )

    expected_features = getattr(
        model,
        "n_features_in_",
        17,
    )

    if X.shape[1] != expected_features:
        raise ValueError(
            f"Biomarker model expects {expected_features} features, "
            f"received {X.shape[1]}."
        )

    if not np.all(
        np.isfinite(X)
    ):
        raise ValueError(
            "Biomarker input contains NaN or infinite values."
        )

    proba = model.predict_proba(
        X
    )

    proba = np.asarray(
        proba,
        dtype=np.float32,
    )

    if proba.ndim != 2:
        raise ValueError(
            f"Expected 2D probability output, got {proba.shape}"
        )

    if proba.shape[1] != 2:
        raise ValueError(
            "The current saved biomarker artifact is binary. "
            f"Expected 2 probability columns, got {proba.shape[1]}."
        )

    if not np.all(
        np.isfinite(proba)
    ):
        raise ValueError(
            "Biomarker probabilities contain NaN or infinite values."
        )

    if np.any(
        proba < 0
    ):
        raise ValueError(
            "Biomarker probabilities contain negative values."
        )

    row_sums = proba.sum(
        axis=1,
        keepdims=True,
    )

    if np.any(
        row_sums <= 0
    ):
        raise ValueError(
            "Biomarker probability rows have invalid sums."
        )

    proba = proba / row_sums

    return proba.astype(
        np.float32
    )


# ---------------------------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------------------------

def save_biomarker_model(
    model,
    path: str = DEFAULT_MODEL_PATH,
) -> str:
    """
    Save a trained biomarker model.
    """

    directory = os.path.dirname(
        path
    )

    if directory:
        os.makedirs(
            directory,
            exist_ok=True,
        )

    joblib.dump(
        model,
        path,
    )

    print(
        f"[biomarker_rf] Model saved -> {path}"
    )

    return path


# ---------------------------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------------------------

def load_biomarker_model(
    path: Optional[str] = None,
):
    """
    Load and repair the already-trained biomarker model.

    The original pickle is never overwritten.
    """

    path = path or DEFAULT_MODEL_PATH

    if not os.path.exists(
        path
    ):
        raise FileNotFoundError(
            f"Biomarker model not found at '{path}'."
        )

    print(
        f"[biomarker_rf] Loading trained model: {path}"
    )

    model = joblib.load(
        path
    )

    print(
        f"[biomarker_rf] Model type: "
        f"{type(model).__name__}"
    )

    print(
        f"[biomarker_rf] Classes: "
        f"{getattr(model, 'classes_', None)}"
    )

    print(
        f"[biomarker_rf] Features: "
        f"{getattr(model, 'n_features_in_', None)}"
    )

    model = _repair_loaded_biomarker_model(
        model
    )

    print(
        "[biomarker_rf] Biomarker model ready for inference."
    )

    return model
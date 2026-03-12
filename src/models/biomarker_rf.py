"""
Biomarker-based risk prediction model using XGBoost or Random Forest.

Trains on clinical tabular data (HbA1c, blood pressure, cholesterol, etc.)
to predict the DR severity grade (0–4).
"""

from typing import Optional, Tuple

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

from src.config import load_settings

settings = load_settings()


def build_biomarker_model(model_type: Optional[str] = None) -> object:
    """Create an XGBoost or Random Forest classifier based on settings.

    Args:
        model_type: 'xgboost' or 'random_forest'. Defaults to settings value.

    Returns:
        A scikit-learn compatible classifier instance.
    """
    cfg = settings["biomarker_model"]
    model_type = model_type or cfg["type"]

    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            random_state=cfg["random_state"],
            use_label_encoder=False,
            eval_metric="mlogloss",
            objective="multi:softproba",
        )
    elif model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            random_state=cfg["random_state"],
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def train_biomarker_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    model_type: Optional[str] = None,
) -> object:
    """Train the biomarker model.

    Args:
        X_train: Training feature matrix.
        y_train: Training labels.
        X_val: Optional validation features (used for early stopping with XGBoost).
        y_val: Optional validation labels.
        model_type: Override the model type from settings.

    Returns:
        Trained classifier.
    """
    model = build_biomarker_model(model_type)

    cfg = settings["biomarker_model"]
    model_type = model_type or cfg["type"]

    if model_type == "xgboost" and X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    else:
        model.fit(X_train, y_train)

    return model


def evaluate_biomarker_model(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, str]:
    """Evaluate the model and return accuracy + classification report.

    Returns:
        (accuracy, classification_report_string)
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
        zero_division=0,
    )
    return acc, report


def predict_biomarker_proba(model: object, X: np.ndarray) -> np.ndarray:
    """Return class probability predictions.

    Returns:
        Array of shape (n_samples, n_classes) with probability estimates.
    """
    return model.predict_proba(X)


def save_biomarker_model(model: object, path: Optional[str] = None) -> str:
    """Serialize and save the trained model to disk."""
    path = path or f"{settings['paths']['saved_models']}/biomarker_model.pkl"
    joblib.dump(model, path)
    return path


def load_biomarker_model(path: Optional[str] = None) -> object:
    """Load a previously saved biomarker model from disk."""
    path = path or f"{settings['paths']['saved_models']}/biomarker_model.pkl"
    return joblib.load(path)

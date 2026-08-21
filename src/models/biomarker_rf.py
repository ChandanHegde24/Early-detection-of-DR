"""
biomarker_rf.py — Multi-Class Clinical Biomarker Stacking Ensemble
Classes:
  0: No DR
  1: Mild NPDR
  2: Moderate NPDR
  3: Severe NPDR
  4: Proliferative DR

Stacking: XGBoost + HistGradientBoosting + ExtraTrees + RandomForest → Multinomial Logistic Meta
"""
import os
from typing import Tuple, Optional
import numpy as np
import joblib
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
    from lightgbm import LGBMClassifier
    has_lgbm = True
except ImportError:
    has_lgbm = False

try:
    from src.config import load_settings
    settings = load_settings()
except Exception:
    settings = {
        "biomarker_model": {"type": "stacking", "random_state": 42},
        "paths": {"saved_models": "saved_models"},
    }

DEFAULT_MODEL_PATH = "saved_models/biomarker_model.pkl"
CLASS_NAMES = ["No DR (0)", "Mild NPDR (1)", "Moderate NPDR (2)", "Severe NPDR (3)", "Proliferative DR (4)"]


def build_stacking_ensemble(random_state: int = 42) -> StackingClassifier:
    """Build multi-class stacking ensemble for 5-grade DR prediction."""
    estimators = []

    # 1. XGBoost Multi-class
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        gamma=0.05,
        verbosity=0,
        eval_metric="mlogloss",
        random_state=random_state,
    )
    estimators.append(("xgb", xgb))

    # 2. HistGradientBoosting Multi-class
    hgb = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=15,
        random_state=random_state,
    )
    estimators.append(("hgb", hgb))

    # 3. Extra Trees Multi-class
    et = ExtraTreesClassifier(
        n_estimators=250,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    estimators.append(("et", et))

    # 4. Random Forest Multi-class
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
        final_estimator=LogisticRegression(C=1.0, max_iter=1000, multi_class="multinomial"),
        cv=3,
        stack_method="predict_proba",
        n_jobs=-1,
    )


def train_biomarker_model(X_train, y_train, X_test, y_test):
    """Train 5-class stacking ensemble with cross-validation."""
    rs = settings.get("biomarker_model", {}).get("random_state", 42)
    model = build_stacking_ensemble(rs)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    model.fit(X_train, y_train)
    return model


def evaluate_biomarker_model(model, X_test, y_test) -> Tuple[float, str]:
    """Evaluate on 5 DR severity grades."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)
    return acc, report


def predict_biomarker_proba(model, X: np.ndarray) -> np.ndarray:
    """Returns true 5-class calibrated probability array [p0, p1, p2, p3, p4].

    Args:
        model: Trained multi-class classifier or StackingClassifier
        X: Feature matrix of shape (n_samples, n_features)

    Returns:
        Probability array of shape (n_samples, 5)
    """
    proba = model.predict_proba(X)
    # Ensure shape is (n_samples, 5)
    if proba.shape[1] < 5:
        padded = np.zeros((proba.shape[0], 5))
        padded[:, :proba.shape[1]] = proba
        return padded
    return proba


def save_biomarker_model(model, path: str = DEFAULT_MODEL_PATH) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[biomarker_rf] Multi-Class Model saved → {path}")
    return path


def load_biomarker_model(path: Optional[str] = None):
    path = path or DEFAULT_MODEL_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model found at '{path}'.\n"
            "Run this first:  python -m src.pipeline.train --model biomarker --force\n"
            "Then restart the backend."
        )
    model = joblib.load(path)
    print(f"[biomarker_rf] Multi-Class Model loaded from {path}")
    return model


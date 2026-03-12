"""
Tabular clinical data preprocessing for biomarker-based risk prediction.

Handles:
- Loading clinical CSV data
- Missing value imputation
- Feature scaling and encoding
- Train/test splitting
"""

from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from src.config import load_settings

settings = load_settings()


def load_clinical_data(csv_path: Optional[str] = None) -> pd.DataFrame:
    """Load clinical biomarker data from a CSV file."""
    csv_path = csv_path or settings["paths"]["raw_tabular"]
    df = pd.read_csv(csv_path)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in the clinical dataset.

    - Numeric columns: median imputation
    - Categorical columns: most-frequent imputation
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    return df


def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Label-encode categorical features, returning the DataFrame and encoder mapping."""
    encoders = {}
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    target_col = settings["tabular"]["target"]
    categorical_cols = [c for c in categorical_cols if c != target_col]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def scale_features(X_train: np.ndarray,
                   X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Standardize features using z-score normalization."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def prepare_tabular_data(
    csv_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list]:
    """Full tabular data preparation pipeline.

    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    cfg = settings["tabular"]

    df = load_clinical_data(csv_path)
    df = handle_missing_values(df)
    df, _ = encode_categorical_features(df)

    feature_cols = [f for f in cfg["features"] if f in df.columns]
    target_col = cfg["target"]

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=y,
    )

    X_train, X_test, scaler = scale_features(X_train, X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_cols

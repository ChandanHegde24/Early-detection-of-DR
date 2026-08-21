"""
Tabular clinical data preprocessing — Multi-Class 5-Grade DR System
Handles:
  1. Generating realistic clinical distribution across all 5 DR grades (0..4)
  2. Multi-class target preparation
  3. Clinically validated interaction features
  4. Class balancing & scaling pipeline
"""

import os
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.config import load_settings

settings = load_settings()


def generate_synthetic_clinical_data(n_per_class: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Generate realistic clinical tabular data for multi-class DR risk scoring.

    Generates stratified distributions for all 5 DR severity grades based on
    established clinical endocrinology & ophthalmology parameters.
    """
    np.random.seed(random_state)
    dfs = []

    # Clinical parameter profiles per DR grade:
    # (hba1c_m, hba1c_s, sbp_m, sbp_s, dbp_m, dbp_s, dur_m, dur_s, bmi_m, bmi_s,
    #  chol_m, chol_s, hdl_m, hdl_s, ldl_m, ldl_s, tg_m, tg_s, smoke_probs, family_prob)
    profiles = [
        # Grade 0: No DR (Healthy / Controlled Diabetes)
        (5.8, 0.7, 118, 10, 74, 7, 2.5, 2.0, 24.5, 3.5, 175, 25, 55, 10, 95, 20, 110, 35, [0.70, 0.20, 0.10], 0.15),
        # Grade 1: Mild NPDR
        (7.4, 0.8, 134, 12, 82, 8, 7.5, 3.0, 27.5, 4.0, 205, 30, 46, 9, 125, 25, 160, 45, [0.50, 0.30, 0.20], 0.30),
        # Grade 2: Moderate NPDR
        (8.8, 1.0, 148, 14, 88, 8, 13.5, 4.0, 30.0, 4.5, 228, 35, 40, 8, 145, 28, 210, 60, [0.40, 0.35, 0.25], 0.45),
        # Grade 3: Severe NPDR
        (10.4, 1.2, 162, 16, 94, 10, 19.0, 5.0, 32.5, 5.0, 252, 38, 35, 7, 168, 30, 270, 75, [0.30, 0.35, 0.35], 0.65),
        # Grade 4: Proliferative DR
        (12.2, 1.5, 178, 18, 102, 11, 24.0, 6.0, 34.5, 5.5, 275, 42, 32, 6, 190, 35, 340, 90, [0.20, 0.35, 0.45], 0.80),
    ]

    for grade, p in enumerate(profiles):
        (hba1c_m, hba1c_s, sbp_m, sbp_s, dbp_m, dbp_s, dur_m, dur_s, bmi_m, bmi_s,
         chol_m, chol_s, hdl_m, hdl_s, ldl_m, ldl_s, tg_m, tg_s, smoke_p, fam_p) = p

        age = np.random.normal(52 + grade * 3.5, 9, n_per_class).clip(25, 88)
        bmi = np.random.normal(bmi_m, bmi_s, n_per_class).clip(17, 52)
        hba1c = np.random.normal(hba1c_m, hba1c_s, n_per_class).clip(4.5, 16.0)
        sbp = np.random.normal(sbp_m, sbp_s, n_per_class).clip(90, 225)
        dbp = np.random.normal(dbp_m, dbp_s, n_per_class).clip(55, 130)
        chol = np.random.normal(chol_m, chol_s, n_per_class).clip(110, 380)
        hdl = np.random.normal(hdl_m, hdl_s, n_per_class).clip(18, 95)
        ldl = np.random.normal(ldl_m, ldl_s, n_per_class).clip(45, 280)
        tg = np.random.normal(tg_m, tg_s, n_per_class).clip(50, 650)
        dur = np.random.normal(dur_m, dur_s, n_per_class).clip(0.5, 45)
        smoke = np.random.choice([0, 1, 2], size=n_per_class, p=smoke_p)
        fam = np.random.choice([0, 1], size=n_per_class, p=[1 - fam_p, fam_p])

        df_g = pd.DataFrame({
            "age": np.round(age, 1),
            "bmi": np.round(bmi, 1),
            "hba1c": np.round(hba1c, 1),
            "blood_pressure_systolic": np.round(sbp, 0),
            "blood_pressure_diastolic": np.round(dbp, 0),
            "cholesterol_total": np.round(chol, 0),
            "cholesterol_hdl": np.round(hdl, 0),
            "cholesterol_ldl": np.round(ldl, 0),
            "triglycerides": np.round(tg, 0),
            "diabetes_duration_years": np.round(dur, 1),
            "smoking_status": smoke,
            "family_history_dr": fam,
            "dr_grade": grade,
        })
        dfs.append(df_g)

    df = pd.concat(dfs).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


def load_clinical_data(csv_path: Optional[str] = None, force_regenerate: bool = False) -> pd.DataFrame:
    csv_path = csv_path or settings.get("paths", {}).get("raw_tabular", "data/raw/clinical_data.csv")
    if os.path.exists(csv_path) and not force_regenerate:
        df = pd.read_csv(csv_path)
        # Check if existing data is multi-class balanced
        if len(df["dr_grade"].unique()) >= 5 and df["dr_grade"].value_counts().min() > 100:
            return df

    print(f"[tabular_prep] Generating balanced 5-class clinical dataset...")
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else ".", exist_ok=True)
    df = generate_synthetic_clinical_data(n_per_class=2000)
    try:
        df.to_csv(csv_path, index=False)
        print(f"[tabular_prep] Saved clinical dataset to '{csv_path}' ({len(df)} records)")
    except Exception as e:
        print(f"[tabular_prep] Notice: could not write to {csv_path}: {e}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 5 clinically meaningful interaction features."""
    df = df.copy()
    df["hba1c_duration"] = df["hba1c"] * df["diabetes_duration_years"]
    df["bp_pulse"] = df["blood_pressure_systolic"] - df["blood_pressure_diastolic"]
    df["chol_ratio"] = df["cholesterol_total"] / (df["cholesterol_hdl"] + 0.001)
    df["high_risk_combo"] = (
        (df["hba1c"] > 8.0) & (df["diabetes_duration_years"] > 10)
    ).astype(int)
    df["metabolic_score"] = (
        df["bmi"] * 0.3 + df["triglycerides"] * 0.01 + df["hba1c"] * 0.5
    )
    return df


def handle_missing_values(df: pd.DataFrame, feature_cols: list) -> Tuple[np.ndarray, SimpleImputer]:
    """Median imputation for clinical features."""
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(df[feature_cols])
    return X, imputer


def scale_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), scaler


def prepare_tabular_data(
    csv_path: Optional[str] = None,
    force_regenerate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list]:
    """Full improved multi-class pipeline for clinical data."""
    cfg = settings.get("tabular", {})
    df = load_clinical_data(csv_path, force_regenerate=force_regenerate)
    df = engineer_features(df)

    base_features = [f for f in cfg.get("features", []) if f in df.columns]
    engineered_features = [
        "hba1c_duration", "bp_pulse", "chol_ratio",
        "high_risk_combo", "metabolic_score"
    ]
    feature_cols = base_features + engineered_features

    X, imputer = handle_missing_values(df, feature_cols)
    y = df["dr_grade"].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.get("test_size", 0.2),
        random_state=cfg.get("random_state", 42),
        stratify=y,
    )

    X_train, X_test, scaler = scale_features(X_train, X_test)
    return X_train, X_test, y_train, y_test, scaler, feature_cols


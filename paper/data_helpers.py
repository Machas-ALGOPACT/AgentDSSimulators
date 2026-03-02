"""
data_helpers.py
Standalone data loading for paper/ analysis scripts.
Reads from local CSVs, merges patients.csv for enrichment.
No backend package imports required.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
_PAPER_DIR   = Path(__file__).parent
_ROOT        = _PAPER_DIR.parent
_DATA_DIR    = _ROOT / "agentds-platform" / "backend" / "data" / "healthcare"

# ── Target columns (real column names from data inspection) ──────────────────
TARGET_READMISSION  = "readmit_30d"
TARGET_ED_COST      = "ed_cost_next3y_usd"
TARGET_DISCHARGE    = "discharge_ready_day11"   # NOTE: backend has a typo ("ready_for_discharge")

# ── ID columns to drop before training ───────────────────────────────────────
DROP_READMISSION = ["admission_id", "patient_id"]
DROP_ED_COST     = ["patient_id"]
DROP_DISCHARGE   = ["stay_id", "patient_id"]


def _load_patients() -> pd.DataFrame:
    """Load the common patients reference table."""
    path = _DATA_DIR / "patients.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_readmission(merge_patients: bool = True) -> tuple[pd.DataFrame, str]:
    """
    Returns (df, target_col) for readmission prediction.
    Merges patients.csv by default for age/sex/insurance/zip3.
    """
    df = pd.read_csv(_DATA_DIR / "admissions_train.csv")
    if merge_patients:
        patients = _load_patients()
        if not patients.empty and "patient_id" in df.columns:
            df = df.merge(patients, on="patient_id", how="left")
    # Drop ID columns
    df = df.drop(columns=[c for c in DROP_READMISSION if c in df.columns])
    return df, TARGET_READMISSION


def load_ed_cost(merge_patients: bool = True) -> tuple[pd.DataFrame, str]:
    """
    Returns (df, target_col) for ED cost forecasting.
    Merges patients.csv by default.
    """
    df = pd.read_csv(_DATA_DIR / "ed_cost_train.csv")
    if merge_patients:
        patients = _load_patients()
        if not patients.empty and "patient_id" in df.columns:
            df = df.merge(patients, on="patient_id", how="left")
    df = df.drop(columns=[c for c in DROP_ED_COST if c in df.columns])
    return df, TARGET_ED_COST


def load_discharge(merge_patients: bool = True) -> tuple[pd.DataFrame, str]:
    """
    Returns (df, target_col) for discharge readiness prediction.
    Merges patients.csv by default.
    """
    df = pd.read_csv(_DATA_DIR / "stays_train.csv")
    if merge_patients:
        patients = _load_patients()
        if not patients.empty and "patient_id" in df.columns:
            df = df.merge(patients, on="patient_id", how="left")
    df = df.drop(columns=[c for c in DROP_DISCHARGE if c in df.columns])
    return df, TARGET_DISCHARGE


def get_X_y(df: pd.DataFrame, target_col: str):
    """Split dataframe into X (features) and y (target)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def describe_dataset(df: pd.DataFrame, target_col: str, name: str) -> dict:
    """Return a summary dict about the dataset."""
    X, y = get_X_y(df, target_col)
    numeric_features   = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categoric_features = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    summary = {
        "task": name,
        "n_samples": int(len(df)),
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "numeric_features": numeric_features,
        "categorical_features": categoric_features,
        "target_col": target_col,
        "target_distribution": y.value_counts().to_dict() if y.dtype in ["int64", "object"] else {
            "min": float(y.min()), "max": float(y.max()),
            "mean": float(y.mean()), "std": float(y.std())
        },
    }
    return summary

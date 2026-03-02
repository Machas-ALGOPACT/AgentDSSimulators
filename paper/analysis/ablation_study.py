"""
TASK 8: ablation_study.py
Trains 5 progressively-richer model variants for readmission prediction
to quantify the contribution of each engineered feature.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_helpers import load_readmission, get_X_y

TABLES_DIR   = Path(__file__).parent.parent / "outputs" / "tables"
RANDOM_STATE = 42

# ── Real column names from admissions_train.csv ───────────────────────────────
BASE_FEATURES = ["los_days", "ed_visits_6m", "charlson_band",
                 "acuity_emergent", "discharge_weekday", "primary_dx"]


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["los_squared"]      = df["los_days"] ** 2
    df["ed_visits_rate"]   = df["ed_visits_6m"] / (df["los_days"] + 1)
    df["weekday_binary"]   = df["discharge_weekday"].isin([6, 7]).astype(int)
    return df


def _build_pipe(X_train):
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ], remainder="drop")
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])


def _evaluate(y_test, y_pred, y_proba):
    return {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Macro_F1":  round(f1_score(y_test, y_pred, average="macro"), 4),
        "AUC":       round(roc_auc_score(y_test, y_proba), 4),
    }


def run():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[Ablation Study] Readmission Prediction")

    # Load with patients merged (needed for patient_features step)
    df_full, target_col = load_readmission(merge_patients=True)
    df_full = _add_engineered_features(df_full)

    # ── Patient features from patients.csv ────────────────────────────────────
    patient_features = [c for c in ["age", "sex", "insurance"] if c in df_full.columns]

    # ── Five ablation variants ────────────────────────────────────────────────
    # Each variant defines the feature columns to USE
    variants = [
        ("Base (raw features)",      BASE_FEATURES),
        ("+ los_squared",            BASE_FEATURES + ["los_squared"]),
        ("+ ed_visits_rate",         BASE_FEATURES + ["los_squared", "ed_visits_rate"]),
        ("+ weekday_binary",         BASE_FEATURES + ["los_squared", "ed_visits_rate", "weekday_binary"]),
        ("+ patient features (Full)",BASE_FEATURES + ["los_squared", "ed_visits_rate", "weekday_binary"] + patient_features),
    ]

    rows = []
    for variant_name, feature_cols in variants:
        # Only keep columns that actually exist in df_full
        use_cols = [c for c in feature_cols if c in df_full.columns] + [target_col]
        df_v = df_full[use_cols]
        X, y = get_X_y(df_v, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        pipe = _build_pipe(X_train)
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = _evaluate(y_test, y_pred, y_proba)
        row = {"Variant": variant_name, "n_features": len(use_cols) - 1, **metrics}
        rows.append(row)
        print(f"  {variant_name:<35}  F1={metrics['Macro_F1']:.4f}  AUC={metrics['AUC']:.4f}")

    results = pd.DataFrame(rows)
    csv_path = TABLES_DIR / "ablation_study_readmission.csv"
    results.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    return results


if __name__ == "__main__":
    run()

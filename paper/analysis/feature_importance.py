"""
TASK 2: feature_importance.py
Extracts feature importances from RandomForest, saves top-10 bar charts + CSVs.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_helpers import load_readmission, load_ed_cost, load_discharge, get_X_y

PLOTS_DIR  = Path(__file__).parent.parent / "outputs" / "plots"
TABLES_DIR = Path(__file__).parent.parent / "outputs" / "tables"
RANDOM_STATE = 42


def _build_preprocessor(X):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
                             remainder="drop"), num_cols, cat_cols


def _get_feature_names(preprocessor, num_cols, cat_cols):
    names = list(num_cols)
    ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
    names += list(ohe.get_feature_names_out(cat_cols))
    return names


def _plot_importance(importances, feature_names, title, save_path, top_n=10):
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(idx)), importances[idx], color="#2196F3", edgecolor="white")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx], fontsize=9)
    ax.set_xlabel("Importance", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved plot: {save_path}")


def _run_task(task_key, loader, model_cls, model_kwargs, title):
    df, target_col = loader(merge_patients=True)
    X, y = get_X_y(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    pre, num_cols, cat_cols = _build_preprocessor(X_train)
    model = model_cls(**model_kwargs, random_state=RANDOM_STATE, n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)

    fnames = _get_feature_names(pipe.named_steps["pre"], num_cols, cat_cols)
    importances = pipe.named_steps["model"].feature_importances_

    top10_idx = np.argsort(importances)[::-1][:10]
    top10 = pd.DataFrame({
        "feature":    [fnames[i] for i in top10_idx],
        "importance": importances[top10_idx]
    })

    csv_path = TABLES_DIR / f"feature_importance_{task_key}.csv"
    top10.to_csv(csv_path, index=False)
    print(f"  Saved CSV : {csv_path}")

    png_path = PLOTS_DIR / f"feature_importance_{task_key}.png"
    _plot_importance(importances, fnames, title, png_path)
    return top10


def run():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("readmission",  load_readmission, RandomForestClassifier,
         {"n_estimators": 100}, "Feature Importance — Readmission Prediction"),
        ("ed_cost",      load_ed_cost,     RandomForestRegressor,
         {"n_estimators": 100}, "Feature Importance — ED Cost Forecasting"),
        ("discharge",    load_discharge,   RandomForestClassifier,
         {"n_estimators": 100}, "Feature Importance — Discharge Readiness"),
    ]

    results = {}
    for task_key, loader, model_cls, model_kwargs, title in tasks:
        print(f"\n[Feature Importance] {title}")
        results[task_key] = _run_task(task_key, loader, model_cls, model_kwargs, title)
    return results


if __name__ == "__main__":
    run()

"""
TASKS 6A + 6B: cost_analysis.py
6A: Prediction vs Actual scatter plot for ED cost forecasting.
6B: Residual distribution histogram.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_helpers import load_ed_cost, get_X_y

PLOTS_DIR    = Path(__file__).parent.parent / "outputs" / "plots"
RANDOM_STATE = 42


def _build_pipeline(X_train):
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ], remainder="drop")
    return Pipeline([("pre", pre), ("reg", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))])


def run():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[Cost Analysis] ED Cost Forecasting")

    df, target_col = load_ed_cost(merge_patients=True)
    X, y = get_X_y(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    pipe = _build_pipeline(X_train)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"  MAE = {mae:,.2f}  R² = {r2:.4f}")

    # ── 6A: Prediction vs Actual ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, y_pred, alpha=0.4, s=20, color="#009688", edgecolors="none")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, color="#E53935", lw=1.5, linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Actual Cost (USD)", fontsize=11)
    ax.set_ylabel("Predicted Cost (USD)", fontsize=11)
    ax.set_title(f"Prediction vs Actual — ED Cost\nMAE=${mae:,.0f}  R²={r2:.3f}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path_6a = PLOTS_DIR / "pred_vs_actual_cost.png"
    plt.savefig(path_6a, dpi=150)
    plt.close()
    print(f"  Saved: {path_6a}")

    # ── 6B: Residual Distribution ─────────────────────────────────────────────
    residuals = y_pred - y_test.values
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=40, color="#7B1FA2", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="#E53935", lw=2, linestyle="--", label="Zero error")
    ax.set_xlabel("Residual (Predicted − Actual, USD)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Residual Distribution — ED Cost Forecasting", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    path_6b = PLOTS_DIR / "residual_distribution_cost.png"
    plt.savefig(path_6b, dpi=150)
    plt.close()
    print(f"  Saved: {path_6b}")

    return {"mae": mae, "r2": r2}


if __name__ == "__main__":
    run()

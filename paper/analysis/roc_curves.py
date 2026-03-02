"""
TASK 3: roc_curves.py
ROC curve + AUC for classification tasks (readmission & discharge).
"""

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_helpers import load_readmission, load_discharge, get_X_y

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
    return Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))])


def _plot_roc(y_test, y_proba, task_key, task_name):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#E91E63", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#90A4AE", lw=1.5, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curve — {task_name}", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_path = PLOTS_DIR / f"roc_curve_{task_key}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  AUC={roc_auc:.4f}  Saved: {save_path}")
    return roc_auc


def run():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [
        ("readmission", load_readmission, "Readmission Prediction"),
        ("discharge",   load_discharge,   "Discharge Readiness"),
    ]
    results = {}
    for task_key, loader, task_name in tasks:
        print(f"\n[ROC Curve] {task_name}")
        df, target_col = loader(merge_patients=True)
        X, y = get_X_y(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        pipe = _build_pipeline(X_train)
        pipe.fit(X_train, y_train)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        results[task_key] = _plot_roc(y_test, y_proba, task_key, task_name)
    return results


if __name__ == "__main__":
    run()

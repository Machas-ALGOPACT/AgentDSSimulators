"""
TASK 5: confusion_matrices.py
Confusion matrix heatmaps for classification tasks.
"""

import sys
import numpy as np
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
from sklearn.metrics import confusion_matrix

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


def _plot_cm(cm, class_labels, task_key, task_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    tick_marks = range(len(class_labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_labels, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_labels, fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {task_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_path = PLOTS_DIR / f"confusion_matrix_{task_key}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


def run():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [
        ("readmission", load_readmission, "Readmission Prediction", ["Not Readmitted", "Readmitted"]),
        ("discharge",   load_discharge,   "Discharge Readiness",    ["Not Ready",       "Ready"]),
    ]
    for task_key, loader, task_name, labels in tasks:
        print(f"\n[Confusion Matrix] {task_name}")
        df, target_col = loader(merge_patients=True)
        X, y = get_X_y(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        pipe = _build_pipeline(X_train)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        _plot_cm(cm, labels, task_key, task_name)


if __name__ == "__main__":
    run()

"""
TASK 7: model_comparison.py
Trains Logistic Regression, Random Forest, Gradient Boosting, and a Soft-Voting Ensemble
for classification tasks. Saves Accuracy + Macro F1 comparison CSV.
"""

import sys
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_helpers import load_readmission, load_discharge, get_X_y

TABLES_DIR   = Path(__file__).parent.parent / "outputs" / "tables"
RANDOM_STATE = 42


def _build_preprocessor(X_train):
    num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ], remainder="drop"), num_cols, cat_cols


def _evaluate(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    return {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Macro_F1": round(f1_score(y_test, y_pred, average="macro"), 4),
    }


def run():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [
        ("readmission", load_readmission, "Readmission Prediction"),
        ("discharge",   load_discharge,   "Discharge Readiness"),
    ]

    all_results = {}
    for task_key, loader, task_name in tasks:
        print(f"\n[Model Comparison] {task_name}")
        df, target_col = loader(merge_patients=True)
        X, y = get_X_y(df, target_col)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        pre, _, _ = _build_preprocessor(X_train)

        lr  = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        rf  = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        gb  = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
        ens = VotingClassifier(estimators=[("lr", lr), ("rf", rf), ("gb", gb)], voting="soft")

        models = [
            ("Logistic Regression", lr),
            ("Random Forest",       rf),
            ("Gradient Boosting",   gb),
            ("Ensemble (Voting)",   ens),
        ]

        rows = []
        for name, clf in models:
            pipe = Pipeline([("pre", pre), ("clf", clf)])
            pipe.fit(X_train, y_train)
            metrics = _evaluate(pipe, X_test, y_test)
            row = {"Model": name, **metrics}
            rows.append(row)
            print(f"  {name:<25}  Acc={metrics['Accuracy']:.4f}  F1={metrics['Macro_F1']:.4f}")

        results_df = pd.DataFrame(rows)
        csv_path = TABLES_DIR / f"model_comparison_{task_key}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
        all_results[task_key] = results_df

    return all_results


if __name__ == "__main__":
    run()

"""
TASK 1: pipeline_analysis.py
Logs feature counts, names, class distribution, dataset shape per task.
Saves summaries as JSON to outputs/metrics/
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_helpers import load_readmission, load_ed_cost, load_discharge, describe_dataset, get_X_y

METRICS_DIR = Path(__file__).parent.parent / "outputs" / "metrics"


def run():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [
        ("readmission",  load_readmission,  "Readmission Prediction"),
        ("ed_cost",      load_ed_cost,       "ED Cost Forecasting"),
        ("discharge",    load_discharge,     "Discharge Readiness"),
    ]

    summaries = []
    for key, loader, name in tasks:
        df, target_col = loader(merge_patients=True)
        summary = describe_dataset(df, target_col, name)
        out_path = METRICS_DIR / f"pipeline_summary_{key}.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"  Shape       : {df.shape}")
        print(f"  Features    : {summary['n_features']}")
        print(f"  Target      : {target_col}")
        print(f"  Distribution: {summary['target_distribution']}")
        print(f"  Saved       : {out_path}")
        summaries.append(summary)

    return summaries


if __name__ == "__main__":
    run()

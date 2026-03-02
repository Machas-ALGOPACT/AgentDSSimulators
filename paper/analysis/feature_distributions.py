"""
TASK 9: feature_distributions.py
Feature distribution plots using real column names:
- Boxplot: los_days by readmit_30d
- Histogram: ed_visits_6m distribution
- Bar chart: charlson_band vs readmit_30d
"""

import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_helpers import load_readmission

PLOTS_DIR = Path(__file__).parent.parent / "outputs" / "plots"


def run():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[Feature Distributions] Readmission Dataset")

    df, target_col = load_readmission(merge_patients=True)
    # Re-attach target if dropped (it is kept by default from load_readmission's raw load)
    # Actually load raw to keep target alongside features
    from pathlib import Path as P
    import pandas as _pd
    _data_dir = P(__file__).parent.parent.parent / "agentds-platform" / "backend" / "data" / "healthcare"
    raw = _pd.read_csv(_data_dir / "admissions_train.csv")

    # ── Plot 1: LOS vs Readmission (boxplot) ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    groups = [raw.loc[raw[target_col] == val, "los_days"].values for val in sorted(raw[target_col].unique())]
    labels = ["Not Readmitted (0)", "Readmitted (1)"]
    bp = ax.boxplot(groups, patch_artist=True, notch=False,
                    medianprops=dict(color="#E53935", linewidth=2))
    colors = ["#42A5F5", "#EF5350"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Length of Stay (days)", fontsize=11)
    ax.set_title("LOS by 30-Day Readmission Status", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    p1 = PLOTS_DIR / "los_vs_readmission.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"  Saved: {p1}")

    # ── Plot 2: ED Visits Distribution (histogram) ────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(raw["ed_visits_6m"], bins=20, color="#7E57C2", edgecolor="white", alpha=0.85)
    ax.set_xlabel("ED Visits in Prior 6 Months", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of ED Visits (Prior 6 Months)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    p2 = PLOTS_DIR / "ed_visits_distribution.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"  Saved: {p2}")

    # ── Plot 3: Charlson Band vs Readmission (grouped bar) ────────────────────
    grouped = raw.groupby("charlson_band")[target_col].mean().reset_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(grouped["charlson_band"], grouped[target_col] * 100,
                  color="#26A69A", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Charlson Comorbidity Index Band", fontsize=11)
    ax.set_ylabel("Readmission Rate (%)", fontsize=11)
    ax.set_title("Readmission Rate by Charlson Band", fontsize=13, fontweight="bold")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    p3 = PLOTS_DIR / "charlson_vs_readmission.png"
    plt.savefig(p3, dpi=150)
    plt.close()
    print(f"  Saved: {p3}")


if __name__ == "__main__":
    run()

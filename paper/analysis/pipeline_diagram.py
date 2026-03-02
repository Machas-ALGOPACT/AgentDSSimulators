"""
TASK 12: pipeline_diagram.py
Generates a pipeline flow diagram using matplotlib patches and arrows.
Flow: Data → Preprocessing → Feature Engineering → Model → Evaluation → AI Suggestions → Human Validation
"""

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

PLOTS_DIR = Path(__file__).parent.parent / "outputs" / "plots"

STEPS = [
    ("Raw Data",           "#EF5350", "admissions_train.csv\ned_cost_train.csv\nstays_train.csv"),
    ("Preprocessing",      "#42A5F5", "Imputation\nScaling\nEncoding"),
    ("Feature Engineering","#AB47BC", "los_squared\ned_visits_rate\nweekday_binary"),
    ("Model Training",     "#26A69A", "RandomForest\nGrad. Boosting\nEnsemble"),
    ("Evaluation",         "#FF7043", "AUC · F1\nMAE · R²\nConfusion Matrix"),
    ("AI Suggestions",     "#78909C", "Feature Importance\nAblation Study\nModel Comparison"),
    ("Human Validation",   "#66BB6A", "Expert Review\nClinical Sign-off"),
]


def run():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_xlim(0, len(STEPS) * 2.2)
    ax.set_ylim(-1.5, 2.5)
    ax.axis("off")

    box_w, box_h = 1.8, 1.4
    gap = 2.2

    for i, (name, color, detail) in enumerate(STEPS):
        x = i * gap + 0.2
        y = 0.2
        fancy = mpatches.FancyBboxPatch((x, y), box_w, box_h,
                                        boxstyle="round,pad=0.08",
                                        facecolor=color, edgecolor="white",
                                        linewidth=1.5, alpha=0.92)
        ax.add_patch(fancy)
        ax.text(x + box_w / 2, y + box_h * 0.72, name,
                ha="center", va="center", fontsize=9, fontweight="bold",
                color="white", wrap=True)
        ax.text(x + box_w / 2, y + box_h * 0.28, detail,
                ha="center", va="center", fontsize=6.5,
                color="white", alpha=0.9, linespacing=1.4)
        if i < len(STEPS) - 1:
            ax.annotate("", xy=(x + box_w + (gap - box_w), y + box_h / 2),
                        xytext=(x + box_w, y + box_h / 2),
                        arrowprops=dict(arrowstyle="->", color="#455A64", lw=2))

    ax.set_title("Healthcare ML Pipeline — End-to-End Overview",
                 fontsize=14, fontweight="bold", pad=10, color="#263238")
    plt.tight_layout()
    save_path = PLOTS_DIR / "pipeline_diagram.png"
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
    return save_path


if __name__ == "__main__":
    run()

"""
TASK 10: latex_outputs.py
Generates LaTeX figure and table snippets for all generated outputs.
Saves to outputs/latex_snippets.txt
"""

import sys
from pathlib import Path

OUTPUTS_DIR  = Path(__file__).parent.parent / "outputs"
PLOTS_DIR    = OUTPUTS_DIR / "plots"
TABLES_DIR   = OUTPUTS_DIR / "tables"


FIGURE_CAPTIONS = {
    "pipeline_diagram":              "End-to-end healthcare ML pipeline from raw data ingestion through preprocessing, feature engineering, model training, automated evaluation, AI-generated suggestions, and final human validation.",
    "feature_importance_readmission":"Top-10 most important features for the 30-day hospital readmission classification model (Random Forest). Higher importance indicates greater influence on prediction.",
    "feature_importance_ed_cost":    "Top-10 most important features for the ED cost forecasting regression model (Random Forest Regressor).",
    "feature_importance_discharge":  "Top-10 most important features for the discharge readiness classification model (Random Forest).",
    "roc_curve_readmission":         "Receiver Operating Characteristic (ROC) curve for 30-day readmission prediction. AUC reported in the legend.",
    "roc_curve_discharge":           "ROC curve for discharge readiness (Day 11) prediction. AUC reported in the legend.",
    "pr_curve_readmission":          "Precision-Recall (PR) curve for readmission prediction. Average Precision (AP) is reported; the dashed line indicates the no-skill baseline.",
    "pr_curve_discharge":            "PR curve for discharge readiness prediction.",
    "confusion_matrix_readmission":  "Confusion matrix for the readmission prediction model evaluated on the held-out test set (20\\%).",
    "confusion_matrix_discharge":    "Confusion matrix for the discharge readiness model.",
    "pred_vs_actual_cost":           "Scatter plot of predicted versus actual ED costs (USD) on the held-out test set. The dashed red line denotes perfect prediction.",
    "residual_distribution_cost":    "Distribution of prediction residuals (\\$\\hat{y} - y\\$) for the ED cost forecasting model. A symmetric distribution centred near zero indicates low systematic bias.",
    "los_vs_readmission":            "Box plots of length-of-stay (LOS, days) stratified by 30-day readmission status. Longer stays correlate with higher readmission risk.",
    "ed_visits_distribution":        "Distribution of ED visits in the 6 months prior to the index admission across the readmission training cohort.",
    "charlson_vs_readmission":       "30-day readmission rate (\\%) by Charlson Comorbidity Index (CCI) band. Patients with higher comorbidity burden exhibit markedly elevated readmission rates.",
}

TABLE_CAPTIONS = {
    "feature_importance_readmission": "Top-10 Feature Importances — Readmission Prediction",
    "feature_importance_ed_cost":     "Top-10 Feature Importances — ED Cost Forecasting",
    "feature_importance_discharge":   "Top-10 Feature Importances — Discharge Readiness",
    "model_comparison_readmission":   "Model Comparison — Readmission Prediction",
    "model_comparison_discharge":     "Model Comparison — Discharge Readiness",
    "ablation_study_readmission":     "Ablation Study — Readmission Prediction",
}


def _figure_block(fname, caption, label=None):
    label = label or fname
    return (
        "\\begin{figure}[h]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=0.80\\linewidth]{{figures/{fname}.png}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{fig:{label}}}\n"
        "\\end{figure}"
    )


def _table_block(csv_name, caption, label=None):
    import pandas as pd
    label = label or csv_name
    csv_path = TABLES_DIR / f"{csv_name}.csv"
    if not csv_path.exists():
        return f"% Table {csv_name}.csv not yet generated\n"
    df = pd.read_csv(csv_path)
    col_fmt = "l" + "r" * (len(df.columns) - 1)
    header  = " & ".join(f"\\textbf{{{c}}}" for c in df.columns) + " \\\\"
    rows    = []
    for _, row in df.iterrows():
        rows.append(" & ".join(str(v) for v in row.values) + " \\\\")
    body = "\n    ".join(rows)
    return (
        "\\begin{table}[h]\n"
        "  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{tab:{label}}}\n"
        f"  \\begin{{tabular}}{{{col_fmt}}}\n"
        "    \\hline\n"
        f"    {header}\n"
        "    \\hline\n"
        f"    {body}\n"
        "    \\hline\n"
        "  \\end{tabular}\n"
        "\\end{table}"
    )


def run():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    snippets = []

    snippets.append("% ═══════════════════════════════════════")
    snippets.append("% FIGURES")
    snippets.append("% ═══════════════════════════════════════\n")
    for fname, caption in FIGURE_CAPTIONS.items():
        snippets.append(_figure_block(fname, caption))
        snippets.append("")

    snippets.append("\n% ═══════════════════════════════════════")
    snippets.append("% TABLES")
    snippets.append("% ═══════════════════════════════════════\n")
    for csv_name, caption in TABLE_CAPTIONS.items():
        snippets.append(_table_block(csv_name, caption))
        snippets.append("")

    out_path = OUTPUTS_DIR / "latex_snippets.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(snippets))
    print(f"  Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    run()

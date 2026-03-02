# Healthcare ML Paper — Analysis Suite

Self-contained analysis package that generates all figures, tables, and LaTeX snippets needed for the IEEE paper submission.

## Quick Start

```powershell
# From d:\2025\AgentDS\AgentDSSimulators\
d:\2025\AgentDS\AgentDSSimulators\venv\Scripts\python.exe paper/run_analysis.py
```

## Output Structure

```
paper/
├── run_analysis.py          ← Main entry point
├── data_helpers.py          ← Data loading (no backend imports)
├── analysis/                ← 11 analysis modules
│   ├── pipeline_analysis.py
│   ├── feature_importance.py
│   ├── roc_curves.py
│   ├── pr_curves.py
│   ├── confusion_matrices.py
│   ├── cost_analysis.py
│   ├── model_comparison.py
│   ├── ablation_study.py
│   ├── feature_distributions.py
│   ├── pipeline_diagram.py
│   └── latex_outputs.py
└── outputs/
    ├── plots/               ← ~15 PNG figures
    ├── tables/              ← CSV tables
    ├── metrics/             ← JSON summaries
    └── latex_snippets.txt   ← Ready-to-paste LaTeX
```

## Generated Outputs

| File | Description |
|---|---|
| `plots/pipeline_diagram.png` | 7-stage ML pipeline diagram |
| `plots/feature_importance_*.png` | Top-10 features per task |
| `plots/roc_curve_*.png` | ROC + AUC for classification tasks |
| `plots/pr_curve_*.png` | Precision-Recall curves |
| `plots/confusion_matrix_*.png` | Confusion matrix heatmaps |
| `plots/pred_vs_actual_cost.png` | ED cost scatter plot |
| `plots/residual_distribution_cost.png` | Residual histogram |
| `plots/los_vs_readmission.png` | LOS boxplot by readmission |
| `plots/ed_visits_distribution.png` | ED visits histogram |
| `plots/charlson_vs_readmission.png` | Charlson band vs readmission rate |
| `tables/feature_importance_*.csv` | Feature importance rankings |
| `tables/model_comparison_*.csv` | LR vs RF vs GB vs Ensemble |
| `tables/ablation_study_readmission.csv` | 5-variant ablation results |
| `metrics/pipeline_summary_*.json` | Dataset summaries |
| `latex_snippets.txt` | LaTeX \\figure and \\table blocks |

## Data Source

All data loaded from `agentds-platform/backend/data/healthcare/`:
- `admissions_train.csv` — Readmission (5,000 rows)
- `ed_cost_train.csv` — ED Cost (2,000 rows)
- `stays_train.csv` — Discharge Readiness (1,000 rows)
- `patients.csv` — Patient demographics (merged for feature enrichment)

> **Note:** `stays_train.csv` uses column `discharge_ready_day11` (not `ready_for_discharge` as referenced in the backend service — this is a known mismatch).

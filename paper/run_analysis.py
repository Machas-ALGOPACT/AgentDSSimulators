"""
run_analysis.py
===============================================================================
Main entry point for the healthcare ML paper analysis.
Runs all 13 tasks from Doc.docx in order.

Usage (from project root with venv activated):
    d:\\2025\\AgentDS\\AgentDSSimulators\\venv\\Scripts\\python.exe paper/run_analysis.py

Or from within paper/ directory:
    python run_analysis.py
===============================================================================
"""

import sys
import io
import time
from pathlib import Path

# Force UTF-8 output regardless of Windows console code page
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure paper/ is on the path
PAPER_DIR = Path(__file__).parent
sys.path.insert(0, str(PAPER_DIR))

# ── Imports ───────────────────────────────────────────────────────────────────
from analysis import pipeline_analysis
from analysis import feature_importance
from analysis import roc_curves
from analysis import pr_curves
from analysis import confusion_matrices
from analysis import cost_analysis
from analysis import model_comparison
from analysis import ablation_study
from analysis import feature_distributions
from analysis import latex_outputs
from analysis import pipeline_diagram


BANNER = """
+==================================================================+
|        Healthcare ML Paper -- Analysis Suite                     |
|        Tasks 1-13 from Doc.docx (all 3 challenges)              |
+==================================================================+
"""


def run_task(name: str, fn, results: dict):
    """Run a single task module and record its output."""
    print(f"\n{'─'*65}")
    print(f"  TASK: {name}")
    print(f"{'─'*65}")
    t0 = time.time()
    try:
        out = fn()
        elapsed = time.time() - t0
        results[name] = {"status": "OK", "elapsed_s": round(elapsed, 1), "output": out}
        print(f"  ✓ Done in {elapsed:.1f}s")
    except Exception as exc:
        elapsed = time.time() - t0
        results[name] = {"status": "ERROR", "elapsed_s": round(elapsed, 1), "error": str(exc)}
        print(f"  ✗ ERROR: {exc}")
        import traceback
        traceback.print_exc()
    return results


def print_summary(results: dict, outputs_dir: Path):
    """Print final summary of all tasks."""
    print(f"\n{'═'*65}")
    print("  SUMMARY")
    print(f"{'═'*65}")
    ok  = [k for k, v in results.items() if v["status"] == "OK"]
    err = [k for k, v in results.items() if v["status"] == "ERROR"]
    print(f"  ✓ Passed : {len(ok)}/{len(results)}")
    if err:
        print(f"  ✗ Failed : {', '.join(err)}")

    print(f"\n  Output files in: {outputs_dir}")
    for folder in ["plots", "tables", "metrics"]:
        d = outputs_dir / folder
        if d.exists():
            files = list(d.glob("*"))
            print(f"    {folder}/ ({len(files)} files)")
            for f in sorted(files):
                print(f"      {f.name}")
    latex_file = outputs_dir / "latex_snippets.txt"
    if latex_file.exists():
        print(f"    latex_snippets.txt")
    print(f"{'═'*65}\n")


def main():
    print(BANNER)
    outputs_dir = PAPER_DIR / "outputs"
    results = {}

    TASKS = [
        ("TASK 1 — Pipeline Analysis",          pipeline_analysis.run),
        ("TASK 2 — Feature Importance",         feature_importance.run),
        ("TASK 3 — ROC Curves",                 roc_curves.run),
        ("TASK 4 — Precision-Recall Curves",    pr_curves.run),
        ("TASK 5 — Confusion Matrices",         confusion_matrices.run),
        ("TASK 6 — Cost Analysis",              cost_analysis.run),
        ("TASK 7 — Model Comparison",           model_comparison.run),
        ("TASK 8 — Ablation Study",             ablation_study.run),
        ("TASK 9 — Feature Distributions",      feature_distributions.run),
        ("TASK 12 — Pipeline Diagram",          pipeline_diagram.run),
        ("TASK 10 — LaTeX Outputs",             latex_outputs.run),   # last: reads CSVs
    ]

    for task_name, task_fn in TASKS:
        run_task(task_name, task_fn, results)

    print_summary(results, outputs_dir)
    return 0 if all(v["status"] == "OK" for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

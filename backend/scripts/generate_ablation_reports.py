"""Regenerate GSFS ablation comparison tables and plots directly from saved metrics JSON."""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
ABLATION_DIR = ROOT_DIR / "experiments" / "ablation"
METRICS_PATH = ABLATION_DIR / "ablation_metrics.json"
SUMMARY_PATH = ABLATION_DIR / "ablation_summary.md"
PLOT_PATH = ABLATION_DIR / "ablation_comparison_plot.png"


def regenerate_ablation_reports():
    if not METRICS_PATH.exists():
        print(f"Error: {METRICS_PATH} does not exist. Run scripts/run_ablation_experiments.py first.")
        sys.exit(1)

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    # 1. Summary Markdown Table
    md_lines = [
        "# GSFS Ablation Study Benchmark Table",
        "",
        "| ID | Configuration Description | Test Accuracy | Test Macro-F1 | Test ROC-AUC | Held-Out Macro-F1 | Accepted Updates | Rejected Updates | Rejection % | Aggregation Events | Theoretical Bytes Avoided |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for r in results:
        md_lines.append(
            f"| `{r['id']}` | **{r['name']}** | {r['accuracy']*100:.2f}% | {r['macro_f1']:.4f} | {r['roc_auc']:.4f} | {r['held_out_macro_f1']:.4f} | {r['accepted_updates']}/{r['total_updates']} | {r['rejected_updates']} | {r['rejection_percentage']}% | {r['aggregation_events']}/5 | {r['theoretical_bytes_avoided']:,} B |"
        )
    md_lines.append("")
    summary_md = "\n".join(md_lines)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"Generated: {SUMMARY_PATH}")

    # 2. Comparative Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    names = [r["name"].replace("GSFS ", "").replace(" (No GSFS)", "") for r in results]
    f1s = [r["macro_f1"] for r in results]
    rejs = [r["rejection_percentage"] for r in results]

    y_pos = np.arange(len(names))
    ax1.barh(y_pos, f1s, color="#F26522", alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Test Macro-F1 Score", fontweight="bold")
    ax1.set_xlim(0.85, 0.90)
    ax1.set_title("Detection Performance Across Ablations", fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="x")

    ax2.barh(y_pos, rejs, color="#3B82F6", alpha=0.85)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Client Update Rejection Rate (%)", fontweight="bold")
    ax2.set_xlim(0, 100)
    ax2.set_title("Update Rejection Impact Across Ablations", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    plt.close(fig)
    print(f"Generated: {PLOT_PATH}")


if __name__ == "__main__":
    regenerate_ablation_reports()

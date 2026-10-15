"""Standalone script to regenerate final experiment comparison tables, markdown reports, and summary figures

Directly reads stored experiment artifacts in experiments/final/:
- experiments/final/metrics.json
- experiments/final/gsfs_sensitivity.json

And regenerates:
- experiments/final/summary_table.md
- experiments/final/gsfs_sensitivity_plot.png
- Console summary
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
FINAL_DIR = ROOT_DIR / "experiments" / "final"
METRICS_PATH = FINAL_DIR / "metrics.json"
SENSITIVITY_PATH = FINAL_DIR / "gsfs_sensitivity.json"
SUMMARY_TABLE_PATH = FINAL_DIR / "summary_table.md"
SENSITIVITY_PLOT_PATH = FINAL_DIR / "gsfs_sensitivity_plot.png"


def regenerate_summary_table(metrics: dict) -> str:
    models = [
        ("logistic_regression", "Centralized Logistic Regression (Linear Baseline)"),
        ("random_forest", "Centralized Random Forest (Non-Neural Ensemble Baseline)"),
        ("centralized_nn", "Centralized PyTorch Neural Network (MLP Reference)"),
        ("fedavg", "Federated Learning - FedAvg (Multi-Seed Mean ± Std)"),
        ("fedopt", "Federated Learning - FedOpt / FedAdam (Multi-Seed Mean ± Std)"),
        ("gsfs", "Federated Learning - GSFS (Multi-Seed Mean ± Std)"),
    ]

    lines = [
        "# Final Experimental Benchmark Summary: Centralized vs. Federated Threat Detection",
        "",
        "## 1. Primary Evaluation Benchmark (Test Set: 4,800 Samples | In-Scenario Interpolation)",
        "",
        "| Architecture / Model | Accuracy | Macro-F1 | Weighted-F1 | Malicious Recall | Malicious Precision | ROC-AUC | PR-AUC | Inference Latency (ms) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for key, name in models:
        m_info = metrics.get(key, {})
        agg = m_info.get("multi_seed_aggregate")
        test_m = m_info.get("test", {})

        if agg:
            acc_str = f"{agg['accuracy_mean']*100:.2f}% ± {agg['accuracy_std']*100:.2f}%"
            f1_str = f"{agg['macro_f1_mean']:.4f} ± {agg['macro_f1_std']:.4f}"
            roc_str = f"{agg['roc_auc_mean']:.4f} ± {agg['roc_auc_std']:.4f}"
            pr_str = f"{agg['pr_auc_mean']:.4f} ± {agg['pr_auc_std']:.4f}"
            wf1_str = f"{test_m.get('weighted_f1', 0.0):.4f}"
            rec_str = f"{test_m.get('malicious_recall', 0.0)*100:.2f}%"
            prec_str = f"{test_m.get('malicious_precision', 0.0)*100:.2f}%"
            lat_str = f"{test_m.get('inference_latency_ms', 0.0):.4f}"
        else:
            acc_str = f"{test_m.get('accuracy', 0.0)*100:.2f}%"
            f1_str = f"{test_m.get('macro_f1', 0.0):.4f}"
            wf1_str = f"{test_m.get('weighted_f1', 0.0):.4f}"
            rec_str = f"{test_m.get('malicious_recall', 0.0)*100:.2f}%"
            prec_str = f"{test_m.get('malicious_precision', 0.0)*100:.2f}%"
            roc_str = f"{test_m.get('roc_auc', 0.0):.4f}"
            pr_str = f"{test_m.get('pr_auc', 0.0):.4f}"
            lat_str = f"{test_m.get('inference_latency_ms', 0.0):.4f}"

        lines.append(f"| **{name}** | {acc_str} | {f1_str} | {wf1_str} | {rec_str} | {prec_str} | {roc_str} | {pr_str} | {lat_str} |")

    lines.extend([
        "",
        "## 2. Cross-Scenario Generalization Benchmark (Scenario-Held-Out Split: 6,166 Samples)",
        "",
        "| Architecture / Model | Held-out Accuracy | Held-out Macro-F1 | Held-out Malicious Recall | Held-out Malicious Precision | Held-out ROC-AUC |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for key, name in models:
        m_info = metrics.get(key, {})
        agg = m_info.get("multi_seed_aggregate")
        held_m = m_info.get("held_out", {})

        if agg:
            acc_str = f"{agg['held_out_accuracy_mean']*100:.2f}% ± {agg['held_out_accuracy_std']*100:.2f}%"
            f1_str = f"{agg['held_out_macro_f1_mean']:.4f} ± {agg['held_out_macro_f1_std']:.4f}"
            rec_str = f"{held_m.get('malicious_recall', 0.0)*100:.2f}%"
            prec_str = f"{held_m.get('malicious_precision', 0.0)*100:.2f}%"
            roc_str = f"{held_m.get('roc_auc', 0.0):.4f}"
        else:
            acc_str = f"{held_m.get('accuracy', 0.0)*100:.2f}%"
            f1_str = f"{held_m.get('macro_f1', 0.0):.4f}"
            rec_str = f"{held_m.get('malicious_recall', 0.0)*100:.2f}%"
            prec_str = f"{held_m.get('malicious_precision', 0.0)*100:.2f}%"
            roc_str = f"{held_m.get('roc_auc', 0.0):.4f}"

        lines.append(f"| **{name}** | {acc_str} | {f1_str} | {rec_str} | {prec_str} | {roc_str} |")

    content = "\n".join(lines) + "\n"
    with open(SUMMARY_TABLE_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[Summary Table] Successfully updated: {SUMMARY_TABLE_PATH}")
    return content


def regenerate_sensitivity_plot(sensitivity: list):
    labels = [f"T={int(s['config']['thresh']*100)}%\n{s['config']['mode']}\nβ={s['config']['beta']}" for s in sensitivity]
    f1_vals = [s["macro_f1"] for s in sensitivity]
    rej_vals = [s["rejection_percentage"] for s in sensitivity]

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5), dpi=150)
    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax1.bar(x - width/2, f1_vals, width, label="Test Macro-F1", color="#F26522")
    ax1.set_ylabel("Macro-F1 Score", color="#F26522", fontweight="bold")
    ax1.set_ylim(0.8, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, rej_vals, width, label="Rejection Rate (%)", color="#3B82F6")
    ax2.set_ylabel("Update Rejection %", color="#3B82F6", fontweight="bold")
    ax2.set_ylim(0, 100)

    fig.suptitle("GSFS Client Selection Sensitivity: Threshold & Trigger Mode Impact", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(SENSITIVITY_PLOT_PATH)
    plt.close(fig)
    print(f"[Sensitivity Plot] Successfully generated: {SENSITIVITY_PLOT_PATH}")


def main():
    if not METRICS_PATH.exists():
        print(f"Error: {METRICS_PATH} does not exist. Run scripts/run_final_experiments.py first.")
        sys.exit(1)

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    regenerate_summary_table(metrics)

    if SENSITIVITY_PATH.exists():
        with open(SENSITIVITY_PATH, "r", encoding="utf-8") as f:
            sensitivity = json.load(f)
        regenerate_sensitivity_plot(sensitivity)


if __name__ == "__main__":
    main()

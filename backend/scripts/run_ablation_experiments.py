"""Master Ablation Studies Suite for GSFS in Federated IoT Intrusion Detection.

Evaluates 9 controlled configurations against the IoT-23 dataset:
1. FedAvg Baseline (No GSFS filtering)
2. GSFS Default (OR mode, perf=0.0, beta=0.0, thresh=0.60, pool=10, staleness=2)
3. GSFS No Performance Trigger (Gradient trigger only)
4. GSFS No Gradient Trigger (Performance trigger only)
5. GSFS Strict AND Mode (Both triggers must pass simultaneously)
6. GSFS Small Update Pool (Capacity=5 instead of 10)
7. GSFS Strict Staleness (Staleness=1 instead of 2)
8. GSFS High Quorum Threshold (80% client quorum required)
9. GSFS Low Quorum Threshold (40% client quorum required)

Outputs:
- experiments/ablation/ablation_metrics.json
- experiments/ablation/ablation_summary.md
- experiments/ablation/ablation_comparison_plot.png
- experiments/ablation/confusion_matrices/cm_ablation_*.png
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.detection.data_loader import DEFAULT_FEATURE_COLUMNS, LABEL_MAPPING
from src.detection.pipeline import build_preprocessor
from src.federated.config import FederatedConfig, get_gsfs_config
from src.federated.model import IoTThreatClassifier, get_model_parameters, estimate_parameter_size_bytes
from src.federated.client import IoTFlowerClient
from src.federated.server import FederatedServerCoordinator
from src.federated.partitioner import partition_train_data

OUTPUT_DIR = ROOT_DIR / "experiments" / "ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "confusion_matrices").mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))

    return {
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_wt),
        "benign_recall": float(r_cls[0]),
        "benign_precision": float(p_cls[0]),
        "malicious_recall": float(r_cls[1]),
        "malicious_precision": float(p_cls[1]),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
    }


def plot_cm(cm: List[List[int]], title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(4, 3.5), dpi=120)
    cm_arr = np.array(cm)
    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    classes = ["Benign", "Malicious"]
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    thresh = cm_arr.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_arr[i, j]:,}", ha="center", va="center",
                    color="white" if cm_arr[i, j] > thresh else "black", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def run_ablation_suite():
    print("=" * 80)
    print("STARTING GSFS ABLATION STUDIES EXPERIMENT SUITE")
    print("=" * 80)

    # 1. Load Data
    train_path = ROOT_DIR / "processed" / "train.csv"
    val_path = ROOT_DIR / "processed" / "val.csv"
    test_path = ROOT_DIR / "processed" / "test.csv"

    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df[DEFAULT_FEATURE_COLUMNS])
    X_val = preprocessor.transform(val_df[DEFAULT_FEATURE_COLUMNS])
    X_test = preprocessor.transform(test_df[DEFAULT_FEATURE_COLUMNS])

    y_val = val_df["label"].str.lower().str.strip().map(LABEL_MAPPING).to_numpy()
    y_test = test_df["label"].str.lower().str.strip().map(LABEL_MAPPING).to_numpy()

    # Scenario-held-out split (unseen scenarios)
    held_out_scenarios = ["CTU-Honeypot-Capture-5-1", "CTU-IoT-Malware-Capture-34-1", "CTU-IoT-Malware-Capture-3-1"]
    held_df = pd.concat([train_df, val_df, test_df])
    held_df = held_df[held_df["scenario"].isin(held_out_scenarios)].copy()
    X_held = preprocessor.transform(held_df[DEFAULT_FEATURE_COLUMNS])
    y_held = held_df["label"].str.lower().str.strip().map(LABEL_MAPPING).to_numpy()

    input_dim = X_train.shape[1]
    param_size_bytes = estimate_parameter_size_bytes(get_model_parameters(IoTThreatClassifier(input_dim=input_dim)))

    # Partition clients
    client_dfs = partition_train_data(train_df, num_clients=5, strategy="scenario_grouped", random_seed=42)

    # 2. Define Ablation Matrix
    ablation_matrix = [
        {
            "id": "1_fedavg_baseline",
            "name": "FedAvg Baseline (No GSFS)",
            "type": "fedavg",
            "cfg": FederatedConfig(random_seed=42, num_rounds=5),
        },
        {
            "id": "2_gsfs_default_or",
            "name": "GSFS Default (OR Mode)",
            "type": "gsfs",
            "cfg": get_gsfs_config(
                aggregation_threshold=0.60, perf_threshold=0.0, grad_beta=0.0,
                trigger_mode="OR", max_pool_size=10, max_staleness=2, warmup_rounds=1,
                random_seed=42, num_rounds=5,
            ),
        },
        {
            "id": "3_gsfs_no_perf_trigger",
            "name": "GSFS No Performance Trigger (Gradient Only)",
            "type": "gsfs",
            "cfg": get_gsfs_config(
                aggregation_threshold=0.60, perf_threshold=-999.0, grad_beta=0.0,
                trigger_mode="AND", max_pool_size=10, max_staleness=2, warmup_rounds=1,
                random_seed=42, num_rounds=5,
            ),
        },
        {
            "id": "4_gsfs_no_grad_trigger",
            "name": "GSFS No Gradient Trigger (Performance Only)",
            "type": "gsfs",
            "cfg": get_gsfs_config(
                aggregation_threshold=0.60, perf_threshold=0.0, grad_beta=999.0,
                trigger_mode="OR", max_pool_size=10, max_staleness=2, warmup_rounds=1,
                random_seed=42, num_rounds=5,
            ),
        },
        {
            "id": "5_gsfs_strict_and_mode",
            "name": "GSFS Strict AND Mode (Both Triggers Required)",
            "type": "gsfs",
            "cfg": get_gsfs_config(
                aggregation_threshold=0.60, perf_threshold=0.0, grad_beta=0.0,
                trigger_mode="AND", max_pool_size=10, max_staleness=2, warmup_rounds=1,
                random_seed=42, num_rounds=5,
            ),
        },
        {
            "id": "6_gsfs_small_pool",
            "name": "GSFS Small Update Pool (Capacity=5)",
            "type": "gsfs",
            "cfg": get_gsfs_config(
                aggregation_threshold=0.60, perf_threshold=0.0, grad_beta=0.0,
                trigger_mode="OR", max_pool_size=5, max_staleness=2, warmup_rounds=1,
                random_seed=42, num_rounds=5,
            ),
        },
        {
            "id": "7_gsfs_strict_staleness",
            "name": "GSFS Strict Staleness (Staleness=1)",
            "type": "gsfs",
            "cfg": get_gsfs_config(
                aggregation_threshold=0.60, perf_threshold=0.0, grad_beta=0.0,
                trigger_mode="OR", max_pool_size=10, max_staleness=1, warmup_rounds=1,
                random_seed=42, num_rounds=5,
            ),
        },
        {
            "id": "8_gsfs_high_quorum",
            "name": "GSFS High Quorum (80% Clients Required)",
            "type": "gsfs",
            "cfg": get_gsfs_config(
                aggregation_threshold=0.80, perf_threshold=0.0, grad_beta=0.0,
                trigger_mode="OR", max_pool_size=10, max_staleness=2, warmup_rounds=1,
                random_seed=42, num_rounds=5,
            ),
        },
        {
            "id": "9_gsfs_low_quorum",
            "name": "GSFS Low Quorum (40% Clients Required)",
            "type": "gsfs",
            "cfg": get_gsfs_config(
                aggregation_threshold=0.40, perf_threshold=0.0, grad_beta=0.0,
                trigger_mode="OR", max_pool_size=10, max_staleness=2, warmup_rounds=1,
                random_seed=42, num_rounds=5,
            ),
        },
    ]

    results = []

    for item in ablation_matrix:
        a_id = item["id"]
        a_name = item["name"]
        a_type = item["type"]
        cfg = item["cfg"]

        print(f"\n--- Running Ablation Experiment [{a_id}]: {a_name} ---")

        # Create fresh clients
        f_clients = {}
        for cid in range(5):
            c_df = client_dfs[cid]
            c_X = preprocessor.transform(c_df[DEFAULT_FEATURE_COLUMNS])
            c_y = c_df["label"].str.lower().str.strip().map(LABEL_MAPPING).to_numpy()
            c_model = IoTThreatClassifier(input_dim=input_dim)
            f_clients[cid] = IoTFlowerClient(cid, c_model, c_X, c_y, local_epochs=2, batch_size=64, learning_rate=0.001)

        coordinator = FederatedServerCoordinator(cfg, f_clients, X_val, y_val, input_dim=input_dim)
        t_start = time.time()
        best_model, hist, comm = coordinator.run_federated_rounds()
        duration = round(time.time() - t_start, 2)

        best_model.eval()
        with torch.no_grad():
            test_probs = torch.softmax(best_model(torch.tensor(X_test, dtype=torch.float32)), dim=1)[:, 1].numpy()
            held_probs = torch.softmax(best_model(torch.tensor(X_held, dtype=torch.float32)), dim=1)[:, 1].numpy()

        test_m = compute_metrics(y_test, test_probs, threshold=0.50)
        held_m = compute_metrics(y_held, held_probs, threshold=0.50)

        # Plot confusion matrix
        plot_cm(test_m["confusion_matrix"], f"{a_name}", OUTPUT_DIR / "confusion_matrices" / f"cm_ablation_{a_id}.png")

        total_opportunities = 5 * 5  # 5 rounds * 5 clients = 25 opportunities
        if a_type == "gsfs":
            strat_hist = getattr(coordinator.strategy, "round_history", [])
            accepted = sum(r.get("updates_selected", 0) for r in strat_hist)
            rejected = sum(r.get("updates_rejected", 0) for r in strat_hist)
            agg_events = sum(1 for r in strat_hist if r.get("aggregated_this_round", False))
            rejections_pct = round((rejected / total_opportunities) * 100, 1)
            avoided_bytes = rejected * param_size_bytes
        else:
            accepted = total_opportunities
            rejected = 0
            agg_events = 5
            rejections_pct = 0.0
            avoided_bytes = 0

        res_entry = {
            "id": a_id,
            "name": a_name,
            "type": a_type,
            "duration_seconds": duration,
            "accuracy": test_m["accuracy"],
            "macro_f1": test_m["macro_f1"],
            "roc_auc": test_m["roc_auc"],
            "pr_auc": test_m["pr_auc"],
            "held_out_accuracy": held_m["accuracy"],
            "held_out_macro_f1": held_m["macro_f1"],
            "total_updates": total_opportunities,
            "accepted_updates": accepted,
            "rejected_updates": rejected,
            "rejection_percentage": rejections_pct,
            "aggregation_events": agg_events,
            "rounds_without_aggregation": 5 - agg_events,
            "theoretical_bytes_avoided": avoided_bytes,
            "test_metrics": test_m,
            "held_out_metrics": held_m,
        }
        results.append(res_entry)
        print(f"  -> Acc: {test_m['accuracy']*100:.2f}% | Macro-F1: {test_m['macro_f1']:.4f} | ROC-AUC: {test_m['roc_auc']:.4f} | Rejected: {rejections_pct}% | Aggregations: {agg_events}/5")

    # 3. Save JSON Metrics
    with open(OUTPUT_DIR / "ablation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 4. Generate Summary Markdown Table
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
    with open(OUTPUT_DIR / "ablation_summary.md", "w", encoding="utf-8") as f:
        f.write(summary_md)

    # 5. Generate Comparative Plot
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
    fig.savefig(OUTPUT_DIR / "ablation_comparison_plot.png")
    plt.close(fig)

    print("\nSUCCESS: All 9 GSFS ablation studies completed and saved to experiments/ablation/")


if __name__ == "__main__":
    run_ablation_suite()

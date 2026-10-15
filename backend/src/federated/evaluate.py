"""Evaluation utilities for PyTorch models and federated rounds."""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.detection.evaluate import compute_metrics, evaluate_per_scenario, plot_confusion_matrix


def evaluate_torch_model(
    model: nn.Module,
    X_data: Union[np.ndarray, torch.Tensor],
    y_data: Union[np.ndarray, torch.Tensor],
    batch_size: int = 256,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evaluate a PyTorch model on a dataset and return standard metrics.
    """
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    if isinstance(X_data, np.ndarray):
        X_tensor = torch.tensor(X_data, dtype=torch.float32)
    else:
        X_tensor = X_data.float()

    if isinstance(y_data, np.ndarray):
        y_tensor = torch.tensor(y_data, dtype=torch.long)
    else:
        y_tensor = y_data.long()

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_probs = []
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(dev)
            batch_y = batch_y.to(dev)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)

            probs = torch.softmax(logits, dim=-1)[:, 1]  # P(malicious)
            preds = torch.argmax(logits, dim=-1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    num_total = len(dataset)
    avg_loss = float(total_loss / max(1, num_total))
    y_true_np = np.array(all_targets)
    y_pred_np = np.array(all_preds)
    y_prob_np = np.array(all_probs)

    metrics = compute_metrics(y_true_np, y_pred_np, y_prob_np)
    metrics["loss"] = avg_loss
    return metrics


def plot_federated_round_metrics(
    round_history: List[Dict[str, Any]],
    save_path: Union[str, Path],
) -> None:
    """Plot validation loss and Macro-F1 across federated rounds."""
    df = pd.DataFrame(round_history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Plot 1: Validation Loss
    ax1.plot(df["round"], df["val_loss"], marker="o", color="#d62728", lw=2, label="Validation Loss")
    ax1.set_xlabel("Federated Round", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.set_title("Global Model Validation Loss vs. Round", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Plot 2: Macro-F1 & Accuracy
    ax2.plot(df["round"], df["val_macro_f1"], marker="s", color="#1f77b4", lw=2, label="Validation Macro-F1")
    ax2.plot(df["round"], df["val_accuracy"], marker="^", color="#2ca02c", lw=2, linestyle="--", label="Validation Accuracy")
    ax2.set_xlabel("Federated Round", fontsize=11)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("Global Model Validation Accuracy & Macro-F1", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_comparison(
    fedavg_rounds: Union[List[Dict[str, Any]], pd.DataFrame, str, Path],
    fedopt_rounds: Union[List[Dict[str, Any]], pd.DataFrame, str, Path],
    save_path: Union[str, Path],
    opt_name: str = "FedAdam",
) -> None:
    """
    Plot side-by-side comparison of Validation Loss and Macro-F1 curves between FedAvg and FedOpt.
    """
    if isinstance(fedavg_rounds, (str, Path)):
        df_avg = pd.read_csv(fedavg_rounds)
    elif isinstance(fedavg_rounds, list):
        df_avg = pd.DataFrame(fedavg_rounds)
    else:
        df_avg = fedavg_rounds

    if isinstance(fedopt_rounds, (str, Path)):
        df_opt = pd.read_csv(fedopt_rounds)
    elif isinstance(fedopt_rounds, list):
        df_opt = pd.DataFrame(fedopt_rounds)
    else:
        df_opt = fedopt_rounds

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # 1. Validation Loss Convergence
    ax1.plot(df_avg["round"], df_avg["val_loss"], marker="o", color="#1f77b4", lw=2, label="FedAvg")
    ax1.plot(df_opt["round"], df_opt["val_loss"], marker="s", color="#d62728", lw=2, linestyle="--", label=f"FedOpt ({opt_name})")
    ax1.set_xlabel("Federated Round", fontsize=11)
    ax1.set_ylabel("Validation Loss (Cross-Entropy)", fontsize=11)
    ax1.set_title("Validation Loss Convergence: FedAvg vs. FedOpt", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Validation Macro-F1 Convergence
    ax2.plot(df_avg["round"], df_avg["val_macro_f1"], marker="o", color="#1f77b4", lw=2, label="FedAvg")
    ax2.plot(df_opt["round"], df_opt["val_macro_f1"], marker="s", color="#2ca02c", lw=2, linestyle="--", label=f"FedOpt ({opt_name})")
    ax2.set_xlabel("Federated Round", fontsize=11)
    ax2.set_ylabel("Validation Macro-F1", fontsize=11)
    ax2.set_title("Validation Macro-F1 Convergence: FedAvg vs. FedOpt", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, bbox_inches="tight")
    plt.close(fig)


def plot_4way_benchmark_comparison(
    metrics_summary: Dict[str, Dict[str, float]],
    save_path: Union[str, Path],
) -> None:
    """
    Plot grouped bar chart comparing multiple methods (Centralized RF, NN, FedAvg, FedOpt, etc.).
    """
    plot_5way_benchmark_comparison(metrics_summary, save_path)


def plot_5way_benchmark_comparison(
    metrics_summary: Dict[str, Dict[str, float]],
    save_path: Union[str, Path],
) -> None:
    """
    Plot grouped bar chart comparing up to 5 methods:
    Centralized RF, Centralized NN, FedAvg, FedOpt, and GSFS.
    """
    methods = list(metrics_summary.keys())
    metric_keys = ["accuracy", "macro_f1", "malicious_recall", "malicious_precision", "roc_auc"]
    metric_labels = ["Accuracy", "Macro-F1", "Malicious Recall", "Malicious Precision", "ROC-AUC"]

    x = np.arange(len(metric_labels))
    n_methods = len(methods)
    width = 0.8 / max(1, n_methods)
    palette = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#1b9e77", "#66a61e"]

    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=150)

    for i, method_name in enumerate(methods):
        values = [metrics_summary[method_name].get(k, 0.0) * 100 for k in metric_keys]
        offset = (i - (n_methods - 1) / 2) * width
        rects = ax.bar(
            x + offset,
            values,
            width,
            label=method_name,
            color=palette[i % len(palette)],
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
        )
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    rotation=0,
                )

    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Comprehensive Benchmark Comparison (IoT-23 Test Set)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, bbox_inches="tight")
    plt.close(fig)


def plot_gsfs_selection_dynamics(
    round_history: Union[List[Dict[str, Any]], pd.DataFrame, str, Path],
    save_path: Union[str, Path],
) -> None:
    """
    Plot per-round GSFS client selection, pool pending count, and gradient threshold dynamics.
    """
    if isinstance(round_history, (str, Path)):
        df = pd.read_csv(round_history)
    elif isinstance(round_history, list):
        df = pd.DataFrame(round_history)
    else:
        df = round_history

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # 1. Update Lifecycle Counts
    rounds = df["round"]
    ax1.plot(rounds, df.get("updates_generated", [5]*len(df)), marker="o", color="#1f77b4", lw=2, label="Generated")
    ax1.plot(rounds, df.get("updates_selected", [0]*len(df)), marker="s", color="#2ca02c", lw=2, label="Selected")
    ax1.plot(rounds, df.get("updates_aggregated", [0]*len(df)), marker="^", color="#d62728", lw=2, linestyle="--", label="Aggregated")
    ax1.plot(rounds, df.get("updates_pending_in_pool", [0]*len(df)), marker="d", color="#ff7f0e", lw=1.5, linestyle=":", label="Pending in Pool")

    ax1.set_xlabel("Federated Round", fontsize=11)
    ax1.set_ylabel("Client Update Count", fontsize=11)
    ax1.set_title("GSFS Client Selection & Update Pool Lifecycle", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Gradient Statistics & Threshold Evolution
    if "grad_threshold" in df.columns:
        ax2.plot(rounds, df["moving_grad_mean"], marker="o", color="#7570b3", lw=2, label="Moving Mean")
        ax2.plot(rounds, df["grad_threshold"], marker="s", color="#e7298a", lw=2, linestyle="--", label="Active Threshold")
        if "moving_grad_std" in df.columns:
            mean = df["moving_grad_mean"]
            std = df["moving_grad_std"]
            ax2.fill_between(rounds, mean - std, mean + std, color="#7570b3", alpha=0.15, label="±1 Std Dev")

        ax2.set_xlabel("Federated Round", fontsize=11)
        ax2.set_ylabel("Parameter Displacement Norm (L2)", fontsize=11)
        ax2.set_title("Fleet Gradient Statistics & Threshold Evolution", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, bbox_inches="tight")
    plt.close(fig)


def plot_3way_convergence_comparison(
    fedavg_rounds: Union[List[Dict[str, Any]], pd.DataFrame, str, Path],
    fedopt_rounds: Union[List[Dict[str, Any]], pd.DataFrame, str, Path],
    gsfs_rounds: Union[List[Dict[str, Any]], pd.DataFrame, str, Path],
    save_path: Union[str, Path],
) -> None:
    """
    Plot side-by-side comparison of Validation Loss and Macro-F1 across FedAvg, FedOpt, and GSFS.
    """
    def _to_df(r):
        if isinstance(r, (str, Path)):
            return pd.read_csv(r)
        elif isinstance(r, list):
            return pd.DataFrame(r)
        return r

    df_avg = _to_df(fedavg_rounds)
    df_opt = _to_df(fedopt_rounds)
    df_gsfs = _to_df(gsfs_rounds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # 1. Validation Loss Convergence
    if not df_avg.empty and "val_loss" in df_avg.columns:
        ax1.plot(df_avg["round"], df_avg["val_loss"], marker="o", color="#1f77b4", lw=2, label="FedAvg")
    if not df_opt.empty and "val_loss" in df_opt.columns:
        ax1.plot(df_opt["round"], df_opt["val_loss"], marker="s", color="#d62728", lw=2, linestyle="--", label="FedOpt (FedAdam)")
    if not df_gsfs.empty and "val_loss" in df_gsfs.columns:
        ax1.plot(df_gsfs["round"], df_gsfs["val_loss"], marker="^", color="#2ca02c", lw=2, linestyle="-.", label="GSFS")

    ax1.set_xlabel("Federated Round", fontsize=11)
    ax1.set_ylabel("Validation Loss (Cross-Entropy)", fontsize=11)
    ax1.set_title("Validation Loss: FedAvg vs. FedOpt vs. GSFS", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # 2. Validation Macro-F1 Convergence
    if not df_avg.empty and "val_macro_f1" in df_avg.columns:
        ax2.plot(df_avg["round"], df_avg["val_macro_f1"], marker="o", color="#1f77b4", lw=2, label="FedAvg")
    if not df_opt.empty and "val_macro_f1" in df_opt.columns:
        ax2.plot(df_opt["round"], df_opt["val_macro_f1"], marker="s", color="#d62728", lw=2, linestyle="--", label="FedOpt (FedAdam)")
    if not df_gsfs.empty and "val_macro_f1" in df_gsfs.columns:
        ax2.plot(df_gsfs["round"], df_gsfs["val_macro_f1"], marker="^", color="#2ca02c", lw=2, linestyle="-.", label="GSFS")

    ax2.set_xlabel("Federated Round", fontsize=11)
    ax2.set_ylabel("Validation Macro-F1", fontsize=11)
    ax2.set_title("Validation Macro-F1: FedAvg vs. FedOpt vs. GSFS", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, bbox_inches="tight")
    plt.close(fig)


"""Comprehensive evaluation suite for threat detection models.

Features:
- Multi-metric calculation: Accuracy, per-class Precision/Recall/F1, Macro-F1, Weighted-F1.
- ROC-AUC and PR-AUC computation for probabilistic classifiers.
- Confusion matrix extraction and high-resolution visualization.
- Threshold search strictly on validation set.
- Per-scenario breakdown analysis.
- Scenario-held-out evaluation to empirically measure cross-scenario generalization vs random-split interpolation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


def compute_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_prob: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Class 0 = Benign
    Class 1 = Malicious
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class metrics
    prec_benign = float(precision_score(y_true, y_pred, pos_label=0, zero_division=0))
    rec_benign = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    f1_benign = float(f1_score(y_true, y_pred, pos_label=0, zero_division=0))

    prec_mal = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_mal = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_mal = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "benign": {
            "precision": prec_benign,
            "recall": rec_benign,
            "f1_score": f1_benign,
            "support": int((y_true == 0).sum()),
        },
        "malicious": {
            "precision": prec_mal,
            "recall": rec_mal,
            "f1_score": f1_mal,
            "support": int((y_true == 1).sum()),
        },
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = None

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["pr_auc"] = None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_metric: str = "macro_f1",
) -> Tuple[float, Dict[str, Any]]:
    """
    Find optimal classification probability threshold using validation data.

    Searches thresholds in [0.10, 0.90] to balance precision and recall.
    """
    thresholds = np.linspace(0.10, 0.90, 81)
    best_th = 0.50
    best_score = -1.0
    best_metrics: Dict[str, Any] = {}

    for th in thresholds:
        preds = (y_prob >= th).astype(int)
        m = compute_metrics(y_true, preds, y_prob)

        if target_metric == "macro_f1":
            score = m["macro_f1"]
        elif target_metric == "f1_malicious":
            score = m["malicious"]["f1_score"]
        else:
            score = m["accuracy"]

        if score > best_score:
            best_score = score
            best_th = float(th)
            best_metrics = m

    return best_th, best_metrics


def evaluate_per_scenario(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    scenarios: Union[np.ndarray, pd.Series],
) -> Dict[str, Any]:
    """Compute per-scenario performance metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scenarios = np.asarray(scenarios)

    unique_scenarios = sorted(np.unique(scenarios))
    results = {}

    for sc in unique_scenarios:
        mask = scenarios == sc
        sc_true = y_true[mask]
        sc_pred = y_pred[mask]
        n_samples = int(mask.sum())
        n_mal = int((sc_true == 1).sum())
        n_ben = int((sc_true == 0).sum())

        acc = float(accuracy_score(sc_true, sc_pred))

        # Precision/recall if classes exist
        rec_mal = float(recall_score(sc_true, sc_pred, pos_label=1, zero_division=0)) if n_mal > 0 else None
        prec_mal = float(precision_score(sc_true, sc_pred, pos_label=1, zero_division=0)) if n_mal > 0 else None
        f1_mal = float(f1_score(sc_true, sc_pred, pos_label=1, zero_division=0)) if n_mal > 0 else None

        results[sc] = {
            "samples": n_samples,
            "benign_count": n_ben,
            "malicious_count": n_mal,
            "accuracy": round(acc, 4),
            "malicious_recall": round(rec_mal, 4) if rec_mal is not None else "N/A",
            "malicious_precision": round(prec_mal, 4) if prec_mal is not None else "N/A",
            "malicious_f1": round(f1_mal, 4) if f1_mal is not None else "N/A",
        }

    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Union[str, Path],
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.85)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    fig.colorbar(cax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=13, pad=15, fontweight="bold")
    plt.tight_layout()

    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, bbox_inches="tight")
    plt.close(fig)


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Union[str, Path],
    model_name: str = "Model",
) -> None:
    """Plot ROC curve and Precision-Recall curve side by side."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # ROC Curve
    ax1.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=11)
    ax1.set_title(f"ROC Curve - {model_name}", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # PR Curve
    baseline_pr = (y_true == 1).mean()
    ax2.plot(rec, prec, color="#ff7f0e", lw=2, label=f"PR (AUC = {pr_auc:.4f})")
    ax2.axhline(y=baseline_pr, color="gray", linestyle="--", lw=1.5, label=f"Baseline ({baseline_pr:.2f})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall", fontsize=11)
    ax2.set_ylabel("Precision", fontsize=11)
    ax2.set_title(f"Precision-Recall Curve - {model_name}", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_p = Path(save_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_p, bbox_inches="tight")
    plt.close(fig)

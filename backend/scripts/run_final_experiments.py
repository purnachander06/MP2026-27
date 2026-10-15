"""Comprehensive Final Research Experiments Pipeline (Phase 9 Part D).

Evaluates:
1. Centralized Logistic Regression (Linear baseline with StandardScaler)
2. Centralized Random Forest (Ensemble baseline, 100 trees, threshold 0.60)
3. Centralized PyTorch Neural Network (2-layer MLP 125->64->32->2)
4. Federated FedAvg (10 rounds, 5 Non-IID clients, 2 local epochs)
5. Federated FedOpt / FedAdam (10 rounds, server momentum)
6. Federated GSFS (10 rounds, Update Pool, Gradient-Selective triggers)

Across Two Evaluation Regimes:
- Regime 1: Random Benchmark Split (In-Scenario Interpolation, 4,800 test samples)
- Regime 2: Scenario-Held-Out Split (Generalization to Unseen Capture Scenarios)

Also executes:
- Multi-seed trials (seeds 42, 101, 2024) with Mean +/- Std reporting
- GSFS Trigger Sensitivity Grid (Thresholds: 40%, 60%, 80%; OR vs AND; Beta: -0.5, 0.0, 0.5)
- Communication overhead measurement (Transmitted upload/download vs Theoretical avoided)
- Confusion matrix and ROC / Precision-Recall curve visualizations
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

# Project root
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.detection.data_loader import (
    load_dataset_split,
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_NUMERIC_FEATURES,
    DEFAULT_CATEGORICAL_FEATURES,
    LABEL_MAPPING,
)
from src.detection.pipeline import build_preprocessor
from src.federated.config import FederatedConfig, get_fedopt_config, get_gsfs_config
from src.federated.partitioner import partition_train_data
from src.federated.model import (
    IoTThreatClassifier,
    get_model_parameters,
    set_model_parameters,
    estimate_parameter_size_bytes,
)
from src.federated.client import IoTFlowerClient
from src.federated.server import FederatedServerCoordinator
from src.federated.evaluate import evaluate_torch_model


OUTPUT_DIR = ROOT_DIR / "experiments" / "final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "confusion_matrices").mkdir(parents=True, exist_ok=True)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute complete classification and probability calibration metrics."""
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    acc = float(accuracy_score(y_true, preds))
    macro_p = float(precision_score(y_true, preds, average="macro", zero_division=0))
    macro_r = float(recall_score(y_true, preds, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))

    weighted_p = float(precision_score(y_true, preds, average="weighted", zero_division=0))
    weighted_r = float(recall_score(y_true, preds, average="weighted", zero_division=0))
    weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))

    mal_p = float(precision_score(y_true, preds, pos_label=1, zero_division=0))
    mal_r = float(recall_score(y_true, preds, pos_label=1, zero_division=0))
    mal_f1 = float(f1_score(y_true, preds, pos_label=1, zero_division=0))

    ben_p = float(precision_score(y_true, preds, pos_label=0, zero_division=0))
    ben_r = float(recall_score(y_true, preds, pos_label=0, zero_division=0))
    ben_f1 = float(f1_score(y_true, preds, pos_label=0, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, probs))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, probs))
    except Exception:
        pr_auc = 0.0

    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "malicious": {
            "precision": mal_p,
            "recall": mal_r,
            "f1": mal_f1,
            "support": int(np.sum(y_true == 1)),
        },
        "benign": {
            "precision": ben_p,
            "recall": ben_r,
            "f1": ben_f1,
            "support": int(np.sum(y_true == 0)),
        },
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "total": int(len(y_true)),
        },
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def measure_inference_latency(predict_fn, sample_data, num_runs=50) -> float:
    """Measure inference latency per sample in milliseconds."""
    # Warmup
    for _ in range(5):
        _ = predict_fn(sample_data[:50])

    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = predict_fn(sample_data[:100])
    t1 = time.perf_counter()
    return round(((t1 - t0) / (num_runs * 100)) * 1000, 4)


def plot_confusion_matrix_heatmap(cm_dict: Dict[str, int], title: str, save_path: Path):
    """Save clean annotated confusion matrix image."""
    cm = np.array([
        [cm_dict["tn"], cm_dict["fp"]],
        [cm_dict["fn"], cm_dict["tp"]]
    ])
    fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    classes = ["Benign (0)", "Malicious (1)"]
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="Ground Truth",
        xlabel="Predicted Label",
    )
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def run_experiment_suite():
    print("=" * 80)
    print("STARTING PHASE 9 COMPREHENSIVE RESEARCH EXPERIMENTS SUITE")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1] Loading IoT-23 Processed Dataset Splits...")
    train_path = ROOT_DIR / "processed" / "train.csv"
    val_path = ROOT_DIR / "processed" / "val.csv"
    test_path = ROOT_DIR / "processed" / "test.csv"
    meta_path = ROOT_DIR / "processed" / "feature_metadata.json"

    raw_train_df = pd.read_csv(train_path)
    X_train_raw, y_train, meta_train = load_dataset_split(train_path, meta_path)
    X_val_raw, y_val, meta_val = load_dataset_split(val_path, meta_path)
    X_test_raw, y_test, meta_test = load_dataset_split(test_path, meta_path)

    y_train_np = y_train.to_numpy()
    y_val_np = y_val.to_numpy()
    y_test_np = y_test.to_numpy()

    print(f"  Train: {len(X_train_raw):,} samples (Benign: {np.sum(y_train_np==0):,}, Malicious: {np.sum(y_train_np==1):,})")
    print(f"  Val:   {len(X_val_raw):,} samples (Benign: {np.sum(y_val_np==0):,}, Malicious: {np.sum(y_val_np==1):,})")
    print(f"  Test:  {len(X_test_raw):,} samples (Benign: {np.sum(y_test_np==0):,}, Malicious: {np.sum(y_test_np==1):,})")

    # Fit Preprocessor exclusively on Train
    preprocessor = build_preprocessor(
        numeric_features=DEFAULT_NUMERIC_FEATURES,
        categorical_features=DEFAULT_CATEGORICAL_FEATURES,
        scale_numeric=True,
    )
    preprocessor.fit(X_train_raw)

    X_train_all = preprocessor.transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)
    input_dim = X_train_all.shape[1]
    print(f"  Preprocessed Feature Dimension: {input_dim} (StandardScaler + OneHot)")

    # Prepare Scenario-Held-Out Split
    print("\n[Step 2] Preparing Scenario-Held-Out Generalization Split...")
    held_out_scenarios = [
        "CTU-Honeypot-Capture-5-1",
        "CTU-IoT-Malware-Capture-34-1",
        "CTU-IoT-Malware-Capture-3-1",
    ]
    # Check scenario column in test or combined sampled dataset
    sampled_path = ROOT_DIR / "processed" / "iot23_sampled.csv"
    sampled_df = pd.read_csv(sampled_path)
    held_out_df = sampled_df[sampled_df["scenario"].isin(held_out_scenarios)]
    X_held_out_raw = held_out_df[DEFAULT_FEATURE_COLUMNS]
    y_held_out_np = held_out_df["label"].str.lower().str.strip().map(LABEL_MAPPING).to_numpy()
    X_held_out = preprocessor.transform(X_held_out_raw)
    print(f"  Held-out Scenarios: {held_out_scenarios}")
    print(f"  Held-out Samples: {len(held_out_df):,} (Benign: {np.sum(y_held_out_np==0):,}, Malicious: {np.sum(y_held_out_np==1):,})")

    all_model_results = {}
    curves_data = {}

    # -------------------------------------------------------------------------
    # Model 1: Centralized Logistic Regression
    # -------------------------------------------------------------------------
    print("\n[Step 3] Evaluating Model 1: Centralized Logistic Regression...")
    lr_pipe = joblib.load(ROOT_DIR / "models" / "baseline_lr_pipeline.joblib")
    t0 = time.time()
    lr_test_probs = lr_pipe.predict_proba(X_test_raw)[:, 1]
    lr_held_probs = lr_pipe.predict_proba(X_held_out_raw)[:, 1]
    lr_latency = measure_inference_latency(lambda d: lr_pipe.predict_proba(d), X_test_raw)

    lr_test_m = compute_comprehensive_metrics(y_test_np, lr_test_probs, threshold=0.50)
    lr_held_m = compute_comprehensive_metrics(y_held_out_np, lr_held_probs, threshold=0.50)
    lr_test_m["inference_latency_ms"] = lr_latency
    lr_test_m["model_size_bytes"] = (ROOT_DIR / "models" / "baseline_lr_pipeline.joblib").stat().st_size
    lr_test_m["train_time_seconds"] = 1.85  # Documented training time

    all_model_results["logistic_regression"] = {
        "display_name": "Centralized Logistic Regression",
        "family": "centralized_linear",
        "random_benchmark": lr_test_m,
        "scenario_held_out": lr_held_m,
    }
    curves_data["logistic_regression"] = (y_test_np, lr_test_probs, "Logistic Regression")
    plot_confusion_matrix_heatmap(
        lr_test_m["confusion_matrix"],
        "Logistic Regression (Test Set)",
        OUTPUT_DIR / "confusion_matrices" / "cm_logistic_regression.png",
    )
    print(f"  Acc: {lr_test_m['accuracy']*100:.2f}% | Macro-F1: {lr_test_m['macro_f1']:.4f} | ROC-AUC: {lr_test_m['roc_auc']:.4f}")

    # -------------------------------------------------------------------------
    # Model 2: Centralized Random Forest
    # -------------------------------------------------------------------------
    print("\n[Step 4] Evaluating Model 2: Centralized Random Forest...")
    rf_pipe = joblib.load(ROOT_DIR / "models" / "best_model_pipeline.joblib")
    rf_test_probs = rf_pipe.predict_proba(X_test_raw)[:, 1]
    rf_held_probs = rf_pipe.predict_proba(X_held_out_raw)[:, 1]
    rf_latency = measure_inference_latency(lambda d: rf_pipe.predict_proba(d), X_test_raw)

    # Threshold tuned on validation set in Phase 2 = 0.60
    rf_test_m = compute_comprehensive_metrics(y_test_np, rf_test_probs, threshold=0.60)
    rf_held_m = compute_comprehensive_metrics(y_held_out_np, rf_held_probs, threshold=0.60)
    rf_test_m["inference_latency_ms"] = rf_latency
    rf_test_m["model_size_bytes"] = (ROOT_DIR / "models" / "best_model_pipeline.joblib").stat().st_size
    rf_test_m["train_time_seconds"] = 3.65

    all_model_results["random_forest"] = {
        "display_name": "Centralized Random Forest",
        "family": "centralized_ensemble",
        "random_benchmark": rf_test_m,
        "scenario_held_out": rf_held_m,
    }
    curves_data["random_forest"] = (y_test_np, rf_test_probs, "Random Forest (tau=0.60)")
    plot_confusion_matrix_heatmap(
        rf_test_m["confusion_matrix"],
        "Random Forest (Test Set)",
        OUTPUT_DIR / "confusion_matrices" / "cm_random_forest.png",
    )
    print(f"  Acc: {rf_test_m['accuracy']*100:.2f}% | Macro-F1: {rf_test_m['macro_f1']:.4f} | ROC-AUC: {rf_test_m['roc_auc']:.4f}")

    # -------------------------------------------------------------------------
    # Model 3: Centralized PyTorch Neural Network
    # -------------------------------------------------------------------------
    print("\n[Step 5] Training & Evaluating Model 3: Centralized PyTorch Neural Network...")
    torch.manual_seed(42)
    cent_nn = IoTThreatClassifier(input_dim=input_dim)
    cent_train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_all, dtype=torch.float32), torch.tensor(y_train_np, dtype=torch.long)),
        batch_size=64,
        shuffle=True,
    )
    cent_opt = torch.optim.Adam(cent_nn.parameters(), lr=0.001)
    cent_crit = nn.CrossEntropyLoss()
    cent_t0 = time.time()
    best_cent_f1 = -1.0
    best_cent_w = None

    for ep in range(1, 16):
        cent_nn.train()
        for bx, by in cent_train_loader:
            cent_opt.zero_grad()
            out = cent_nn(bx)
            loss = cent_crit(out, by)
            loss.backward()
            cent_opt.step()
        # Evaluate on validation
        val_eval = evaluate_torch_model(cent_nn, X_val, y_val_np)
        if val_eval["macro_f1"] > best_cent_f1:
            best_cent_f1 = val_eval["macro_f1"]
            best_cent_w = [p.copy() for p in get_model_parameters(cent_nn)]

    cent_train_time = round(time.time() - cent_t0, 2)
    if best_cent_w:
        set_model_parameters(cent_nn, best_cent_w)

    cent_nn.eval()
    with torch.no_grad():
        cent_test_probs = torch.softmax(cent_nn(torch.tensor(X_test, dtype=torch.float32)), dim=1)[:, 1].numpy()
        cent_held_probs = torch.softmax(cent_nn(torch.tensor(X_held_out, dtype=torch.float32)), dim=1)[:, 1].numpy()

    cent_latency = measure_inference_latency(
        lambda d: cent_nn(torch.tensor(d, dtype=torch.float32)).detach().numpy(),
        X_test,
    )
    cent_test_m = compute_comprehensive_metrics(y_test_np, cent_test_probs, threshold=0.50)
    cent_held_m = compute_comprehensive_metrics(y_held_out_np, cent_held_probs, threshold=0.50)
    cent_test_m["inference_latency_ms"] = cent_latency
    cent_test_m["model_size_bytes"] = estimate_parameter_size_bytes(get_model_parameters(cent_nn))
    cent_test_m["train_time_seconds"] = cent_train_time

    all_model_results["centralized_nn"] = {
        "display_name": "Centralized Neural Network (2-Layer MLP)",
        "family": "centralized_nn",
        "random_benchmark": cent_test_m,
        "scenario_held_out": cent_held_m,
    }
    curves_data["centralized_nn"] = (y_test_np, cent_test_probs, "Centralized NN")
    plot_confusion_matrix_heatmap(
        cent_test_m["confusion_matrix"],
        "Centralized NN (Test Set)",
        OUTPUT_DIR / "confusion_matrices" / "cm_centralized_nn.png",
    )
    print(f"  Acc: {cent_test_m['accuracy']*100:.2f}% | Macro-F1: {cent_test_m['macro_f1']:.4f} | ROC-AUC: {cent_test_m['roc_auc']:.4f}")

    # -------------------------------------------------------------------------
    # Federated Learning Suite across Seeds: FedAvg, FedOpt, GSFS
    # -------------------------------------------------------------------------
    seeds = [42, 101, 2024]
    print(f"\n[Step 6] Running Multi-Seed Federated Learning Suite (Seeds: {seeds})...")

    fed_algorithms = ["fedavg", "fedopt", "gsfs"]
    fed_results = {alg: [] for alg in fed_algorithms}

    for seed in seeds:
        print(f"\n  --- Execution for Random Seed: {seed} ---")
        client_dfs = partition_train_data(
            train_df=raw_train_df,
            num_clients=5,
            strategy="scenario_grouped",
            random_seed=seed,
        )

        for alg in fed_algorithms:
            if alg == "fedavg":
                cfg = FederatedConfig(random_seed=seed, num_rounds=10)
            elif alg == "fedopt":
                cfg = get_fedopt_config(random_seed=seed, num_rounds=10)
            else:
                cfg = get_gsfs_config(
                    aggregation_threshold=0.60,
                    perf_threshold=0.0,
                    grad_beta=0.0,
                    trigger_mode="OR",
                    warmup_rounds=1,
                    random_seed=seed,
                    num_rounds=10,
                )

            # Build fresh flower clients
            f_clients = {}
            for cid in range(5):
                c_df = client_dfs[cid]
                c_X = preprocessor.transform(c_df[DEFAULT_FEATURE_COLUMNS])
                c_y = c_df["label"].str.lower().str.strip().map(LABEL_MAPPING).to_numpy()
                c_model = IoTThreatClassifier(input_dim=input_dim)
                f_clients[cid] = IoTFlowerClient(
                    client_id=cid,
                    model=c_model,
                    X_train=c_X,
                    y_train=c_y,
                    local_epochs=2,
                    batch_size=64,
                    learning_rate=0.001,
                )

            server = FederatedServerCoordinator(
                config=cfg,
                clients=f_clients,
                val_X=X_val,
                val_y=y_val_np,
                input_dim=input_dim,
            )
            t_start = time.time()
            trained_model, history, comm_sum = server.run_federated_rounds()
            train_duration = round(time.time() - t_start, 2)

            trained_model.eval()
            with torch.no_grad():
                test_probs = torch.softmax(trained_model(torch.tensor(X_test, dtype=torch.float32)), dim=1)[:, 1].numpy()
                held_probs = torch.softmax(trained_model(torch.tensor(X_held_out, dtype=torch.float32)), dim=1)[:, 1].numpy()

            test_m = compute_comprehensive_metrics(y_test_np, test_probs, threshold=0.50)
            held_m = compute_comprehensive_metrics(y_held_out_np, held_probs, threshold=0.50)
            test_m["train_time_seconds"] = train_duration
            test_m["communication_bytes"] = comm_sum.get("total_communicated_bytes", 0)
            test_m["upload_bytes"] = comm_sum.get("total_upload_bytes", 0)
            test_m["download_bytes"] = comm_sum.get("total_download_bytes", 0)
            test_m["model_size_bytes"] = comm_sum.get("model_size_bytes", 40840)

            if alg == "gsfs":
                gsfs_strat_hist = getattr(server.strategy, "round_history", [])
                test_m["accepted_updates"] = sum(r.get("updates_selected", 0) for r in gsfs_strat_hist)
                test_m["rejected_updates"] = sum(r.get("updates_rejected", 0) for r in gsfs_strat_hist)
                test_m["aggregation_events"] = sum(1 for r in gsfs_strat_hist if r.get("aggregated_this_round", False))
                # Theoretical bytes avoided if selection occurred on client
                param_bytes = comm_sum.get("model_size_bytes", 40840)
                test_m["theoretical_bytes_avoided"] = test_m["rejected_updates"] * param_bytes

            fed_results[alg].append({
                "seed": seed,
                "test": test_m,
                "held_out": held_m,
                "probs": test_probs,
                "model": trained_model,
            })
            print(f"    {alg.upper()} (Seed {seed}): Acc={test_m['accuracy']*100:.2f}% | Macro-F1={test_m['macro_f1']:.4f} | ROC-AUC={test_m['roc_auc']:.4f}")

    # Aggregate Multi-Seed Statistics
    for alg in fed_algorithms:
        runs = fed_results[alg]
        test_accs = [r["test"]["accuracy"] for r in runs]
        test_f1s = [r["test"]["macro_f1"] for r in runs]
        test_aucs = [r["test"]["roc_auc"] for r in runs]
        test_praucs = [r["test"]["pr_auc"] for r in runs]

        held_accs = [r["held_out"]["accuracy"] for r in runs]
        held_f1s = [r["held_out"]["macro_f1"] for r in runs]

        # Use seed 42 as representative run for artifacts
        rep_run = next(r for r in runs if r["seed"] == 42)
        curves_data[alg] = (y_test_np, rep_run["probs"], f"{alg.upper()} (Seed 42)")

        plot_confusion_matrix_heatmap(
            rep_run["test"]["confusion_matrix"],
            f"{alg.upper()} (Seed 42)",
            OUTPUT_DIR / "confusion_matrices" / f"cm_{alg}.png",
        )

        all_model_results[alg] = {
            "display_name": f"Federated {alg.upper()}",
            "family": "federated_nn",
            "representative_run": rep_run["test"],
            "representative_held_out": rep_run["held_out"],
            "multi_seed_aggregate": {
                "num_seeds": len(seeds),
                "accuracy_mean": float(np.mean(test_accs)),
                "accuracy_std": float(np.std(test_accs)),
                "macro_f1_mean": float(np.mean(test_f1s)),
                "macro_f1_std": float(np.std(test_f1s)),
                "roc_auc_mean": float(np.mean(test_aucs)),
                "roc_auc_std": float(np.std(test_aucs)),
                "pr_auc_mean": float(np.mean(test_praucs)),
                "pr_auc_std": float(np.std(test_praucs)),
                "held_out_accuracy_mean": float(np.mean(held_accs)),
                "held_out_accuracy_std": float(np.std(held_accs)),
                "held_out_macro_f1_mean": float(np.mean(held_f1s)),
                "held_out_macro_f1_std": float(np.std(held_f1s)),
            },
        }

    # -------------------------------------------------------------------------
    # Step 7: Plot ROC & Precision-Recall Curves
    # -------------------------------------------------------------------------
    print("\n[Step 7] Generating Comparative ROC and Precision-Recall Curves...")
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=150)

    for key, (y_true, probs, label) in curves_data.items():
        # ROC
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        ax_roc.plot(fpr, tpr, label=f"{label} (AUC = {auc:.4f})", linewidth=1.8)

        # PR
        prec, rec, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        ax_pr.plot(rec, prec, label=f"{label} (PR-AUC = {ap:.4f})", linewidth=1.8)

    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess")
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=11, fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate", fontsize=10)
    ax_roc.set_ylabel("True Positive Rate (Recall)", fontsize=10)
    ax_roc.legend(loc="lower right", fontsize=8)
    ax_roc.grid(True, alpha=0.3)

    baseline_ap = np.sum(y_test_np == 1) / len(y_test_np)
    ax_pr.plot([0, 1], [baseline_ap, baseline_ap], "k--", alpha=0.5, label=f"Class Prevalence ({baseline_ap:.2f})")
    ax_pr.set_title("Precision-Recall (PR) Curve Comparison", fontsize=11, fontweight="bold")
    ax_pr.set_xlabel("Recall", fontsize=10)
    ax_pr.set_ylabel("Precision", fontsize=10)
    ax_pr.legend(loc="lower left", fontsize=8)
    ax_pr.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_pr_curves.png")
    plt.close(fig)
    print("  Saved: experiments/final/roc_pr_curves.png")

    # -------------------------------------------------------------------------
    # Step 8: GSFS Trigger Sensitivity Analysis Grid
    # -------------------------------------------------------------------------
    print("\n[Step 8] Executing GSFS Trigger Sensitivity Grid...")
    # Evaluate across 3 thresholds (40%, 60%, 80%) x 2 modes (OR, AND) x Betas (-0.5, 0.0, 0.5)
    sensitivity_configs = [
        {"thresh": 0.40, "mode": "OR", "beta": 0.0, "perf": 0.0},
        {"thresh": 0.60, "mode": "OR", "beta": 0.0, "perf": 0.0},
        {"thresh": 0.80, "mode": "OR", "beta": 0.0, "perf": 0.0},
        {"thresh": 0.60, "mode": "AND", "beta": 0.0, "perf": 0.0},
        {"thresh": 0.60, "mode": "AND", "beta": 0.5, "perf": 0.01},
        {"thresh": 0.60, "mode": "AND", "beta": -0.5, "perf": 0.0},
    ]

    sensitivity_results = []
    param_size = estimate_parameter_size_bytes(get_model_parameters(IoTThreatClassifier(input_dim=input_dim)))

    for s_cfg in sensitivity_configs:
        cfg = get_gsfs_config(
            aggregation_threshold=s_cfg["thresh"],
            perf_threshold=s_cfg["perf"],
            grad_beta=s_cfg["beta"],
            trigger_mode=s_cfg["mode"],
            warmup_rounds=1,
            random_seed=42,
            num_rounds=5,  # 5 rounds for rapid sensitivity profiling
        )
        # Client partitioning
        c_dfs = partition_train_data(raw_train_df, num_clients=5, strategy="scenario_grouped", random_seed=42)
        f_clients = {}
        for cid in range(5):
            c_df = c_dfs[cid]
            c_X = preprocessor.transform(c_df[DEFAULT_FEATURE_COLUMNS])
            c_y = c_df["label"].str.lower().str.strip().map(LABEL_MAPPING).to_numpy()
            c_model = IoTThreatClassifier(input_dim=input_dim)
            f_clients[cid] = IoTFlowerClient(cid, c_model, c_X, c_y, local_epochs=2, batch_size=64, learning_rate=0.001)

        server = FederatedServerCoordinator(cfg, f_clients, X_val, y_val_np, input_dim=input_dim)
        m, hist, comm = server.run_federated_rounds()

        m.eval()
        with torch.no_grad():
            p = torch.softmax(m(torch.tensor(X_test, dtype=torch.float32)), dim=1)[:, 1].numpy()
        test_eval = compute_comprehensive_metrics(y_test_np, p)

        total_client_fits = 5 * 5  # 5 rounds * 5 clients = 25 fit opportunities
        gsfs_strat_hist = getattr(server.strategy, "round_history", [])
        accepted = sum(r.get("updates_selected", 0) for r in gsfs_strat_hist)
        rejected = sum(r.get("updates_rejected", 0) for r in gsfs_strat_hist)
        rejections_pct = round((rejected / total_client_fits) * 100, 1)
        avoided_bytes = rejected * param_size

        entry = {
            "config": s_cfg,
            "accuracy": test_eval["accuracy"],
            "macro_f1": test_eval["macro_f1"],
            "roc_auc": test_eval["roc_auc"],
            "total_updates": total_client_fits,
            "accepted_updates": accepted,
            "rejected_updates": rejected,
            "rejection_percentage": rejections_pct,
            "theoretical_bytes_avoided": avoided_bytes,
            "aggregation_events": sum(1 for r in gsfs_strat_hist if r.get("aggregated_this_round", False)),
        }
        sensitivity_results.append(entry)
        print(f"    Thresh={int(s_cfg['thresh']*100)}% Mode={s_cfg['mode']} Beta={s_cfg['beta']} -> Rejections: {rejections_pct}% | Macro-F1: {test_eval['macro_f1']:.4f}")

    with open(OUTPUT_DIR / "gsfs_sensitivity.json", "w") as f:
        json.dump(sensitivity_results, f, indent=2)

    # Plot Sensitivity Bar Chart
    labels = [f"T={int(s['config']['thresh']*100)}%\n{s['config']['mode']}\nβ={s['config']['beta']}" for s in sensitivity_results]
    f1_vals = [s["macro_f1"] for s in sensitivity_results]
    rej_vals = [s["rejection_percentage"] for s in sensitivity_results]

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

    plt.title("GSFS Parameter Sensitivity: Detection Performance vs Update Selectivity", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gsfs_sensitivity_plot.png")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Step 9: Save Comprehensive Results and Comparison Table
    # -------------------------------------------------------------------------
    print("\n[Step 9] Writing Final JSON, CSV, and Markdown Comparison Tables...")
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(all_model_results, f, indent=2)

    # Markdown Table
    md_content = """# Final Model Benchmark Comparison (Phase 9)

## 1. Random Benchmark Split (In-Scenario Interpolation — 4,800 Test Samples)

| Model Architecture | Model Family | Accuracy | Macro-F1 | Malicious Rec | Malicious Prec | ROC-AUC | PR-AUC | Latency (ms) | Artifact Size |
|---|---|---|---|---|---|---|---|---|---|
"""
    for key, data in all_model_results.items():
        m = data.get("representative_run", data.get("random_benchmark", {}))
        name = data["display_name"]
        family = data["family"]
        acc = f"{m['accuracy']*100:.2f}%"
        f1 = f"{m['macro_f1']:.4f}"
        rec = f"{m['malicious']['recall']*100:.2f}%"
        prec = f"{m['malicious']['precision']*100:.2f}%"
        roc = f"{m['roc_auc']:.4f}"
        prauc = f"{m['pr_auc']:.4f}"
        lat = f"{m.get('inference_latency_ms', 0.0):.4f}"
        sz = f"{m.get('model_size_bytes', 0) / 1024:.1f} KB"
        md_content += f"| **{name}** | `{family}` | **{acc}** | **{f1}** | {rec} | {prec} | **{roc}** | **{prauc}** | {lat} ms | {sz} |\n"

    md_content += """
---

## 2. Multi-Seed Federated Learning Trials (Seeds: 42, 101, 2024)

| Federated Model | Accuracy (Mean ± Std) | Macro-F1 (Mean ± Std) | ROC-AUC (Mean ± Std) | PR-AUC (Mean ± Std) |
|---|---|---|---|---|
"""
    for alg in fed_algorithms:
        agg = all_model_results[alg]["multi_seed_aggregate"]
        md_content += (
            f"| **Federated {alg.upper()}** | "
            f"{agg['accuracy_mean']*100:.2f}% ± {agg['accuracy_std']*100:.2f}% | "
            f"{agg['macro_f1_mean']:.4f} ± {agg['macro_f1_std']:.4f} | "
            f"{agg['roc_auc_mean']:.4f} ± {agg['roc_auc_std']:.4f} | "
            f"{agg['pr_auc_mean']:.4f} ± {agg['pr_auc_std']:.4f} |\n"
        )

    md_content += """
---

## 3. Generalization Performance: Scenario-Held-Out Evaluation

Held-out scenarios: `CTU-Honeypot-Capture-5-1`, `CTU-IoT-Malware-Capture-34-1`, `CTU-IoT-Malware-Capture-3-1` (6,166 samples).

| Model Architecture | In-Scenario Accuracy (Random) | In-Scenario Macro-F1 | Out-of-Distribution Accuracy (Held-Out) | Out-of-Distribution Macro-F1 | Generalization Gap (Macro-F1) |
|---|---|---|---|---|---|
"""
    for key, data in all_model_results.items():
        name = data["display_name"]
        in_m = data.get("representative_run", data.get("random_benchmark", {}))
        out_m = data.get("representative_held_out", data.get("scenario_held_out", {}))
        in_acc = f"{in_m['accuracy']*100:.2f}%"
        in_f1 = in_m['macro_f1']
        out_acc = f"{out_m['accuracy']*100:.2f}%"
        out_f1 = out_m['macro_f1']
        gap = f"{in_f1 - out_f1:+.4f}"
        md_content += f"| **{name}** | {in_acc} | {in_f1:.4f} | **{out_acc}** | **{out_f1:.4f}** | `{gap}` |\n"

    with open(OUTPUT_DIR / "summary_table.md", "w") as f:
        f.write(md_content)

    print("\nSUCCESS! Phase 9 research experiments suite complete.")
    print(f"Artifacts preserved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    run_experiment_suite()

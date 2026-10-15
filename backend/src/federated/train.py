"""Complete workflow for Federated Learning (FedAvg) and centralized benchmark comparison.

Workflow:
1. Loads IoT-23 splits (train: 33,602, val: 9,601, test: 4,800).
2. Fits preprocessing pipeline on train data only, producing aligned 125-dim features.
3. Partitions training data into 5 Non-IID clients using scenario-aware grouping.
4. Trains a Centralized Neural Network reference on the pooled training set.
5. Runs 10 rounds of Federated Learning with simulated Flower clients using FedAvg.
6. Compares Centralized Random Forest, Centralized NN, and Federated FedAvg on Test set.
7. Logs communication measurements, round metrics, and saves all reproducible artifacts.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import joblib

from src.detection.data_loader import (
    load_dataset_split,
    load_feature_metadata,
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_NUMERIC_FEATURES,
    DEFAULT_CATEGORICAL_FEATURES,
    LABEL_MAPPING,
)
from src.detection.pipeline import build_preprocessor
from src.detection.evaluate import plot_confusion_matrix
from src.federated.config import FederatedConfig
from src.federated.partitioner import (
    partition_train_data,
    get_partition_summary,
    save_partition_metadata,
)
from src.federated.model import (
    IoTThreatClassifier,
    get_model_parameters,
    set_model_parameters,
    count_parameters,
    estimate_parameter_size_bytes,
)
from src.federated.client import IoTFlowerClient
from src.federated.server import FederatedServerCoordinator
from src.federated.evaluate import (
    evaluate_torch_model,
    plot_federated_round_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FedAvgTrainer")


def train_centralized_neural_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    random_seed: int = 42,
    device: str = "cpu",
) -> Tuple[IoTThreatClassifier, Dict[str, Any], float]:
    """
    Train a centralized PyTorch neural network as an architecture reference.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    dev = torch.device(device)
    model = IoTThreatClassifier(input_dim=input_dim)
    model.to(dev)

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"Training Centralized Neural Network ({epochs} epochs)...")
    t0 = time.time()
    best_val_macro_f1 = -1.0
    best_weights = None

    for ep in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        # Validation check
        val_m = evaluate_torch_model(model, X_val, y_val, device=device)
        if val_m["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_m["macro_f1"]
            best_weights = [p.copy() for p in get_model_parameters(model)]

    train_time = round(time.time() - t0, 2)
    if best_weights is not None:
        set_model_parameters(model, best_weights)

    val_metrics = evaluate_torch_model(model, X_val, y_val, device=device)
    val_metrics["train_time_seconds"] = train_time
    logger.info(f"Centralized NN Train Complete: Val Acc={val_metrics['accuracy']*100:.2f}%, Val Macro-F1={val_metrics['macro_f1']:.4f} in {train_time}s")
    return model, val_metrics, train_time


def run_federated_experiment(config: Optional[FederatedConfig] = None) -> Dict[str, Any]:
    """Execute complete Phase 3 Federated Learning pipeline."""
    cfg = config or FederatedConfig()
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.experiments_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    logger.info("Loading IoT-23 processed data...")
    raw_train_df = pd.read_csv(cfg.train_path)
    X_train_raw, y_train, meta_train = load_dataset_split(cfg.train_path, cfg.metadata_path)
    X_val_raw, y_val, meta_val = load_dataset_split(cfg.val_path, cfg.metadata_path)
    X_test_raw, y_test, meta_test = load_dataset_split(cfg.test_path, cfg.metadata_path)

    # 2. Fit Preprocessor ONCE on training split only
    logger.info("Fitting ColumnTransformer preprocessor on training data only...")
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
    logger.info(f"Preprocessed input feature dimension: {input_dim}")

    # 3. Partition training data into Non-IID clients
    logger.info(f"Partitioning training set into {cfg.num_clients} clients ({cfg.partition_strategy})...")
    client_dfs = partition_train_data(
        train_df=raw_train_df,
        num_clients=cfg.num_clients,
        strategy=cfg.partition_strategy,
        random_seed=cfg.random_seed,
        alpha=cfg.dirichlet_alpha,
    )
    partition_summary = get_partition_summary(client_dfs)
    partition_json_path = cfg.experiments_dir / "client_partitions.json"
    save_partition_metadata(partition_summary, partition_json_path)

    # 4. Prepare Flower Clients
    flower_clients: Dict[int, IoTFlowerClient] = {}
    for cid in range(cfg.num_clients):
        c_df = client_dfs[cid]
        c_X_raw = c_df[DEFAULT_FEATURE_COLUMNS]
        c_X = preprocessor.transform(c_X_raw)
        c_y = c_df["label"].str.lower().str.strip().map(LABEL_MAPPING).to_numpy()

        c_model = IoTThreatClassifier(
            input_dim=input_dim,
            hidden_dims=cfg.hidden_dims,
            dropout_rate=cfg.dropout_rate,
        )
        flower_clients[cid] = IoTFlowerClient(
            client_id=cid,
            model=c_model,
            X_train=c_X,
            y_train=c_y,
            local_epochs=cfg.local_epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            device=cfg.device,
        )

    # 5. Train Centralized Reference Neural Network
    cent_nn_model, cent_val_m, cent_train_time = train_centralized_neural_network(
        X_train=X_train_all,
        y_train=y_train.to_numpy(),
        X_val=X_val,
        y_val=y_val.to_numpy(),
        input_dim=input_dim,
        epochs=cfg.num_rounds * cfg.local_epochs,  # Equivalent local epoch budget
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        random_seed=cfg.random_seed,
        device=cfg.device,
    )
    cent_test_m = evaluate_torch_model(cent_nn_model, X_test, y_test.to_numpy(), device=cfg.device)

    # 6. Execute Federated Learning (FedAvg) Simulation
    logger.info("Initializing Federated Coordinator...")
    server = FederatedServerCoordinator(
        config=cfg,
        clients=flower_clients,
        val_X=X_val,
        val_y=y_val.to_numpy(),
        input_dim=input_dim,
    )
    fed_model, round_history, comm_summary = server.run_federated_rounds()

    # 7. Evaluate Federated Model on Global Test Set
    logger.info("Evaluating Federated Model on Global Test Set...")
    fed_test_m = evaluate_torch_model(fed_model, X_test, y_test.to_numpy(), device=cfg.device)
    logger.info(
        f"Federated Model Test Evaluation: Acc={fed_test_m['accuracy']*100:.2f}% | "
        f"Macro-F1={fed_test_m['macro_f1']:.4f} | "
        f"Malicious Rec={fed_test_m['malicious']['recall']*100:.2f}% | "
        f"Malicious Prec={fed_test_m['malicious']['precision']*100:.2f}% | "
        f"ROC-AUC={fed_test_m['roc_auc']:.4f}"
    )

    # Load Phase 2 Centralized Random Forest metrics for comparison
    rf_test_m = {}
    rf_metrics_path = Path("experiments/baseline/metrics.json")
    if rf_metrics_path.exists():
        with rf_metrics_path.open(encoding="utf-8") as f:
            base_m = json.load(f)
            rf_test_m = base_m.get("test_evaluation", {}).get("random_forest", {})

    # 8. Save Artifacts
    # A. Save model checkpoint
    model_save_path = cfg.models_dir / "fedavg_model.pt"
    torch.save(fed_model.state_dict(), model_save_path)
    logger.info(f"Saved global federated model to {model_save_path}")

    # B. Save metadata
    metadata_save_path = cfg.models_dir / "fedavg_metadata.json"
    fed_metadata = {
        "model_architecture": "IoTThreatClassifier",
        "input_dim": input_dim,
        "hidden_dims": list(cfg.hidden_dims),
        "dropout_rate": cfg.dropout_rate,
        "num_classes": 2,
        "parameter_count": comm_summary["parameter_count"],
        "num_clients": cfg.num_clients,
        "num_rounds": cfg.num_rounds,
        "local_epochs": cfg.local_epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "random_seed": cfg.random_seed,
        "partition_strategy": cfg.partition_strategy,
        "model_file": model_save_path.name,
    }
    with metadata_save_path.open("w", encoding="utf-8") as f:
        json.dump(fed_metadata, f, indent=2)

    # C. Save round_metrics.csv
    round_df = pd.DataFrame(round_history)
    round_csv_path = cfg.experiments_dir / "round_metrics.csv"
    round_df.to_csv(round_csv_path, index=False)

    # D. Save config.json
    config_save_path = cfg.experiments_dir / "config.json"
    with config_save_path.open("w", encoding="utf-8") as f:
        json.dump({
            "num_clients": cfg.num_clients,
            "num_rounds": cfg.num_rounds,
            "local_epochs": cfg.local_epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "random_seed": cfg.random_seed,
            "partition_strategy": cfg.partition_strategy,
            "device": cfg.device,
        }, f, indent=2)

    # E. Save metrics.json
    full_metrics = {
        "federated_test_metrics": fed_test_m,
        "centralized_nn_test_metrics": cent_test_m,
        "centralized_rf_test_metrics": rf_test_m,
        "communication_summary": comm_summary,
        "round_history": round_history,
    }
    metrics_json_path = cfg.experiments_dir / "metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)

    # F. Generate plots
    round_plot_path = cfg.experiments_dir / "round_metrics.png"
    plot_federated_round_metrics(round_history, round_plot_path)

    cm_plot_path = cfg.experiments_dir / "confusion_matrix.png"
    cm_arr = np.array(fed_test_m["confusion_matrix"]["matrix"])
    plot_confusion_matrix(
        cm=cm_arr,
        class_names=["Benign", "Malicious"],
        save_path=cm_plot_path,
        title="Test Confusion Matrix - Federated FedAvg NN",
    )

    # G. Generate Markdown report
    report_md_path = cfg.experiments_dir / "report.md"
    with report_md_path.open("w", encoding="utf-8") as f:
        f.write(_build_federated_report(full_metrics, partition_summary, cfg))
    logger.info(f"Generated Federated Learning report: {report_md_path}")

    return full_metrics


def _build_federated_report(
    metrics: Dict[str, Any],
    partition_summary: Dict[str, Any],
    cfg: FederatedConfig,
) -> str:
    """Build comprehensive Markdown report for Phase 3 FedAvg."""
    fed_te = metrics["federated_test_metrics"]
    cent_te = metrics["centralized_nn_test_metrics"]
    rf_te = metrics.get("centralized_rf_test_metrics", {})
    comm = metrics["communication_summary"]

    lines = [
        "# Phase 3: Federated Learning with FedAvg - Experiment Report",
        "",
        "## 1. Executive Summary & Context",
        "This experiment evaluates **Federated Averaging (FedAvg)** using simulated Flower clients on the **IoT-23** cybersecurity dataset.",
        "The objective is to establish a verified federated learning baseline on decentralized, non-IID client partitions before introducing communication-reduction techniques (GSFS) or LLM explanation layers in subsequent phases.",
        "",
        "---",
        "",
        "## 2. Experimental Setup & Hyperparameters",
        "",
        f"- **Federated Algorithm:** Sample-Weighted FedAvg (Flower architecture)",
        f"- **Number of Simulated Clients:** `{cfg.num_clients}`",
        f"- **Federated Rounds:** `{cfg.num_rounds}`",
        f"- **Participation Fraction:** `{cfg.fraction_fit}` (100% participation: all 5 clients in every round)",
        f"- **Local Epochs per Round:** `{cfg.local_epochs}` (Total local passes: `{cfg.num_rounds * cfg.local_epochs}`)",
        f"- **Batch Size:** `{cfg.batch_size}`",
        f"- **Optimizer & Learning Rate:** Adam ($lr = {cfg.learning_rate}$)",
        f"- **Loss Function:** PyTorch Cross-Entropy Loss",
        f"- **Neural Architecture:** `IoTThreatClassifier` (Input: 125 $\\to$ 64 $\\to$ 32 $\\to$ 2 logits)",
        f"- **Parameter Count:** `{comm['parameter_count']:,}` parameters",
        f"- **Execution Device:** `{cfg.device.upper()}`",
        f"- **Random Seed:** `{cfg.random_seed}`",
        "",
        "---",
        "",
        "## 3. Non-IID Client Partitioning Breakdown",
        "",
        "Training records (`33,602` samples) were partitioned among 5 clients using **scenario-aware grouping**.",
        "Each scenario's network traffic is preserved intact within a specific client to realistically emulate heterogeneous IoT deployments.",
        "",
        "| Client ID | Total Records | Benign Samples | Malicious Samples | Benign % | Malicious % | Assigned Scenarios Count |",
        "|---|---|---|---|---|---|---|",
    ]

    for cid_str, c_info in sorted(partition_summary["clients"].items()):
        cid = c_info["client_id"]
        lines.append(
            f"| Client {cid} | {c_info['total_samples']:,} | {c_info['benign_samples']:,} | {c_info['malicious_samples']:,} | "
            f"{c_info['benign_ratio']*100:.1f}% | {c_info['malicious_ratio']*100:.1f}% | {c_info['scenarios_count']} scenarios |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Fair Benchmark Comparison on Global Test Set (4,800 Samples)",
        "",
        "We compare three models on the same unseen test split:",
        "1. **Centralized Random Forest (Phase 2):** High-capacity ensemble tree baseline.",
        "2. **Centralized Neural Network:** Same architecture and training budget as federated NN, but trained on centrally pooled data.",
        "3. **Federated Neural Network (FedAvg):** Decentralized client training aggregated via FedAvg.",
        "",
        "| Metric | Centralized Random Forest | Centralized Neural Network | Federated NN (FedAvg) |",
        "|---|---|---|---|",
    ])

    def fmt_pct(val):
        return f"{val*100:.2f}%" if isinstance(val, (int, float)) else "N/A"

    def fmt_f(val):
        return f"{val:.4f}" if isinstance(val, (int, float)) else "N/A"

    lines.append(f"| **Accuracy** | {fmt_pct(rf_te.get('accuracy'))} | {fmt_pct(cent_te.get('accuracy'))} | **{fmt_pct(fed_te.get('accuracy'))}** |")
    lines.append(f"| **Macro-F1** | {fmt_f(rf_te.get('macro_f1'))} | {fmt_f(cent_te.get('macro_f1'))} | **{fmt_f(fed_te.get('macro_f1'))}** |")
    lines.append(f"| **Weighted-F1** | {fmt_f(rf_te.get('weighted_f1'))} | {fmt_f(cent_te.get('weighted_f1'))} | **{fmt_f(fed_te.get('weighted_f1'))}** |")
    lines.append(f"| **Malicious Precision** | {fmt_pct(rf_te.get('malicious', {}).get('precision'))} | {fmt_pct(cent_te.get('malicious', {}).get('precision'))} | **{fmt_pct(fed_te.get('malicious', {}).get('precision'))}** |")
    lines.append(f"| **Malicious Recall** | {fmt_pct(rf_te.get('malicious', {}).get('recall'))} | {fmt_pct(cent_te.get('malicious', {}).get('recall'))} | **{fmt_pct(fed_te.get('malicious', {}).get('recall'))}** |")
    lines.append(f"| **Malicious F1** | {fmt_f(rf_te.get('malicious', {}).get('f1_score'))} | {fmt_f(cent_te.get('malicious', {}).get('f1_score'))} | **{fmt_f(fed_te.get('malicious', {}).get('f1_score'))}** |")
    lines.append(f"| **Benign Precision** | {fmt_pct(rf_te.get('benign', {}).get('precision'))} | {fmt_pct(cent_te.get('benign', {}).get('precision'))} | **{fmt_pct(fed_te.get('benign', {}).get('precision'))}** |")
    lines.append(f"| **Benign Recall** | {fmt_pct(rf_te.get('benign', {}).get('recall'))} | {fmt_pct(cent_te.get('benign', {}).get('recall'))} | **{fmt_pct(fed_te.get('benign', {}).get('recall'))}** |")
    lines.append(f"| **Benign F1** | {fmt_f(rf_te.get('benign', {}).get('f1_score'))} | {fmt_f(cent_te.get('benign', {}).get('f1_score'))} | **{fmt_f(fed_te.get('benign', {}).get('f1_score'))}** |")
    lines.append(f"| **ROC-AUC** | {fmt_f(rf_te.get('roc_auc'))} | {fmt_f(cent_te.get('roc_auc'))} | **{fmt_f(fed_te.get('roc_auc'))}** |")
    lines.append(f"| **PR-AUC** | {fmt_f(rf_te.get('pr_auc'))} | {fmt_f(cent_te.get('pr_auc'))} | **{fmt_f(fed_te.get('pr_auc'))}** |")

    lines.extend([
        "",
        "> [!NOTE]",
        "> **Architecture vs. Federation Effect:** The centralized neural network and the federated neural network use identical architectures and feature sets. The centralized Random Forest operates on tree decision boundaries. Comparing the Federated NN against the Centralized NN isolates the direct statistical cost of decentralized non-IID learning.",
        "",
        "---",
        "",
        "## 5. Communication Measurement & Overhead Analysis",
        "",
        "Communication measurement is a primary research axis for this project (setting the baseline for subsequent GSFS gradient-pruning experiments).",
        "",
        "| Communication Metric | Measured Value |",
        "|---|---|",
        f"| Model Parameter Count | `{comm['parameter_count']:,}` parameters |",
        f"| Single Model Payload Size | `{comm['model_size_bytes'] / 1024:.2f} KB` (`{comm['model_size_bytes']:,}` bytes) |",
        f"| Rounds Executed | `{comm['num_rounds']}` rounds |",
        f"| Client Updates Aggregated | `{comm['num_rounds'] * comm['total_clients']}` updates |",
        f"| Estimated Total Upload Payload (Clients $\\to$ Server) | `{comm['total_upload_bytes'] / (1024 * 1024):.2f} MB` (`{comm['total_upload_bytes']:,}` bytes) |",
        f"| Estimated Total Download Payload (Server $\\to$ Clients) | `{comm['total_download_bytes'] / (1024 * 1024):.2f} MB` (`{comm['total_download_bytes']:,}` bytes) |",
        f"| **Total Communicated Payload** | **`{comm['total_communicated_bytes'] / (1024 * 1024):.2f} MB`** |",
        f"| Total Client Compute Time | `{comm['total_client_training_time_seconds']} seconds` |",
        "",
        "> [!IMPORTANT]",
        "> **Measurement Scope:** Payload sizes represent raw serialized parameter tensors (Float32). Real network transports incur additional gRPC/TLS envelope headers and network latency.",
        "",
        "---",
        "",
        "## 6. Key Scientific Limitations",
        "",
        "1. **Simulated In-Process Clients:** Clients run in local memory simulation rather than distributed physical IoT devices.",
        "2. **Centralized Preprocessing Assumption:** In this initial baseline, `ColumnTransformer` feature encodings and scalers were fit on pooled training data. In real-world zero-trust federated networks, feature encodings must use public schemas or decentralized private quantile sketches.",
        "3. **In-Scenario Test Interpolation:** The global test set includes samples from represented scenarios, evaluating in-distribution threat detection. Zero-shot generalization to unseen IoT scenarios drops as demonstrated in Phase 2.",
    ])

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="IoT-23 FedAvg Federated Learning Experiment")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    config = FederatedConfig(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        random_seed=args.seed,
        device=args.device,
    )

    run_federated_experiment(config)


if __name__ == "__main__":
    main()

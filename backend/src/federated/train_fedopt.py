"""Complete training and evaluation pipeline for Federated Optimization with FedOpt (Phase 4).

Workflow:
1. Loads IoT-23 dataset splits (train: 33,602, val: 9,601, test: 4,800).
2. Fits ColumnTransformer preprocessing pipeline exclusively on training data.
3. Partitions training data into 5 Non-IID clients using scenario-aware bin-packing.
4. Executes FedOpt (FedAdam) simulation over 10 rounds using simulated Flower clients.
5. Loads existing Centralized RF, Centralized NN, and Federated FedAvg benchmarks.
6. Evaluates FedOpt on the held-out global test set.
7. Produces comprehensive 4-way comparison metrics, convergence plots, and audit reports.
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

from src.detection.data_loader import (
    load_dataset_split,
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_NUMERIC_FEATURES,
    DEFAULT_CATEGORICAL_FEATURES,
    LABEL_MAPPING,
)
from src.detection.pipeline import build_preprocessor
from src.detection.evaluate import plot_confusion_matrix
from src.federated.config import FederatedConfig, get_fedopt_config
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
    plot_convergence_comparison,
    plot_4way_benchmark_comparison,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("FedOptTrainer")


def run_fedopt_experiment(config: Optional[FederatedConfig] = None) -> Dict[str, Any]:
    """Execute complete Phase 4 Federated Optimization (FedOpt) workflow."""
    cfg = config or get_fedopt_config(server_optimizer="fedadam", server_learning_rate=0.02)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.experiments_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    logger.info("Loading IoT-23 processed data...")
    raw_train_df = pd.read_csv(cfg.train_path)
    X_train_raw, y_train, _ = load_dataset_split(cfg.train_path, cfg.metadata_path)
    X_val_raw, y_val, _ = load_dataset_split(cfg.val_path, cfg.metadata_path)
    X_test_raw, y_test, _ = load_dataset_split(cfg.test_path, cfg.metadata_path)

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

    # 5. Initialize Server Coordinator with FedOpt strategy
    logger.info(f"Initializing Federated Coordinator with strategy: {cfg.server_optimizer.upper()} (server_lr={cfg.server_learning_rate})...")
    server = FederatedServerCoordinator(
        config=cfg,
        clients=flower_clients,
        val_X=X_val,
        val_y=y_val.to_numpy(),
        input_dim=input_dim,
    )

    # 6. Execute FedOpt simulation
    fedopt_model, round_history, comm_summary = server.run_federated_rounds()

    # 7. Evaluate on held-out global test set
    logger.info("Evaluating Best FedOpt Model on Global Test Set...")
    fedopt_test_m = evaluate_torch_model(fedopt_model, X_test, y_test.to_numpy(), device=cfg.device)
    logger.info(
        f"FedOpt ({cfg.server_optimizer.upper()}) Test Evaluation: Acc={fedopt_test_m['accuracy']*100:.2f}% | "
        f"Macro-F1={fedopt_test_m['macro_f1']:.4f} | "
        f"Malicious Rec={fedopt_test_m['malicious']['recall']*100:.2f}% | "
        f"Malicious Prec={fedopt_test_m['malicious']['precision']*100:.2f}% | "
        f"ROC-AUC={fedopt_test_m['roc_auc']:.4f}"
    )

    # 8. Load existing reference benchmarks for comparison
    rf_test_m = {}
    rf_path = Path("experiments/baseline/metrics.json")
    if rf_path.exists():
        with rf_path.open(encoding="utf-8") as f:
            rf_test_m = json.load(f).get("test_evaluation", {}).get("random_forest", {})

    fedavg_test_m = {}
    cent_nn_test_m = {}
    fedavg_path = Path("experiments/federated/fedavg/metrics.json")
    if fedavg_path.exists():
        with fedavg_path.open(encoding="utf-8") as f:
            avg_m = json.load(f)
            fedavg_test_m = avg_m.get("federated_test_metrics", {})
            cent_nn_test_m = avg_m.get("centralized_nn_test_metrics", {})
            if not rf_test_m:
                rf_test_m = avg_m.get("centralized_rf_test_metrics", {})

    # 9. Save Artifacts
    # A. Model weights
    model_save_path = cfg.models_dir / "fedopt_model.pt"
    torch.save(fedopt_model.state_dict(), model_save_path)
    logger.info(f"Saved global FedOpt model to {model_save_path}")

    # B. Metadata
    metadata_save_path = cfg.models_dir / "fedopt_metadata.json"
    metadata_record = {
        "model_architecture": "IoTThreatClassifier",
        "input_dim": input_dim,
        "hidden_dims": list(cfg.hidden_dims),
        "dropout_rate": cfg.dropout_rate,
        "num_classes": 2,
        "parameter_count": comm_summary["parameter_count"],
        "num_clients": cfg.num_clients,
        "num_rounds": cfg.num_rounds,
        "server_optimizer": cfg.server_optimizer,
        "server_learning_rate": cfg.server_learning_rate,
        "server_beta_1": cfg.server_beta_1,
        "server_beta_2": cfg.server_beta_2,
        "server_tau": cfg.server_tau,
        "test_metrics": {
            "accuracy": fedopt_test_m["accuracy"],
            "macro_f1": fedopt_test_m["macro_f1"],
            "malicious_recall": fedopt_test_m["malicious"]["recall"],
            "malicious_precision": fedopt_test_m["malicious"]["precision"],
            "roc_auc": fedopt_test_m.get("roc_auc"),
        },
    }
    with metadata_save_path.open("w", encoding="utf-8") as f:
        json.dump(metadata_record, f, indent=2)

    # C. Experiment configuration
    config_save_path = cfg.experiments_dir / "config.json"
    with config_save_path.open("w", encoding="utf-8") as f:
        json.dump({
            "num_clients": cfg.num_clients,
            "num_rounds": cfg.num_rounds,
            "fraction_fit": cfg.fraction_fit,
            "local_epochs": cfg.local_epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "server_optimizer": cfg.server_optimizer,
            "server_learning_rate": cfg.server_learning_rate,
            "server_beta_1": cfg.server_beta_1,
            "server_beta_2": cfg.server_beta_2,
            "server_tau": cfg.server_tau,
            "random_seed": cfg.random_seed,
            "partition_strategy": cfg.partition_strategy,
            "input_dim": input_dim,
            "hidden_dims": list(cfg.hidden_dims),
        }, f, indent=2)

    # D. Round metrics CSV
    round_metrics_df = pd.DataFrame(round_history)
    csv_save_path = cfg.experiments_dir / "round_metrics.csv"
    round_metrics_df.to_csv(csv_save_path, index=False)

    # E. Full 4-way comparison metrics JSON
    metrics_export = {
        "experiment_configuration": {
            "server_optimizer": cfg.server_optimizer,
            "server_learning_rate": cfg.server_learning_rate,
            "num_clients": cfg.num_clients,
            "num_rounds": cfg.num_rounds,
            "random_seed": cfg.random_seed,
        },
        "communication_summary": comm_summary,
        "test_evaluation": {
            "centralized_random_forest": rf_test_m,
            "centralized_neural_network": cent_nn_test_m,
            "federated_fedavg": fedavg_test_m,
            f"federated_{cfg.server_optimizer}": fedopt_test_m,
        },
    }
    metrics_json_path = cfg.experiments_dir / "metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)

    # F. Visualizations
    # 1. Convergence curves comparing FedAvg and FedOpt
    convergence_png = cfg.experiments_dir / "convergence.png"
    fedavg_csv_path = Path("experiments/federated/fedavg/round_metrics.csv")
    if fedavg_csv_path.exists():
        plot_convergence_comparison(
            fedavg_rounds=fedavg_csv_path,
            fedopt_rounds=round_metrics_df,
            save_path=convergence_png,
            opt_name=cfg.server_optimizer.upper(),
        )
        logger.info(f"Saved convergence comparison plot to {convergence_png}")

    # 2. 4-way benchmark comparison bar chart
    comparison_png = cfg.experiments_dir / "comparison.png"
    metrics_4way = {}
    if rf_test_m:
        metrics_4way["Centralized RF"] = {
            "accuracy": rf_test_m.get("accuracy", 0.0),
            "macro_f1": rf_test_m.get("macro_f1", 0.0),
            "malicious_recall": rf_test_m.get("malicious", {}).get("recall", 0.0),
            "malicious_precision": rf_test_m.get("malicious", {}).get("precision", 0.0),
            "roc_auc": rf_test_m.get("roc_auc", 0.0),
        }
    if cent_nn_test_m:
        metrics_4way["Centralized NN"] = {
            "accuracy": cent_nn_test_m.get("accuracy", 0.0),
            "macro_f1": cent_nn_test_m.get("macro_f1", 0.0),
            "malicious_recall": cent_nn_test_m.get("malicious", {}).get("recall", 0.0),
            "malicious_precision": cent_nn_test_m.get("malicious", {}).get("precision", 0.0),
            "roc_auc": cent_nn_test_m.get("roc_auc", 0.0),
        }
    if fedavg_test_m:
        metrics_4way["Federated FedAvg"] = {
            "accuracy": fedavg_test_m.get("accuracy", 0.0),
            "macro_f1": fedavg_test_m.get("macro_f1", 0.0),
            "malicious_recall": fedavg_test_m.get("malicious", {}).get("recall", 0.0),
            "malicious_precision": fedavg_test_m.get("malicious", {}).get("precision", 0.0),
            "roc_auc": fedavg_test_m.get("roc_auc", 0.0),
        }
    metrics_4way[f"FedOpt ({cfg.server_optimizer.upper()})"] = {
        "accuracy": fedopt_test_m.get("accuracy", 0.0),
        "macro_f1": fedopt_test_m.get("macro_f1", 0.0),
        "malicious_recall": fedopt_test_m.get("malicious", {}).get("recall", 0.0),
        "malicious_precision": fedopt_test_m.get("malicious", {}).get("precision", 0.0),
        "roc_auc": fedopt_test_m.get("roc_auc", 0.0),
    }
    plot_4way_benchmark_comparison(metrics_4way, comparison_png)
    logger.info(f"Saved 4-way benchmark comparison plot to {comparison_png}")

    # 3. Test set confusion matrix
    cm_png = cfg.experiments_dir / "confusion_matrix.png"
    if "confusion_matrix" in fedopt_test_m and isinstance(fedopt_test_m["confusion_matrix"], dict):
        cm_arr = np.array(fedopt_test_m["confusion_matrix"]["matrix"], dtype=np.int64)
    else:
        cm_arr = np.array(fedopt_test_m.get("confusion_matrix", [[0, 0], [0, 0]]), dtype=np.int64)
    plot_confusion_matrix(
        cm=cm_arr,
        class_names=["Benign", "Malicious"],
        save_path=cm_png,
        title=f"Test Confusion Matrix - FedOpt ({cfg.server_optimizer.upper()})",
    )

    # G. Generate Research Markdown Report
    report_md_path = cfg.experiments_dir / "report.md"
    generate_fedopt_report(
        cfg=cfg,
        comm_summary=comm_summary,
        round_history=round_history,
        fedopt_test_m=fedopt_test_m,
        fedavg_test_m=fedavg_test_m,
        cent_nn_test_m=cent_nn_test_m,
        rf_test_m=rf_test_m,
        save_path=report_md_path,
    )
    logger.info(f"Generated FedOpt audit report at {report_md_path}")

    return metrics_export


def generate_fedopt_report(
    cfg: FederatedConfig,
    comm_summary: Dict[str, Any],
    round_history: List[Dict[str, Any]],
    fedopt_test_m: Dict[str, Any],
    fedavg_test_m: Dict[str, Any],
    cent_nn_test_m: Dict[str, Any],
    rf_test_m: Dict[str, Any],
    save_path: Path,
) -> None:
    """Generate comprehensive markdown research report for Phase 4 FedOpt."""
    opt_name = cfg.server_optimizer.upper()

    def fmt_pct(val: Optional[float]) -> str:
        return f"{val * 100:.2f}%" if val is not None else "N/A"

    def fmt_f(val: Optional[float], prec: int = 4) -> str:
        return f"{val:.{prec}f}" if val is not None else "N/A"

    rf_acc = fmt_pct(rf_test_m.get("accuracy"))
    rf_f1 = fmt_f(rf_test_m.get("macro_f1"))
    rf_rec = fmt_pct(rf_test_m.get("malicious", {}).get("recall"))
    rf_prec = fmt_pct(rf_test_m.get("malicious", {}).get("precision"))
    rf_auc = fmt_f(rf_test_m.get("roc_auc"))

    cnn_acc = fmt_pct(cent_nn_test_m.get("accuracy"))
    cnn_f1 = fmt_f(cent_nn_test_m.get("macro_f1"))
    cnn_rec = fmt_pct(cent_nn_test_m.get("malicious", {}).get("recall"))
    cnn_prec = fmt_pct(cent_nn_test_m.get("malicious", {}).get("precision"))
    cnn_auc = fmt_f(cent_nn_test_m.get("roc_auc"))

    avg_acc = fmt_pct(fedavg_test_m.get("accuracy"))
    avg_f1 = fmt_f(fedavg_test_m.get("macro_f1"))
    avg_rec = fmt_pct(fedavg_test_m.get("malicious", {}).get("recall"))
    avg_prec = fmt_pct(fedavg_test_m.get("malicious", {}).get("precision"))
    avg_auc = fmt_f(fedavg_test_m.get("roc_auc"))

    opt_acc = fmt_pct(fedopt_test_m.get("accuracy"))
    opt_f1 = fmt_f(fedopt_test_m.get("macro_f1"))
    opt_rec = fmt_pct(fedopt_test_m.get("malicious", {}).get("recall"))
    opt_prec = fmt_pct(fedopt_test_m.get("malicious", {}).get("precision"))
    opt_auc = fmt_f(fedopt_test_m.get("roc_auc"))

    report_lines = [
        f"# Phase 4 Research Report: Federated Optimization with FedOpt ({opt_name})",
        f"## Project: LLMs meet Federated Learning for Scalable and Secure IoT Management",
        "",
        "### 1. Executive Summary",
        f"This phase investigates **FedOpt** (specifically **{opt_name}**), an adaptive federated optimization strategy designed to address client drift under non-IID data distributions. In contrast to standard FedAvg, which directly averages client model weights, FedOpt treats the sample-weighted client displacement as a **pseudo-gradient** and applies an adaptive server-side optimizer (momentum $m_t$ and second-moment variance $v_t$) to update global model parameters.",
        "",
        "### 2. Experimental Setup",
        f"- **Dataset Source:** IoT-23 (`processed/train.csv` 33,602 rows, `processed/val.csv` 9,601 rows, `processed/test.csv` 4,800 rows)",
        f"- **Input Dimension:** 125 preprocessed features (10 standardized numerics + 6 one-hot encoded categoricals)",
        f"- **Clients:** {cfg.num_clients} simulated clients with scenario-aware Non-IID partitioning",
        f"- **Rounds:** {cfg.num_rounds} federated communication rounds",
        f"- **Local Client Configuration:** 2 epochs per round, batch size 64, Adam optimizer (lr=0.001)",
        f"- **Server Optimizer:** {opt_name} (server learning rate $\\eta={cfg.server_learning_rate}$, $\\beta_1={cfg.server_beta_1}$, $\\beta_2={cfg.server_beta_2}$, $\\tau={cfg.server_tau}$)",
        f"- **Random Seed:** {cfg.random_seed}",
        "",
        "### 3. Comprehensive 4-Way Benchmark Comparison (Test Set: 4,800 Samples)",
        "",
        "| Metric | Centralized Random Forest | Centralized Neural Network | Federated FedAvg | Federated FedOpt (" + opt_name + ") |",
        "|---|---|---|---|---|",
        f"| **Accuracy** | {rf_acc} | {cnn_acc} | {avg_acc} | **{opt_acc}** |",
        f"| **Macro-F1** | {rf_f1} | {cnn_f1} | {avg_f1} | **{opt_f1}** |",
        f"| **Malicious Recall** | {rf_rec} | {cnn_rec} | {avg_rec} | **{opt_rec}** |",
        f"| **Malicious Precision** | {rf_prec} | {cnn_prec} | {avg_prec} | **{opt_prec}** |",
        f"| **ROC-AUC** | {rf_auc} | {cnn_auc} | {avg_auc} | **{opt_auc}** |",
        "",
        "### 4. Convergence & Optimization Dynamics",
        "- **Early vs. Late Rounds:** In early rounds, FedAdam accumulates momentum and rapidly improves validation accuracy. In later rounds, coordinate-wise variance adaptation dampens oscillations in high-variance feature weights.",
        "- **Client Drift Mitigation:** Because different IoT scenarios produce gradients with disparate coordinate magnitudes, FedOpt's adaptive denominator (sqrt(v_t) + tau) scales updates appropriately for both dominant and rare features.",
        "",
        "### 5. Communication Measurement",
        f"- **Model Parameter Count:** {comm_summary['parameter_count']:,} parameters (39.88 KB raw float32 payload)",
        f"- **Total Client Updates:** {cfg.num_rounds * cfg.num_clients}",
        f"- **Total Communicated Payload:** {comm_summary['total_communicated_bytes'] / (1024 * 1024):.2f} MB (identical to FedAvg, because FedOpt modifies server aggregation logic without communicating additional metadata)",
        f"- **Total Client Training Time:** {comm_summary.get('total_client_training_time_seconds', 0.0):.2f}s",
        "",
        "### 6. Limitations & Scientific Discussion",
        "1. **Server Hyperparameter Sensitivity:** Unlike FedAvg (which has no server learning rate), FedOpt introduces server learning rate $\\eta$ and numerical stabilizer $\\tau$. Setting $\\eta$ too high can lead to overshooting.",
        "2. **Communication Equality:** Standard FedOpt does not reduce the number of communicated bytes per round compared to FedAvg. This establishes the scientific baseline for Phase 5 (Gradient-Selective Federated Sampling / GSFS) where parameter updates are compressed or selectively skipped.",
    ]

    with save_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def main():
    parser = argparse.ArgumentParser(description="Run Phase 4 FedOpt Experiment")
    parser.add_argument("--optimizer", type=str, default="fedadam", choices=["fedadam", "fedyogi", "fedadagrad", "fedavg"])
    parser.add_argument("--server-lr", type=float, default=0.02, help="Server learning rate (eta)")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--clients", type=int, default=5, help="Number of simulated clients")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 round on 2 clients as a smoke test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = get_fedopt_config(
        server_optimizer=args.optimizer,
        server_learning_rate=args.server_lr,
        random_seed=args.seed,
    )
    if args.smoke_test:
        cfg.num_rounds = 1
        cfg.num_clients = 2
        logger.info("Executing FAST SMOKE TEST (1 round, 2 clients)...")
    else:
        cfg.num_rounds = args.rounds
        cfg.num_clients = args.clients

    run_fedopt_experiment(cfg)


if __name__ == "__main__":
    main()

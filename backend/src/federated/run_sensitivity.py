"""GSFS Sensitivity Analysis across thresholds, trigger modes, and beta values."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np
import torch

from src.detection.data_loader import (
    load_dataset_split,
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_NUMERIC_FEATURES,
    DEFAULT_CATEGORICAL_FEATURES,
    LABEL_MAPPING,
)
from src.detection.pipeline import build_preprocessor
from src.federated.config import get_gsfs_config
from src.federated.partitioner import partition_train_data
from src.federated.model import IoTThreatClassifier
from src.federated.client import IoTFlowerClient
from src.federated.server import FederatedServerCoordinator
from src.federated.gsfs_strategy import GSFSStrategy
from src.federated.evaluate import evaluate_torch_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("GSFSSensitivity")


def run_sensitivity_experiments():
    """Run controlled sensitivity analysis for GSFS hyperparameters."""
    base_cfg = get_gsfs_config()

    # Load data once
    raw_train_df = pd.read_csv(base_cfg.train_path)
    X_train_raw, y_train, _ = load_dataset_split(base_cfg.train_path, base_cfg.metadata_path)
    X_val_raw, y_val, _ = load_dataset_split(base_cfg.val_path, base_cfg.metadata_path)
    X_test_raw, y_test, _ = load_dataset_split(base_cfg.test_path, base_cfg.metadata_path)

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

    client_dfs = partition_train_data(
        train_df=raw_train_df,
        num_clients=base_cfg.num_clients,
        strategy=base_cfg.partition_strategy,
        random_seed=base_cfg.random_seed,
    )

    configs_to_test = [
        {"name": "GSFS (Threshold 40%, OR, Beta 0.0)", "threshold": 0.40, "mode": "OR", "beta": 0.0, "perf_thresh": 0.0},
        {"name": "GSFS (Threshold 60%, OR, Beta 0.0) [Baseline]", "threshold": 0.60, "mode": "OR", "beta": 0.0, "perf_thresh": 0.0},
        {"name": "GSFS (Threshold 80%, OR, Beta 0.0)", "threshold": 0.80, "mode": "OR", "beta": 0.0, "perf_thresh": 0.0},
        {"name": "GSFS (Threshold 60%, AND, Beta 0.0)", "threshold": 0.60, "mode": "AND", "beta": 0.0, "perf_thresh": 0.0},
        {"name": "GSFS (Threshold 60%, OR, Beta +0.5)", "threshold": 0.60, "mode": "OR", "beta": 0.5, "perf_thresh": 0.0},
        {"name": "GSFS (Threshold 60%, OR, Strict Perf 0.01)", "threshold": 0.60, "mode": "OR", "beta": 0.0, "perf_thresh": 0.01},
    ]

    sensitivity_results = []

    for item in configs_to_test:
        logger.info(f"--- Running Sensitivity Test: {item['name']} ---")
        cfg = get_gsfs_config(
            aggregation_threshold=item["threshold"],
            trigger_mode=item["mode"],
            grad_beta=item["beta"],
            perf_threshold=item["perf_thresh"],
            num_rounds=10,
        )

        # Fresh clients
        flower_clients = {}
        for cid in range(cfg.num_clients):
            c_df = client_dfs[cid]
            c_X = preprocessor.transform(c_df[DEFAULT_FEATURE_COLUMNS])
            c_y = c_df["label"].str.lower().str.strip().map(LABEL_MAPPING).to_numpy()
            c_model = IoTThreatClassifier(input_dim=input_dim, hidden_dims=cfg.hidden_dims, dropout_rate=cfg.dropout_rate)
            flower_clients[cid] = IoTFlowerClient(client_id=cid, model=c_model, X_train=c_X, y_train=c_y, local_epochs=cfg.local_epochs, batch_size=cfg.batch_size, learning_rate=cfg.learning_rate)

        server = FederatedServerCoordinator(
            config=cfg,
            clients=flower_clients,
            val_X=X_val,
            val_y=y_val.to_numpy(),
            input_dim=input_dim,
        )

        model, history, comm = server.run_federated_rounds()
        strat: GSFSStrategy = server.strategy

        val_m = evaluate_torch_model(model, X_val, y_val.to_numpy())
        test_m = evaluate_torch_model(model, X_test, y_test.to_numpy())

        sel_df = pd.DataFrame(strat.client_selection_history)
        tot_gen = len(sel_df)
        tot_sel = int(sel_df["combined_triggered"].sum()) if not sel_df.empty else 0
        tot_rej = tot_gen - tot_sel

        rec = {
            "name": item["name"],
            "aggregation_threshold": item["threshold"],
            "trigger_mode": item["mode"],
            "grad_beta": item["beta"],
            "perf_threshold": item["perf_thresh"],
            "total_updates_generated": tot_gen,
            "total_updates_selected": tot_sel,
            "total_updates_rejected": tot_rej,
            "selection_rate_pct": round((tot_sel / max(1, tot_gen)) * 100, 2),
            "aggregation_events": len(strat.aggregation_history),
            "val_accuracy": round(val_m["accuracy"], 4),
            "val_macro_f1": round(val_m["macro_f1"], 4),
            "val_loss": round(val_m["loss"], 4),
            "test_accuracy": round(test_m["accuracy"], 4),
            "test_macro_f1": round(test_m["macro_f1"], 4),
            "test_malicious_recall": round(test_m["malicious"]["recall"], 4),
            "test_roc_auc": round(test_m.get("roc_auc", 0.0), 4),
        }
        sensitivity_results.append(rec)
        logger.info(f"Result for {item['name']}: Val Acc={rec['val_accuracy']*100:.2f}%, Val F1={rec['val_macro_f1']:.4f}, Agg Events={rec['aggregation_events']}/10")

    # Save sensitivity results
    out_path = Path("experiments/federated/gsfs/sensitivity_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sensitivity_results, f, indent=2)
    logger.info(f"Saved sensitivity analysis to {out_path}")

    # Append sensitivity table to report.md
    report_md_path = Path("experiments/federated/gsfs/report.md")
    if report_md_path.exists():
        with open(report_md_path, "r", encoding="utf-8") as f:
            content = f.read()

        table_md = "\n\n### Hyperparameter Sensitivity Analysis (Controlled Experiments)\n\n"
        table_md += "| Configuration | Thresh (%) | Mode | Beta | Perf Thresh | Selected / Gen | Selection % | Agg Events | Val Macro-F1 | Test Acc | Test Recall | Test ROC-AUC |\n"
        table_md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
        for r in sensitivity_results:
            table_md += f"| **{r['name']}** | {int(r['aggregation_threshold']*100)}% | {r['trigger_mode']} | {r['grad_beta']} | {r['perf_threshold']} | {r['total_updates_selected']}/{r['total_updates_generated']} | {r['selection_rate_pct']}% | {r['aggregation_events']}/10 | {r['val_macro_f1']:.4f} | {r['test_accuracy']*100:.2f}% | {r['test_malicious_recall']*100:.2f}% | {r['test_roc_auc']:.4f} |\n"

        if "### Hyperparameter Sensitivity Analysis" not in content:
            content = content.replace("### Reproducibility Commands", table_md + "\n---\n\n### Reproducibility Commands")
            with open(report_md_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("Appended sensitivity table to report.md.")


if __name__ == "__main__":
    run_sensitivity_experiments()

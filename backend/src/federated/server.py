"""Federated Server coordinator executing FedAvg across simulated IoT clients."""

import logging
import time
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import torch

from flwr.common import (
    Parameters,
    FitRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy import Strategy

from src.federated.model import (
    IoTThreatClassifier,
    get_model_parameters,
    set_model_parameters,
    count_parameters,
    estimate_parameter_size_bytes,
)
from src.federated.client import IoTFlowerClient
from src.federated.strategy import (
    aggregate_fedavg,
    create_strategy,
    LoggingFedAvg,
)
from src.federated.evaluate import evaluate_torch_model
from src.federated.config import FederatedConfig

logger = logging.getLogger("FederatedServer")


class FederatedServerCoordinator:
    """
    Coordinates simulated federated learning rounds with Flower clients using
    configurable aggregation strategies (FedAvg, FedAdam, FedYogi, FedAdagrad).
    """

    def __init__(
        self,
        config: FederatedConfig,
        clients: Dict[int, IoTFlowerClient],
        val_X: np.ndarray,
        val_y: np.ndarray,
        input_dim: int = 125,
        strategy: Optional[Strategy] = None,
    ):
        self.config = config
        self.clients = clients
        self.val_X = val_X
        self.val_y = val_y
        self.input_dim = input_dim

        # Fix PyTorch seed for global model initialization
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

        self.global_model = IoTThreatClassifier(
            input_dim=input_dim,
            hidden_dims=config.hidden_dims,
            dropout_rate=config.dropout_rate,
        )
        self.global_model.to(torch.device(config.device))

        initial_ndarrays = get_model_parameters(self.global_model)
        initial_params = ndarrays_to_parameters(initial_ndarrays)

        # Initialize strategy
        if strategy is not None:
            self.strategy = strategy
        else:
            self.strategy = create_strategy(
                strategy_name=config.server_optimizer,
                initial_parameters=initial_params,
                server_learning_rate=config.server_learning_rate,
                beta_1=config.server_beta_1,
                beta_2=config.server_beta_2,
                tau=config.server_tau,
                accept_failures=config.accept_failures,
                aggregation_threshold=getattr(config, "gsfs_aggregation_threshold", 0.60),
                perf_threshold=getattr(config, "gsfs_perf_threshold", 0.0),
                grad_beta=getattr(config, "gsfs_grad_beta", 0.0),
                trigger_mode=getattr(config, "gsfs_trigger_mode", "OR"),
                warmup_rounds=getattr(config, "gsfs_warmup_rounds", 1),
                max_staleness=getattr(config, "gsfs_max_staleness", 2),
                max_pool_size=getattr(config, "gsfs_max_pool_size", 10),
            )

        self.num_params = count_parameters(self.global_model)
        self.round_history: List[Dict[str, Any]] = []
        self.best_val_macro_f1 = -1.0
        self.best_val_loss = float("inf")
        self.best_model_parameters: Optional[List[np.ndarray]] = None

    def run_federated_rounds(self) -> Tuple[IoTThreatClassifier, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute configured federated learning rounds.

        Returns:
            global_model: Best performing global model.
            round_history: Metrics per round.
            communication_summary: Total communication payload statistics.
        """
        num_rounds = self.config.num_rounds
        num_clients = len(self.clients)
        current_parameters = get_model_parameters(self.global_model)
        model_payload_bytes = estimate_parameter_size_bytes(current_parameters)

        logger.info(
            f"Starting FedAvg simulation: {num_clients} clients, {num_rounds} rounds, "
            f"{self.num_params:,} parameters ({model_payload_bytes/1024:.1f} KB per model)."
        )

        total_upload_bytes = 0
        total_download_bytes = 0
        total_client_training_time = 0.0

        for round_idx in range(1, num_rounds + 1):
            round_t0 = time.time()
            logger.info(f"--- Federated Round [{round_idx}/{num_rounds}] ---")

            # 1. Client selection (fraction_fit)
            participating_ids = sorted(list(self.clients.keys()))
            num_selected = len(participating_ids)

            # Communication: Server broadcasts global model parameters to participating clients
            download_bytes_round = model_payload_bytes * num_selected
            total_download_bytes += download_bytes_round

            # 2. Local client optimization
            fit_results: List[Tuple[List[np.ndarray], int]] = []
            round_client_metrics: List[Dict[str, Any]] = []

            for cid in participating_ids:
                client = self.clients[cid]
                fit_config = {
                    "local_epochs": self.config.local_epochs,
                    "batch_size": self.config.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "round": round_idx,
                }
                c_t0 = time.time()
                updated_params, num_samples, metrics = client.fit(current_parameters, fit_config)
                c_dur = time.time() - c_t0
                total_client_training_time += c_dur

                fit_results.append((updated_params, num_samples))
                round_client_metrics.append(metrics)

            # Communication: Clients upload their trained updates to server
            upload_bytes_round = model_payload_bytes * len(fit_results)
            total_upload_bytes += upload_bytes_round

            # 3. Server aggregation using configured strategy (FedAvg or FedOpt: FedAdam/FedYogi/FedAdagrad)
            flower_results = [
                (
                    None,
                    FitRes(
                        status=Status(Code.OK, ""),
                        parameters=ndarrays_to_parameters(params),
                        num_examples=num_samples,
                        metrics=client_metrics,
                    ),
                )
                for (params, num_samples), client_metrics in zip(fit_results, round_client_metrics)
            ]

            aggregated_params, strat_metrics = self.strategy.aggregate_fit(
                server_round=round_idx,
                results=flower_results,
                failures=[],
            )
            if aggregated_params is not None:
                current_parameters = parameters_to_ndarrays(aggregated_params)
            else:
                # Fallback to direct weighted average if strategy returned None
                current_parameters = aggregate_fedavg(fit_results)

            set_model_parameters(self.global_model, current_parameters)

            round_duration = time.time() - round_t0

            # 4. Global Validation Evaluation
            val_metrics = evaluate_torch_model(
                self.global_model,
                self.val_X,
                self.val_y,
                batch_size=256,
                device=self.config.device,
            )

            round_rec = {
                "round": round_idx,
                "participating_clients": num_selected,
                "total_examples_trained": sum(n for _, n in fit_results),
                "round_duration_seconds": round(round_duration, 3),
                "val_loss": round(val_metrics["loss"], 4),
                "val_accuracy": round(val_metrics["accuracy"], 4),
                "val_macro_f1": round(val_metrics["macro_f1"], 4),
                "val_malicious_recall": round(val_metrics["malicious"]["recall"], 4),
                "val_malicious_precision": round(val_metrics["malicious"]["precision"], 4),
                "val_malicious_f1": round(val_metrics["malicious"]["f1_score"], 4),
                "upload_bytes": upload_bytes_round,
                "download_bytes": download_bytes_round,
            }
            self.round_history.append(round_rec)

            logger.info(
                f"Round {round_idx} Complete in {round_duration:.2f}s | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']*100:.2f}% | "
                f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | "
                f"Malicious Rec: {val_metrics['malicious']['recall']*100:.2f}%"
            )

            # Track best model parameters
            if val_metrics["macro_f1"] > self.best_val_macro_f1:
                self.best_val_macro_f1 = val_metrics["macro_f1"]
                self.best_val_loss = val_metrics["loss"]
                self.best_model_parameters = [p.copy() for p in current_parameters]

        # Set global model to best performing checkpoint
        if self.best_model_parameters is not None:
            set_model_parameters(self.global_model, self.best_model_parameters)

        communication_summary = {
            "num_rounds": num_rounds,
            "total_clients": num_clients,
            "parameter_count": self.num_params,
            "model_size_bytes": model_payload_bytes,
            "total_upload_bytes": total_upload_bytes,
            "total_download_bytes": total_download_bytes,
            "total_communicated_bytes": total_upload_bytes + total_download_bytes,
            "total_client_training_time_seconds": round(total_client_training_time, 2),
        }

        return self.global_model, self.round_history, communication_summary

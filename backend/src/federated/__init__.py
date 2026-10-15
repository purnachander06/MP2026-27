"""Federated Learning (FedAvg) package for IoT-23 dataset."""

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
from src.federated.strategy import aggregate_fedavg, LoggingFedAvg
from src.federated.server import FederatedServerCoordinator
from src.federated.evaluate import evaluate_torch_model, plot_federated_round_metrics

from src.federated.performance_trigger import PerformanceTrigger
from src.federated.gradient_statistics import (
    compute_param_deltas,
    compute_gradient_norm,
    GradientStatisticsTracker,
)
from src.federated.update_pool import ClientUpdate, UpdatePool
from src.federated.gsfs_strategy import (
    combine_triggers,
    calculate_required_updates,
    aggregate_gsfs_deltas,
    GSFSStrategy,
)

__all__ = [
    "FederatedConfig",
    "partition_train_data",
    "get_partition_summary",
    "save_partition_metadata",
    "IoTThreatClassifier",
    "get_model_parameters",
    "set_model_parameters",
    "count_parameters",
    "estimate_parameter_size_bytes",
    "IoTFlowerClient",
    "aggregate_fedavg",
    "LoggingFedAvg",
    "FederatedServerCoordinator",
    "evaluate_torch_model",
    "plot_federated_round_metrics",
    "PerformanceTrigger",
    "compute_param_deltas",
    "compute_gradient_norm",
    "GradientStatisticsTracker",
    "ClientUpdate",
    "UpdatePool",
    "combine_triggers",
    "calculate_required_updates",
    "aggregate_gsfs_deltas",
    "GSFSStrategy",
]

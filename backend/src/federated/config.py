"""Configuration dataclass for Federated Learning (FedAvg) experiment."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional


@dataclass
class FederatedConfig:
    """Experiment and model configuration for FedAvg simulation."""

    # Federated setup
    num_clients: int = 5
    num_rounds: int = 10
    fraction_fit: float = 1.0  # 1.0 means all 5 clients participate in every round
    local_epochs: int = 2      # Local training epochs per round
    batch_size: int = 64
    learning_rate: float = 0.001
    random_seed: int = 42

    # Model architecture
    hidden_dims: Tuple[int, ...] = (64, 32)
    dropout_rate: float = 0.15

    # Partitioning
    partition_strategy: str = "scenario_grouped"  # 'scenario_grouped' (default) or 'dirichlet'
    dirichlet_alpha: float = 0.5

    # Server Optimizer (FedOpt) settings
    server_optimizer: str = "fedavg"  # "fedavg", "fedadam", "fedyogi", "fedadagrad"
    server_learning_rate: float = 0.02  # eta: server-side learning rate
    server_beta_1: float = 0.9          # beta_1: momentum factor for 1st moment
    server_beta_2: float = 0.99         # beta_2: decay factor for 2nd moment
    server_tau: float = 1e-3            # tau: numerical stabilizer
    accept_failures: bool = True        # whether to accept rounds with partial client failures

    # GSFS settings
    gsfs_aggregation_threshold: float = 0.60  # e.g., 0.60 * 5 = 3 clients
    gsfs_perf_threshold: float = 0.0          # Performance improvement threshold
    gsfs_grad_beta: float = 0.0               # Gradient statistic multiplier (mean + beta * std)
    gsfs_trigger_mode: str = "OR"             # "OR" or "AND" combination mode
    gsfs_warmup_rounds: int = 1               # Number of warm-up rounds before filtering
    gsfs_max_staleness: int = 2               # Max rounds an update can remain in pool
    gsfs_max_pool_size: int = 10              # Maximum capacity of the update pool

    # File paths
    train_path: Path = Path("processed/train.csv")
    val_path: Path = Path("processed/val.csv")
    test_path: Path = Path("processed/test.csv")
    metadata_path: Path = Path("processed/feature_metadata.json")
    models_dir: Path = Path("models/federated")
    experiments_dir: Path = Path("experiments/federated/fedavg")

    # Execution device
    device: str = "cpu"

    def __post_init__(self):
        # Automatically resolve relative paths when run from subdirectories
        if not self.train_path.exists():
            for cand in [Path("../processed/train.csv"), Path("../../processed/train.csv")]:
                if cand.exists():
                    self.train_path = cand
                    self.val_path = cand.parent / "val.csv"
                    self.test_path = cand.parent / "test.csv"
                    self.metadata_path = cand.parent / "feature_metadata.json"
                    break
        if not self.models_dir.exists():
            for cand in [Path("../models/federated"), Path("../../backend/models/federated")]:
                if cand.exists():
                    self.models_dir = cand
                    break


def get_fedopt_config(
    server_optimizer: str = "fedadam",
    server_learning_rate: float = 0.02,
    server_beta_1: float = 0.9,
    server_beta_2: float = 0.99,
    server_tau: float = 1e-3,
    **kwargs,
) -> FederatedConfig:
    """
    Factory function returning a configured FederatedConfig instance for FedOpt.
    """
    opt_name = server_optimizer.lower().strip()
    return FederatedConfig(
        server_optimizer=opt_name,
        server_learning_rate=server_learning_rate,
        server_beta_1=server_beta_1,
        server_beta_2=server_beta_2,
        server_tau=server_tau,
        experiments_dir=Path("experiments/federated/fedopt"),
        **kwargs,
    )


def get_gsfs_config(
    aggregation_threshold: float = 0.60,
    perf_threshold: float = 0.0,
    grad_beta: float = 0.0,
    trigger_mode: str = "OR",
    warmup_rounds: int = 1,
    max_staleness: int = 2,
    max_pool_size: int = 10,
    **kwargs,
) -> FederatedConfig:
    """
    Factory function returning a configured FederatedConfig instance for GSFS (Phase 5).
    """
    return FederatedConfig(
        server_optimizer="gsfs",
        gsfs_aggregation_threshold=aggregation_threshold,
        gsfs_perf_threshold=perf_threshold,
        gsfs_grad_beta=grad_beta,
        gsfs_trigger_mode=trigger_mode.upper().strip(),
        gsfs_warmup_rounds=warmup_rounds,
        gsfs_max_staleness=max_staleness,
        gsfs_max_pool_size=max_pool_size,
        experiments_dir=Path("experiments/federated/gsfs"),
        **kwargs,
    )


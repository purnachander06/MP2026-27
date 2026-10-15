"""Non-IID client partitioning for Federated Learning on IoT-23.

Features:
- Deterministic, reproducible partitioning with configurable random seed.
- Defensible scenario-aware partitioning (scenario_grouped):
  Assigns intact capture scenarios to clients via balanced bin-packing.
  Each client represents a distinct IoT network deployment with heterogeneous,
  non-IID traffic and attack characteristics.
- Alternative Dirichlet label-skew partitioning.
- Strict data integrity:
  1. Every training record belongs to exactly one client.
  2. No duplicate record assignments.
  3. Zero record loss (sum of client records == total training records).
  4. Validation and test sets remain strictly outside client partitions.
- Full reporting of per-client sample counts, class distributions, and scenario composition.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def partition_scenario_grouped(
    df: pd.DataFrame,
    num_clients: int = 5,
    scenario_col: str = "scenario",
    random_seed: int = 42,
) -> Dict[int, pd.DataFrame]:
    """
    Scenario-aware Non-IID partitioning: assigns whole scenarios to clients.

    Uses a seeded greedy bin-packing heuristic to distribute scenarios across clients
    so that sample counts are balanced while preserving intact scenario environments.
    """
    if scenario_col not in df.columns:
        raise ValueError(f"Scenario column '{scenario_col}' not found in DataFrame.")

    # Get sample count per scenario
    sc_counts = df[scenario_col].value_counts().to_dict()

    # Deterministic shuffling with seed before sorting for tie-breaking
    rng = np.random.RandomState(random_seed)
    sc_items = list(sc_counts.items())
    rng.shuffle(sc_items)
    # Sort descending by sample count
    sc_items.sort(key=lambda x: x[1], reverse=True)

    client_scenarios: Dict[int, List[str]] = {i: [] for i in range(num_clients)}
    client_totals: Dict[int, int] = {i: 0 for i in range(num_clients)}

    # Greedy assignment: assign largest remaining scenario to client with smallest total
    for sc_name, count in sc_items:
        min_client = min(client_totals.keys(), key=lambda c: client_totals[c])
        client_scenarios[min_client].append(sc_name)
        client_totals[min_client] += count

    # Build client DataFrames
    client_dfs: Dict[int, pd.DataFrame] = {}
    for cid in range(num_clients):
        assigned_sc = client_scenarios[cid]
        c_df = df[df[scenario_col].isin(assigned_sc)].copy()
        # Shuffle client data locally for training
        c_df = c_df.sample(frac=1.0, random_state=random_seed + cid).reset_index(drop=True)
        client_dfs[cid] = c_df

    return client_dfs


def partition_dirichlet(
    df: pd.DataFrame,
    num_clients: int = 5,
    alpha: float = 0.5,
    label_col: str = "label",
    random_seed: int = 42,
) -> Dict[int, pd.DataFrame]:
    """
    Dirichlet-based label-skew non-IID partitioning.
    Distributes records of each class according to Dirichlet(alpha).
    """
    rng = np.random.RandomState(random_seed)
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    unique_labels = df[label_col].unique()

    for lbl in unique_labels:
        lbl_indices = df[df[label_col] == lbl].index.to_numpy()
        rng.shuffle(lbl_indices)

        # Sample proportions from Dirichlet distribution
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Normalize and convert to split sizes
        split_sizes = (proportions * len(lbl_indices)).astype(int)
        # Fix rounding difference
        diff = len(lbl_indices) - split_sizes.sum()
        split_sizes[rng.choice(num_clients, size=diff, replace=True)] += 1

        start_idx = 0
        for cid, size in enumerate(split_sizes):
            client_indices[cid].extend(lbl_indices[start_idx : start_idx + size])
            start_idx += size

    client_dfs: Dict[int, pd.DataFrame] = {}
    for cid in range(num_clients):
        c_df = df.loc[client_indices[cid]].copy()
        c_df = c_df.sample(frac=1.0, random_state=random_seed + cid).reset_index(drop=True)
        client_dfs[cid] = c_df

    return client_dfs


def partition_train_data(
    train_df: pd.DataFrame,
    num_clients: int = 5,
    strategy: str = "scenario_grouped",
    random_seed: int = 42,
    alpha: float = 0.5,
) -> Dict[int, pd.DataFrame]:
    """
    Master entrypoint to partition training data among federated clients.
    """
    strategy_clean = strategy.lower().strip()
    if strategy_clean in ("scenario_grouped", "scenario"):
        client_dfs = partition_scenario_grouped(
            df=train_df,
            num_clients=num_clients,
            scenario_col="scenario",
            random_seed=random_seed,
        )
    elif strategy_clean in ("dirichlet", "label_skew"):
        client_dfs = partition_dirichlet(
            df=train_df,
            num_clients=num_clients,
            alpha=alpha,
            label_col="label",
            random_seed=random_seed,
        )
    else:
        raise ValueError(f"Unknown partitioning strategy '{strategy}'. Supported: 'scenario_grouped', 'dirichlet'.")

    # Verify partition integrity
    total_partitioned = sum(len(df) for df in client_dfs.values())
    if total_partitioned != len(train_df):
        raise ValueError(
            f"Partition integrity check failed: Total partitioned rows ({total_partitioned}) != Train rows ({len(train_df)})"
        )

    return client_dfs


def get_partition_summary(client_dfs: Dict[int, pd.DataFrame]) -> Dict[str, Any]:
    """
    Calculate detailed statistics for each client partition.
    """
    summary: Dict[str, Any] = {}
    total_records = 0

    for cid, c_df in client_dfs.items():
        n_samples = len(c_df)
        total_records += n_samples

        labels = c_df["label"].str.lower().str.strip()
        n_benign = int((labels == "benign").sum())
        n_malicious = int((labels == "malicious").sum())

        benign_ratio = round(n_benign / n_samples, 4) if n_samples > 0 else 0.0
        malicious_ratio = round(n_malicious / n_samples, 4) if n_samples > 0 else 0.0

        scenarios_present = sorted(c_df["scenario"].unique().tolist()) if "scenario" in c_df.columns else []

        summary[f"client_{cid}"] = {
            "client_id": cid,
            "total_samples": n_samples,
            "benign_samples": n_benign,
            "malicious_samples": n_malicious,
            "benign_ratio": benign_ratio,
            "malicious_ratio": malicious_ratio,
            "scenarios_count": len(scenarios_present),
            "scenarios": scenarios_present,
        }

    return {
        "num_clients": len(client_dfs),
        "total_records": total_records,
        "clients": summary,
    }


def save_partition_metadata(summary: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """Save client partition statistics to JSON."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved client partition metadata to {out_p}")

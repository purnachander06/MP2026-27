"""Gradient-Selective Federated Sampling (GSFS) Strategy for Flower.

Implements a four-component federated client selection and aggregation strategy:
1. Performance-Change Trigger (evaluates local loss reduction).
2. Gradient-Statistics Trigger (evaluates parameter displacement norm vs moving stats).
3. Update Pool (manages buffering, deduplication, and staleness).
4. Aggregation Threshold (triggers sample-weighted delta aggregation at threshold, e.g. 60%).
"""

import math
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
import flwr as fl
from flwr.common import (
    Parameters,
    FitRes,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from src.federated.model import estimate_parameter_size_bytes
from src.federated.performance_trigger import PerformanceTrigger
from src.federated.gradient_statistics import (
    compute_param_deltas,
    compute_gradient_norm,
    GradientStatisticsTracker,
)
from src.federated.update_pool import ClientUpdate, UpdatePool

import logging
logger = logging.getLogger("GSFS.Strategy")


def combine_triggers(perf_triggered: bool, grad_triggered: bool, mode: str = "OR") -> bool:
    """
    Combine performance and gradient triggers using explicit logical rule.

    Args:
        perf_triggered: Whether performance improvement threshold was met.
        grad_triggered: Whether gradient statistical threshold was met.
        mode: Logical combination operator ('OR' or 'AND').

    Returns:
        Boolean indicating whether the client update is selected.
    """
    m = mode.upper().strip()
    if m == "OR":
        return bool(perf_triggered or grad_triggered)
    elif m == "AND":
        return bool(perf_triggered and grad_triggered)
    else:
        raise ValueError(f"Unsupported trigger combination mode '{mode}'. Choose 'OR' or 'AND'.")


def calculate_required_updates(aggregation_threshold: float, eligible_client_count: int) -> int:
    """
    Calculate minimum updates required to trigger server aggregation:
        required_updates = ceil(aggregation_threshold * eligible_client_count)

    Args:
        aggregation_threshold: Fraction of clients required (e.g., 0.60 for 60%).
        eligible_client_count: Total participating / eligible clients in round.

    Returns:
        Integer count of required updates (minimum 1).
    """
    if eligible_client_count <= 0:
        return 0
    thresh = float(aggregation_threshold)
    if thresh <= 0.0:
        return 1
    return max(1, math.ceil(thresh * eligible_client_count))


def aggregate_gsfs_deltas(
    current_parameters: List[np.ndarray],
    updates: List[ClientUpdate],
) -> List[np.ndarray]:
    """
    Apply sample-weighted parameter deltas to current global model parameters:
        theta_new = theta_global + sum_{k} (n_k / sum_j n_j) * Delta_k

    Args:
        current_parameters: Active global model weights.
        updates: List of eligible ClientUpdate instances.

    Returns:
        Updated global parameters as List of NumPy arrays matching original float32 dtype.
    """
    if not updates:
        raise ValueError("Cannot aggregate empty updates in GSFS.")

    total_samples = sum(u.num_samples for u in updates)
    if total_samples <= 0:
        raise ValueError("Total sample count across aggregated client updates must be positive.")

    # Initialize zero accumulators for weighted deltas
    first_deltas = updates[0].parameter_deltas
    weighted_deltas = [np.zeros_like(d, dtype=np.float64) for d in first_deltas]

    for u in updates:
        weight = float(u.num_samples) / float(total_samples)
        for i, delta in enumerate(u.parameter_deltas):
            if delta.shape != current_parameters[i].shape:
                raise ValueError(
                    f"Delta shape mismatch at layer {i}: delta {delta.shape} vs param {current_parameters[i].shape}"
                )
            weighted_deltas[i] += delta.astype(np.float64) * weight

    # Add weighted deltas to current global parameters
    new_parameters = []
    for i, curr_p in enumerate(current_parameters):
        updated_layer = curr_p.astype(np.float64) + weighted_deltas[i]
        new_parameters.append(updated_layer.astype(curr_p.dtype))

    return new_parameters


class GSFSStrategy(FedAvg):
    """
    Flower Strategy implementing Gradient-Selective Federated Sampling (GSFS).
    """

    def __init__(
        self,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 2,
        min_available_clients: int = 2,
        aggregation_threshold: float = 0.60,
        perf_threshold: float = 0.0,
        grad_beta: float = 0.0,
        trigger_mode: str = "OR",
        warmup_rounds: int = 1,
        max_staleness: int = 2,
        max_pool_size: int = 10,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        evaluate_fn: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            evaluate_fn=evaluate_fn,
            accept_failures=accept_failures,
            **kwargs,
        )
        self.aggregation_threshold = float(aggregation_threshold)
        self.perf_threshold = float(perf_threshold)
        self.grad_beta = float(grad_beta)
        self.trigger_mode = trigger_mode.upper().strip()
        self.warmup_rounds = max(0, int(warmup_rounds))
        self.max_staleness = max(1, int(max_staleness))
        self.max_pool_size = max(1, int(max_pool_size))

        # Core GSFS Modules
        self.performance_trigger = PerformanceTrigger(threshold=self.perf_threshold)
        self.gradient_tracker = GradientStatisticsTracker(beta=self.grad_beta)
        self.update_pool = UpdatePool(max_pool_size=self.max_pool_size, max_staleness=self.max_staleness)

        # Global Model State Tracking
        if initial_parameters is not None:
            self.current_parameters: List[np.ndarray] = [
                p.astype(np.float32) for p in parameters_to_ndarrays(initial_parameters)
            ]
        else:
            self.current_parameters: List[np.ndarray] = []
        self.current_model_version: int = 0

        # Histories for audit and reporting
        self.round_history: List[Dict[str, Any]] = []
        self.client_selection_history: List[Dict[str, Any]] = []
        self.aggregation_history: List[Dict[str, Any]] = []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Evaluate client updates via GSFS triggers, manage the update pool,
        and perform sample-weighted delta aggregation when the aggregation threshold is met.
        """
        if not results:
            if not self.accept_failures and failures:
                logger.error(f"GSFS Round {server_round}: All client fits failed or returned empty.")
                return None, {}
            # Return current global parameters unchanged
            return ndarrays_to_parameters(self.current_parameters), {}

        # If current parameters not set, initialize from first result
        if not self.current_parameters:
            first_params = parameters_to_ndarrays(results[0][1].parameters)
            self.current_parameters = [p.astype(np.float32) for p in first_params]

        num_eligible_clients = len(results)
        round_selected_count = 0
        round_rejected_count = 0
        accepted_grad_norms = []
        accepted_client_losses = []

        # 1. Process and evaluate each client's update
        for client_proxy, fit_res in results:
            local_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            local_ndarrays = [p.astype(np.float32) for p in local_ndarrays]
            num_samples = fit_res.num_examples
            metrics = fit_res.metrics or {}

            cid = int(metrics.get("client_id", id(client_proxy) % 10000))
            init_loss = metrics.get("initial_loss", None)
            final_loss = metrics.get("final_loss", metrics.get("train_loss", 0.5))

            # Validate finite values to prevent poisoned/non-finite model updates
            if not all(np.all(np.isfinite(p)) for p in local_ndarrays):
                logger.warning(f"Client {cid} submitted non-finite (NaN/Inf) parameters; rejecting update.")
                round_rejected_count += 1
                continue

            # Compute deltas and gradient norm
            try:
                deltas = compute_param_deltas(local_ndarrays, self.current_parameters)
                grad_norm = compute_gradient_norm(deltas=deltas)
            except Exception as e:
                logger.error(f"Failed computing deltas for Client {cid}: {e}")
                round_rejected_count += 1
                continue

            # Evaluate Performance Trigger
            perf_trig, perf_imp, perf_reason = self.performance_trigger.evaluate(
                client_id=cid,
                initial_loss=init_loss,
                final_loss=final_loss,
                round_idx=server_round,
            )

            # Evaluate Gradient-Statistics Trigger
            grad_trig, grad_thresh, grad_reason = self.gradient_tracker.evaluate(grad_norm)

            # Combine triggers with Warm-Up policy
            if server_round <= self.warmup_rounds:
                combined_triggered = True
                selection_reason = f"warmup_round_{server_round}"
            else:
                combined_triggered = combine_triggers(perf_trig, grad_trig, mode=self.trigger_mode)
                selection_reason = (
                    f"{self.trigger_mode}_accepted(perf={perf_trig},grad={grad_trig})"
                    if combined_triggered
                    else f"{self.trigger_mode}_rejected(perf={perf_trig},grad={grad_trig})"
                )

            # Record selection log
            selection_record = {
                "round": server_round,
                "client_id": cid,
                "num_samples": num_samples,
                "initial_loss": round(float(init_loss), 4) if init_loss is not None else None,
                "final_loss": round(float(final_loss), 4) if final_loss is not None else None,
                "loss_improvement": round(float(perf_imp), 4),
                "perf_triggered": perf_trig,
                "perf_reason": perf_reason,
                "grad_norm": round(float(grad_norm), 6) if np.isfinite(grad_norm) else None,
                "grad_threshold": round(float(grad_thresh), 6),
                "grad_triggered": grad_trig,
                "grad_reason": grad_reason,
                "combined_triggered": combined_triggered,
                "selection_reason": selection_reason,
            }
            self.client_selection_history.append(selection_record)

            if combined_triggered:
                client_update = ClientUpdate(
                    client_id=cid,
                    round=server_round,
                    model_version=self.current_model_version,
                    parameter_deltas=deltas,
                    num_samples=num_samples,
                    perf_improvement=perf_imp,
                    grad_norm=grad_norm,
                    trigger_decisions={
                        "perf": perf_trig,
                        "grad": grad_trig,
                        "combined": combined_triggered,
                    },
                    selection_reason=selection_reason,
                )
                accepted, pool_reason = self.update_pool.add_update(client_update)
                if accepted:
                    round_selected_count += 1
                    accepted_grad_norms.append(grad_norm)
                    if final_loss is not None:
                        accepted_client_losses.append((cid, final_loss))
                else:
                    round_rejected_count += 1
            else:
                round_rejected_count += 1

        # Synchronize fleet statistics at round boundary
        for g_norm in accepted_grad_norms:
            self.gradient_tracker.update_statistics(g_norm)
        for c_id, f_loss in accepted_client_losses:
            self.performance_trigger.record_client_loss(c_id, f_loss)

        # 2. Prune stale updates from pool
        pruned_count = self.update_pool.prune_stale(current_round=server_round)

        # 3. Check Aggregation Threshold
        required_updates = calculate_required_updates(
            self.aggregation_threshold,
            num_eligible_clients,
        )
        eligible_updates = self.update_pool.get_eligible_updates(
            current_model_version=self.current_model_version,
            current_round=server_round,
        )

        aggregated = False
        updates_aggregated_count = 0
        samples_aggregated_count = 0

        if len(eligible_updates) >= required_updates:
            # Threshold met: perform sample-weighted delta aggregation
            logger.info(
                f"GSFS Round {server_round}: Aggregation Threshold MET ({len(eligible_updates)}/{required_updates} updates). "
                f"Aggregating {len(eligible_updates)} client updates."
            )
            self.current_parameters = aggregate_gsfs_deltas(
                self.current_parameters,
                eligible_updates,
            )
            updates_aggregated_count = len(eligible_updates)
            samples_aggregated_count = sum(u.num_samples for u in eligible_updates)
            self.update_pool.mark_consumed(eligible_updates)
            self.current_model_version += 1
            aggregated = True

            self.aggregation_history.append({
                "round": server_round,
                "model_version": self.current_model_version,
                "clients_aggregated": [u.client_id for u in eligible_updates],
                "updates_aggregated": updates_aggregated_count,
                "samples_aggregated": samples_aggregated_count,
            })
        else:
            logger.info(
                f"GSFS Round {server_round}: Aggregation Threshold NOT met "
                f"({len(eligible_updates)}/{required_updates} updates). Model unchanged. Updates stay pending in pool."
            )

        # 4. Communication & Round Accounting
        model_size_bytes = estimate_parameter_size_bytes(self.current_parameters)
        pool_status = self.update_pool.get_status(current_round=server_round)

        # All selected clients transmitted their model update for server trigger evaluation
        upload_bytes_round = model_size_bytes * num_eligible_clients
        download_bytes_round = model_size_bytes * num_eligible_clients

        round_record = {
            "round": server_round,
            "strategy": "GSFS",
            "participating_clients": num_eligible_clients,
            "failed_clients": len(failures),
            "updates_generated": num_eligible_clients,
            "updates_selected": round_selected_count,
            "updates_rejected": round_rejected_count,
            "updates_pending_in_pool": pool_status["unconsumed_count"],
            "required_threshold_count": required_updates,
            "aggregated_this_round": aggregated,
            "updates_aggregated": updates_aggregated_count,
            "total_samples_aggregated": samples_aggregated_count,
            "moving_grad_mean": round(self.gradient_tracker.moving_mean, 6),
            "moving_grad_std": round(self.gradient_tracker.moving_std, 6),
            "grad_threshold": round(self.gradient_tracker.compute_threshold(), 6),
            "model_size_bytes": model_size_bytes,
            "upload_bytes_round": upload_bytes_round,
            "download_bytes_round": download_bytes_round,
        }
        self.round_history.append(round_record)

        return ndarrays_to_parameters(self.current_parameters), {
            "aggregated": aggregated,
            "updates_aggregated": updates_aggregated_count,
            "pool_pending": pool_status["unconsumed_count"],
        }

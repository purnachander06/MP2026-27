"""Gradient-statistics trigger and parameter displacement tracker for GSFS.

Measures the Euclidean (L2) displacement between client parameters and received
global parameters, maintaining moving mean and standard deviation across
the fleet to identify high-information model updates.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging

logger = logging.getLogger("GSFS.GradientStatistics")


def compute_param_deltas(
    local_params: List[np.ndarray],
    global_params: List[np.ndarray],
) -> List[np.ndarray]:
    """
    Compute parameter displacement vectors:
        Delta = local_params - global_params

    Args:
        local_params: Tensors updated by local client optimization.
        global_params: Reference global parameters sent by server.

    Returns:
        List of layer-wise delta NumPy arrays with matching shapes and float32 dtype.
    """
    if len(local_params) != len(global_params):
        raise ValueError(
            f"Layer count mismatch: local={len(local_params)} vs global={len(global_params)}"
        )

    deltas = []
    for i, (l_p, g_p) in enumerate(zip(local_params, global_params)):
        if l_p.shape != g_p.shape:
            raise ValueError(
                f"Shape mismatch at layer {i}: local {l_p.shape} vs global {g_p.shape}"
            )
        delta_layer = (l_p.astype(np.float64) - g_p.astype(np.float64)).astype(np.float32)
        deltas.append(delta_layer)

    return deltas


def compute_gradient_norm(
    local_params: Optional[List[np.ndarray]] = None,
    global_params: Optional[List[np.ndarray]] = None,
    deltas: Optional[List[np.ndarray]] = None,
) -> float:
    """
    Compute global Euclidean (L2) norm of parameter displacement:
        ||Delta||_2 = sqrt( sum_{layers} sum_{weights} (Delta_i)^2 )

    Args:
        local_params: Updated local parameters (optional if deltas provided).
        global_params: Received global parameters (optional if deltas provided).
        deltas: Pre-computed parameter displacement layers.

    Returns:
        Scalar Euclidean L2 norm as float.
    """
    if deltas is None:
        if local_params is None or global_params is None:
            raise ValueError("Must provide either pre-computed deltas or both local_params and global_params.")
        deltas = compute_param_deltas(local_params, global_params)

    squared_sum = 0.0
    for d in deltas:
        if not np.all(np.isfinite(d)):
            return float("nan")
        squared_sum += float(np.sum(np.square(d, dtype=np.float64)))

    return float(np.sqrt(squared_sum))


class GradientStatisticsTracker:
    """
    Tracks fleet-wide gradient norm statistics and computes dynamic thresholds.

    Threshold Formula:
        threshold = moving_mean + beta * moving_std

    Selection Rule:
        triggered = (gradient_norm >= threshold)
    """

    def __init__(
        self,
        beta: float = 0.0,
        min_history: int = 2,
    ):
        """
        Args:
            beta: Multiplier for moving standard deviation (default 0.0 = mean).
                  beta > 0 selects updates with above-average gradient norm.
                  beta < 0 is more permissive, accepting moderate updates.
            min_history: Minimum historical observations before statistical filtering.
        """
        self.beta = float(beta)
        self.min_history = max(1, int(min_history))
        self.history: List[float] = []

    @property
    def moving_mean(self) -> float:
        """Moving mean of recorded gradient norms."""
        if not self.history:
            return 0.0
        return float(np.mean(self.history))

    @property
    def moving_std(self) -> float:
        """Moving sample standard deviation of recorded gradient norms."""
        if len(self.history) < 2:
            return 0.0
        return float(np.std(self.history, ddof=1))

    def compute_threshold(self) -> float:
        """
        Calculate current statistical gradient threshold:
            threshold = moving_mean + beta * moving_std
        """
        if len(self.history) < self.min_history:
            return 0.0
        return float(self.moving_mean + self.beta * self.moving_std)

    def evaluate(self, grad_norm: float) -> Tuple[bool, float, str]:
        """
        Evaluate if a candidate gradient norm satisfies the statistical threshold.

        Args:
            grad_norm: L2 norm of the client's parameter displacement.

        Returns:
            Tuple of:
                - is_triggered (bool): True if update meets threshold.
                - threshold (float): Active gradient threshold.
                - reason (str): Audit reason.
        """
        if not np.isfinite(grad_norm):
            return False, 0.0, "non_finite_gradient_norm"

        # If insufficient historical observations, allow for baseline accumulation
        if len(self.history) < self.min_history:
            return True, 0.0, f"insufficient_history_{len(self.history)}_lt_{self.min_history}"

        threshold = self.compute_threshold()
        is_triggered = bool(grad_norm >= threshold)
        reason = (
            f"grad_norm_{grad_norm:.4f}_gte_threshold_{threshold:.4f}"
            if is_triggered
            else f"grad_norm_{grad_norm:.4f}_lt_threshold_{threshold:.4f}"
        )
        return is_triggered, threshold, reason

    def update_statistics(self, grad_norm: float) -> None:
        """
        Incorporate a verified finite gradient norm into moving statistics.
        """
        if np.isfinite(grad_norm):
            self.history.append(float(grad_norm))
        else:
            logger.warning(f"Ignored non-finite gradient norm: {grad_norm}")

    def reset(self) -> None:
        """Clear recorded historical gradient norms."""
        self.history.clear()

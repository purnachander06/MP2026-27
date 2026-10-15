"""Performance-change trigger for Gradient-Selective Federated Sampling (GSFS).

Evaluates whether a client's local training produced a statistically or
operationally meaningful improvement in local model performance without
leaking any global test or validation data.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging

logger = logging.getLogger("GSFS.PerformanceTrigger")


class PerformanceTrigger:
    """
    Evaluates client local performance changes against a configurable threshold.

    Sign Convention:
        performance_improvement = initial_loss - final_loss
        A positive improvement indicates that the model's loss decreased during
        local training, demonstrating positive task progress on local data.

    Trigger Rule:
        triggered = (performance_improvement >= threshold)
    """

    def __init__(
        self,
        threshold: float = 0.0,
        first_round_policy: str = "allow",
    ):
        """
        Args:
            threshold: Minimum loss reduction required to trigger (default 0.0).
            first_round_policy: Behavior when historical baseline is missing.
                - 'allow': Trigger evaluates to True on first round / unbenchmarked clients.
                - 'strict': Evaluates only on within-round initial vs final loss.
        """
        self.threshold = float(threshold)
        self.first_round_policy = first_round_policy.lower().strip()
        self.client_history: Dict[int, List[float]] = {}

    def compute_improvement(self, initial_loss: float, final_loss: float) -> float:
        """
        Calculate loss improvement with explicit sign convention:
            improvement = initial_loss - final_loss
        """
        if not (np.isfinite(initial_loss) and np.isfinite(final_loss)):
            raise ValueError(f"Non-finite loss encountered: initial={initial_loss}, final={final_loss}")
        return float(initial_loss - final_loss)

    def evaluate(
        self,
        client_id: int,
        initial_loss: Optional[float] = None,
        final_loss: Optional[float] = None,
        round_idx: int = 1,
    ) -> Tuple[bool, float, str]:
        """
        Evaluate if a client's training satisfies the performance-change trigger.

        Args:
            client_id: Simulated client identifier.
            initial_loss: Loss on received global parameters prior to local fit.
            final_loss: Loss on updated local parameters after local fit.
            round_idx: Current federated round index.

        Returns:
            Tuple of:
                - is_triggered (bool): True if update meets threshold.
                - improvement (float): Measured improvement value.
                - reason (str): Human-readable audit explanation.
        """
        if final_loss is None or not np.isfinite(final_loss):
            return False, 0.0, "non_finite_or_missing_final_loss"

        # 1. Within-round evaluation (preferred when client provides initial loss)
        if initial_loss is not None and np.isfinite(initial_loss):
            improvement = self.compute_improvement(initial_loss, final_loss)
            is_triggered = bool(improvement >= self.threshold)
            reason = (
                f"within_round_improvement_{improvement:+.4f}_gte_{self.threshold:.4f}"
                if is_triggered
                else f"within_round_improvement_{improvement:+.4f}_lt_{self.threshold:.4f}"
            )
            return is_triggered, improvement, reason

        # 2. Historical round-over-round comparison if client has history
        if client_id in self.client_history and len(self.client_history[client_id]) > 0:
            prev_loss = self.client_history[client_id][-1]
            improvement = float(prev_loss - final_loss)
            is_triggered = bool(improvement >= self.threshold)
            reason = (
                f"historical_improvement_{improvement:+.4f}_gte_{self.threshold:.4f}"
                if is_triggered
                else f"historical_improvement_{improvement:+.4f}_lt_{self.threshold:.4f}"
            )
            return is_triggered, improvement, reason

        # 3. First-round handling when baseline is absent
        if round_idx == 1 or self.first_round_policy == "allow":
            return True, 0.0, "first_round_unbenchmarked_allowed"

        return False, 0.0, "missing_baseline_strict_rejected"

    def record_client_loss(self, client_id: int, loss: float) -> None:
        """Record final local loss into client's historical trajectory."""
        if np.isfinite(loss):
            if client_id not in self.client_history:
                self.client_history[client_id] = []
            self.client_history[client_id].append(float(loss))

    def reset(self) -> None:
        """Reset historical client trajectories."""
        self.client_history.clear()

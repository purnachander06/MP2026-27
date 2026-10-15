"""Server-side Update Pool for Gradient-Selective Federated Sampling (GSFS).

Manages storage, deduplication, staleness expiration, and aggregation staging
for eligible client model updates.
"""

from dataclasses import dataclass, field
import time
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import logging

logger = logging.getLogger("GSFS.UpdatePool")


@dataclass
class ClientUpdate:
    """Represents an eligible client update stored in the GSFS update pool."""
    client_id: int
    round: int
    model_version: int
    parameter_deltas: List[np.ndarray]
    num_samples: int
    perf_improvement: float
    grad_norm: float
    trigger_decisions: Dict[str, bool]
    selection_reason: str
    timestamp: float = field(default_factory=time.time)
    consumed: bool = False


class UpdatePool:
    """
    In-memory pool buffering eligible client updates until the aggregation threshold is met.

    Guarantees:
        - Updates are consumed exactly once (no double-counting).
        - Stale updates older than max_staleness rounds are expired.
        - Deduplication per client per model version.
        - Strict capacity enforcement (bounded memory).
    """

    def __init__(
        self,
        max_pool_size: int = 10,
        max_staleness: int = 2,
    ):
        """
        Args:
            max_pool_size: Maximum capacity of the pool.
            max_staleness: Maximum allowed round lag before an update is expired.
        """
        self.max_pool_size = max(1, int(max_pool_size))
        self.max_staleness = max(1, int(max_staleness))
        # Stored updates indexed by client_id for deduplication
        self.pool: Dict[int, ClientUpdate] = {}
        self.total_received = 0
        self.total_accepted = 0
        self.total_rejected = 0
        self.total_expired = 0
        self.total_aggregated = 0

    def add_update(self, update: ClientUpdate) -> Tuple[bool, str]:
        """
        Add a candidate client update to the pool after structural verification.

        Args:
            update: Verified ClientUpdate instance.

        Returns:
            Tuple of (accepted: bool, reason: str).
        """
        self.total_received += 1

        # 1. Validate parameter finiteness
        for i, delta in enumerate(update.parameter_deltas):
            if not np.all(np.isfinite(delta)):
                self.total_rejected += 1
                reason = f"rejected_non_finite_tensor_layer_{i}"
                logger.warning(f"Client {update.client_id} update rejected: {reason}")
                return False, reason

        # 2. Check for existing update from this client (deduplication)
        if update.client_id in self.pool:
            existing = self.pool[update.client_id]
            if not existing.consumed:
                if update.round >= existing.round:
                    # Replace with fresher update
                    logger.info(
                        f"Client {update.client_id}: Replacing pending update from round {existing.round} "
                        f"with fresh update from round {update.round}."
                    )
                    self.pool[update.client_id] = update
                    self.total_accepted += 1
                    return True, "replaced_older_unconsumed_update"
                else:
                    self.total_rejected += 1
                    return False, "rejected_duplicate_older_than_pool"

        # 3. Check pool capacity
        unconsumed_count = sum(1 for u in self.pool.values() if not u.consumed)
        if unconsumed_count >= self.max_pool_size:
            # Try pruning stale updates first
            pruned = self.prune_stale(current_round=update.round)
            unconsumed_count = sum(1 for u in self.pool.values() if not u.consumed)
            if unconsumed_count >= self.max_pool_size:
                self.total_rejected += 1
                reason = f"rejected_pool_overflow_capacity_{self.max_pool_size}"
                logger.warning(f"Client {update.client_id} update rejected: {reason}")
                return False, reason

        self.pool[update.client_id] = update
        self.total_accepted += 1
        logger.debug(f"Client {update.client_id} update accepted into pool (round={update.round}).")
        return True, "accepted_into_pool"

    def get_eligible_updates(
        self,
        current_model_version: int,
        current_round: int,
    ) -> List[ClientUpdate]:
        """
        Retrieve unconsumed updates that are within the allowed staleness window.

        Args:
            current_model_version: Current active global model version.
            current_round: Current federated round.

        Returns:
            List of valid, unconsumed ClientUpdate instances.
        """
        eligible = []
        for update in self.pool.values():
            if update.consumed:
                continue
            staleness = current_round - update.round
            if staleness > self.max_staleness:
                continue
            # Accept updates trained on current model version or compatible within staleness
            if update.model_version == current_model_version or staleness <= self.max_staleness:
                eligible.append(update)

        return eligible

    def prune_stale(self, current_round: int) -> int:
        """
        Remove updates that exceed max_staleness or are already consumed.

        Args:
            current_round: Current federated round.

        Returns:
            Number of updates pruned.
        """
        to_remove = []
        for cid, update in self.pool.items():
            if update.consumed:
                to_remove.append(cid)
            elif (current_round - update.round) > self.max_staleness:
                to_remove.append(cid)
                self.total_expired += 1
                logger.info(
                    f"Update from Client {cid} (round {update.round}) expired due to staleness "
                    f"({current_round - update.round} > {self.max_staleness})."
                )

        for cid in to_remove:
            del self.pool[cid]

        return len(to_remove)

    def mark_consumed(self, updates: List[ClientUpdate]) -> None:
        """
        Mark updates as consumed after aggregation to guarantee single-use.
        """
        for u in updates:
            u.consumed = True
            if u.client_id in self.pool:
                self.pool[u.client_id].consumed = True
            self.total_aggregated += 1

    def clear(self) -> None:
        """Clear all entries in the pool."""
        self.pool.clear()

    def get_status(self, current_round: int = 1) -> Dict[str, Any]:
        """
        Return structured metrics summarizing pool state.
        """
        unconsumed = [u for u in self.pool.values() if not u.consumed]
        return {
            "total_in_pool": len(self.pool),
            "unconsumed_count": len(unconsumed),
            "unconsumed_client_ids": sorted([u.client_id for u in unconsumed]),
            "total_received": self.total_received,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "total_expired": self.total_expired,
            "total_aggregated": self.total_aggregated,
        }

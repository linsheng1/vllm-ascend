from collections import deque, Counter
from dataclasses import dataclass
from typing import Counter


@dataclass
class ExpertActivationRecord:
    """Record of expert activations at a point in time."""
    layer_id: int
    expert_ids: list[int]


class SlidingWindowCounter:
    """
    Tracks expert activations per layer using a sliding window.

    For each layer, maintains a deque of activation records.
    When computing hot experts, counts occurrences in the window.

    Note: This class is NOT thread-safe. Do not share instances across threads
    without external synchronization.

    Note: The `expert_ids` list in each record counts duplicates multiple times.
    For example, recording [0, 0, 0] will increment expert 0's count by 3.
    """

    def __init__(self, num_layers: int, num_experts: int, window_size: int):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.window_size = window_size
        # Per-layer deques of activation records
        self._windows: list[deque[ExpertActivationRecord]] = [
            deque(maxlen=window_size) for _ in range(num_layers)
        ]

    def record(self, layer_id: int, expert_ids: list[int]) -> None:
        """Record expert activations for a layer at current token."""
        record = ExpertActivationRecord(layer_id=layer_id, expert_ids=expert_ids)
        self._windows[layer_id].append(record)

    def get_counts(self, layer_id: int) -> Counter[int]:
        """Get expert activation counts for a layer in current window."""
        counts: Counter[int] = Counter()
        for record in self._windows[layer_id]:
            for expert_id in record.expert_ids:
                counts[expert_id] += 1
        return counts

    def get_hot_expert_ids(self, layer_id: int, top_k: int) -> list[int]:
        """Get the top_k hottest expert IDs for a layer."""
        counts = self.get_counts(layer_id)
        # Sort by count descending, then by expert_id for stability
        sorted_experts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        return [expert_id for expert_id, _ in sorted_experts[:top_k]]

    def reset(self) -> None:
        """Reset all counters for a new request."""
        for window in self._windows:
            window.clear()
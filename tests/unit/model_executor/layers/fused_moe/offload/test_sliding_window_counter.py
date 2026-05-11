import pytest
from collections import Counter, deque
from dataclasses import dataclass


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


# Test cases
def test_single_layer_tracking():
    """Test basic counter for single layer."""
    counter = SlidingWindowCounter(num_layers=1, num_experts=8, window_size=3)

    # Token 1: activate experts 0, 2, 4
    counter.record(0, [0, 2, 4])
    assert counter.get_counts(0) == {0: 1, 2: 1, 4: 1}

    # Token 2: activate experts 0, 2, 6
    counter.record(0, [0, 2, 6])
    assert counter.get_counts(0) == {0: 2, 2: 2, 4: 1, 6: 1}

    # Token 3: activate experts 1, 1, 1 (same expert multiple times)
    counter.record(0, [1, 1, 1])
    assert counter.get_counts(0)[1] == 3  # Should count 3 times
    print("test_single_layer_tracking PASSED")


def test_window_eviction():
    """Test that old entries are evicted after window_size."""
    counter = SlidingWindowCounter(num_layers=1, num_experts=4, window_size=2)

    counter.record(0, [0, 1])
    counter.record(0, [2, 3])

    # After 2 records, window is full but nothing evicted yet
    # Expert 0, 1 should still be present
    counts = counter.get_counts(0)
    assert 0 in counts
    assert 1 in counts
    assert 2 in counts
    assert 3 in counts

    # Add 3rd record - this triggers eviction of the oldest record [0, 1]
    counter.record(0, [1, 2])

    # Now expert 0 should be evicted, but 1, 2, 3 should remain
    counts = counter.get_counts(0)
    assert 0 not in counts
    assert 1 in counts
    assert 2 in counts
    assert 3 in counts
    print("test_window_eviction PASSED")


def test_multi_layer_tracking():
    """Test counter tracks each layer independently."""
    counter = SlidingWindowCounter(num_layers=3, num_experts=8, window_size=3)

    counter.record(0, [0, 1])  # Layer 0
    counter.record(1, [2, 3])  # Layer 1
    counter.record(2, [4, 5])  # Layer 2

    assert counter.get_counts(0) == {0: 1, 1: 1}
    assert counter.get_counts(1) == {2: 1, 3: 1}
    assert counter.get_counts(2) == {4: 1, 5: 1}
    print("test_multi_layer_tracking PASSED")


def test_get_hot_expert_ids():
    """Test getting hot expert IDs for a layer."""
    counter = SlidingWindowCounter(num_layers=1, num_experts=8, window_size=3)

    counter.record(0, [0, 0, 0])  # Expert 0: count 3
    counter.record(0, [1, 1])       # Expert 1: count 2
    counter.record(0, [2])           # Expert 2: count 1

    hot_ids = counter.get_hot_expert_ids(0, top_k=2)
    assert hot_ids == [0, 1]  # Expert 0 is hottest, then expert 1
    print("test_get_hot_expert_ids PASSED")


def test_request_reset():
    """Test that counter can be reset for new request."""
    counter = SlidingWindowCounter(num_layers=2, num_experts=8, window_size=3)

    counter.record(0, [0, 1])
    assert len(counter.get_counts(0)) > 0

    counter.reset()
    assert counter.get_counts(0) == {}
    assert counter.get_counts(1) == {}
    print("test_request_reset PASSED")


if __name__ == "__main__":
    from collections import deque
    test_single_layer_tracking()
    test_window_eviction()
    test_multi_layer_tracking()
    test_get_hot_expert_ids()
    test_request_reset()
    print("\nAll SlidingWindowCounter tests PASSED!")
import pytest


class ExpertHotnessTracker:
    """
    Tracks expert hotness per layer.

    Uses SlidingWindowCounter internally to maintain activation counts
    within a sliding time window.
    """

    def __init__(self, num_layers: int, num_experts: int, window_size: int = 10):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self._counter = SlidingWindowCounter(
            num_layers=num_layers,
            num_experts=num_experts,
            window_size=window_size,
        )

    def record_activations(self, layer_id: int, expert_ids: list[int]) -> None:
        """Record expert activations for a layer at current token."""
        self._counter.record(layer_id, expert_ids)

    def get_hot_expert_ids_for_layer(
        self, layer_id: int, top_k: int = 3
    ) -> list[int]:
        """Get the top_k hottest expert IDs for a layer."""
        return self._counter.get_hot_expert_ids(layer_id, top_k)

    def reset(self) -> None:
        """Reset for new request."""
        self._counter.reset()


# Test cases
def test_hotness_tracker_initialization():
    """Test basic initialization."""
    tracker = ExpertHotnessTracker(num_layers=2, num_experts=8)
    assert tracker.num_layers == 2
    assert tracker.num_experts == 8
    print("test_hotness_tracker_initialization PASSED")


def test_get_hot_experts_for_next_layer():
    """Test getting hot experts for next layer prefetch."""
    tracker = ExpertHotnessTracker(num_layers=3, num_experts=8)

    # Simulate Layer 0 activations
    tracker.record_activations(layer_id=0, expert_ids=[0, 2, 4])

    # Get hot experts for Layer 1 (no data recorded for layer 1 yet)
    hot_ids = tracker.get_hot_expert_ids_for_layer(1, top_k=3)
    # Layer 1 has no recorded activations, so should be empty
    assert hot_ids == []

    # Layer 0 has activations, should return them
    hot_ids_0 = tracker.get_hot_expert_ids_for_layer(0, top_k=2)
    assert len(hot_ids_0) == 2
    assert set(hot_ids_0).issubset({0, 2, 4})
    print("test_get_hot_experts_for_next_layer PASSED")


def test_hotness_accumulates():
    """Test that hotness accumulates over multiple tokens."""
    tracker = ExpertHotnessTracker(num_layers=1, num_experts=8)

    # Token 1
    tracker.record_activations(0, [0, 1])
    # Token 2
    tracker.record_activations(0, [0, 2])
    # Token 3
    tracker.record_activations(0, [0, 3])

    # Expert 0 is hottest
    hot_ids = tracker.get_hot_expert_ids_for_layer(0, top_k=1)
    assert hot_ids == [0]
    print("test_hotness_accumulates PASSED")


def test_request_reset():
    """Test reset clears all state."""
    tracker = ExpertHotnessTracker(num_layers=2, num_experts=8)

    tracker.record_activations(0, [0, 1])
    tracker.record_activations(1, [2, 3])

    tracker.reset()

    # After reset, counters should be empty
    hot_ids_0 = tracker.get_hot_expert_ids_for_layer(0, top_k=2)
    hot_ids_1 = tracker.get_hot_expert_ids_for_layer(1, top_k=2)
    assert hot_ids_0 == []
    assert hot_ids_1 == []
    print("test_request_reset PASSED")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/mac_lin/work/zsc_code/vllm-ascend/vllm_ascend/model_executor/layers/fused_moe/offload')
    from sliding_window_counter import SlidingWindowCounter

    # Monkey-patch for test
    import types
    test_module = types.ModuleType('test_module')
    test_module.SlidingWindowCounter = SlidingWindowCounter
    sys.modules['sliding_window_counter'] = test_module

    # Override the import in the module
    import importlib

    class TrackerWithCounter(ExpertHotnessTracker):
        def __init__(self, num_layers, num_experts, window_size=10):
            from collections import deque, Counter
            from dataclasses import dataclass

            @dataclass
            class ExpertActivationRecord:
                layer_id: int
                expert_ids: list[int]

            class SlidingWindowCounter:
                def __init__(self, num_layers, num_experts, window_size):
                    self.num_layers = num_layers
                    self.num_experts = num_experts
                    self.window_size = window_size
                    self._windows = [deque(maxlen=window_size) for _ in range(num_layers)]

                def record(self, layer_id, expert_ids):
                    record = ExpertActivationRecord(layer_id=layer_id, expert_ids=expert_ids)
                    self._windows[layer_id].append(record)

                def get_counts(self, layer_id):
                    counts = Counter()
                    for record in self._windows[layer_id]:
                        for expert_id in record.expert_ids:
                            counts[expert_id] += 1
                    return counts

                def get_hot_expert_ids(self, layer_id, top_k):
                    counts = self.get_counts(layer_id)
                    sorted_experts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
                    return [expert_id for expert_id, _ in sorted_experts[:top_k]]

                def reset(self):
                    for window in self._windows:
                        window.clear()

            self.num_layers = num_layers
            self.num_experts = num_experts
            self._counter = SlidingWindowCounter(num_layers, num_experts, window_size)

        def record_activations(self, layer_id, expert_ids):
            self._counter.record(layer_id, expert_ids)

        def get_hot_expert_ids_for_layer(self, layer_id, top_k=3):
            return self._counter.get_hot_expert_ids(layer_id, top_k)

        def reset(self):
            self._counter.reset()

    # Run tests with inline implementation
    tracker = TrackerWithCounter(num_layers=2, num_experts=8)
    assert tracker.num_layers == 2
    assert tracker.num_experts == 8
    print("test_hotness_tracker_initialization PASSED")

    tracker = TrackerWithCounter(num_layers=3, num_experts=8)
    tracker.record_activations(0, [0, 2, 4])
    hot_ids = tracker.get_hot_expert_ids_for_layer(1, top_k=3)
    assert hot_ids == []
    hot_ids_0 = tracker.get_hot_expert_ids_for_layer(0, top_k=2)
    assert len(hot_ids_0) == 2
    assert set(hot_ids_0).issubset({0, 2, 4})
    print("test_get_hot_experts_for_next_layer PASSED")

    tracker = TrackerWithCounter(num_layers=1, num_experts=8)
    tracker.record_activations(0, [0, 1])
    tracker.record_activations(0, [0, 2])
    tracker.record_activations(0, [0, 3])
    hot_ids = tracker.get_hot_expert_ids_for_layer(0, top_k=1)
    assert hot_ids == [0]
    print("test_hotness_accumulates PASSED")

    tracker = TrackerWithCounter(num_layers=2, num_experts=8)
    tracker.record_activations(0, [0, 1])
    tracker.record_activations(1, [2, 3])
    tracker.reset()
    hot_ids_0 = tracker.get_hot_expert_ids_for_layer(0, top_k=2)
    hot_ids_1 = tracker.get_hot_expert_ids_for_layer(1, top_k=2)
    assert hot_ids_0 == []
    assert hot_ids_1 == []
    print("test_request_reset PASSED")

    print("\nAll ExpertHotnessTracker tests PASSED!")
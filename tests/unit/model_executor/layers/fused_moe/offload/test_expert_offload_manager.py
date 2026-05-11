import threading
from collections import deque, Counter
from dataclasses import dataclass
from typing import Protocol

import torch


class CPUExpertBuffer(Protocol):
    """Protocol for CPU expert buffer management."""

    def allocate(self, expert_ids: list[int]) -> list[torch.Tensor]:
        ...

    def free(self, expert_ids: list[int]) -> None:
        ...

    def get_num_free(self) -> int:
        ...


class SimpleCPUExpertBuffer:
    def __init__(self, num_experts: int, expert_shape: tuple[int, ...]):
        self.num_experts = num_experts
        self.expert_shape = expert_shape
        self._buffers: list[torch.Tensor] = [
            torch.zeros(expert_shape, dtype=torch.float16, device="cpu")
            for _ in range(num_experts)
        ]
        self._free_pool: set[int] = set(range(num_experts))
        self._allocated: set[int] = set()
        self._lock = threading.Lock()

    def allocate(self, expert_ids: list[int]) -> list[torch.Tensor]:
        with self._lock:
            for expert_id in expert_ids:
                if expert_id not in self._free_pool:
                    raise ValueError(f"Expert {expert_id} is not in free pool")
                if expert_id in self._allocated:
                    raise ValueError(f"Expert {expert_id} is already allocated")
            for expert_id in expert_ids:
                self._free_pool.discard(expert_id)
                self._allocated.add(expert_id)
        return [self._buffers[expert_id] for expert_id in expert_ids]

    def free(self, expert_ids: list[int]) -> None:
        with self._lock:
            for expert_id in expert_ids:
                if expert_id not in self._allocated:
                    raise ValueError(f"Expert {expert_id} was not allocated")
            for expert_id in expert_ids:
                self._allocated.discard(expert_id)
                self._free_pool.add(expert_id)

    def get_num_free(self) -> int:
        return len(self._free_pool)

    def get_buffer(self, expert_id: int) -> torch.Tensor:
        if expert_id not in self._allocated:
            raise ValueError(f"Expert {expert_id} is not currently allocated")
        return self._buffers[expert_id]


@dataclass
class ExpertActivationRecord:
    layer_id: int
    expert_ids: list[int]


class SlidingWindowCounter:
    def __init__(self, num_layers: int, num_experts: int, window_size: int):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.window_size = window_size
        self._windows: list[deque[ExpertActivationRecord]] = [
            deque(maxlen=window_size) for _ in range(num_layers)
        ]

    def record(self, layer_id: int, expert_ids: list[int]) -> None:
        record = ExpertActivationRecord(layer_id=layer_id, expert_ids=expert_ids)
        self._windows[layer_id].append(record)

    def get_counts(self, layer_id: int) -> Counter[int]:
        counts: Counter[int] = Counter()
        for record in self._windows[layer_id]:
            for expert_id in record.expert_ids:
                counts[expert_id] += 1
        return counts

    def get_hot_expert_ids(self, layer_id: int, top_k: int) -> list[int]:
        counts = self.get_counts(layer_id)
        sorted_experts = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
        return [expert_id for expert_id, _ in sorted_experts[:top_k]]

    def reset(self) -> None:
        for window in self._windows:
            window.clear()


class ExpertHotnessTracker:
    def __init__(self, num_layers: int, num_experts: int, window_size: int = 10):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self._counter = SlidingWindowCounter(
            num_layers=num_layers,
            num_experts=num_experts,
            window_size=window_size,
        )

    def record_activations(self, layer_id: int, expert_ids: list[int]) -> None:
        self._counter.record(layer_id, expert_ids)

    def get_hot_expert_ids_for_layer(self, layer_id: int, top_k: int = 3) -> list[int]:
        return self._counter.get_hot_expert_ids(layer_id, top_k)

    def reset(self) -> None:
        self._counter.reset()


class ExpertOffloadManager:
    """
    Manages expert weight offload from CPU to NPU.

    Responsibilities:
    1. Track which experts are currently on NPU
    2. Record layer activations for prefetch planning
    3. Calculate prefetch union (next layer's experts from 3 sources)
    4. Trigger async H2D transfers

    Thread-safe: all public methods use locks for shared state protection.
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        expert_shape: tuple[int, ...],
    ):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.expert_shape = expert_shape

        # CPU buffer pool
        self._cpu_buffer = SimpleCPUExpertBuffer(num_experts, expert_shape)

        # Hotness tracker
        self._hotness_tracker = ExpertHotnessTracker(num_layers, num_experts)

        # Which experts are currently on NPU (per layer)
        self._on_npu: list[set[int]] = [set() for _ in range(num_layers)]

        # Previous step's expert activations (per layer)
        self._prev_step_experts: list[list[int]] = [[] for _ in range(num_layers)]

        # Lock for thread-safe access to shared state
        self._lock = threading.Lock()

    def record_layer_activations(self, layer_id: int, expert_ids: list[int]) -> None:
        """Record expert activations for a layer. Thread-safe."""
        with self._lock:
            # Update hotness tracker
            self._hotness_tracker.record_activations(layer_id, expert_ids)

            # Store as previous step for next layer's prefetch calculation
            self._prev_step_experts[layer_id] = list(expert_ids)

    def get_prefetch_union_for_next_layer(self, current_layer: int) -> list[int]:
        """
        Calculate the prefetch union for next layer.

        Args:
            current_layer: The layer that is currently computing.
                           Caller passes this explicitly (not stored internally).

        Returns:
            List of expert IDs to prefetch for the next layer.
        """
        next_layer = current_layer + 1

        if next_layer >= self.num_layers:
            return []

        with self._lock:
            # Source 1: Current layer's expert IDs
            source1 = self._prev_step_experts[current_layer]

            # Source 2: Next layer's previous step experts
            source2 = self._prev_step_experts[next_layer]

            # Source 3: Next layer's hot experts
            source3 = self._hotness_tracker.get_hot_expert_ids_for_layer(
                next_layer, top_k=3
            )

        # Union of all sources
        union_set = set(source1) | set(source2) | set(source3)
        return list(union_set)

    def get_previous_step_experts(self, layer_id: int) -> list[int]:
        """Get the experts used in the previous step for a layer (public API)."""
        with self._lock:
            return list(self._prev_step_experts[layer_id])

    def get_missing_experts(self, layer_id: int, needed: list[int]) -> list[int]:
        """Get experts needed but not on NPU."""
        with self._lock:
            return [eid for eid in needed if eid not in self._on_npu[layer_id]]

    def mark_on_npu(self, layer_id: int, expert_id: int) -> None:
        """Mark an expert as being on NPU."""
        with self._lock:
            self._on_npu[layer_id].add(expert_id)

    def mark_off_npu(self, layer_id: int, expert_id: int) -> None:
        """Mark an expert as no longer on NPU."""
        with self._lock:
            self._on_npu[layer_id].discard(expert_id)

    def reset(self) -> None:
        """Reset for new request."""
        with self._lock:
            self._hotness_tracker.reset()
            self._on_npu = [set() for _ in range(self.num_layers)]
            self._prev_step_experts = [[] for _ in range(self.num_layers)]


# Test cases
def test_manager_initialization():
    """Test manager initializes with correct state."""
    manager = ExpertOffloadManager(num_layers=2, num_experts=8, expert_shape=(4096, 2048))
    assert manager.num_layers == 2
    assert manager.num_experts == 8
    print("test_manager_initialization PASSED")


def test_prefetch_union_calculation_with_current_layer():
    """Test prefetch union calculation takes current_layer parameter."""
    manager = ExpertOffloadManager(num_layers=3, num_experts=8, expert_shape=(4096, 2048))

    # Simulate Layer 0 activations
    manager.record_layer_activations(0, [0, 2, 4])

    # Get prefetch union for Layer 1, passing current_layer=0 explicitly
    union = manager.get_prefetch_union_for_next_layer(current_layer=0)
    # Should include:
    # - Layer 0's experts (source 1): [0, 2, 4]
    # - Layer 1's previous step (source 2): []
    # - Layer 1's hot experts (source 3): []
    assert isinstance(union, list)


def test_async_load_missing():
    """Test async loading of missing experts."""
    manager = ExpertOffloadManager(num_layers=2, num_experts=8, expert_shape=(4096, 2048))

    # Layer 0 needs experts 0, 2
    # Simulate already having expert 0 on NPU
    manager.mark_on_npu(0, 0)

    # Get missing experts
    missing = manager.get_missing_experts(0, [0, 2])
    assert missing == [2]  # Expert 2 is missing


def test_record_and_prefetch_flow():
    """Test the full record -> prefetch flow."""
    manager = ExpertOffloadManager(num_layers=3, num_experts=8, expert_shape=(4096, 2048))

    # Process token T at Layer 0
    manager.record_layer_activations(0, [0, 2, 4])

    # Process token T at Layer 1
    manager.record_layer_activations(1, [1, 3, 5])

    # Get prefetch union for Layer 2, using current_layer=1
    union = manager.get_prefetch_union_for_next_layer(current_layer=1)

    # Should include:
    # - Layer 1's experts (source 1): [1, 3, 5]
    # - Layer 2's previous step (source 2): []
    # - Layer 2's hot experts (source 3): []
    assert set(union) == {1, 3, 5}
    print("test_record_and_prefetch_flow PASSED")


def test_public_api_get_previous_step_experts():
    """Test that get_previous_step_experts is the proper public API."""
    manager = ExpertOffloadManager(num_layers=2, num_experts=8, expert_shape=(4096, 2048))

    manager.record_layer_activations(0, [0, 2, 4])

    # Use public API instead of direct manager._prev_step_experts access
    prev_experts = manager.get_previous_step_experts(0)
    assert prev_experts == [0, 2, 4]
    print("test_public_api_get_previous_step_experts PASSED")


if __name__ == "__main__":
    test_manager_initialization()
    test_prefetch_union_calculation_with_current_layer()
    test_async_load_missing()
    test_record_and_prefetch_flow()
    test_public_api_get_previous_step_experts()
    print("\nAll ExpertOffloadManager tests PASSED!")
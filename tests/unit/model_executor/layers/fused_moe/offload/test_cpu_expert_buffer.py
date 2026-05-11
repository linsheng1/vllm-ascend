import threading
from collections import deque
from typing import Protocol

import torch


class CPUExpertBuffer(Protocol):
    """Protocol for CPU expert buffer management."""

    def allocate(self, expert_ids: list[int]) -> list[torch.Tensor]:
        """Allocate CPU buffers for expert IDs."""
        ...

    def free(self, expert_ids: list[int]) -> None:
        """Free CPU buffers for expert IDs."""
        ...

    def get_num_free(self) -> int:
        """Get number of free buffers."""
        ...


class SimpleCPUExpertBuffer:
    """
    Simple CPU expert buffer using set-based pool for O(1) operations.

    Manages a pool of CPU tensors for expert weights.
    Thread-safe with a SINGLE lock for all operations.

    Note: Uses single lock to ensure compound operations (allocate/free
    multiple expert IDs) are atomic. All expert_ids are validated BEFORE
    any state is modified to prevent inconsistent buffer state.
    """

    def __init__(self, num_experts: int, expert_shape: tuple[int, ...]):
        self.num_experts = num_experts
        self.expert_shape = expert_shape

        # Pre-allocate all CPU tensors
        self._buffers: list[torch.Tensor] = [
            torch.zeros(expert_shape, dtype=torch.float16, device="cpu")
            for _ in range(num_experts)
        ]

        # Pool of free buffer indices (use set for O(1) operations)
        self._free_pool: set[int] = set(range(num_experts))

        # Set of allocated buffer indices
        self._allocated: set[int] = set()

        # SINGLE lock for thread safety - covers ALL operations
        self._lock = threading.Lock()

    def allocate(self, expert_ids: list[int]) -> list[torch.Tensor]:
        """
        Allocate CPU buffers for expert IDs. Thread-safe and atomic.

        Validates ALL expert_ids BEFORE modifying any state to prevent
        inconsistent buffer state on partial failure.
        """
        with self._lock:
            # Phase 1: Validate ALL expert_ids first (fail-fast)
            for expert_id in expert_ids:
                if expert_id not in self._free_pool:
                    raise ValueError(f"Expert {expert_id} is not in free pool")
                if expert_id in self._allocated:
                    raise ValueError(f"Expert {expert_id} is already allocated")

            # Phase 2: Update state ONLY after all validations pass
            for expert_id in expert_ids:
                self._free_pool.discard(expert_id)  # O(1) set operation
                self._allocated.add(expert_id)

        return [self._buffers[expert_id] for expert_id in expert_ids]

    def free(self, expert_ids: list[int]) -> None:
        """
        Free CPU buffers for expert IDs. Thread-safe and atomic.

        Validates ALL expert_ids BEFORE modifying any state to prevent
        inconsistent buffer state on partial failure.
        """
        with self._lock:
            # Phase 1: Validate ALL expert_ids first (fail-fast)
            for expert_id in expert_ids:
                if expert_id not in self._allocated:
                    raise ValueError(f"Expert {expert_id} was not allocated")

            # Phase 2: Update state ONLY after all validations pass
            for expert_id in expert_ids:
                self._allocated.discard(expert_id)
                self._free_pool.add(expert_id)  # O(1) set operation

    def get_num_free(self) -> int:
        """Get number of free buffers."""
        return len(self._free_pool)

    def get_buffer(self, expert_id: int) -> torch.Tensor:
        """Get buffer tensor for expert ID (must be allocated)."""
        if expert_id not in self._allocated:
            raise ValueError(f"Expert {expert_id} is not currently allocated")
        return self._buffers[expert_id]


# Test cases
def test_cpu_buffer_allocation():
    """Test CPU buffer allocation."""
    buffer = SimpleCPUExpertBuffer(num_experts=8, expert_shape=(4096, 2048))

    # Allocate buffers for experts 0, 2, 4
    allocated = buffer.allocate([0, 2, 4])
    assert len(allocated) == 3
    assert allocated[0].shape == (4096, 2048)
    assert allocated[0].device == torch.device("cpu")
    print("test_cpu_buffer_allocation PASSED")


def test_cpu_buffer_free():
    """Test CPU buffer deallocation."""
    buffer = SimpleCPUExpertBuffer(num_experts=8, expert_shape=(4096, 2048))

    # Allocate and then free
    buffer.allocate([0, 2, 4])
    buffer.free([0, 2, 4])

    # Should be able to reallocate
    allocated = buffer.allocate([0, 2, 4])
    assert len(allocated) == 3
    print("test_cpu_buffer_free PASSED")


def test_cpu_buffer_allocation_tracking():
    """Test that allocation is tracked correctly."""
    buffer = SimpleCPUExpertBuffer(num_experts=8, expert_shape=(4096, 2048))

    # Initially all free
    assert buffer.get_num_free() == 8

    # Allocate 3
    buffer.allocate([0, 2, 4])
    assert buffer.get_num_free() == 5

    # Allocate 2 more
    buffer.allocate([1, 3])
    assert buffer.get_num_free() == 3
    print("test_cpu_buffer_allocation_tracking PASSED")


def test_free_invalid_ids():
    """Test freeing IDs that weren't allocated."""
    buffer = SimpleCPUExpertBuffer(num_experts=8, expert_shape=(4096, 2048))

    # Allocate 0, 2, 4
    buffer.allocate([0, 2, 4])

    # Try to free 1 (not allocated) - should raise
    try:
        buffer.free([1])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("test_free_invalid_ids PASSED")


def test_concurrent_allocate_free():
    """Test concurrent allocate and free from multiple threads."""
    buffer = SimpleCPUExpertBuffer(num_experts=8, expert_shape=(4096, 2048))
    errors = []

    def allocate_free():
        try:
            ids = [0, 1, 2, 3]
            buffer.allocate(ids)
            buffer.free(ids)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=allocate_free) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Errors: {errors}"
    print("test_concurrent_allocate_free PASSED")


if __name__ == "__main__":
    test_cpu_buffer_allocation()
    test_cpu_buffer_free()
    test_cpu_buffer_allocation_tracking()
    test_free_invalid_ids()
    test_concurrent_allocate_free()
    print("\nAll CPUExpertBuffer tests PASSED!")
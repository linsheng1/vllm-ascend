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
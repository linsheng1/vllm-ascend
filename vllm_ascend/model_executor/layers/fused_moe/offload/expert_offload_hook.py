import threading
from collections import deque, Counter
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ExpertActivationRecord:
    layer_id: int
    expert_ids: list[int]


class SlidingWindowCounter:
    """Tracks expert activations per layer using a sliding window."""

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
    """Tracks expert hotness per layer using SlidingWindowCounter."""

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


class SimpleCPUExpertBuffer:
    """CPU expert buffer with set-based O(1) pool management."""

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


class ExpertOffloadManager:
    """Manages expert weight offload from CPU to NPU."""

    def __init__(self, num_layers: int, num_experts: int, expert_shape: tuple[int, ...]):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.expert_shape = expert_shape
        self._cpu_buffer = SimpleCPUExpertBuffer(num_experts, expert_shape)
        self._hotness_tracker = ExpertHotnessTracker(num_layers, num_experts)
        self._on_npu: list[set[int]] = [set() for _ in range(num_layers)]
        self._prev_step_experts: list[list[int]] = [[] for _ in range(num_layers)]
        self._lock = threading.Lock()

    def record_layer_activations(self, layer_id: int, expert_ids: list[int]) -> None:
        with self._lock:
            self._hotness_tracker.record_activations(layer_id, expert_ids)
            self._prev_step_experts[layer_id] = list(expert_ids)

    def get_prefetch_union_for_next_layer(self, current_layer: int) -> list[int]:
        next_layer = current_layer + 1
        if next_layer >= self.num_layers:
            return []
        with self._lock:
            source1 = self._prev_step_experts[current_layer]
            source2 = self._prev_step_experts[next_layer]
            source3 = self._hotness_tracker.get_hot_expert_ids_for_layer(next_layer, top_k=3)
        return list(set(source1) | set(source2) | set(source3))

    def get_previous_step_experts(self, layer_id: int) -> list[int]:
        with self._lock:
            return list(self._prev_step_experts[layer_id])

    def get_missing_experts(self, layer_id: int, needed: list[int]) -> list[int]:
        with self._lock:
            return [eid for eid in needed if eid not in self._on_npu[layer_id]]

    def mark_on_npu(self, layer_id: int, expert_id: int) -> None:
        with self._lock:
            self._on_npu[layer_id].add(expert_id)

    def mark_off_npu(self, layer_id: int, expert_id: int) -> None:
        with self._lock:
            self._on_npu[layer_id].discard(expert_id)

    def reset(self) -> None:
        with self._lock:
            self._hotness_tracker.reset()
            self._on_npu = [set() for _ in range(self.num_layers)]
            self._prev_step_experts = [[] for _ in range(self.num_layers)]


class ExpertOffloadHook:
    """
    Hooks into MoE forward pass to orchestrate offload with ping-pong buffers.

    Ping-Pong Buffer Design:
    - Each expert has 2 NPU buffers (A and B)
    - While compute uses buffer A, transfer fills buffer B
    - swap_buffers() switches active buffer after transfer completes
    - torch.npu.Event ensures compute finishes before transfer starts
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        expert_shape: tuple[int, ...],
    ):
        self.manager = ExpertOffloadManager(
            num_layers=num_layers,
            num_experts=num_experts,
            expert_shape=expert_shape,
        )
        self._current_layer: int = 0

        # H2D transfer references
        self._expert_weights_cpu: Optional[dict[int, torch.Tensor]] = None

        # Ping-pong buffers on NPU
        self._npu_buffers_a: dict[int, torch.Tensor] = {}
        self._npu_buffers_b: dict[int, torch.Tensor] = {}
        self._active_buffer: dict[int, str] = {}

        # torch.npu.Event per expert
        self._compute_complete_events: dict[int, torch.npu.Event] = {}

        # torch.npu.Stream for async transfers
        self._transfer_stream: Optional[torch.npu.Stream] = None

        self._buffer_lock = threading.Lock()
        self._init_ping_pong_buffers(expert_shape)

    def _init_ping_pong_buffers(self, expert_shape: tuple[int, ...]) -> None:
        """Initialize ping-pong NPU buffers and events for each expert."""
        # Use CPU fallback if NPU not available
        device = "cpu"
        if hasattr(torch, 'npu') and torch.npu.is_available():
            device = "npu"

        for expert_id in range(self.manager.num_experts):
            # Allocate two buffers per expert
            self._npu_buffers_a[expert_id] = torch.zeros(
                expert_shape, dtype=torch.float16, device=device
            )
            self._npu_buffers_b[expert_id] = torch.zeros(
                expert_shape, dtype=torch.float16, device=device
            )
            # Start with buffer A as active
            self._active_buffer[expert_id] = "A"
            # Create event to signal compute completion
            self._compute_complete_events[expert_id] = torch.npu.Event(
                enable_timing=False, blocking=False
            )
        # Create dedicated transfer stream for async H2D/D2H
        self._transfer_stream = torch.npu.Stream(device=device, priority=0)

    def get_active_buffer(self, expert_id: int) -> torch.Tensor:
        with self._buffer_lock:
            buf = self._active_buffer[expert_id]
            return self._npu_buffers_a[expert_id] if buf == "A" else self._npu_buffers_b[expert_id]

    def get_inactive_buffer(self, expert_id: int) -> torch.Tensor:
        with self._buffer_lock:
            buf = "B" if self._active_buffer[expert_id] == "A" else "A"
            return self._npu_buffers_a[expert_id] if buf == "A" else self._npu_buffers_b[expert_id]

    def record_compute_complete(self, expert_id: int) -> None:
        if expert_id in self._compute_complete_events:
            self._compute_complete_events[expert_id].record()

    def swap_buffers(self, expert_id: int) -> None:
        with self._buffer_lock:
            self._active_buffer[expert_id] = (
                "B" if self._active_buffer[expert_id] == "A" else "A"
            )

    def _async_h2d_transfer_with_sync(
        self,
        expert_ids: list[int],
        expert_weights_cpu: dict[int, torch.Tensor],
    ) -> None:
        if self._transfer_stream is None:
            return

        with torch.npu.stream(self._transfer_stream):
            for expert_id in expert_ids:
                compute_event = self._compute_complete_events.get(expert_id)
                if compute_event is not None:
                    compute_event.synchronize()

                inactive_buf = self.get_inactive_buffer(expert_id)
                cpu_tensor = expert_weights_cpu.get(expert_id)
                if cpu_tensor is not None:
                    inactive_buf.copy_(cpu_tensor)

                self.manager.mark_on_npu(self._current_layer, expert_id)
                self.swap_buffers(expert_id)

    def set_buffers(self, cpu_buffers: dict[int, torch.Tensor]) -> None:
        self._expert_weights_cpu = cpu_buffers

    def on_routing(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        expert_ids = topk_ids.unique().tolist()
        self.manager.record_layer_activations(layer_id, expert_ids)

    def on_layer_compute_start(self, layer_id: int) -> None:
        self._current_layer = layer_id
        needed = self.manager.get_previous_step_experts(layer_id)
        missing = self.manager.get_missing_experts(layer_id, needed)

        if missing and self._expert_weights_cpu is not None:
            self._async_h2d_transfer_with_sync(missing, self._expert_weights_cpu)

    def on_layer_compute_end(self, layer_id: int) -> None:
        prev_experts = self.manager.get_previous_step_experts(layer_id)
        for expert_id in prev_experts:
            self.record_compute_complete(expert_id)

        if layer_id < self.manager.num_layers - 1:
            prefetch_union = self.manager.get_prefetch_union_for_next_layer(
                current_layer=layer_id
            )
            if prefetch_union and self._expert_weights_cpu is not None:
                self._async_h2d_transfer_with_sync(prefetch_union, self._expert_weights_cpu)

    def on_expert_compute_end(self, expert_id: int) -> None:
        self.record_compute_complete(expert_id)

    def reset(self) -> None:
        self.manager.reset()
        self._current_layer = 0
        with self._buffer_lock:
            for expert_id in self._active_buffer:
                self._active_buffer[expert_id] = "A"
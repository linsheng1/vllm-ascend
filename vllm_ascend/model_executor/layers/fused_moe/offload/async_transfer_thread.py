import threading
from collections import deque
from typing import Optional

import torch


class AsyncTransferThread(threading.Thread):
    """
    Background thread for async H2D/D2H expert weight transfers.

    Uses collections.deque for O(1) enqueue/popleft operations instead of
    list.pop(0) which is O(n).

    Usage:
        # H2D transfer (CPU -> NPU):
        async_thread.enqueue_h2d(expert_ids, expert_weights_cpu, expert_weights_npu)

        # D2H transfer (NPU -> CPU):
        async_thread.enqueue_d2h(expert_ids, expert_weights_npu, expert_weights_cpu)
    """

    def __init__(
        self,
        name: str = "ExpertTransferThread",
    ):
        super().__init__(daemon=True, name=name)
        # Use deque for O(1) popleft() instead of list.pop(0) which is O(n)
        self._h2d_queue: deque[tuple[list[int], dict, dict]] = deque()
        self._d2h_queue: deque[tuple[list[int], dict, dict]] = deque()
        self._lock = threading.Lock()
        self._h2d_event = threading.Event()
        self._d2h_event = threading.Event()
        self._stop_event = threading.Event()

    def enqueue_h2d(
        self,
        expert_ids: list[int],
        expert_weights_cpu: dict[int, torch.Tensor],
        expert_weights_npu: dict[int, torch.Tensor],
    ) -> None:
        """Enqueue H2D (Host-to-Device) transfer for expert weights."""
        with self._lock:
            self._h2d_queue.append((expert_ids, expert_weights_cpu, expert_weights_npu))
            self._h2d_event.set()

    def enqueue_d2h(
        self,
        expert_ids: list[int],
        expert_weights_npu: dict[int, torch.Tensor],
        expert_weights_cpu: dict[int, torch.Tensor],
    ) -> None:
        """Enqueue D2H (Device-to-Host) transfer for expert weights."""
        with self._lock:
            self._d2h_queue.append((expert_ids, expert_weights_npu, expert_weights_cpu))
            self._d2h_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            # Process H2D transfers
            if self._h2d_event.wait(timeout=0.1):
                with self._lock:
                    if self._h2d_queue:
                        expert_ids, cpu_bufs, npu_bufs = self._h2d_queue.popleft()
                        if not self._h2d_queue:
                            self._h2d_event.clear()
                if expert_ids:
                    self._do_h2d_transfer(expert_ids, cpu_bufs, npu_bufs)

            # Process D2H transfers
            if self._d2h_event.wait(timeout=0.1):
                with self._lock:
                    if self._d2h_queue:
                        expert_ids, npu_bufs, cpu_bufs = self._d2h_queue.popleft()
                        if not self._d2h_queue:
                            self._d2h_event.clear()
                if expert_ids:
                    self._do_d2h_transfer(expert_ids, npu_bufs, cpu_bufs)

    def _do_h2d_transfer(
        self,
        expert_ids: list[int],
        expert_weights_cpu: dict[int, torch.Tensor],
        expert_weights_npu: dict[int, torch.Tensor],
    ) -> None:
        """Perform H2D (Host-to-Device) transfer."""
        for expert_id in expert_ids:
            if expert_id in expert_weights_cpu and expert_id in expert_weights_npu:
                npu_tensor = expert_weights_npu[expert_id]
                cpu_tensor = expert_weights_cpu[expert_id]
                npu_tensor.copy_(cpu_tensor)

    def _do_d2h_transfer(
        self,
        expert_ids: list[int],
        expert_weights_npu: dict[int, torch.Tensor],
        expert_weights_cpu: dict[int, torch.Tensor],
    ) -> None:
        """Perform D2H (Device-to-Host) transfer."""
        for expert_id in expert_ids:
            if expert_id in expert_weights_npu and expert_id in expert_weights_cpu:
                cpu_tensor = expert_weights_cpu[expert_id]
                npu_tensor = expert_weights_npu[expert_id]
                cpu_tensor.copy_(npu_tensor)

    def stop(self) -> None:
        """Signal the thread to stop."""
        self._stop_event.set()
        self.join(timeout=1.0)


class ExpertTransferThreadPool:
    """
    Dedicated thread pool for expert weight H2D/D2H transfers.

    Unlike KVTransferThread which handles token-specific KV cache,
    this pool handles layer-specific expert weights that are shared
    across all tokens using a given layer.

    Thread-safe with O(1) queue operations using deque.
    """

    def __init__(
        self,
        num_threads: int = 2,
        name: str = "ExpertTransferPool",
    ):
        self._num_threads = num_threads
        self._threads: list[AsyncTransferThread] = []
        self._h2d_queue: deque[tuple[int, list[int], dict, dict]] = deque()
        self._d2h_queue: deque[tuple[int, list[int], dict, dict]] = deque()
        self._lock = threading.Lock()
        self._h2d_event = threading.Event()
        self._d2h_event = threading.Event()
        self._stop_event = threading.Event()

        # Start worker threads
        for i in range(num_threads):
            worker = AsyncTransferThread(name=f"{name}-{i}")
            worker._h2d_queue = self._h2d_queue
            worker._d2h_queue = self._d2h_queue
            worker._h2d_event = self._h2d_event
            worker._d2h_event = self._d2h_event
            worker._stop_event = self._stop_event
            worker._lock = self._lock
            worker.start()
            self._threads.append(worker)

    def enqueue_h2d(
        self,
        layer_id: int,
        expert_ids: list[int],
        expert_weights_cpu: dict[int, torch.Tensor],
        expert_weights_npu: dict[int, torch.Tensor],
    ) -> None:
        """Enqueue H2D transfer for expert weights."""
        with self._lock:
            self._h2d_queue.append((layer_id, expert_ids, expert_weights_cpu, expert_weights_npu))
            self._h2d_event.set()

    def enqueue_d2h(
        self,
        layer_id: int,
        expert_ids: list[int],
        expert_weights_npu: dict[int, torch.Tensor],
        expert_weights_cpu: dict[int, torch.Tensor],
    ) -> None:
        """Enqueue D2H transfer for expert weights."""
        with self._lock:
            self._d2h_queue.append((layer_id, expert_ids, expert_weights_npu, expert_weights_cpu))
            self._d2h_event.set()

    def stop(self) -> None:
        """Stop all worker threads."""
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=2.0)
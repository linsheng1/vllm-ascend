from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - real runtime always has torch.
    torch = None  # type: ignore[assignment]

try:
    from vllm.logger import logger
except Exception:  # pragma: no cover - used by standalone unit tests.
    logger = logging.getLogger(__name__)


@dataclass
class FineMoEDataRecord:
    step: int
    layer: int
    num_tokens: int
    topk: int
    num_total_experts: int
    needed: list[int]
    topk_ids: list[list[int]]
    topk_weights: list[list[float]]
    expert_map_sparse: list[tuple[int, float]]
    timestamp: float


class FineMoEDataSimulator:
    """Collects FineMoE-style expert-map data without doing offload.

    FineMoE uses per-iteration, per-layer expert probability trajectories to
    guide later prefetch decisions. At this hook point vLLM Ascend exposes the
    selected top-k experts and weights, so this collector stores a sparse
    expert-map approximation keyed by decode step and layer.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        topk: int,
        decode_max_tokens: int = 1,
        dump_path: str | None = None,
        dump_interval_steps: int = 198,
        log_updates: bool = True,
    ) -> None:
        self.enabled = enabled
        self.topk = topk
        self.decode_max_tokens = decode_max_tokens
        self.dump_path = dump_path
        self.dump_interval_steps = dump_interval_steps
        self.log_updates = log_updates
        self.records: list[FineMoEDataRecord] = []
        self.step = 0
        self._last_layer: int | None = None
        self._last_dump_step = -1

    def record(
        self,
        *,
        layer_idx: int,
        topk_ids: Any,
        topk_weights: Any,
        num_total_experts: int,
    ) -> FineMoEDataRecord | None:
        if not self.enabled:
            return None

        num_tokens = int(topk_ids.size(0))
        if self.decode_max_tokens > 0 and num_tokens > self.decode_max_tokens:
            if self.log_updates:
                logger.info(
                    "[FINEMOE-SIM] skip layer=%s tokens=%s decode_max_tokens=%s",
                    layer_idx,
                    num_tokens,
                    self.decode_max_tokens,
                )
            return None

        if self._last_layer is not None and layer_idx <= self._last_layer:
            self.step += 1
        self._last_layer = layer_idx

        ids = _tensor_to_nested_ints(topk_ids)
        weights = _tensor_to_nested_floats(topk_weights)
        needed = sorted({expert_id for row in ids for expert_id in row})
        sparse_map = _build_sparse_expert_map(ids, weights)

        record = FineMoEDataRecord(
            step=self.step,
            layer=layer_idx,
            num_tokens=num_tokens,
            topk=len(ids[0]) if ids else self.topk,
            num_total_experts=num_total_experts,
            needed=needed,
            topk_ids=ids,
            topk_weights=weights,
            expert_map_sparse=sparse_map,
            timestamp=time.time(),
        )
        self.records.append(record)

        if self.log_updates:
            logger.warning(
                "[FINEMOE-SIM] step=%d layer=%d needed=%s sparse_map=%s",
                record.step,
                record.layer,
                record.needed,
                record.expert_map_sparse,
            )

        self._maybe_dump()
        return record

    def dump(self) -> None:
        if not self.dump_path:
            return
        if torch is None:
            raise RuntimeError("torch is required to dump FineMoE simulation data")

        directory = os.path.dirname(self.dump_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "format": "finemoe_data_sim_v1",
            "description": (
                "Sparse FineMoE expert-map data collected from topk_ids and "
                "topk_weights. This is for offline trajectory similarity "
                "analysis only and does not perform expert offload."
            ),
            "records": [asdict(record) for record in self.records],
        }
        torch.save(payload, self.dump_path)
        self._last_dump_step = self.step

    def _maybe_dump(self) -> None:
        if not self.dump_path or self.dump_interval_steps <= 0:
            return
        if self.step <= 0 or self.step == self._last_dump_step:
            return
        if self.step % self.dump_interval_steps == 0:
            self.dump()


_FINEMOE_DATA_SIMULATOR: FineMoEDataSimulator | None = None


def maybe_init_finemoe_data_simulator(
    *,
    enabled: bool,
    topk: int,
    decode_max_tokens: int = 1,
    dump_path: str | None = None,
    dump_interval_steps: int = 198,
    log_updates: bool = True,
) -> None:
    global _FINEMOE_DATA_SIMULATOR
    if not enabled:
        _FINEMOE_DATA_SIMULATOR = None
        return
    _FINEMOE_DATA_SIMULATOR = FineMoEDataSimulator(
        enabled=enabled,
        topk=topk,
        decode_max_tokens=decode_max_tokens,
        dump_path=dump_path,
        dump_interval_steps=dump_interval_steps,
        log_updates=log_updates,
    )


def has_finemoe_data_simulator() -> bool:
    return _FINEMOE_DATA_SIMULATOR is not None


def get_finemoe_data_simulator() -> FineMoEDataSimulator:
    if _FINEMOE_DATA_SIMULATOR is None:
        raise RuntimeError("FineMoE data simulator is not initialized")
    return _FINEMOE_DATA_SIMULATOR


def maybe_record_finemoe_data(layer: Any, topk_ids: Any, topk_weights: Any) -> None:
    if _FINEMOE_DATA_SIMULATOR is None:
        return
    layer_idx = int(getattr(layer, "moe_instance_id", getattr(layer, "layer_id", 0)))
    num_total_experts = int(getattr(layer, "global_num_experts", getattr(layer, "local_num_experts", 0)))
    _FINEMOE_DATA_SIMULATOR.record(
        layer_idx=layer_idx,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        num_total_experts=num_total_experts,
    )


def _tensor_to_nested_ints(tensor: Any) -> list[list[int]]:
    values = tensor.detach().cpu().tolist()
    return [[int(item) for item in row] for row in values]


def _tensor_to_nested_floats(tensor: Any) -> list[list[float]]:
    values = tensor.detach().cpu().tolist()
    return [[round(float(item), 8) for item in row] for row in values]


def _build_sparse_expert_map(
    topk_ids: list[list[int]],
    topk_weights: list[list[float]],
) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ids_row, weights_row in zip(topk_ids, topk_weights):
        for expert_id, weight in zip(ids_row, weights_row):
            scores[expert_id] = scores.get(expert_id, 0.0) + float(weight)
    return [(expert_id, round(score, 8)) for expert_id, score in sorted(scores.items())]

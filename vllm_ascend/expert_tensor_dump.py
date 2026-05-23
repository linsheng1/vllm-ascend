"""Dump per-layer MoE top-k expert tensors for offline overlap analysis."""

from __future__ import annotations

import atexit
from pathlib import Path

import torch
from vllm.logger import logger

from vllm_ascend.ascend_config import get_ascend_config


class ExpertTensorDumper:

    def __init__(self) -> None:
        config = get_ascend_config()
        self.enabled = bool(getattr(config, "expert_tensor_dump_enabled", False))
        self.path = getattr(config, "expert_tensor_dump_path", None) or "expert_tensor_dump.pt"
        self.flush_interval = int(getattr(config, "expert_tensor_dump_flush_interval", 198))
        self.max_tokens = int(getattr(config, "expert_tensor_dump_max_tokens", 1))
        self.records: list[dict[str, int | torch.Tensor]] = []
        self.layer_steps: dict[int, int] = {}
        self.layer_ids_by_object: dict[int, int] = {}
        self.last_flushed_step = 0
        if self.enabled:
            atexit.register(self.flush)

    def record(self, layer, topk_ids: torch.Tensor) -> None:
        if not self.enabled:
            return
        if self.max_tokens > 0 and topk_ids.size(0) > self.max_tokens:
            return
        layer_idx = self._layer_idx(layer)
        step = self.layer_steps.get(layer_idx, 0) + 1
        self.layer_steps[layer_idx] = step
        self.records.append(
            {
                "step": step,
                "layer": layer_idx,
                "experts": topk_ids.detach().to(dtype=torch.int32, device="cpu"),
            }
        )
        self._maybe_flush()

    def _layer_idx(self, layer) -> int:
        for attr in ("moe_instance_id", "layer_id"):
            value = getattr(layer, attr, None)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        key = id(layer)
        if key not in self.layer_ids_by_object:
            self.layer_ids_by_object[key] = len(self.layer_ids_by_object)
        return self.layer_ids_by_object[key]

    def _maybe_flush(self) -> None:
        if self.flush_interval == 0 or not self.layer_steps:
            return
        completed_step = min(self.layer_steps.values())
        if completed_step - self.last_flushed_step < self.flush_interval:
            return
        self.flush()
        self.last_flushed_step = completed_step

    def flush(self) -> None:
        if not self.enabled or not self.records:
            return
        path = Path(self.path).expanduser()
        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "num_layers": len(self.layer_steps),
                "flush_interval": self.flush_interval,
                "max_tokens": self.max_tokens,
            },
            "records": self.records,
        }
        torch.save(payload, path)
        logger.info("Saved expert tensor dump with %d records to %s", len(self.records), path)


_DUMPER: ExpertTensorDumper | None = None


def get_expert_tensor_dumper() -> ExpertTensorDumper:
    global _DUMPER
    if _DUMPER is None:
        _DUMPER = ExpertTensorDumper()
    return _DUMPER


def record_expert_topk_tensor(layer, topk_ids: torch.Tensor) -> None:
    get_expert_tensor_dumper().record(layer, topk_ids)

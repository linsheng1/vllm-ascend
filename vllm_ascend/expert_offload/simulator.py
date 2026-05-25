"""Expert offload simulation for hit/miss accounting only."""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from vllm.logger import logger
except ModuleNotFoundError:
    import logging

    logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExpertOffloadSimulationRecord:
    layer_idx: int
    step: int
    needed: list[int]
    hits: list[int]
    misses: list[int]
    loaded: list[int]
    evicted: list[int]
    resident: list[int]
    hit_rate: float


class ExpertOffloadSimulator:
    """Virtual expert cache used only for statistics.

    The simulator mirrors the original offload cache update at the expert-id
    level, but it never stores weights and never copies CPU tensors to NPU.
    """

    def __init__(self, num_device_experts: int, log_updates: bool = True):
        self.num_device_experts = num_device_experts
        self.log_updates = log_updates
        self.layer_steps: dict[int, int] = {}
        self.layer_log2phy: dict[int, list[int]] = {}
        self.records: list[ExpertOffloadSimulationRecord] = []
        self.w13_weights_cpu: list = []
        self.w2_weights_cpu: list = []

    def record(
        self,
        layer_idx: int,
        num_total_experts: int,
        topk_ids: torch.Tensor,
    ) -> ExpertOffloadSimulationRecord:
        log2phy = self._get_log2phy(layer_idx, num_total_experts)
        step = self.layer_steps.get(layer_idx, 0) + 1
        self.layer_steps[layer_idx] = step

        needed = sorted({int(eid) for eid in topk_ids.detach().cpu().flatten().tolist()})
        slot_owner = {
            slot: eid
            for eid, slot in enumerate(log2phy)
            if slot >= 0
        }
        on_device = set(slot_owner.values())
        hits = sorted(set(needed) & on_device)
        misses = sorted(set(needed) - on_device)
        reusable_slots = [
            slot for slot, eid in slot_owner.items()
            if eid not in needed
        ]

        loaded: list[int] = []
        evicted: list[int] = []
        for eid in misses:
            if not reusable_slots:
                break
            slot = reusable_slots.pop()
            old_eid = slot_owner[slot]
            log2phy[old_eid] = -1
            log2phy[eid] = slot
            slot_owner[slot] = eid
            loaded.append(eid)
            evicted.append(old_eid)

        resident = sorted(eid for eid, slot in enumerate(log2phy) if slot >= 0)
        record = ExpertOffloadSimulationRecord(
            layer_idx=layer_idx,
            step=step,
            needed=needed,
            hits=hits,
            misses=misses,
            loaded=loaded,
            evicted=evicted,
            resident=resident,
            hit_rate=(len(hits) / len(needed)) if needed else 1.0,
        )
        self.records.append(record)
        if self.log_updates:
            logger.warning(
                "[EXPERT-OFFLOAD-SIM] layer=%d step=%d needed=%s hits=%s "
                "misses=%s loaded=%s evicted=%s hit_rate=%.6f resident=%s",
                record.layer_idx,
                record.step,
                record.needed,
                record.hits,
                record.misses,
                record.loaded,
                record.evicted,
                record.hit_rate,
                record.resident,
            )
        return record

    def _get_log2phy(self, layer_idx: int, num_total_experts: int) -> list[int]:
        log2phy = self.layer_log2phy.get(layer_idx)
        if log2phy is not None:
            return log2phy
        log2phy = [-1] * num_total_experts
        for eid in range(min(self.num_device_experts, num_total_experts)):
            log2phy[eid] = eid
        self.layer_log2phy[layer_idx] = log2phy
        return log2phy


_EXPERT_OFFLOAD_SIMULATOR: ExpertOffloadSimulator | None = None


def maybe_init_expert_offload_simulator(num_device_experts: int, log_updates: bool = True):
    global _EXPERT_OFFLOAD_SIMULATOR
    if _EXPERT_OFFLOAD_SIMULATOR is None:
        _EXPERT_OFFLOAD_SIMULATOR = ExpertOffloadSimulator(
            num_device_experts=num_device_experts,
            log_updates=log_updates,
        )
        logger.info(
            "Expert offload simulation enabled with num_device_experts=%d",
            num_device_experts,
        )


def has_expert_offload_simulator() -> bool:
    return _EXPERT_OFFLOAD_SIMULATOR is not None


def get_expert_offload_simulator() -> ExpertOffloadSimulator:
    assert _EXPERT_OFFLOAD_SIMULATOR is not None, (
        "Expert offload simulator is not initialized"
    )
    return _EXPERT_OFFLOAD_SIMULATOR


def maybe_record_expert_offload_simulation(layer, topk_ids: torch.Tensor) -> None:
    if _EXPERT_OFFLOAD_SIMULATOR is None:
        return
    layer_idx = getattr(layer, "moe_instance_id", None)
    if layer_idx is None:
        layer_idx = getattr(layer, "layer_id", id(layer))
    num_total_experts = getattr(
        layer,
        "global_num_experts",
        getattr(layer, "local_num_experts", 0),
    )
    if num_total_experts <= 0:
        return
    _EXPERT_OFFLOAD_SIMULATOR.record(
        layer_idx=int(layer_idx),
        num_total_experts=int(num_total_experts),
        topk_ids=topk_ids,
    )

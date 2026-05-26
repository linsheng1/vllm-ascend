"""Expert offload simulation for hit/miss accounting only."""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    from vllm.logger import logger
except ModuleNotFoundError:
    import logging

    logger = logging.getLogger(__name__)

from vllm_ascend.expert_offload.lrc_policy import LRCExpertCachePolicy

try:
    from vllm_ascend.ascend_forward_context import _EXTRA_CTX
except (ImportError, ModuleNotFoundError):
    _EXTRA_CTX = None


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
    policy_step: int


class ExpertOffloadSimulator:
    """Virtual expert cache used only for statistics.

    The simulator mirrors the original offload cache update at the expert-id
    level, but it never stores weights and never copies CPU tensors to NPU.
    """

    def __init__(
        self,
        num_device_experts: int,
        log_updates: bool = True,
        topk: int = 1,
        num_layers: int = 1,
        recent_window: int = 32,
        ema_beta: float = 0.9,
        recent_weight: float = 1.0,
        ema_weight: float = 0.5,
        router_weight: float = 0.3,
        age_weight: float = 0.01,
    ):
        if topk < 1:
            raise ValueError("topk must be >= 1")
        self.num_device_experts = num_device_experts
        self.log_updates = log_updates
        self.layer_steps: dict[int, int] = {}
        self.layer_log2phy: dict[int, list[int]] = {}
        self.layer_policies: dict[int, LRCExpertCachePolicy] = {}
        self.records: list[ExpertOffloadSimulationRecord] = []
        self.w13_weights_cpu: list = []
        self.w2_weights_cpu: list = []
        self.topk = topk
        self.offload_threshold = num_device_experts // topk
        self.num_layers = num_layers
        self.recent_window = recent_window
        self.ema_beta = ema_beta
        self.recent_weight = recent_weight
        self.ema_weight = ema_weight
        self.router_weight = router_weight
        self.age_weight = age_weight

    def record(
        self,
        layer_idx: int,
        num_total_experts: int,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor | None = None,
    ) -> ExpertOffloadSimulationRecord | None:
        if topk_ids.size(0) > self.offload_threshold:
            if self.log_updates:
                logger.warning(
                    "[EXPERT-OFFLOAD-SIM] skip layer=%d topk_shape=%s "
                    "offload_threshold=%d",
                    layer_idx,
                    tuple(topk_ids.shape),
                    self.offload_threshold,
                )
            return None

        log2phy = self._get_log2phy(layer_idx, num_total_experts)
        step = self.layer_steps.get(layer_idx, 0) + 1
        self.layer_steps[layer_idx] = step

        policy = self._get_policy(layer_idx, num_total_experts)
        topk_ids_cpu = topk_ids.detach().cpu()
        use_router_scores = (
            topk_weights is not None
            and self.router_weight != 0
            and not getattr(_EXTRA_CTX, "capturing", False)
        )
        topk_weights_cpu = topk_weights.detach().cpu() if use_router_scores else None
        router_scores = topk_weights_cpu.tolist() if topk_weights_cpu is not None else None
        needed_set = policy.observe(
            layer_idx=0,
            topk_ids=topk_ids_cpu.tolist(),
            router_scores=router_scores,
        )
        needed = sorted(needed_set)
        slot_owner = {
            slot: eid
            for eid, slot in enumerate(log2phy)
            if slot >= 0
        }
        on_device = set(slot_owner.values())
        hit_set = needed_set & on_device
        miss_set = needed_set - on_device
        hits = sorted(hit_set)
        misses = sorted(miss_set)

        loaded: list[int] = []
        evicted: list[int] = []
        for eid in miss_set:
            victim = policy.choose_victim(
                layer_idx=0,
                slot_owner=slot_owner,
                protected=needed_set,
            )
            slot = int(log2phy[victim]) if victim is not None else -1
            if slot < 0:
                break
            log2phy[victim] = -1
            log2phy[eid] = slot
            slot_owner[slot] = eid
            loaded.append(eid)
            evicted.append(victim)

        resident = sorted(eid for eid, slot in enumerate(log2phy) if slot >= 0)
        policy_step = policy.layer_step(0)
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
            policy_step=policy_step,
        )
        self.records.append(record)
        if self.log_updates:
            logger.warning(
                "[EXPERT-OFFLOAD-SIM] layer=%d step=%d policy_step=%d needed=%s "
                "hits=%s misses=%s loaded=%s evicted=%s hit_rate=%.6f resident=%s",
                record.layer_idx,
                record.step,
                record.policy_step,
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

    def _get_policy(self, layer_idx: int, num_total_experts: int) -> LRCExpertCachePolicy:
        policy = self.layer_policies.get(layer_idx)
        if policy is not None:
            return policy
        policy = LRCExpertCachePolicy(
            num_layers=1,
            num_experts=num_total_experts,
            cache_size=self.num_device_experts,
            topk=self.topk,
            recent_window=self.recent_window,
            ema_beta=self.ema_beta,
            recent_weight=self.recent_weight,
            ema_weight=self.ema_weight,
            router_weight=self.router_weight,
            age_weight=self.age_weight,
        )
        self.layer_policies[layer_idx] = policy
        return policy


_EXPERT_OFFLOAD_SIMULATOR: ExpertOffloadSimulator | None = None


def maybe_init_expert_offload_simulator(
    num_device_experts: int,
    log_updates: bool = True,
    topk: int = 1,
    num_layers: int = 1,
    recent_window: int = 32,
    ema_beta: float = 0.9,
    recent_weight: float = 1.0,
    ema_weight: float = 0.5,
    router_weight: float = 0.3,
    age_weight: float = 0.01,
):
    global _EXPERT_OFFLOAD_SIMULATOR
    if _EXPERT_OFFLOAD_SIMULATOR is None:
        _EXPERT_OFFLOAD_SIMULATOR = ExpertOffloadSimulator(
            num_device_experts=num_device_experts,
            log_updates=log_updates,
            topk=topk,
            num_layers=num_layers,
            recent_window=recent_window,
            ema_beta=ema_beta,
            recent_weight=recent_weight,
            ema_weight=ema_weight,
            router_weight=router_weight,
            age_weight=age_weight,
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


def maybe_record_expert_offload_simulation(
    layer,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
) -> None:
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
        topk_weights=topk_weights,
    )

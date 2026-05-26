import torch

from vllm_ascend.expert_offload.simulator import ExpertOffloadSimulator


def test_simulator_tracks_hits_and_misses_without_weight_buffers():
    simulator = ExpertOffloadSimulator(
        num_device_experts=4,
        log_updates=False,
    )

    first = simulator.record(
        layer_idx=0,
        num_total_experts=8,
        topk_ids=torch.tensor([[0, 1], [4, 4]], dtype=torch.int32),
    )
    second = simulator.record(
        layer_idx=0,
        num_total_experts=8,
        topk_ids=torch.tensor([[1, 4], [5, 5]], dtype=torch.int32),
    )

    assert first.step == 1
    assert first.hits == [0, 1]
    assert first.misses == [4]
    assert first.loaded == [4]
    assert first.evicted == [2]
    assert first.hit_rate == 2 / 3

    assert second.step == 2
    assert second.hits == [1, 4]
    assert second.misses == [5]
    assert second.loaded == [5]
    assert second.evicted == [3]
    assert second.hit_rate == 2 / 3

    assert simulator.w13_weights_cpu == []
    assert simulator.w2_weights_cpu == []


def test_simulator_keeps_layer_state_separate():
    simulator = ExpertOffloadSimulator(
        num_device_experts=2,
        log_updates=False,
    )

    layer0 = simulator.record(
        layer_idx=0,
        num_total_experts=4,
        topk_ids=torch.tensor([[0, 2]], dtype=torch.int32),
    )
    layer1 = simulator.record(
        layer_idx=1,
        num_total_experts=4,
        topk_ids=torch.tensor([[1, 2]], dtype=torch.int32),
    )

    assert layer0.step == 1
    assert layer1.step == 1
    assert layer0.hits == [0]
    assert layer1.hits == [1]


def test_simulator_uses_lrc_hotness_to_choose_victim():
    simulator = ExpertOffloadSimulator(
        num_device_experts=2,
        log_updates=False,
        recent_window=4,
        recent_weight=1.0,
        ema_weight=0.0,
        router_weight=0.0,
        age_weight=0.0,
    )

    simulator.record(
        layer_idx=0,
        num_total_experts=4,
        topk_ids=torch.tensor([[1]], dtype=torch.int32),
    )
    miss = simulator.record(
        layer_idx=0,
        num_total_experts=4,
        topk_ids=torch.tensor([[2]], dtype=torch.int32),
    )

    assert miss.loaded == [2]
    assert miss.evicted == [0]
    assert miss.resident == [1, 2]


def test_simulator_skips_prefill_like_real_offload_decode_path():
    simulator = ExpertOffloadSimulator(
        num_device_experts=4,
        topk=2,
        log_updates=False,
    )

    skipped = simulator.record(
        layer_idx=0,
        num_total_experts=8,
        topk_ids=torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32),
    )

    assert skipped is None
    assert simulator.records == []
    assert simulator.layer_policies == {}

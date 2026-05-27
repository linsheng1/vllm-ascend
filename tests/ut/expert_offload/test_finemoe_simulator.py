import tempfile
import unittest
import pickle
import sys
import types
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


class _FakeTensor:

    def __init__(self, data):
        self._data = data
        rows = len(data)
        cols = len(data[0]) if rows and isinstance(data[0], list) else 0
        self.shape = (rows, cols)

    def size(self, dim=None):
        return self.shape if dim is None else self.shape[dim]

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._data


def _tensor(data, dtype=None):
    return _FakeTensor(data)


def _ones(shape, dtype=None):
    rows, cols = shape
    return _FakeTensor([[1.0 for _ in range(cols)] for _ in range(rows)])


def _save(payload, path):
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def _load(path, map_location=None):
    with open(path, "rb") as f:
        return pickle.load(f)


fake_torch = types.SimpleNamespace(
    Tensor=_FakeTensor,
    int32="int32",
    float32="float32",
    tensor=_tensor,
    ones=_ones,
    save=_save,
    load=_load,
)
sys.modules.setdefault("torch", fake_torch)
torch = fake_torch

module_path = REPO_ROOT / "vllm_ascend" / "expert_offload" / "finemoe_simulator.py"
spec = importlib.util.spec_from_file_location("finemoe_simulator", module_path)
finemoe_simulator = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
sys.modules["finemoe_simulator"] = finemoe_simulator
spec.loader.exec_module(finemoe_simulator)
FineMoEDataSimulator = finemoe_simulator.FineMoEDataSimulator


class FineMoEDataSimulatorTest(unittest.TestCase):

    def test_records_decode_trajectory_by_step_and_layer(self):
        simulator = FineMoEDataSimulator(
            enabled=True,
            topk=2,
            dump_interval_steps=0,
        )
        topk_ids = torch.tensor([[3, 1]], dtype=torch.int32)
        topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)

        record = simulator.record(
            layer_idx=4,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            num_total_experts=8,
        )

        self.assertIsNotNone(record)
        self.assertEqual(record.step, 0)
        self.assertEqual(record.layer, 4)
        self.assertEqual(record.needed, [1, 3])
        self.assertEqual(record.topk_ids, [[3, 1]])
        self.assertEqual(record.topk_weights, [[0.7, 0.3]])

        second = simulator.record(
            layer_idx=5,
            topk_ids=torch.tensor([[2, 1]], dtype=torch.int32),
            topk_weights=torch.tensor([[0.6, 0.4]], dtype=torch.float32),
            num_total_experts=8,
        )
        self.assertEqual(second.step, 0)
        self.assertEqual(second.layer, 5)

        third = simulator.record(
            layer_idx=4,
            topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
            topk_weights=torch.tensor([[0.9, 0.1]], dtype=torch.float32),
            num_total_experts=8,
        )
        self.assertEqual(third.step, 1)

    def test_skips_large_batches_to_keep_decode_data_clean(self):
        simulator = FineMoEDataSimulator(
            enabled=True,
            topk=2,
            decode_max_tokens=1,
            dump_interval_steps=0,
        )

        record = simulator.record(
            layer_idx=0,
            topk_ids=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
            topk_weights=torch.ones((2, 2), dtype=torch.float32),
            num_total_experts=8,
        )

        self.assertIsNone(record)
        self.assertEqual(simulator.records, [])

    def test_periodic_dump_contains_all_seen_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/finemoe.pt"
            simulator = FineMoEDataSimulator(
                enabled=True,
                topk=2,
                dump_path=path,
                dump_interval_steps=2,
            )
            for _ in range(3):
                simulator.record(
                    layer_idx=0,
                    topk_ids=torch.tensor([[1, 2]], dtype=torch.int32),
                    topk_weights=torch.tensor([[0.8, 0.2]], dtype=torch.float32),
                    num_total_experts=8,
                )

            payload = torch.load(path, map_location="cpu")
            self.assertEqual(payload["format"], "finemoe_data_sim_v1")
            self.assertEqual(len(payload["records"]), 3)
            self.assertEqual(payload["records"][0]["layer"], 0)


if __name__ == "__main__":
    unittest.main()

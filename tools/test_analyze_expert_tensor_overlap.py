#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from analyze_expert_tensor_overlap import summarize


class AnalyzeExpertTensorOverlapTest(unittest.TestCase):

    def test_summarize_scenarios(self):
        payload = {
            "metadata": {"num_layers": 2, "topk": 3},
            "records": [
                {"step": 1, "layer": 0, "experts": torch.tensor([[1, 2, 3]], dtype=torch.int32)},
                {"step": 1, "layer": 1, "experts": torch.tensor([[4, 5, 6]], dtype=torch.int32)},
                {"step": 2, "layer": 0, "experts": torch.tensor([2, 3, 7], dtype=torch.int32)},
                {"step": 2, "layer": 1, "experts": torch.tensor([4, 8, 9], dtype=torch.int32)},
                {"step": 3, "layer": 0, "experts": torch.tensor([3, 7, 10], dtype=torch.int32)},
                {"step": 3, "layer": 1, "experts": torch.tensor([8, 9, 11], dtype=torch.int32)},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dump.pt"
            torch.save(payload, path)
            summary = summarize(str(path), lag=1)

        prev_step = summary.scenarios["prev_step_same_layer"]
        self.assertEqual(prev_step.pairs, 4)
        self.assertEqual(prev_step.overlap, 7)
        self.assertEqual(prev_step.current_count, 12)
        self.assertAlmostEqual(prev_step.rate, 7 / 12)

        prev_layer = summary.scenarios["same_step_prev_layer"]
        self.assertEqual(prev_layer.pairs, 3)
        self.assertEqual(prev_layer.overlap, 0)
        self.assertEqual(prev_layer.current_count, 9)

        union = summary.scenarios["union_prev_step_or_prev_layer"]
        self.assertEqual(union.pairs, 5)
        self.assertEqual(union.overlap, 7)
        self.assertEqual(union.current_count, 15)


if __name__ == "__main__":
    unittest.main()

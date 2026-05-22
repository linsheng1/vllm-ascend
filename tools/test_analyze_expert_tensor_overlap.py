#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from analyze_expert_tensor_overlap import summarize


class AnalyzeExpertTensorOverlapTest(unittest.TestCase):

    def test_summarize_lag_one(self):
        payload = {
            "metadata": {"num_layers": 2, "topk": 3},
            "records": [
                {"step": 1, "layer": 0, "experts": torch.tensor([1, 2, 3], dtype=torch.int32)},
                {"step": 1, "layer": 1, "experts": torch.tensor([4, 5, 6], dtype=torch.int32)},
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

        self.assertEqual(summary.global_bucket.pairs, 4)
        self.assertEqual(summary.global_bucket.overlap, 7)
        self.assertEqual(summary.global_bucket.current_count, 12)
        self.assertAlmostEqual(summary.global_bucket.rate, 7 / 12)
        self.assertAlmostEqual(summary.layers[0].rate, 4 / 6)
        self.assertAlmostEqual(summary.layers[1].rate, 3 / 6)
        self.assertAlmostEqual(summary.steps[2].rate, 3 / 6)
        self.assertAlmostEqual(summary.steps[3].rate, 4 / 6)


if __name__ == "__main__":
    unittest.main()

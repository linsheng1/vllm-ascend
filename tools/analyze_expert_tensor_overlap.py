#!/usr/bin/env python3
"""Analyze overlap between per-step expert tensors dumped by vllm-ascend."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


SCENARIOS = (
    "prev_step_same_layer",
    "same_step_prev_layer",
    "union_prev_step_or_prev_layer",
)


@dataclass
class Event:
    scenario: str
    step: int
    layer: int
    overlap: int
    current_count: int
    previous_count: int


@dataclass
class Bucket:
    pairs: int = 0
    overlap: int = 0
    current_count: int = 0
    previous_count: int = 0

    def add(self, event: Event) -> None:
        self.pairs += 1
        self.overlap += event.overlap
        self.current_count += event.current_count
        self.previous_count += event.previous_count

    @property
    def rate(self) -> float:
        return self.overlap / self.current_count if self.current_count else 0.0

    @property
    def jaccard(self) -> float:
        union = self.current_count + self.previous_count - self.overlap
        return self.overlap / union if union else 0.0


@dataclass
class Summary:
    metadata: dict
    scenarios: dict[str, Bucket]
    layers: dict[str, dict[int, Bucket]]
    steps: dict[str, dict[int, Bucket]]


def _torch_load(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_set(value) -> set[int]:
    if isinstance(value, torch.Tensor):
        return {int(item) for item in value.cpu().reshape(-1).tolist()}
    return {int(item) for item in value}


def load_records(path: str) -> tuple[dict, dict[tuple[int, int], set[int]]]:
    payload = _torch_load(path)
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError("expected a dict payload with a 'records' field")
    by_key: dict[tuple[int, int], set[int]] = {}
    for record in payload["records"]:
        by_key[(int(record["step"]), int(record["layer"]))] = _as_set(record["experts"])
    return dict(payload.get("metadata", {})), by_key


def summarize(path: str, lag: int = 1) -> Summary:
    if lag < 1:
        raise ValueError("lag must be >= 1")
    metadata, by_key = load_records(path)
    events: list[Event] = []
    for (step, layer), current in sorted(by_key.items()):
        prev_step = by_key.get((step - lag, layer))
        prev_layer = by_key.get((step, layer - 1))
        if prev_step is not None:
            events.append(Event("prev_step_same_layer", step, layer, len(current & prev_step), len(current), len(prev_step)))
        if prev_layer is not None:
            events.append(Event("same_step_prev_layer", step, layer, len(current & prev_layer), len(current), len(prev_layer)))
        if prev_step is not None or prev_layer is not None:
            ref = (prev_step or set()) | (prev_layer or set())
            events.append(Event("union_prev_step_or_prev_layer", step, layer, len(current & ref), len(current), len(ref)))

    scenarios: dict[str, Bucket] = defaultdict(Bucket)
    layers: dict[str, dict[int, Bucket]] = defaultdict(lambda: defaultdict(Bucket))
    steps: dict[str, dict[int, Bucket]] = defaultdict(lambda: defaultdict(Bucket))
    for event in events:
        scenarios[event.scenario].add(event)
        layers[event.scenario][event.layer].add(event)
        steps[event.scenario][event.step].add(event)
    return Summary(
        metadata=metadata,
        scenarios=dict(scenarios),
        layers={scenario: dict(bucket) for scenario, bucket in layers.items()},
        steps={scenario: dict(bucket) for scenario, bucket in steps.items()},
    )


def _bucket_dict(bucket: Bucket) -> dict[str, int | float]:
    return {
        "pairs": bucket.pairs,
        "overlap": bucket.overlap,
        "current_count": bucket.current_count,
        "previous_count": bucket.previous_count,
        "rate": bucket.rate,
        "jaccard": bucket.jaccard,
    }


def _write_csv(path: str, rows: Iterable[dict[str, int | float | str]]) -> None:
    rows = list(rows)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print(summary: Summary, lag: int, top_steps: int) -> None:
    print("GLOBAL")
    print("scenario,lag,pairs,overlap,current,previous,rate,jaccard")
    for scenario in SCENARIOS:
        bucket = summary.scenarios.get(scenario, Bucket())
        print(
            f"{scenario},{lag},{bucket.pairs},{bucket.overlap},{bucket.current_count},"
            f"{bucket.previous_count},{bucket.rate:.6f},{bucket.jaccard:.6f}"
        )
    print("\nPER_LAYER")
    print("scenario,layer,pairs,overlap,current,previous,rate,jaccard")
    for scenario in SCENARIOS:
        for layer, bucket in sorted(summary.layers.get(scenario, {}).items()):
            print(
                f"{scenario},{layer},{bucket.pairs},{bucket.overlap},{bucket.current_count},"
                f"{bucket.previous_count},{bucket.rate:.6f},{bucket.jaccard:.6f}"
            )
    print("\nPER_STEP")
    print("scenario,step,layer_pairs,overlap,current,previous,rate,jaccard")
    for scenario in SCENARIOS:
        scenario_steps = summary.steps.get(scenario, {})
        for step, bucket in sorted(scenario_steps.items())[:top_steps]:
            print(
                f"{scenario},{step},{bucket.pairs},{bucket.overlap},{bucket.current_count},"
                f"{bucket.previous_count},{bucket.rate:.6f},{bucket.jaccard:.6f}"
            )
        if len(scenario_steps) > top_steps:
            print(f"... truncated {len(scenario_steps) - top_steps} {scenario} steps; use --top-steps to show more")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump")
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--top-steps", type=int, default=30)
    parser.add_argument("--layer-csv")
    parser.add_argument("--step-csv")
    args = parser.parse_args()
    summary = summarize(args.dump, lag=args.lag)
    if args.layer_csv:
        _write_csv(
            args.layer_csv,
            (
                {"scenario": scenario, "layer": layer, **_bucket_dict(bucket)}
                for scenario in SCENARIOS
                for layer, bucket in sorted(summary.layers.get(scenario, {}).items())
            ),
        )
    if args.step_csv:
        _write_csv(
            args.step_csv,
            (
                {"scenario": scenario, "step": step, **_bucket_dict(bucket)}
                for scenario in SCENARIOS
                for step, bucket in sorted(summary.steps.get(scenario, {}).items())
            ),
        )
    if args.json:
        print(json.dumps({
            "metadata": summary.metadata,
            "lag": args.lag,
            "global": {scenario: _bucket_dict(summary.scenarios.get(scenario, Bucket())) for scenario in SCENARIOS},
        }, indent=2, sort_keys=True))
    else:
        _print(summary, args.lag, args.top_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

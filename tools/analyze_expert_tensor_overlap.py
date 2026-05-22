#!/usr/bin/env python3
"""Analyze overlap between per-step expert tensors dumped by expert offload."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch


@dataclass
class OverlapEvent:
    step: int
    prev_step: int
    layer: int
    overlap: int
    current_count: int
    previous_count: int

    @property
    def rate(self) -> float:
        return self.overlap / self.current_count if self.current_count else 0.0

    @property
    def jaccard(self) -> float:
        union = self.current_count + self.previous_count - self.overlap
        return self.overlap / union if union else 0.0


@dataclass
class Bucket:
    pairs: int = 0
    overlap: int = 0
    current_count: int = 0
    previous_count: int = 0

    def add(self, event: OverlapEvent) -> None:
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
    events: list[OverlapEvent]
    layers: dict[int, Bucket]
    steps: dict[int, Bucket]
    global_bucket: Bucket


def _torch_load(path: str) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _as_expert_set(value) -> set[int]:
    if isinstance(value, torch.Tensor):
        return {int(item) for item in value.cpu().tolist()}
    return {int(item) for item in value}


def load_records(path: str) -> tuple[dict, dict[tuple[int, int], set[int]]]:
    payload = _torch_load(path)
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError("expected a dict payload with a 'records' field")

    by_key: dict[tuple[int, int], set[int]] = {}
    for record in payload["records"]:
        step = int(record["step"])
        layer = int(record["layer"])
        by_key[(step, layer)] = _as_expert_set(record["experts"])
    return dict(payload.get("metadata", {})), by_key


def summarize(path: str, lag: int = 1) -> Summary:
    if lag < 1:
        raise ValueError("lag must be >= 1")
    metadata, by_key = load_records(path)
    events: list[OverlapEvent] = []

    for (step, layer), current in sorted(by_key.items()):
        previous = by_key.get((step - lag, layer))
        if previous is None:
            continue
        events.append(
            OverlapEvent(
                step=step,
                prev_step=step - lag,
                layer=layer,
                overlap=len(current & previous),
                current_count=len(current),
                previous_count=len(previous),
            )
        )

    layers: dict[int, Bucket] = defaultdict(Bucket)
    steps: dict[int, Bucket] = defaultdict(Bucket)
    global_bucket = Bucket()
    for event in events:
        layers[event.layer].add(event)
        steps[event.step].add(event)
        global_bucket.add(event)

    return Summary(
        metadata=metadata,
        events=events,
        layers=dict(layers),
        steps=dict(steps),
        global_bucket=global_bucket,
    )


def _write_csv(path: str, rows: Iterable[dict[str, int | float]]) -> None:
    rows = list(rows)
    if not rows:
        Path(path).write_text("", encoding="utf-8")
        return
    with Path(path).open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _bucket_json(bucket: Bucket) -> dict[str, int | float]:
    return {
        "pairs": bucket.pairs,
        "overlap": bucket.overlap,
        "current_count": bucket.current_count,
        "previous_count": bucket.previous_count,
        "rate": bucket.rate,
        "jaccard": bucket.jaccard,
    }


def _json_summary(summary: Summary, lag: int) -> dict:
    return {
        "metadata": summary.metadata,
        "lag": lag,
        "global": _bucket_json(summary.global_bucket),
        "layers": {
            str(layer): _bucket_json(bucket)
            for layer, bucket in sorted(summary.layers.items())
        },
        "steps": {
            str(step): _bucket_json(bucket)
            for step, bucket in sorted(summary.steps.items())
        },
    }


def _print_table(summary: Summary, lag: int, top_steps: int) -> None:
    print("GLOBAL")
    print(
        f"  lag={lag} pairs={summary.global_bucket.pairs} "
        f"overlap={summary.global_bucket.overlap} current={summary.global_bucket.current_count} "
        f"previous={summary.global_bucket.previous_count} rate={summary.global_bucket.rate:.6f} "
        f"jaccard={summary.global_bucket.jaccard:.6f}"
    )

    print("\nPER_LAYER")
    print("layer,pairs,overlap,current,previous,rate,jaccard")
    for layer, bucket in sorted(summary.layers.items()):
        print(
            f"{layer},{bucket.pairs},{bucket.overlap},{bucket.current_count},"
            f"{bucket.previous_count},{bucket.rate:.6f},{bucket.jaccard:.6f}"
        )

    print("\nPER_STEP")
    print("step,layer_pairs,overlap,current,previous,rate,jaccard")
    for step, bucket in sorted(summary.steps.items())[:top_steps]:
        print(
            f"{step},{bucket.pairs},{bucket.overlap},{bucket.current_count},"
            f"{bucket.previous_count},{bucket.rate:.6f},{bucket.jaccard:.6f}"
        )
    if len(summary.steps) > top_steps:
        print(f"... truncated {len(summary.steps) - top_steps} steps; use --top-steps to show more")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", help="expert tensor dump path produced by expert offload")
    parser.add_argument("--lag", type=int, default=1, help="step distance to compare")
    parser.add_argument("--json", action="store_true", help="print JSON instead of tables")
    parser.add_argument("--top-steps", type=int, default=30, help="number of per-step rows to print")
    parser.add_argument("--layer-csv", help="write per-layer overlap stats as CSV")
    parser.add_argument("--step-csv", help="write per-step overlap stats as CSV")
    args = parser.parse_args()

    summary = summarize(args.dump, lag=args.lag)

    if args.layer_csv:
        _write_csv(
            args.layer_csv,
            (
                {
                    "layer": layer,
                    "pairs": bucket.pairs,
                    "overlap": bucket.overlap,
                    "current": bucket.current_count,
                    "previous": bucket.previous_count,
                    "rate": bucket.rate,
                    "jaccard": bucket.jaccard,
                }
                for layer, bucket in sorted(summary.layers.items())
            ),
        )
    if args.step_csv:
        _write_csv(
            args.step_csv,
            (
                {
                    "step": step,
                    "layer_pairs": bucket.pairs,
                    "overlap": bucket.overlap,
                    "current": bucket.current_count,
                    "previous": bucket.previous_count,
                    "rate": bucket.rate,
                    "jaccard": bucket.jaccard,
                }
                for step, bucket in sorted(summary.steps.items())
            ),
        )

    if args.json:
        print(json.dumps(_json_summary(summary, args.lag), indent=2, sort_keys=True))
    else:
        _print_table(summary, args.lag, args.top_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

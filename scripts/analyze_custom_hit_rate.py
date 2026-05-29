#!/usr/bin/env python3
"""Evaluate FineMoE-style expert prediction from FINEMOE-SIM logs."""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


SIM_RE = re.compile(
    r"\[FINEMOE-SIM\]\s+step=(?P<step>\d+)\s+layer=(?P<layer>\d+)\s+"
    r"needed=(?P<needed>\[[^\]]*\])\s+sparse_map=(?P<sparse>\[[^\]]*\])"
)
HTTP_OK_RE = re.compile(r'HTTP/[0-9.]+"\s+200\s+OK|HTTP.*200\s+OK')


@dataclass(frozen=True)
class Record:
    request: int
    step: int
    layer: int
    needed: tuple[int, ...]
    sparse_map: dict[int, float]


History = dict[tuple[int, int], Record]
LayerScores = dict[int, Counter[int]]


def parse_records(path: Path) -> list[Record]:
    records: list[Record] = []
    request = 0
    for line in path.open("r", encoding="utf-8", errors="replace"):
        if HTTP_OK_RE.search(line):
            request += 1
            continue

        match = SIM_RE.search(line)
        if not match:
            continue

        sparse = ast.literal_eval(match.group("sparse"))
        records.append(
            Record(
                request=request,
                step=int(match.group("step")),
                layer=int(match.group("layer")),
                needed=tuple(int(x) for x in ast.literal_eval(match.group("needed"))),
                sparse_map={int(expert): float(score) for expert, score in sparse},
            )
        )
    return records


def top_weighted_experts(record: Record, limit: int) -> set[int]:
    return {
        expert
        for expert, _ in sorted(
            record.sparse_map.items(),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
    }


def select_similarity_aware_experts(
    scores: dict[int, float],
    *,
    similarity: float,
    min_experts: int,
    capacity: int,
) -> set[int]:
    if capacity <= 0 or not scores:
        return set()

    delta = max(0.0, min(1.0 - similarity, 1.0))
    total = sum(max(score, 0.0) for score in scores.values())
    normalizer = total if total > 0.0 else 1.0
    selected: set[int] = set()
    cumulative = 0.0

    for expert, score in sorted(scores.items(), key=lambda item: (-item[1], item[0])):
        if len(selected) >= capacity:
            break
        selected.add(expert)
        cumulative += max(score, 0.0) / normalizer
        if len(selected) >= min_experts and cumulative >= delta:
            break

    return selected


def cosine(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot = sum(left[k] * right[k] for k in common)
    left_norm = math.sqrt(sum(v * v for v in left.values()))
    right_norm = math.sqrt(sum(v * v for v in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def build_trajectory(history: History, step: int, max_layer_exclusive: int) -> dict[int, float]:
    trajectory: dict[int, float] = {}
    for layer in range(max_layer_exclusive):
        record = history.get((step, layer))
        if record is None:
            continue
        offset = layer * 1000000
        for expert, score in record.sparse_map.items():
            trajectory[offset + expert] = score
    return trajectory


def layer_popularity_prediction(
    layer_scores: LayerScores,
    record: Record,
    *,
    capacity: int,
    min_experts: int,
) -> set[int]:
    scores = {expert: float(score) for expert, score in layer_scores.get(record.layer, {}).items()}
    return select_similarity_aware_experts(
        scores,
        similarity=0.0,
        min_experts=min_experts,
        capacity=capacity,
    )


def predict_finemoe(
    record: Record,
    *,
    history: History,
    seen_steps: list[int],
    layer_scores: LayerScores,
    capacity: int,
    prefetch_distance: int,
    min_experts: int,
) -> tuple[set[int], str, int | None, float]:
    observed_layers = record.layer - prefetch_distance
    if observed_layers <= 0:
        predicted = layer_popularity_prediction(
            layer_scores,
            record,
            capacity=capacity,
            min_experts=min_experts,
        )
        return predicted, "layer_popularity_fallback", None, 0.0

    current = build_trajectory(history, record.step, observed_layers)
    if not current:
        predicted = layer_popularity_prediction(
            layer_scores,
            record,
            capacity=capacity,
            min_experts=min_experts,
        )
        return predicted, "missing_trajectory_fallback", None, 0.0

    best_step: int | None = None
    best_score = -1.0
    for candidate_step in seen_steps:
        if candidate_step >= record.step:
            continue
        candidate = build_trajectory(history, candidate_step, observed_layers)
        if not candidate:
            continue
        score = cosine(current, candidate)
        if score > best_score:
            best_score = score
            best_step = candidate_step

    if best_step is None:
        predicted = layer_popularity_prediction(
            layer_scores,
            record,
            capacity=capacity,
            min_experts=min_experts,
        )
        return predicted, "no_match_fallback", None, 0.0

    matched = history.get((best_step, record.layer))
    if matched is None:
        predicted = layer_popularity_prediction(
            layer_scores,
            record,
            capacity=capacity,
            min_experts=min_experts,
        )
        return predicted, "missing_target_fallback", best_step, best_score

    predicted = select_similarity_aware_experts(
        matched.sparse_map,
        similarity=best_score,
        min_experts=min_experts,
        capacity=capacity,
    )
    return predicted, "trajectory_search", best_step, best_score


def predict_experts(
    record: Record,
    *,
    history: History,
    seen_steps: list[int],
    capacity: int,
    algorithm: str,
    prefetch_distance: int,
    layer_scores: LayerScores,
    min_experts: int,
) -> tuple[set[int], str, int | None, float]:
    if algorithm == "prev_step_same_layer":
        previous = history.get((record.step - 1, record.layer))
        predicted = top_weighted_experts(previous, capacity) if previous is not None else set()
        return predicted, "prev_step_same_layer", record.step - 1 if previous is not None else None, 0.0

    if algorithm == "same_step_prev_layer":
        previous = history.get((record.step, record.layer - 1))
        predicted = top_weighted_experts(previous, capacity) if previous is not None else set()
        return predicted, "same_step_prev_layer", record.step if previous is not None else None, 0.0

    if algorithm == "union_prev_step_or_prev_layer":
        predicted: set[int] = set()
        previous_step = history.get((record.step - 1, record.layer))
        previous_layer = history.get((record.step, record.layer - 1))
        if previous_step is not None:
            predicted.update(top_weighted_experts(previous_step, capacity))
        if previous_layer is not None:
            predicted.update(top_weighted_experts(previous_layer, capacity))
        return set(sorted(predicted)[:capacity]), "union_prev_step_or_prev_layer", record.step - 1, 0.0

    if algorithm == "trajectory_nn":
        # FineMoE-style offline approximation:
        # use current step layers [0, target_layer - d) to find the most similar
        # historical step trajectory, then use that historical target layer.
        observed_layers = record.layer - prefetch_distance
        if observed_layers <= 0:
            return set(), "insufficient_layers", None, 0.0
        current = build_trajectory(history, record.step, observed_layers)
        if not current:
            return set(), "missing_trajectory", None, 0.0

        best_step: int | None = None
        best_score = -1.0
        for candidate_step in seen_steps:
            if candidate_step >= record.step:
                continue
            candidate = build_trajectory(history, candidate_step, observed_layers)
            score = cosine(current, candidate)
            if score > best_score:
                best_score = score
                best_step = candidate_step

        if best_step is None:
            return set(), "no_match", None, 0.0
        matched = history.get((best_step, record.layer))
        predicted = top_weighted_experts(matched, capacity) if matched is not None else set()
        return predicted, "trajectory_nn", best_step, best_score

    if algorithm == "finemoe":
        return predict_finemoe(
            record,
            history=history,
            seen_steps=seen_steps,
            layer_scores=layer_scores,
            capacity=capacity,
            prefetch_distance=prefetch_distance,
            min_experts=min_experts,
        )

    raise ValueError(f"unknown algorithm: {algorithm}")


def analyze(records: list[Record], args: argparse.Namespace) -> dict[str, object]:
    history: History = {}
    layer_scores: LayerScores = defaultdict(Counter)
    seen_steps: list[int] = []
    seen_step_set: set[int] = set()
    global_total = Counter()
    layer_totals: dict[int, Counter[str]] = defaultdict(Counter)
    step_totals: dict[int, Counter[str]] = defaultdict(Counter)
    request_totals: dict[int, Counter[str]] = defaultdict(Counter)
    rows: list[dict[str, object]] = []

    for record in records:
        if record.step not in seen_step_set:
            seen_step_set.add(record.step)
            seen_steps.append(record.step)

        predicted, source, matched_step, similarity = predict_experts(
            record,
            history=history,
            seen_steps=seen_steps,
            capacity=args.capacity,
            algorithm=args.algorithm,
            prefetch_distance=args.prefetch_distance,
            layer_scores=layer_scores,
            min_experts=args.min_experts,
        )
        needed = set(record.needed)
        overlap = predicted & needed
        needed_count = len(needed)

        for bucket in (
            global_total,
            layer_totals[record.layer],
            step_totals[record.step],
            request_totals[record.request],
        ):
            bucket["needed"] += needed_count
            bucket["overlap"] += len(overlap)
            bucket["predicted"] += len(predicted)
            bucket["records"] += 1

        rows.append(
            {
                "request": record.request,
                "step": record.step,
                "layer": record.layer,
                "needed": needed_count,
                "predicted": len(predicted),
                "overlap": len(overlap),
                "hit_rate": f"{len(overlap) / needed_count:.6f}" if needed_count else "0.000000",
                "needed_experts": sorted(needed),
                "predicted_experts": sorted(predicted),
                "overlap_experts": sorted(overlap),
                "source": source,
                "matched_step": matched_step,
                "similarity": f"{similarity:.6f}",
            }
        )
        history[(record.step, record.layer)] = record
        for expert, score in record.sparse_map.items():
            layer_scores[record.layer][expert] += score

    return {
        "rows": rows,
        "global": global_total,
        "by_layer": layer_totals,
        "by_step": step_totals,
        "by_request": request_totals,
    }


def write_counter_csv(path: Path, key_name: str, counters: dict[int, Counter[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[key_name, "records", "overlap", "needed", "predicted", "hit_rate"],
        )
        writer.writeheader()
        for key in sorted(counters):
            counter = counters[key]
            needed = counter["needed"]
            writer.writerow(
                {
                    key_name: key,
                    "records": counter["records"],
                    "overlap": counter["overlap"],
                    "needed": needed,
                    "predicted": counter["predicted"],
                    "hit_rate": f"{counter['overlap'] / needed:.6f}" if needed else "0.000000",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("custom_hit_rate_analysis"))
    parser.add_argument("--capacity", type=int, default=12)
    parser.add_argument(
        "--algorithm",
        choices=(
            "prev_step_same_layer",
            "same_step_prev_layer",
            "union_prev_step_or_prev_layer",
            "trajectory_nn",
            "finemoe",
        ),
        default="finemoe",
    )
    parser.add_argument("--prefetch-distance", type=int, default=3)
    parser.add_argument("--min-experts", type=int, default=6)
    args = parser.parse_args()

    records = parse_records(args.log)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = analyze(records, args)

    detail_path = args.out_dir / f"{args.algorithm}_detail.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "request",
            "step",
            "layer",
            "needed",
            "predicted",
            "overlap",
            "hit_rate",
            "needed_experts",
            "predicted_experts",
            "overlap_experts",
            "source",
            "matched_step",
            "similarity",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["rows"])

    write_counter_csv(args.out_dir / f"{args.algorithm}_by_layer.csv", "layer", result["by_layer"])
    write_counter_csv(args.out_dir / f"{args.algorithm}_by_step.csv", "step", result["by_step"])
    write_counter_csv(args.out_dir / f"{args.algorithm}_by_request.csv", "request", result["by_request"])

    global_counter = result["global"]
    needed = global_counter["needed"]
    print(f"algorithm={args.algorithm}")
    print(f"records={global_counter['records']}")
    print(f"requests={len(result['by_request'])}")
    print(f"layers={len(result['by_layer'])}")
    print(f"steps={len(result['by_step'])}")
    print(f"overlap={global_counter['overlap']} needed={needed} predicted={global_counter['predicted']}")
    print(f"global_hit_rate={global_counter['overlap'] / needed:.6f}" if needed else "global_hit_rate=0.000000")
    print(f"wrote={args.out_dir}")


if __name__ == "__main__":
    main()

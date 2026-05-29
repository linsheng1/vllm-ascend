#!/usr/bin/env python3
"""Compare LRC optimization candidates from FINEMOE-SIM logs.

This script is for offline strategy exploration. It replays each step/layer in
log order and reports expert-cache recall for several cache policies:

* current_lrc: the current default LRC/hotness policy.
* lrc_no_recent: the best simple parameter sweep observed on the tail log.
* online_static_topk: online per-layer popularity cache.
* protected_base_lrc: online protected popular experts + dynamic LRC slots.
* reuse_distance_lrc: LRC with an inter-arrival-distance eviction penalty.
* per_layer_tuned_lrc: layer-specific LRC weights chosen from a small offline
  grid on the same log. This is an optimistic tuning diagnostic.
* oracle_static_topk: whole-log per-layer top experts. This leaks future data
  and should be treated as an upper-bound style diagnostic only.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
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
    router_scores: dict[int, float]


@dataclass(frozen=True)
class LRCParams:
    recent_weight: float = 1.0
    ema_weight: float = 0.5
    router_weight: float = 0.3
    age_weight: float = 0.01
    reuse_distance_weight: float = 0.0


@dataclass
class LayerState:
    resident: set[int]
    recent: deque[int] = field(default_factory=deque)
    recent_counts: Counter[int] = field(default_factory=Counter)
    ema: dict[int, float] = field(default_factory=dict)
    router_score: dict[int, float] = field(default_factory=dict)
    last_seen: dict[int, int] = field(default_factory=dict)
    interval_ema: dict[int, float] = field(default_factory=dict)
    policy_step: int = 0


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

        needed = tuple(int(x) for x in ast.literal_eval(match.group("needed")))
        sparse = ast.literal_eval(match.group("sparse"))
        records.append(
            Record(
                request=request,
                step=int(match.group("step")),
                layer=int(match.group("layer")),
                needed=needed,
                router_scores={int(expert): float(score) for expert, score in sparse},
            )
        )
    return records


def top_experts(scores: Counter[int] | dict[int, float], limit: int) -> set[int]:
    if limit <= 0:
        return set()
    return {
        expert
        for expert, _ in sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
    }


def make_initial_resident(capacity: int, mode: str) -> set[int]:
    if mode == "first_n":
        return set(range(max(capacity, 0)))
    return set()


def hotness(state: LayerState, expert: int, params: LRCParams) -> float:
    age = state.policy_step - state.last_seen.get(expert, -1)
    reuse_distance = state.interval_ema.get(expert, float(age + 1))
    return (
        params.recent_weight * state.recent_counts.get(expert, 0)
        + params.ema_weight * state.ema.get(expert, 0.0)
        + params.router_weight * state.router_score.get(expert, 0.0)
        - params.age_weight * age
        - params.reuse_distance_weight * reuse_distance
    )


def choose_victim(
    state: LayerState,
    *,
    protected: set[int],
    params: LRCParams,
) -> int | None:
    candidates = [expert for expert in state.resident if expert not in protected]
    if not candidates:
        return None
    return min(candidates, key=lambda expert: hotness(state, expert, params))


def observe(
    state: LayerState,
    record: Record,
    *,
    recent_window: int,
    ema_beta: float,
    interval_beta: float,
) -> None:
    state.policy_step += 1
    active = set(record.needed)

    for expert, score in record.router_scores.items():
        state.router_score[expert] = score

    for expert in active:
        previous = state.last_seen.get(expert)
        if previous is not None:
            interval = state.policy_step - previous
            old = state.interval_ema.get(expert, float(interval))
            state.interval_ema[expert] = interval_beta * old + (1.0 - interval_beta) * interval
        state.last_seen[expert] = state.policy_step
        state.ema[expert] = ema_beta * state.ema.get(expert, 0.0) + (1.0 - ema_beta)
        state.recent.append(expert)
        state.recent_counts[expert] += 1

    known = set(state.ema) | active | set(state.router_score)
    for expert in known - active:
        state.ema[expert] = ema_beta * state.ema.get(expert, 0.0)

    while len(state.recent) > recent_window:
        old = state.recent.popleft()
        state.recent_counts[old] -= 1
        if state.recent_counts[old] <= 0:
            del state.recent_counts[old]


def fill_cache(
    state: LayerState,
    misses: set[int],
    *,
    capacity: int,
    protected: set[int],
    params: LRCParams,
) -> tuple[list[int], list[int]]:
    loaded: list[int] = []
    evicted: list[int] = []
    if capacity <= 0:
        return loaded, evicted

    for expert in sorted(misses):
        while len(state.resident) >= capacity and expert not in state.resident:
            victim = choose_victim(state, protected=protected, params=params)
            if victim is None:
                break
            state.resident.remove(victim)
            evicted.append(victim)
        if len(state.resident) < capacity:
            state.resident.add(expert)
            loaded.append(expert)
    return loaded, evicted


def run_lrc_like(
    records: list[Record],
    *,
    capacity: int,
    initial_resident: str,
    recent_window: int,
    ema_beta: float,
    interval_beta: float,
    params: LRCParams,
    reset_on_request: bool = False,
    protected_base_k: int = 0,
) -> tuple[list[dict[str, object]], dict[int, Counter[str]], Counter[str]]:
    states: dict[tuple[int, int], LayerState] = {}
    layer_scores: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    rows: list[dict[str, object]] = []
    by_layer: dict[int, Counter[str]] = defaultdict(Counter)
    total = Counter()

    for record in records:
        request_key = record.request if reset_on_request else -1
        state_key = (request_key, record.layer)
        score_key = (request_key, record.layer)
        if state_key not in states:
            states[state_key] = LayerState(make_initial_resident(capacity, initial_resident))
        state = states[state_key]

        protected_base = top_experts(layer_scores[score_key], protected_base_k)
        for expert in protected_base:
            if len(state.resident) < capacity:
                state.resident.add(expert)
        resident_before = set(state.resident)
        needed = set(record.needed)
        hits = needed & resident_before
        misses = needed - resident_before

        observe(
            state,
            record,
            recent_window=recent_window,
            ema_beta=ema_beta,
            interval_beta=interval_beta,
        )
        protected = needed | protected_base
        loaded, evicted = fill_cache(
            state,
            misses,
            capacity=capacity,
            protected=protected,
            params=params,
        )

        for expert, score in record.router_scores.items():
            layer_scores[score_key][expert] += score

        add_totals(total, by_layer[record.layer], len(hits), len(misses), len(needed))
        rows.append(make_row("lrc_like", record, hits, misses, resident_before, loaded, evicted, state.resident))

    return rows, by_layer, total


def run_online_static(
    records: list[Record],
    *,
    capacity: int,
    reset_on_request: bool = False,
) -> tuple[list[dict[str, object]], dict[int, Counter[str]], Counter[str]]:
    layer_scores: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    rows: list[dict[str, object]] = []
    by_layer: dict[int, Counter[str]] = defaultdict(Counter)
    total = Counter()

    for record in records:
        request_key = record.request if reset_on_request else -1
        score_key = (request_key, record.layer)
        resident = top_experts(layer_scores[score_key], capacity)
        needed = set(record.needed)
        hits = needed & resident
        misses = needed - resident

        for expert, score in record.router_scores.items():
            layer_scores[score_key][expert] += score

        add_totals(total, by_layer[record.layer], len(hits), len(misses), len(needed))
        rows.append(make_row("online_static_topk", record, hits, misses, resident, [], [], resident))

    return rows, by_layer, total


def run_oracle_static(
    records: list[Record],
    *,
    capacity: int,
) -> tuple[list[dict[str, object]], dict[int, Counter[str]], Counter[str]]:
    layer_scores: dict[int, Counter[int]] = defaultdict(Counter)
    for record in records:
        for expert, score in record.router_scores.items():
            layer_scores[record.layer][expert] += score

    residents = {layer: top_experts(scores, capacity) for layer, scores in layer_scores.items()}
    rows: list[dict[str, object]] = []
    by_layer: dict[int, Counter[str]] = defaultdict(Counter)
    total = Counter()

    for record in records:
        resident = residents.get(record.layer, set())
        needed = set(record.needed)
        hits = needed & resident
        misses = needed - resident
        add_totals(total, by_layer[record.layer], len(hits), len(misses), len(needed))
        rows.append(make_row("oracle_static_topk", record, hits, misses, resident, [], [], resident))

    return rows, by_layer, total


def tune_layers(
    records: list[Record],
    *,
    capacity: int,
    initial_resident: str,
    recent_window: int,
    ema_beta: float,
    interval_beta: float,
) -> dict[int, LRCParams]:
    grid = [
        LRCParams(1.0, 0.5, 0.3, 0.01, 0.0),
        LRCParams(0.0, 2.0, 0.3, 0.01, 0.0),
        LRCParams(0.0, 1.0, 0.0, 0.01, 0.0),
        LRCParams(0.5, 1.0, 0.3, 0.01, 0.0),
        LRCParams(0.0, 2.0, 0.3, 0.01, 0.05),
        LRCParams(0.0, 2.0, 0.3, 0.01, 0.10),
    ]
    by_layer_records: dict[int, list[Record]] = defaultdict(list)
    for record in records:
        by_layer_records[record.layer].append(record)

    tuned: dict[int, LRCParams] = {}
    for layer, layer_records in by_layer_records.items():
        best_rate = -1.0
        best_params = grid[0]
        for params in grid:
            _, _, total = run_lrc_like(
                layer_records,
                capacity=capacity,
                initial_resident=initial_resident,
                recent_window=recent_window,
                ema_beta=ema_beta,
                interval_beta=interval_beta,
                params=params,
            )
            needed = total["needed"]
            rate = total["hits"] / needed if needed else 0.0
            if rate > best_rate:
                best_rate = rate
                best_params = params
        tuned[layer] = best_params
    return tuned


def run_per_layer_tuned(
    records: list[Record],
    *,
    capacity: int,
    initial_resident: str,
    recent_window: int,
    ema_beta: float,
    interval_beta: float,
) -> tuple[list[dict[str, object]], dict[int, Counter[str]], Counter[str]]:
    tuned = tune_layers(
        records,
        capacity=capacity,
        initial_resident=initial_resident,
        recent_window=recent_window,
        ema_beta=ema_beta,
        interval_beta=interval_beta,
    )
    rows: list[dict[str, object]] = []
    by_layer: dict[int, Counter[str]] = defaultdict(Counter)
    total = Counter()
    all_rows: list[dict[str, object]] = []
    for layer in sorted(tuned):
        layer_records = [record for record in records if record.layer == layer]
        layer_rows, _, _ = run_lrc_like(
            layer_records,
            capacity=capacity,
            initial_resident=initial_resident,
            recent_window=recent_window,
            ema_beta=ema_beta,
            interval_beta=interval_beta,
            params=tuned[layer],
        )
        for row in layer_rows:
            row["strategy"] = "per_layer_tuned_lrc"
            row["params"] = params_to_string(tuned[layer])
        all_rows.extend(layer_rows)

    all_rows.sort(key=lambda row: (int(row["step"]), int(row["layer"])))
    for row in all_rows:
        hits = int(row["hits"])
        misses = int(row["misses"])
        needed = int(row["needed"])
        layer = int(row["layer"])
        total["hits"] += hits
        total["misses"] += misses
        total["needed"] += needed
        total["records"] += 1
        by_layer[layer]["hits"] += hits
        by_layer[layer]["misses"] += misses
        by_layer[layer]["needed"] += needed
        by_layer[layer]["records"] += 1
        rows.append(row)
    return rows, by_layer, total


def add_totals(total: Counter[str], layer_total: Counter[str], hits: int, misses: int, needed: int) -> None:
    for bucket in (total, layer_total):
        bucket["hits"] += hits
        bucket["misses"] += misses
        bucket["needed"] += needed
        bucket["records"] += 1


def make_row(
    strategy: str,
    record: Record,
    hits: set[int],
    misses: set[int],
    resident_before: set[int],
    loaded: list[int],
    evicted: list[int],
    resident_after: set[int],
) -> dict[str, object]:
    needed = len(record.needed)
    return {
        "strategy": strategy,
        "request": record.request,
        "step": record.step,
        "layer": record.layer,
        "needed": needed,
        "hits": len(hits),
        "misses": len(misses),
        "hit_rate": f"{len(hits) / needed:.6f}" if needed else "0.000000",
        "needed_experts": list(record.needed),
        "hit_experts": sorted(hits),
        "resident_before": sorted(resident_before),
        "loaded": loaded,
        "evicted": evicted,
        "resident_after": sorted(resident_after),
        "params": "",
    }


def params_to_string(params: LRCParams) -> str:
    return (
        f"recent={params.recent_weight},ema={params.ema_weight},"
        f"router={params.router_weight},age={params.age_weight},"
        f"reuse_distance={params.reuse_distance_weight}"
    )


def summarize_strategy(strategy: str, total: Counter[str]) -> dict[str, object]:
    needed = total["needed"]
    return {
        "strategy": strategy,
        "records": total["records"],
        "hits": total["hits"],
        "misses": total["misses"],
        "needed": needed,
        "hit_rate": f"{total['hits'] / needed:.6f}" if needed else "0.000000",
    }


def run_strategy(
    strategy: str,
    records: list[Record],
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[int, Counter[str]], Counter[str]]:
    if strategy == "current_lrc":
        return run_lrc_like(
            records,
            capacity=args.capacity,
            initial_resident=args.initial_resident,
            recent_window=args.recent_window,
            ema_beta=args.ema_beta,
            interval_beta=args.interval_beta,
            params=LRCParams(),
            reset_on_request=args.reset_on_request,
        )
    if strategy == "lrc_no_recent":
        return run_lrc_like(
            records,
            capacity=args.capacity,
            initial_resident=args.initial_resident,
            recent_window=args.recent_window,
            ema_beta=args.ema_beta,
            interval_beta=args.interval_beta,
            params=LRCParams(0.0, 2.0, 0.3, 0.01, 0.0),
            reset_on_request=args.reset_on_request,
        )
    if strategy == "online_static_topk":
        return run_online_static(records, capacity=args.capacity, reset_on_request=args.reset_on_request)
    if strategy == "protected_base_lrc":
        return run_lrc_like(
            records,
            capacity=args.capacity,
            initial_resident=args.initial_resident,
            recent_window=args.recent_window,
            ema_beta=args.ema_beta,
            interval_beta=args.interval_beta,
            params=LRCParams(0.0, 2.0, 0.3, 0.01, 0.0),
            reset_on_request=args.reset_on_request,
            protected_base_k=args.base_k,
        )
    if strategy == "reuse_distance_lrc":
        return run_lrc_like(
            records,
            capacity=args.capacity,
            initial_resident=args.initial_resident,
            recent_window=args.recent_window,
            ema_beta=args.ema_beta,
            interval_beta=args.interval_beta,
            params=LRCParams(0.0, 2.0, 0.3, 0.01, args.reuse_distance_weight),
            reset_on_request=args.reset_on_request,
        )
    if strategy == "per_layer_tuned_lrc":
        return run_per_layer_tuned(
            records,
            capacity=args.capacity,
            initial_resident=args.initial_resident,
            recent_window=args.recent_window,
            ema_beta=args.ema_beta,
            interval_beta=args.interval_beta,
        )
    if strategy == "oracle_static_topk":
        return run_oracle_static(records, capacity=args.capacity)
    raise ValueError(f"unknown strategy: {strategy}")


def write_detail(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "strategy",
        "request",
        "step",
        "layer",
        "needed",
        "hits",
        "misses",
        "hit_rate",
        "needed_experts",
        "hit_experts",
        "resident_before",
        "loaded",
        "evicted",
        "resident_after",
        "params",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_by_layer(path: Path, layer_results: dict[str, dict[int, Counter[str]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["strategy", "layer", "records", "hits", "misses", "needed", "hit_rate"],
        )
        writer.writeheader()
        for strategy in sorted(layer_results):
            for layer in sorted(layer_results[strategy]):
                counter = layer_results[strategy][layer]
                needed = counter["needed"]
                writer.writerow(
                    {
                        "strategy": strategy,
                        "layer": layer,
                        "records": counter["records"],
                        "hits": counter["hits"],
                        "misses": counter["misses"],
                        "needed": needed,
                        "hit_rate": f"{counter['hits'] / needed:.6f}" if needed else "0.000000",
                    }
                )


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["strategy", "records", "hits", "misses", "needed", "hit_rate"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("lrc_strategy_sweep"))
    parser.add_argument("--capacity", type=int, default=12)
    parser.add_argument("--base-k", type=int, default=4)
    parser.add_argument("--reuse-distance-weight", type=float, default=0.05)
    parser.add_argument("--initial-resident", choices=("first_n", "empty"), default="first_n")
    parser.add_argument("--reset-on-request", action="store_true")
    parser.add_argument("--recent-window", type=int, default=32)
    parser.add_argument("--ema-beta", type=float, default=0.9)
    parser.add_argument("--interval-beta", type=float, default=0.8)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "current_lrc",
            "lrc_no_recent",
            "online_static_topk",
            "protected_base_lrc",
            "reuse_distance_lrc",
            "per_layer_tuned_lrc",
            "oracle_static_topk",
        ],
        choices=[
            "current_lrc",
            "lrc_no_recent",
            "online_static_topk",
            "protected_base_lrc",
            "reuse_distance_lrc",
            "per_layer_tuned_lrc",
            "oracle_static_topk",
        ],
    )
    parser.add_argument("--write-detail", action="store_true")
    args = parser.parse_args()

    records = parse_records(args.log)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    layer_results: dict[str, dict[int, Counter[str]]] = {}
    detail_rows: list[dict[str, object]] = []

    for strategy in args.strategies:
        rows, by_layer, total = run_strategy(strategy, records, args)
        for row in rows:
            row["strategy"] = strategy
        summary = summarize_strategy(strategy, total)
        summary_rows.append(summary)
        layer_results[strategy] = by_layer
        if args.write_detail:
            detail_rows.extend(rows)
        print(
            f"{strategy}: hit_rate={summary['hit_rate']} "
            f"hits={summary['hits']} misses={summary['misses']} needed={summary['needed']}"
        )

    write_summary(args.out_dir / "strategy_summary.csv", summary_rows)
    write_by_layer(args.out_dir / "strategy_by_layer.csv", layer_results)
    if args.write_detail:
        write_detail(args.out_dir / "strategy_detail.csv", detail_rows)
    print(f"wrote={args.out_dir}")


if __name__ == "__main__":
    main()

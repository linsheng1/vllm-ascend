#!/usr/bin/env python3
"""Simulate LRC expert-cache hit rate from FINEMOE-SIM logs."""

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


@dataclass
class LayerState:
    resident: set[int]
    recent: deque[int] = field(default_factory=deque)
    recent_counts: Counter[int] = field(default_factory=Counter)
    ema: dict[int, float] = field(default_factory=dict)
    router_score: dict[int, float] = field(default_factory=dict)
    last_seen: dict[int, int] = field(default_factory=dict)
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
        router_scores = {int(expert): float(score) for expert, score in sparse}
        records.append(
            Record(
                request=request,
                step=int(match.group("step")),
                layer=int(match.group("layer")),
                needed=needed,
                router_scores=router_scores,
            )
        )
    return records


def hotness(
    state: LayerState,
    expert: int,
    *,
    recent_weight: float,
    ema_weight: float,
    router_weight: float,
    age_weight: float,
) -> float:
    age = state.policy_step - state.last_seen.get(expert, -1)
    return (
        recent_weight * state.recent_counts.get(expert, 0)
        + ema_weight * state.ema.get(expert, 0.0)
        + router_weight * state.router_score.get(expert, 0.0)
        - age_weight * age
    )


def choose_victim(
    state: LayerState,
    protected: set[int],
    *,
    recent_weight: float,
    ema_weight: float,
    router_weight: float,
    age_weight: float,
) -> int | None:
    candidates = [expert for expert in state.resident if expert not in protected]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda expert: hotness(
            state,
            expert,
            recent_weight=recent_weight,
            ema_weight=ema_weight,
            router_weight=router_weight,
            age_weight=age_weight,
        ),
    )


def update_policy(
    state: LayerState,
    record: Record,
    *,
    capacity: int,
    recent_window: int,
    ema_beta: float,
    recent_weight: float,
    ema_weight: float,
    router_weight: float,
    age_weight: float,
) -> tuple[set[int], set[int], list[int], list[int], set[int]]:
    needed = set(record.needed)
    before = set(state.resident)
    hits = needed & before
    misses = needed - before

    state.policy_step += 1
    for expert, score in record.router_scores.items():
        state.router_score[expert] = score

    active = set(record.needed)
    for expert in active:
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

    loaded: list[int] = []
    evicted: list[int] = []
    for expert in sorted(misses):
        if capacity <= 0:
            break
        while len(state.resident) >= capacity and expert not in state.resident:
            victim = choose_victim(
                state,
                protected=needed,
                recent_weight=recent_weight,
                ema_weight=ema_weight,
                router_weight=router_weight,
                age_weight=age_weight,
            )
            if victim is None:
                break
            state.resident.remove(victim)
            evicted.append(victim)
        if len(state.resident) < capacity:
            state.resident.add(expert)
            loaded.append(expert)

    return hits, misses, loaded, evicted, before


def simulate(records: list[Record], args: argparse.Namespace) -> dict[str, object]:
    states: dict[tuple[int, int], LayerState] = {}
    layer_totals: dict[int, Counter[str]] = defaultdict(Counter)
    step_totals: dict[int, Counter[str]] = defaultdict(Counter)
    request_totals: dict[int, Counter[str]] = defaultdict(Counter)
    rows: list[dict[str, object]] = []
    global_total = Counter()

    for record in records:
        state_key = (record.request if args.reset_on_request else -1, record.layer)
        if state_key not in states:
            initial = set(range(max(args.capacity, 0))) if args.initial_resident == "first_n" else set()
            states[state_key] = LayerState(resident=initial)
        state = states[state_key]

        hits, misses, loaded, evicted, resident_before = update_policy(
            state,
            record,
            capacity=args.capacity,
            recent_window=args.recent_window,
            ema_beta=args.ema_beta,
            recent_weight=args.recent_weight,
            ema_weight=args.ema_weight,
            router_weight=args.router_weight,
            age_weight=args.age_weight,
        )
        needed_count = len(record.needed)
        hit_count = len(hits)
        miss_count = len(misses)
        rate = hit_count / needed_count if needed_count else 0.0

        for bucket in (
            global_total,
            layer_totals[record.layer],
            step_totals[record.step],
            request_totals[record.request],
        ):
            bucket["needed"] += needed_count
            bucket["hits"] += hit_count
            bucket["misses"] += miss_count
            bucket["records"] += 1

        rows.append(
            {
                "request": record.request,
                "step": record.step,
                "layer": record.layer,
                "needed": needed_count,
                "hits": hit_count,
                "misses": miss_count,
                "hit_rate": f"{rate:.6f}",
                "needed_experts": list(record.needed),
                "resident_before": sorted(resident_before),
                "loaded": loaded,
                "evicted": evicted,
                "resident_after": sorted(state.resident),
            }
        )

    return {
        "rows": rows,
        "global": global_total,
        "by_layer": layer_totals,
        "by_step": step_totals,
        "by_request": request_totals,
    }


def write_counter_csv(path: Path, key_name: str, counters: dict[int, Counter[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[key_name, "records", "hits", "misses", "needed", "hit_rate"])
        writer.writeheader()
        for key in sorted(counters):
            counter = counters[key]
            needed = counter["needed"]
            writer.writerow(
                {
                    key_name: key,
                    "records": counter["records"],
                    "hits": counter["hits"],
                    "misses": counter["misses"],
                    "needed": needed,
                    "hit_rate": f"{counter['hits'] / needed:.6f}" if needed else "0.000000",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("lrc_hit_rate_analysis"))
    parser.add_argument("--capacity", type=int, default=12)
    parser.add_argument("--initial-resident", choices=("first_n", "empty"), default="first_n")
    parser.add_argument("--reset-on-request", action="store_true")
    parser.add_argument("--recent-window", type=int, default=32)
    parser.add_argument("--ema-beta", type=float, default=0.9)
    parser.add_argument("--recent-weight", type=float, default=1.0)
    parser.add_argument("--ema-weight", type=float, default=0.5)
    parser.add_argument("--router-weight", type=float, default=0.3)
    parser.add_argument("--age-weight", type=float, default=0.01)
    args = parser.parse_args()

    records = parse_records(args.log)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = simulate(records, args)

    detail_path = args.out_dir / "lrc_detail.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "request",
            "step",
            "layer",
            "needed",
            "hits",
            "misses",
            "hit_rate",
            "needed_experts",
            "resident_before",
            "loaded",
            "evicted",
            "resident_after",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["rows"])

    write_counter_csv(args.out_dir / "lrc_by_layer.csv", "layer", result["by_layer"])
    write_counter_csv(args.out_dir / "lrc_by_step.csv", "step", result["by_step"])
    write_counter_csv(args.out_dir / "lrc_by_request.csv", "request", result["by_request"])

    global_counter = result["global"]
    needed = global_counter["needed"]
    print(f"records={global_counter['records']}")
    print(f"requests={len(result['by_request'])}")
    print(f"layers={len(result['by_layer'])}")
    print(f"steps={len(result['by_step'])}")
    print(f"hits={global_counter['hits']} misses={global_counter['misses']} needed={needed}")
    print(f"global_hit_rate={global_counter['hits'] / needed:.6f}" if needed else "global_hit_rate=0.000000")
    print(f"wrote={args.out_dir}")


if __name__ == "__main__":
    main()

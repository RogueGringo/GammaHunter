# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""CLI: generate JSONL reachability datasets under RESEARCH mode.

No training, no model claims — writes labeled graph instances only.
Hop distance + is_ood follow ADR-001 (see docs/ADR-001-metrics-and-compute.md).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterator, Optional

from reachability_gen.adr_invariants import (
    HOP_UNREACHABLE,
    ID_HOP_MAX,
    ID_HOP_MIN,
    K_TRAIN_MAX,
    OOD_HOP_VALUES,
    is_ood_hop,
)
from reachability_gen.encode import edge_hash as compute_edge_hash
from reachability_gen.encode import encode_instance
from reachability_gen.graph import adjacency_list, er_digraph, hop_distance
from reachability_gen.schema import ReachabilityExample
from reachability_gen.splits import (
    P_VALUES,
    SPLITS,
    derive_example_seed,
    get_split_spec,
)


def _reachability_matrix(n: int, edges: list[tuple[int, int]]) -> list[list[bool]]:
    """Full directed reachability (incl. diagonal True) via multi-source BFS."""
    adj = adjacency_list(n, edges)
    mat = [[False] * n for _ in range(n)]
    for s in range(n):
        seen = mat[s]
        q: deque[int] = deque([s])
        seen[s] = True
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
    return mat


def _hop_matrix(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """All-pairs shortest-path hop distances; unreachable = -1; diagonal = 0."""
    adj = adjacency_list(n, edges)
    mat = [[-1] * n for _ in range(n)]
    for s in range(n):
        mat[s][s] = 0
        q: deque[int] = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if mat[s][v] < 0:
                    mat[s][v] = mat[s][u] + 1
                    q.append(v)
    return mat


def _pairs_with_label(mat: list[list[bool]], y: int) -> list[tuple[int, int]]:
    n = len(mat)
    want = bool(y)
    return [(s, t) for s in range(n) for t in range(n) if mat[s][t] == want]


def _hop_buckets(
    hop_mat: list[list[int]],
    pairs: list[tuple[int, int]],
) -> dict[int, list[tuple[int, int]]]:
    buckets: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for s, t in pairs:
        k = hop_mat[s][t]
        if k >= 0:
            buckets[k].append((s, t))
    return buckets


def _prefer_hop_pool(
    buckets: dict[int, list[tuple[int, int]]],
    preferred_hops: list[int],
    rng: random.Random,
) -> tuple[int, int, int] | None:
    """Pick (s,t,hop) from preferred hop buckets if any are non-empty."""
    available = [k for k in preferred_hops if buckets.get(k)]
    if not available:
        return None
    k = available[rng.randrange(len(available))]
    s, t = buckets[k][rng.randrange(len(buckets[k]))]
    return s, t, k


def _stratified_sample(
    n: int,
    p: float,
    target_y: int,
    rng: random.Random,
    max_rejects: int = 10_000,
    *,
    prefer_id_hops: bool = True,
    prefer_ood_hops: bool = False,
) -> tuple[list[tuple[int, int]], int, int, int, int, bool]:
    """Sample ER digraph, then pick (s,t) from the target-label pool.

    For y=1, prefers stratified hop buckets (ID [2,6] or OOD {8,12,16}) when
    the pool permits; otherwise any reachable pair. Returns
    (edges, s, t, hop_distance, n_attempts, success).
    """
    id_hops = list(range(ID_HOP_MIN, ID_HOP_MAX + 1))
    ood_hops = list(OOD_HOP_VALUES)

    for attempt in range(1, max_rejects + 1):
        edges = er_digraph(n, p, rng)
        mat = _reachability_matrix(n, edges)
        hop_mat = _hop_matrix(n, edges)
        pool = _pairs_with_label(mat, target_y)
        if not pool:
            continue

        if target_y == 1:
            buckets = _hop_buckets(hop_mat, pool)
            preferred: list[int] = []
            if prefer_ood_hops:
                preferred.extend(ood_hops)
            if prefer_id_hops:
                preferred.extend(id_hops)
            # Round-robin preference: try preferred first, then any hop >= 0.
            picked = _prefer_hop_pool(buckets, preferred, rng) if preferred else None
            if picked is None:
                # Fall back: any reachable pair (incl. hop 0 / hop 1).
                s, t = pool[rng.randrange(len(pool))]
                k = hop_mat[s][t]
            else:
                s, t, k = picked
            return edges, s, t, k, attempt, True

        # y=0: unreachable; hop sentinel.
        s, t = pool[rng.randrange(len(pool))]
        return edges, s, t, HOP_UNREACHABLE, attempt, True

    return [], -1, -1, HOP_UNREACHABLE, max_rejects, False


def _any_sample(
    n: int,
    p: float,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], int, int, int, int]:
    """Fallback: one ER draw + uniform (s,t); true label may break 50/50."""
    edges = er_digraph(n, p, rng)
    mat = _reachability_matrix(n, edges)
    s = rng.randrange(n)
    t = rng.randrange(n)
    y = 1 if mat[s][t] else 0
    if y == 1:
        k = hop_distance(n, edges, s, t)
        hop = int(k) if k is not None else HOP_UNREACHABLE
    else:
        hop = HOP_UNREACHABLE
    return edges, s, t, y, hop


def generate_split(
    split: str,
    n_per_cell: int,
    *,
    max_rejects: int = 10_000,
    seed_override: Optional[int] = None,
    prefer_id_hops: bool = True,
    prefer_ood_hops: bool = False,
) -> tuple[list[ReachabilityExample], float]:
    """Generate a ≈balanced split via stratified rejection + hop preference.

    For each (n, p) cell, target ceil/floor half positive/negative. When a
    label class is unavailable within max_rejects (ER nearly strongly
    connected), fall back to an unconstrained sample so generation still
    completes; reject_rate and realized label counts remain in the logs.

    Positive examples prefer hop buckets toward ID [2,6] (and optionally OOD
    {8,12,16} when ``prefer_ood_hops``). ``is_ood`` follows ADR-001:
    ``hop_distance > K_TRAIN_MAX`` (unreachable → False).
    """
    spec = get_split_spec(split)
    master_seed = seed_override if seed_override is not None else spec.seed
    # Size-OOD split: also try hop-OOD buckets when graphs are large enough.
    use_ood_hops = prefer_ood_hops or (split == "ood_size")
    examples: list[ReachabilityExample] = []
    total_attempts = 0
    total_accepted = 0
    example_index = 0

    for n in spec.n_values:
        for p in spec.p_values:
            n_pos = n_per_cell // 2
            n_neg = n_per_cell - n_pos
            schedule: list[int] = [1] * n_pos + [0] * n_neg
            cell_rng = random.Random(
                derive_example_seed(master_seed, example_index ^ (n * 1009) ^ int(p * 1e6))
            )
            cell_rng.shuffle(schedule)

            for target_y in schedule:
                ex_seed = derive_example_seed(master_seed, example_index)
                rng = random.Random(ex_seed)
                edges, s, t, hop, attempts, ok = _stratified_sample(
                    n,
                    p,
                    target_y,
                    rng,
                    max_rejects=max_rejects,
                    prefer_id_hops=prefer_id_hops,
                    prefer_ood_hops=use_ood_hops,
                )
                total_attempts += attempts
                if ok:
                    y = target_y
                else:
                    # Best-effort: keep ER marginal; balance may slip on hard cells.
                    edges, s, t, y, hop = _any_sample(n, p, rng)
                    total_attempts += 1
                total_accepted += 1
                eh = compute_edge_hash(edges)
                enc = encode_instance(n, edges, s, t)
                examples.append(
                    ReachabilityExample(
                        split=split,
                        seed=ex_seed,
                        n=n,
                        p=p,
                        edge_hash=eh,
                        s=s,
                        t=t,
                        y=y,
                        hop_distance=hop,
                        is_ood=is_ood_hop(hop, k_train_max=K_TRAIN_MAX),
                        encoding=enc,
                        arm_id=None,
                        arm_meta=None,
                        n_attempts=attempts if ok else attempts + 1,
                    )
                )
                example_index += 1

    reject_rate = (
        (total_attempts - total_accepted) / total_attempts if total_attempts else 0.0
    )
    for ex in examples:
        ex.reject_rate = reject_rate
    return examples, reject_rate


def write_jsonl(
    path: Path,
    examples: Iterator[ReachabilityExample] | list[ReachabilityExample],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), sort_keys=True) + "\n")
            count += 1
    return count


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reachability-gen",
        description=(
            "RESEARCH scaffold: generate directed-graph reachability JSONL "
            "(no training, no model claims)."
        ),
    )
    p.add_argument(
        "--split",
        choices=list(SPLITS),
        default="train",
        help="Which split to generate (default: train).",
    )
    p.add_argument(
        "--n-per-cell",
        type=int,
        default=4,
        help="Examples per (n, p) cell; labels target ≈50/50 (default: 4).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL path (default: data/<split>.jsonl).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override split master seed (default: fixed seed table).",
    )
    p.add_argument(
        "--max-rejects",
        type=int,
        default=10_000,
        help="Max graph rejections per example when a label pool is empty.",
    )
    p.add_argument(
        "--all-splits",
        action="store_true",
        help="Generate all splits into data/<split>.jsonl.",
    )
    p.add_argument(
        "--prefer-ood-hops",
        action="store_true",
        help="Prefer hop-OOD buckets {8,12,16} for y=1 when available.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    splits = list(SPLITS) if args.all_splits else [args.split]
    for split in splits:
        out = args.out
        if out is None or args.all_splits:
            out = Path("data") / f"{split}.jsonl"
        examples, reject_rate = generate_split(
            split,
            args.n_per_cell,
            max_rejects=args.max_rejects,
            seed_override=args.seed if not args.all_splits else None,
            prefer_ood_hops=args.prefer_ood_hops,
        )
        n_written = write_jsonl(out, examples)
        n_pos = sum(ex.y for ex in examples)
        print(
            f"wrote {n_written} examples → {out} "
            f"(reject_rate={reject_rate:.4f}, y=1:{n_pos}, y=0:{n_written - n_pos}, "
            f"p_grid={list(P_VALUES)}, K_TRAIN_MAX={K_TRAIN_MAX})",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Generate a fixed 2k ID reachability JSONL for MEASURE FF/Geo comparison.

Constraints (science_open=False always):
  - 2000 examples: 1600 train / 400 val (field ``split``)
  - Exact 1000 y=1 / 1000 y=0 overall
  - Train 800/800, val 200/200
  - Positives: hop_distance uniform across K in {2,3,4,5,6} (200 overall /
    160 train / 40 val per hop)
  - Negatives: ALL hard — deg(s)>=1, deg(t)>=1, unreachable
  - Fixed seed; writes a generation report JSON

Uses ER digraph sampling with expanded n support for scarce long hops.
MEASURE plumbing only — no science OPEN claims.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.adr_invariants import (
    HOP_UNREACHABLE,
    ID_HOP_MAX,
    ID_HOP_MIN,
    K_TRAIN_MAX,
    SCIENCE_OPEN_DEFAULT,
    is_ood_hop,
)
from reachability_gen.encode import edge_hash as compute_edge_hash
from reachability_gen.encode import encode_instance
from reachability_gen.graph import adjacency_list, er_digraph
from reachability_gen.hard_negatives import is_hard_negative, total_degrees
from reachability_gen.schema import ReachabilityExample
from reachability_gen.splits import P_VALUES, derive_example_seed

# Locked MEASURE seed for regenerability of id_2k.
ID_2K_SEED: int = 42_000

# Quotas
N_TOTAL: int = 2000
N_TRAIN: int = 1600
N_VAL: int = 400
N_POS_TOTAL: int = 1000
N_NEG_TOTAL: int = 1000
N_POS_TRAIN: int = 800
N_NEG_TRAIN: int = 800
N_POS_VAL: int = 200
N_NEG_VAL: int = 200

ID_HOPS: tuple[int, ...] = tuple(range(ID_HOP_MIN, ID_HOP_MAX + 1))  # 2..6
# 1000 pos / 5 hops = 200; train 160 / hop, val 40 / hop
POS_PER_HOP_TOTAL: int = N_POS_TOTAL // len(ID_HOPS)  # 200
POS_PER_HOP_TRAIN: int = N_POS_TRAIN // len(ID_HOPS)  # 160
POS_PER_HOP_VAL: int = N_POS_VAL // len(ID_HOPS)  # 40

# Expanded n support: hop-6 is scarce on n=8; include larger sparse graphs.
N_SUPPORT_SHORT: tuple[int, ...] = (8, 12, 16)
N_SUPPORT_LONG: tuple[int, ...] = (16, 20, 24, 28, 32)
P_SUPPORT_SHORT: tuple[float, ...] = P_VALUES  # 0.15, 0.25, 0.35
P_SUPPORT_LONG: tuple[float, ...] = (0.08, 0.10, 0.12, 0.15, 0.20)


def _hop_matrix(n: int, edges: Sequence[tuple[int, int]]) -> list[list[int]]:
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


def _n_p_for_hop(hop: int, rng: random.Random) -> tuple[int, float]:
    """Prefer larger / sparser graphs for longer hops."""
    if hop >= 5:
        n = rng.choice(N_SUPPORT_LONG)
        p = rng.choice(P_SUPPORT_LONG)
    elif hop == 4:
        n = rng.choice((12, 16, 20, 24))
        p = rng.choice((0.10, 0.12, 0.15, 0.20, 0.25))
    else:
        n = rng.choice(N_SUPPORT_SHORT)
        p = rng.choice(P_SUPPORT_SHORT)
    # Need at least hop+1 nodes for a simple path of length hop.
    if n < hop + 1:
        n = hop + 1
    return n, p


def _n_p_for_neg(rng: random.Random) -> tuple[int, float]:
    # Mix sizes; lower p yields more unreachable hard pairs.
    if rng.random() < 0.5:
        return rng.choice(N_SUPPORT_SHORT), rng.choice(P_SUPPORT_SHORT)
    return rng.choice(N_SUPPORT_LONG), rng.choice(P_SUPPORT_LONG)


def _harvest_graph(
    n: int,
    edges: list[tuple[int, int]],
    *,
    want_hops: Optional[set[int]] = None,
    want_hard_neg: bool = True,
) -> tuple[dict[int, list[tuple[int, int]]], list[tuple[int, int]]]:
    """Return (positives_by_hop, hard_neg_pairs) from one graph."""
    hop_mat = _hop_matrix(n, edges)
    deg = total_degrees(n, edges)
    pos: dict[int, list[tuple[int, int]]] = defaultdict(list)
    negs: list[tuple[int, int]] = []
    hop_filter = want_hops if want_hops is not None else set(ID_HOPS)
    for s in range(n):
        for t in range(n):
            k = hop_mat[s][t]
            if k in hop_filter:
                pos[k].append((s, t))
            elif want_hard_neg and k < 0 and deg[s] >= 1 and deg[t] >= 1:
                # Double-check via helper (reachable check already from hop).
                ok, _ = is_hard_negative(n, edges, s, t, y=0)
                if ok:
                    negs.append((s, t))
    return pos, negs


def _make_example(
    *,
    split: str,
    seed: int,
    n: int,
    p: float,
    edges: list[tuple[int, int]],
    s: int,
    t: int,
    y: int,
    hop: int,
    n_attempts: int = 1,
) -> ReachabilityExample:
    eh = compute_edge_hash(edges)
    enc = encode_instance(n, edges, s, t)
    return ReachabilityExample(
        split=split,
        seed=seed,
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
        n_attempts=n_attempts,
    )


def generate_id_2k(
    *,
    seed: int = ID_2K_SEED,
    max_graph_draws: int = 50_000,
    progress_every: int = 500,
) -> tuple[list[ReachabilityExample], dict[str, Any]]:
    """Generate the locked 2k ID dataset; return (examples, report)."""
    master = random.Random(seed)
    # Pools of (n, p, edges, s, t) keyed by hop for positives; list for negs.
    pos_pools: dict[int, list[tuple[int, float, list[tuple[int, int]], int, int]]] = {
        k: [] for k in ID_HOPS
    }
    neg_pool: list[tuple[int, float, list[tuple[int, int]], int, int]] = []
    seen_keys: set[tuple[str, int, int]] = set()

    need_pos = {k: POS_PER_HOP_TOTAL for k in ID_HOPS}
    need_neg = N_NEG_TOTAL
    draws = 0
    reject_reasons: Counter = Counter()

    def _key(edges, s, t) -> tuple[str, int, int]:
        return (compute_edge_hash(edges), s, t)

    # Fill pools until quotas met.
    while any(len(pos_pools[k]) < need_pos[k] for k in ID_HOPS) or len(neg_pool) < need_neg:
        if draws >= max_graph_draws:
            break
        draws += 1
        # Bias draws toward scarce hop buckets.
        scarce = [k for k in ID_HOPS if len(pos_pools[k]) < need_pos[k]]
        if scarce and (len(neg_pool) >= need_neg or master.random() < 0.7):
            target_hop = scarce[master.randrange(len(scarce))]
            # Prefer the scarcest.
            scarce.sort(key=lambda k: need_pos[k] - len(pos_pools[k]), reverse=True)
            target_hop = scarce[0]
            n, p = _n_p_for_hop(target_hop, master)
            want_hops: Optional[set[int]] = {target_hop}
            # Still collect other hops / negs opportunistically from same graph.
            want_hops = set(ID_HOPS)
        else:
            n, p = _n_p_for_neg(master)
            want_hops = set(ID_HOPS)

        edges = er_digraph(n, p, master)
        pos_by_hop, hard_negs = _harvest_graph(
            n, edges, want_hops=want_hops, want_hard_neg=True
        )

        for k, pairs in pos_by_hop.items():
            if k not in need_pos:
                continue
            master.shuffle(pairs)
            for s, t in pairs:
                if len(pos_pools[k]) >= need_pos[k]:
                    break
                key = _key(edges, s, t)
                if key in seen_keys:
                    reject_reasons["dup_pos"] += 1
                    continue
                seen_keys.add(key)
                # Store a copy of edges (immutable enough after sort).
                pos_pools[k].append((n, p, list(edges), s, t))

        if len(neg_pool) < need_neg:
            master.shuffle(hard_negs)
            for s, t in hard_negs:
                if len(neg_pool) >= need_neg:
                    break
                key = _key(edges, s, t)
                if key in seen_keys:
                    reject_reasons["dup_neg"] += 1
                    continue
                seen_keys.add(key)
                neg_pool.append((n, p, list(edges), s, t))

        if progress_every and draws % progress_every == 0:
            filled_pos = {k: len(pos_pools[k]) for k in ID_HOPS}
            print(
                f"gen_id_2k draws={draws} pos={filled_pos} neg={len(neg_pool)}",
                file=sys.stderr,
            )

    # Verify pools filled.
    shortfalls: dict[str, Any] = {}
    for k in ID_HOPS:
        if len(pos_pools[k]) < need_pos[k]:
            shortfalls[f"pos_hop_{k}"] = {
                "have": len(pos_pools[k]),
                "need": need_pos[k],
            }
    if len(neg_pool) < need_neg:
        shortfalls["neg"] = {"have": len(neg_pool), "need": need_neg}
    if shortfalls:
        raise RuntimeError(
            f"id_2k generation shortfall after {draws} draws: {shortfalls}"
        )

    # Deterministic trim / shuffle of each pool.
    for k in ID_HOPS:
        pool_rng = random.Random(derive_example_seed(seed, k * 1000))
        pool_rng.shuffle(pos_pools[k])
        pos_pools[k] = pos_pools[k][: POS_PER_HOP_TOTAL]
    neg_rng = random.Random(derive_example_seed(seed, 99_000))
    neg_rng.shuffle(neg_pool)
    neg_pool = neg_pool[:N_NEG_TOTAL]

    examples: list[ReachabilityExample] = []
    example_index = 0

    # Assign train/val splits: first POS_PER_HOP_TRAIN → train, rest → val per hop.
    for k in ID_HOPS:
        for i, (n, p, edges, s, t) in enumerate(pos_pools[k]):
            split = "train" if i < POS_PER_HOP_TRAIN else "val"
            ex_seed = derive_example_seed(seed, example_index)
            examples.append(
                _make_example(
                    split=split,
                    seed=ex_seed,
                    n=n,
                    p=p,
                    edges=edges,
                    s=s,
                    t=t,
                    y=1,
                    hop=k,
                )
            )
            example_index += 1

    for i, (n, p, edges, s, t) in enumerate(neg_pool):
        split = "train" if i < N_NEG_TRAIN else "val"
        ex_seed = derive_example_seed(seed, example_index)
        examples.append(
            _make_example(
                split=split,
                seed=ex_seed,
                n=n,
                p=p,
                edges=edges,
                s=s,
                t=t,
                y=0,
                hop=HOP_UNREACHABLE,
            )
        )
        example_index += 1

    # Final shuffle within each split for stable but mixed order in JSONL.
    train = [e for e in examples if e.split == "train"]
    val = [e for e in examples if e.split == "val"]
    random.Random(derive_example_seed(seed, 1)).shuffle(train)
    random.Random(derive_example_seed(seed, 2)).shuffle(val)
    ordered = train + val

    report = build_generation_report(ordered, seed=seed, graph_draws=draws)
    report["reject_reasons"] = dict(reject_reasons)
    report["n_support_short"] = list(N_SUPPORT_SHORT)
    report["n_support_long"] = list(N_SUPPORT_LONG)
    report["science_open"] = False
    return ordered, report


def build_generation_report(
    examples: Sequence[ReachabilityExample] | Sequence[dict[str, Any]],
    *,
    seed: int = ID_2K_SEED,
    graph_draws: Optional[int] = None,
) -> dict[str, Any]:
    """Counts per hop / class / split for the generation report."""
    rows = [
        e.to_dict() if isinstance(e, ReachabilityExample) else dict(e)
        for e in examples
    ]
    by_split: dict[str, Counter] = defaultdict(Counter)
    by_hop: Counter = Counter()
    by_split_hop: dict[str, Counter] = defaultdict(Counter)
    by_split_y: dict[str, Counter] = defaultdict(Counter)
    y_counts: Counter = Counter()
    n_values_used: Counter = Counter()

    for r in rows:
        split = str(r.get("split", ""))
        y = int(r["y"])
        hop = int(r["hop_distance"])
        by_split[split]["total"] += 1
        by_split_y[split][f"y{y}"] += 1
        y_counts[y] += 1
        by_hop[hop] += 1
        by_split_hop[split][hop] += 1
        n_values_used[int(r.get("n", 0))] += 1

    return {
        "seed": seed,
        "n_total": len(rows),
        "n_train": by_split["train"]["total"],
        "n_val": by_split["val"]["total"],
        "y_counts": {str(k): v for k, v in sorted(y_counts.items())},
        "y_by_split": {s: dict(c) for s, c in sorted(by_split_y.items())},
        "hop_counts": {str(k): v for k, v in sorted(by_hop.items())},
        "hop_by_split": {
            s: {str(k): v for k, v in sorted(c.items())}
            for s, c in sorted(by_split_hop.items())
        },
        "n_values_used": {str(k): v for k, v in sorted(n_values_used.items())},
        "graph_draws": graph_draws,
        "quotas": {
            "pos_per_hop_total": POS_PER_HOP_TOTAL,
            "pos_per_hop_train": POS_PER_HOP_TRAIN,
            "pos_per_hop_val": POS_PER_HOP_VAL,
            "neg_train": N_NEG_TRAIN,
            "neg_val": N_NEG_VAL,
        },
        "science_open": False,
    }


def verify_id_2k(
    examples: Sequence[ReachabilityExample] | Sequence[dict[str, Any]] | Path,
    *,
    check_hard_neg: bool = True,
) -> tuple[bool, list[str]]:
    """Verify id_2k constraints. Returns ``(ok, list_of_issues)``."""
    if isinstance(examples, Path):
        rows: list[dict[str, Any]] = []
        with examples.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        rows = [
            e.to_dict() if isinstance(e, ReachabilityExample) else dict(e)
            for e in examples
        ]

    issues: list[str] = []
    if len(rows) != N_TOTAL:
        issues.append(f"n_total={len(rows)} != {N_TOTAL}")

    train = [r for r in rows if r.get("split") == "train"]
    val = [r for r in rows if r.get("split") == "val"]
    if len(train) != N_TRAIN:
        issues.append(f"n_train={len(train)} != {N_TRAIN}")
    if len(val) != N_VAL:
        issues.append(f"n_val={len(val)} != {N_VAL}")

    y1 = sum(1 for r in rows if int(r["y"]) == 1)
    y0 = sum(1 for r in rows if int(r["y"]) == 0)
    if y1 != N_POS_TOTAL or y0 != N_NEG_TOTAL:
        issues.append(f"y counts y1={y1} y0={y0} want {N_POS_TOTAL}/{N_NEG_TOTAL}")

    for split_name, split_rows, n_pos, n_neg in (
        ("train", train, N_POS_TRAIN, N_NEG_TRAIN),
        ("val", val, N_POS_VAL, N_NEG_VAL),
    ):
        sp = sum(1 for r in split_rows if int(r["y"]) == 1)
        sn = sum(1 for r in split_rows if int(r["y"]) == 0)
        if sp != n_pos or sn != n_neg:
            issues.append(
                f"{split_name} y1={sp} y0={sn} want {n_pos}/{n_neg}"
            )

    # Hop uniformity for positives.
    for split_name, split_rows, per_hop in (
        ("overall", rows, POS_PER_HOP_TOTAL),
        ("train", train, POS_PER_HOP_TRAIN),
        ("val", val, POS_PER_HOP_VAL),
    ):
        for k in ID_HOPS:
            c = sum(
                1
                for r in split_rows
                if int(r["y"]) == 1 and int(r["hop_distance"]) == k
            )
            if c != per_hop:
                issues.append(
                    f"{split_name} hop={k} pos count={c} want {per_hop}"
                )

    # Negatives: hop=-1 and hard.
    for r in rows:
        if int(r["y"]) == 0:
            if int(r["hop_distance"]) != HOP_UNREACHABLE:
                issues.append(
                    f"y0 hop={r['hop_distance']} want {HOP_UNREACHABLE} "
                    f"edge_hash={r.get('edge_hash')}"
                )
                break
        else:
            hop = int(r["hop_distance"])
            if hop < ID_HOP_MIN or hop > ID_HOP_MAX:
                issues.append(f"y1 hop={hop} out of ID range")
                break
            if bool(r.get("is_ood", False)):
                issues.append(f"y1 marked is_ood hop={hop}")
                break

    if check_hard_neg:
        from reachability_gen.hard_negatives import classify_y0_row

        for r in rows:
            if int(r["y"]) != 0:
                continue
            ok, reason = classify_y0_row(r)
            if not ok:
                issues.append(
                    f"soft/invalid negative reason={reason} "
                    f"s={r.get('s')} t={r.get('t')} n={r.get('n')}"
                )
                break  # one is enough to fail; avoid huge lists

    # No science_open True in rows (optional field).
    for r in rows:
        if r.get("science_open") is True:
            issues.append("row has science_open=True")
            break

    return len(issues) == 0, issues


def write_jsonl(path: Path, examples: Sequence[ReachabilityExample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), sort_keys=True) + "\n")
            count += 1
    return count


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate data/id_2k.jsonl (MEASURE plumbing; no science OPEN)."
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/id_2k.jsonl"),
        help="Output JSONL path (default: data/id_2k.jsonl).",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/id_2k_generation_report.json"),
        help="Generation report JSON path.",
    )
    p.add_argument("--seed", type=int, default=ID_2K_SEED)
    p.add_argument("--max-graph-draws", type=int, default=50_000)
    p.add_argument(
        "--verify-only",
        type=Path,
        default=None,
        help="Only verify an existing JSONL; do not regenerate.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only is not None:
        ok, issues = verify_id_2k(args.verify_only)
        print(json.dumps({"ok": ok, "issues": issues}, sort_keys=True))
        return 0 if ok else 1

    examples, report = generate_id_2k(
        seed=args.seed, max_graph_draws=args.max_graph_draws
    )
    n = write_jsonl(args.out, examples)
    ok, issues = verify_id_2k(examples)
    report["verify_ok"] = ok
    report["verify_issues"] = issues
    write_report(args.report, report)
    print(
        f"wrote {n} examples → {args.out}; report → {args.report}; "
        f"verify_ok={ok}",
        file=sys.stderr,
    )
    if not ok:
        print(f"VERIFY FAIL: {issues}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "n": n, "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

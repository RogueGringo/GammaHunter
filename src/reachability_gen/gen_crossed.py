# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Generate crossed reachability sets: four queries per graph (MEASURE plumbing).

Each graph holds two node-disjoint paths of its stratum's hop length k,
``s1 → … → t1`` and ``s2 → … → t2``, and contributes four queries:
``(s1, t1)`` and ``(s2, t2)`` are reachable at shortest distance k, while the
crossings ``(s1, t2)`` and ``(s2, t1)`` are unreachable. Every source and every
target thus appears once with each label, so no rule that reads a single
endpoint, however it is computed, can score above 0.5: only the relation
between the two endpoints decides the label. (The paired sets of
``gen_id_disjoint`` balance labels per graph but not per endpoint, and rules on
one endpoint's reach score up to 0.86 on them; see ``reach_cues``.)

Construction: the two paths are planted with two bridges (an off-lane hub
that is a common sink or common source of the two paths, placed so that each
crossing is k apart when edge direction is ignored, as the positives are);
random edges are then proposed over all ordered node pairs in random order and
each is kept with probability p, unless it would lead from a node a source
reaches into a node that reaches a target. The planted paths thus stay the
only links between the two regions: no crossing, shortcut, chord or detour,
so each positive's path is unique and positives gain no connection that the
negatives cannot have. Node ids are then relabelled at random, and the encoding lists
edges in canonical order, so neither ids nor order mark the paths. Splits are
graph-disjoint; negatives are strong hard negatives by construction; encodings
are capped at the set's token limit.

``science_open=false`` always.

Usage::

    python -m reachability_gen.gen_crossed --n-total 20000 --n-val 4000 \\
        --out data/id_crossed_20k.jsonl --report artifacts/id_crossed_20k_generation_report.json
    python -m reachability_gen.gen_crossed --spec extended --n-total 2000 --n-val 2000 \\
        --seed 191000 --out data/extended_crossed_2k.jsonl \\
        --report artifacts/extended_crossed_2k_generation_report.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.adr_invariants import HOP_UNREACHABLE, K_TRAIN_MAX, is_ood_hop
from reachability_gen.encode import edge_hash as compute_edge_hash
from reachability_gen.encode import encode_instance, parse_instance
from reachability_gen.gen_id_2k import _hop_matrix, write_jsonl, write_report
from reachability_gen.gen_id_disjoint import (
    EXTENDED_SPEC,
    ID_HOPS,
    MAX_TOKENS,
    SetSpec,
    endpoint_rule_accuracy,
)
from reachability_gen.hard_negatives import classify_y0_row, row_has_endpoint_cue
from reachability_gen.schema import ReachabilityExample
from reachability_gen.splits import derive_example_seed
from reachability_gen.tokenize import split_encoding_tokens

CROSSED_SEED: int = 190_500
ROWS_PER_GRAPH: int = 4

# Graph sizes are shared by every hop and leave room for both paths of the
# longest hop; density (mean out-degree of the proposed random edges) is too.
CROSSED_ID_SPEC = SetSpec(
    hops=ID_HOPS,
    n_support=(16, 18, 20, 22, 24),
    mean_degree_by_hop={k: (1.5, 2.0, 2.5) for k in ID_HOPS},
    max_tokens=MAX_TOKENS,
)
CROSSED_EXTENDED_SPEC = SetSpec(
    hops=EXTENDED_SPEC.hops,
    n_support=(40, 44, 48),
    mean_degree_by_hop={k: (1.5, 2.0, 2.5) for k in EXTENDED_SPEC.hops},
    max_tokens=EXTENDED_SPEC.max_tokens,
)
SPECS: dict[str, SetSpec] = {"id": CROSSED_ID_SPEC, "extended": CROSSED_EXTENDED_SPEC}

DEFAULT_OUT = Path("data/id_crossed_20k.jsonl")
DEFAULT_REPORT = Path("artifacts/id_crossed_20k_generation_report.json")
N_TOTAL: int = 20_000
N_VAL: int = 4_000


def min_nodes(k: int) -> int:
    """Two node-disjoint k-hop paths plus two bridge hubs."""
    return 2 * (k + 1) + 2


def _distances(adj: list[list[int]], start: int) -> list[int]:
    """BFS distances from ``start`` along ``adj`` (-1 where unreachable)."""
    dist = [-1] * len(adj)
    dist[start] = 0
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def plant_crossed(
    n: int, k: int, p: float, rng: random.Random
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    """Graph with two planted k-hop paths whose crossings stay unreachable.

    Returns ``(edges, (s1, t1), (s2, t2))`` with relabelled node ids.
    """
    if n < min_nodes(k):
        raise ValueError(f"n={n} cannot hold two node-disjoint {k}-hop paths and their bridges")
    lanes = (list(range(0, k + 1)), list(range(k + 1, 2 * k + 2)))
    (s1, t1), (s2, t2) = ((lane[0], lane[-1]) for lane in lanes)
    edges = {(a, b) for lane in lanes for a, b in zip(lane, lane[1:])}
    # Bridges: an off-lane hub that is a common sink or a common source of
    # lane-1 position i and lane-2 position i + 2 (and the mirror image), so
    # each crossing is k apart when direction is ignored, like the positives,
    # while no directed path joins the lanes.
    hubs = (2 * k + 2, 2 * k + 3)
    i, j = rng.randrange(k - 1), rng.randrange(k - 1)
    for hub, a, b in ((hubs[0], lanes[0][i], lanes[1][i + 2]), (hubs[1], lanes[1][j], lanes[0][j + 2])):
        edges |= {(a, hub), (b, hub)} if rng.random() < 0.5 else {(hub, a), (hub, b)}
    fwd: list[list[int]] = [[] for _ in range(n)]
    bwd: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        fwd[a].append(b)
        bwd[b].append(a)

    def refresh() -> tuple[list[int], list[int], list[int], list[int]]:
        return _distances(fwd, s1), _distances(fwd, s2), _distances(bwd, t1), _distances(bwd, t2)

    from1, from2, to1, to2 = refresh()
    proposals = [(u, v) for u in range(n) for v in range(n) if u != v and (u, v) not in edges]
    rng.shuffle(proposals)
    for u, v in proposals:
        if rng.random() >= p:
            continue
        # The planted paths stay the only links from what the sources reach to
        # what reaches the targets: this forbids crossings, shortcuts, chords
        # and detours alike, so positives gain no connection negatives lack.
        if (from1[u] >= 0 or from2[u] >= 0) and (to1[v] >= 0 or to2[v] >= 0):
            continue
        edges.add((u, v))
        fwd[u].append(v)
        bwd[v].append(u)
        from1, from2, to1, to2 = refresh()

    perm = list(range(n))
    rng.shuffle(perm)
    relabelled = sorted((perm[a], perm[b]) for a, b in edges)
    return relabelled, (perm[s1], perm[t1]), (perm[s2], perm[t2])


def _quotas(n_total: int, n_val: int, n_hops: int) -> tuple[int, int]:
    """Graphs per hop overall and in val (four rows per graph)."""
    unit = ROWS_PER_GRAPH * n_hops
    if n_total % unit or n_val % unit or not 0 < n_val <= n_total:
        raise ValueError(
            f"n_total={n_total} and n_val={n_val} must be multiples of {unit}, "
            "with 0 < n_val <= n_total"
        )
    return n_total // unit, n_val // unit


def generate_crossed(
    *,
    seed: int = CROSSED_SEED,
    n_total: int = N_TOTAL,
    n_val: int = N_VAL,
    spec: SetSpec = CROSSED_ID_SPEC,
    max_graph_draws: int = 1_000_000,
) -> tuple[list[ReachabilityExample], dict[str, Any]]:
    """Generate a crossed set; return (examples, report)."""
    graphs_per_hop, graphs_per_hop_val = _quotas(n_total, n_val, len(spec.hops))
    master = random.Random(seed)
    records: dict[int, list[tuple[int, float, list[tuple[int, int]], tuple[int, int], tuple[int, int]]]] = {
        k: [] for k in spec.hops
    }
    seen: set[str] = set()
    rejects: Counter = Counter()
    draws = 0
    while any(len(records[k]) < graphs_per_hop for k in spec.hops):
        if draws >= max_graph_draws:
            raise RuntimeError(f"crossed shortfall after {draws} draws: { {k: len(v) for k, v in records.items()} }")
        draws += 1
        hop = min(spec.hops, key=lambda k: (len(records[k]), k))  # fill evenly
        n = master.choice([m for m in spec.n_support if m >= min_nodes(hop)])
        p = round(master.choice(spec.mean_degree_by_hop[hop]) / (n - 1), 4)
        edges, (s1, t1), (s2, t2) = plant_crossed(n, hop, p, master)
        eh = compute_edge_hash(edges)
        if eh in seen:
            rejects["duplicate_graph"] += 1
            continue
        if len(split_encoding_tokens(encode_instance(n, edges, s1, t1))) > spec.max_tokens:
            rejects["overlong"] += 1
            continue
        seen.add(eh)
        records[hop].append((n, p, edges, (s1, t1), (s2, t2)))

    examples: list[ReachabilityExample] = []
    index = 0
    for k in spec.hops:
        random.Random(derive_example_seed(seed, k * 1000)).shuffle(records[k])
        for i, (n, p, edges, (s1, t1), (s2, t2)) in enumerate(records[k]):
            split = "val" if i < graphs_per_hop_val else "train"
            eh = compute_edge_hash(edges)
            queries = (((s1, t1), 1), ((s2, t2), 1), ((s1, t2), 0), ((s2, t1), 0))
            for (s, t), y in queries:
                hop = k if y == 1 else HOP_UNREACHABLE
                examples.append(
                    ReachabilityExample(
                        split=split,
                        seed=derive_example_seed(seed, index),
                        n=n,
                        p=p,
                        edge_hash=eh,
                        s=s,
                        t=t,
                        y=y,
                        hop_distance=hop,
                        is_ood=is_ood_hop(hop, k_train_max=K_TRAIN_MAX),
                        encoding=encode_instance(n, edges, s, t),
                    )
                )
                index += 1

    train = [e for e in examples if e.split == "train"]
    val = [e for e in examples if e.split == "val"]
    random.Random(derive_example_seed(seed, 1)).shuffle(train)
    random.Random(derive_example_seed(seed, 2)).shuffle(val)
    ordered = train + val
    report = build_report(ordered, seed=seed, graph_draws=draws, spec=spec)
    report["reject_reasons"] = dict(sorted(rejects.items()))
    return ordered, report


def _rows(examples: Sequence[ReachabilityExample] | Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e.to_dict() if isinstance(e, ReachabilityExample) else dict(e) for e in examples]


def endpoint_balance(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Endpoints (per graph) not seen exactly once with each label; 0 when crossed."""
    seen: dict[tuple[str, str, int], list[int]] = {}
    for r in rows:
        seen.setdefault((r["edge_hash"], "s", int(r["s"])), []).append(int(r["y"]))
        seen.setdefault((r["edge_hash"], "t", int(r["t"])), []).append(int(r["y"]))
    off = Counter(kind for (_, kind, _), ys in seen.items() if sorted(ys) != [0, 1])
    return {"sources_unbalanced": off.get("s", 0), "targets_unbalanced": off.get("t", 0)}


def build_report(
    examples: Sequence[ReachabilityExample] | Sequence[dict[str, Any]],
    *,
    seed: int,
    graph_draws: int,
    spec: SetSpec,
) -> dict[str, Any]:
    rows = _rows(examples)
    lens = [len(split_encoding_tokens(r["encoding"])) for r in rows]
    by_split: dict[str, Any] = {}
    for split in ("train", "val"):
        rs = [r for r in rows if r["split"] == split]
        by_split[split] = {
            "rows": len(rs),
            "graphs": len({r["edge_hash"] for r in rs}),
            "y_counts": {str(y): sum(1 for r in rs if int(r["y"]) == y) for y in (0, 1)},
            "hop_counts": {
                str(k): v for k, v in sorted(Counter(int(r["hop_distance"]) for r in rs).items())
            },
        }
    graphs = {r["edge_hash"]: r for r in rows}
    graph_hop = {r["edge_hash"]: int(r["hop_distance"]) for r in rows if int(r["y"]) == 1}
    degree_by_hop: dict[str, float] = {}
    for k in spec.hops:
        degs = []
        for eh, r in graphs.items():
            if graph_hop[eh] == k:
                n, edges, _, _ = parse_instance(r["encoding"])
                degs.append(len(edges) / n)
        degree_by_hop[str(k)] = statistics.fmean(degs) if degs else float("nan")
    negs = [r for r in rows if int(r["y"]) == 0]
    train_graphs = {r["edge_hash"] for r in rows if r["split"] == "train"}
    val_graphs = {r["edge_hash"] for r in rows if r["split"] == "val"}
    return {
        "seed": seed,
        "graph_draws": graph_draws,
        "n_total": len(rows),
        "design": "crossed: two k-hop paths per graph; both crossings are the negatives",
        "by_split": by_split,
        "graphs_shared_by_train_and_val": len(train_graphs & val_graphs),
        "n_values_by_hop": {
            str(k): {
                str(n): c
                for n, c in sorted(Counter(int(graphs[eh]["n"]) for eh, h in graph_hop.items() if h == k).items())
            }
            for k in spec.hops
        },
        "mean_out_degree_by_hop": degree_by_hop,
        "token_len": {"min": min(lens), "max": max(lens), "mean": statistics.fmean(lens), "cap": spec.max_tokens},
        "endpoint_balance": endpoint_balance(rows),
        "negatives_with_endpoint_cue": {"count": sum(1 for r in negs if row_has_endpoint_cue(r)), "of": len(negs)},
        "endpoint_rule_accuracy": {
            split: endpoint_rule_accuracy([r for r in rows if r["split"] == split])
            for split in ("train", "val")
            if any(r["split"] == split for r in rows)
        },
        "spec": {
            "hops": list(spec.hops),
            "n_support": list(spec.n_support),
            "mean_degree_by_hop": {str(k): list(v) for k, v in spec.mean_degree_by_hop.items()},
            "max_tokens": spec.max_tokens,
        },
        "science_open": False,
    }


def verify_crossed(
    examples: Sequence[ReachabilityExample] | Sequence[dict[str, Any]] | Path,
    *,
    n_total: int = N_TOTAL,
    n_val: int = N_VAL,
    spec: SetSpec = CROSSED_ID_SPEC,
) -> tuple[bool, list[str]]:
    """Verify quotas, the crossed structure, distances, splits and the token cap."""
    graphs_per_hop, graphs_per_hop_val = _quotas(n_total, n_val, len(spec.hops))
    if isinstance(examples, Path):
        with examples.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        rows = _rows(examples)
    issues: list[str] = []
    train = [r for r in rows if r.get("split") == "train"]
    val = [r for r in rows if r.get("split") == "val"]
    if (len(rows), len(train), len(val)) != (n_total, n_total - n_val, n_val):
        issues.append(f"sizes total/train/val={len(rows)}/{len(train)}/{len(val)}")
    per_graph: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        per_graph.setdefault(r["edge_hash"], []).append(r)
    for eh, rs in per_graph.items():
        pos = [(int(r["s"]), int(r["t"])) for r in rs if int(r["y"]) == 1]
        neg = {(int(r["s"]), int(r["t"])) for r in rs if int(r["y"]) == 0}
        if len(rs) != ROWS_PER_GRAPH or len(pos) != 2 or len(neg) != 2:
            issues.append(f"graph {eh[:10]}: want 2 positives + 2 negatives, got {len(pos)} + {len(neg)}")
            break
        (s1, t1), (s2, t2) = pos
        if neg != {(s1, t2), (s2, t1)}:
            issues.append(f"graph {eh[:10]}: negatives are not the two crossings")
            break
        if len({r["split"] for r in rs}) != 1:
            issues.append(f"graph {eh[:10]} spans splits")
            break
        n, edges, _, _ = parse_instance(rs[0]["encoding"])
        mat = _hop_matrix(n, edges)
        hops = {int(r["hop_distance"]) for r in rs if int(r["y"]) == 1}
        if len(hops) != 1 or not hops <= set(spec.hops) or mat[s1][t1] != mat[s2][t2] or mat[s1][t1] not in hops:
            issues.append(f"graph {eh[:10]}: positive distances {mat[s1][t1]}, {mat[s2][t2]} vs recorded {hops}")
            break
        if mat[s1][t2] >= 0 or mat[s2][t1] >= 0:
            issues.append(f"graph {eh[:10]}: a crossing is reachable")
            break
    for name, rs, per_hop in (("train", train, graphs_per_hop - graphs_per_hop_val), ("val", val, graphs_per_hop_val)):
        for k in spec.hops:
            c = sum(1 for r in rs if int(r["y"]) == 1 and int(r["hop_distance"]) == k)
            if c != 2 * per_hop:
                issues.append(f"{name} hop={k} positives={c} want {2 * per_hop}")
    for r in rows:
        if int(r["y"]) == 0:
            ok, reason = classify_y0_row(r)
            if not ok or int(r["hop_distance"]) != HOP_UNREACHABLE or row_has_endpoint_cue(r):
                issues.append(f"bad negative ({reason}) graph {r['edge_hash'][:10]}")
                break
        if len(split_encoding_tokens(r["encoding"])) > spec.max_tokens:
            issues.append(f"encoding longer than {spec.max_tokens} tokens")
            break
        if r.get("science_open") is True:
            issues.append("row has science_open=True")
            break
    balance = endpoint_balance(rows)
    if any(balance.values()):
        issues.append(f"endpoint balance broken: {balance}")
    return not issues, issues


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a crossed reachability set: two k-hop paths per graph, both crossings as negatives (MEASURE)."
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--seed", type=int, default=CROSSED_SEED)
    p.add_argument("--n-total", type=int, default=N_TOTAL)
    p.add_argument("--n-val", type=int, default=N_VAL)
    p.add_argument("--spec", choices=sorted(SPECS), default="id",
                   help="id: hops 2-6 on 16-24 nodes (default); extended: hops 8-16 on 40-48 nodes")
    p.add_argument("--max-graph-draws", type=int, default=1_000_000)
    p.add_argument("--verify-only", type=Path, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    sizes = {"n_total": args.n_total, "n_val": args.n_val, "spec": SPECS[args.spec]}
    if args.verify_only is not None:
        ok, issues = verify_crossed(args.verify_only, **sizes)
        print(json.dumps({"ok": ok, "issues": issues}, sort_keys=True))
        return 0 if ok else 1
    examples, report = generate_crossed(seed=args.seed, max_graph_draws=args.max_graph_draws, **sizes)
    n = write_jsonl(args.out, examples)
    ok, issues = verify_crossed(examples, **sizes)
    report["verify_ok"] = ok
    report["verify_issues"] = issues
    write_report(args.report, report)
    print(f"wrote {n} examples to {args.out.as_posix()}; report {args.report.as_posix()}; verify_ok={ok}",
          file=sys.stderr)
    if not ok:
        print(f"VERIFY FAIL: {issues}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "n": n, "out": args.out.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Generate a graph-disjoint, label-paired ID set (MEASURE plumbing).

Same task, encoding, hop strata and hard-negative rule as ``gen_id_2k``, and
by default the same quotas (1600 train / 400 val, 1000 / 1000 labels, 200
positives per hop K∈[2,6]; ``--n-total`` / ``--n-val`` scale them). Only the
composition changes:

- every graph contributes exactly one y=1 and one y=0 query, so graph
  identity and graph size carry no label information (a query-blind
  predictor scores 0.5 by construction);
- every graph appears in exactly one split, so validation measures
  reachability on unseen graphs;
- all hops draw graph sizes from the same range (only density varies with the
  target hop), and encodings are capped at ``MAX_TOKENS``;
- negatives are *strong* hard negatives: besides the total-degree rule, the
  source has an outgoing edge and the target an incoming edge, so no query is
  decidable from one endpoint alone (:func:`hard_negatives.has_endpoint_cue`;
  in ``id_2k`` that one-endpoint rule scores 0.8275 on val).

``science_open=false`` always.

Usage::

    python -m reachability_gen.gen_id_disjoint
    python -m reachability_gen.gen_id_disjoint --verify-only data/id_disjoint_2k.jsonl
    python -m reachability_gen.gen_id_disjoint --n-total 20000 --n-val 4000 \\
        --seed 170000 --out data/id_disjoint_20k.jsonl \\
        --report artifacts/id_disjoint_20k_generation_report.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.adr_invariants import (
    HOP_UNREACHABLE,
    ID_HOP_MAX,
    ID_HOP_MIN,
    K_TRAIN_MAX,
    is_ood_hop,
)
from reachability_gen.encode import edge_hash as compute_edge_hash
from reachability_gen.encode import encode_instance
from reachability_gen.gen_id_2k import _hop_matrix, write_jsonl, write_report
from reachability_gen.graph import er_digraph
from reachability_gen.hard_negatives import (
    is_hard_negative,
    row_has_endpoint_cue,
    total_degrees,
)
from reachability_gen.schema import ReachabilityExample
from reachability_gen.splits import derive_example_seed
from reachability_gen.tokenize import split_encoding_tokens

ID_DISJOINT_SEED: int = 168_000

N_TOTAL: int = 2000  # default size; one positive + one negative per graph
N_VAL: int = 400
ID_HOPS: tuple[int, ...] = tuple(range(ID_HOP_MIN, ID_HOP_MAX + 1))  # 2..6

# Shared size range for every hop; density targets mean out-degree n·p.
N_SUPPORT: tuple[int, ...] = (10, 12, 14, 16, 18, 20)
MEAN_DEGREE_BY_HOP: dict[int, tuple[float, ...]] = {
    2: (2.5, 3.0, 3.5),
    3: (1.8, 2.2, 2.6),
    4: (1.5, 1.8, 2.1),
    5: (1.3, 1.5, 1.7),
    6: (1.1, 1.3, 1.5),
}
MAX_TOKENS: int = 250

DEFAULT_OUT = Path("data/id_disjoint_2k.jsonl")
DEFAULT_REPORT = Path("artifacts/id_disjoint_2k_generation_report.json")


def _encoding_len(n: int, edges: Sequence[tuple[int, int]], s: int, t: int) -> int:
    return len(split_encoding_tokens(encode_instance(n, list(edges), s, t)))


def _in_out_degrees(n: int, edges: Sequence[tuple[int, int]]) -> tuple[list[int], list[int]]:
    out_deg, in_deg = [0] * n, [0] * n
    for u, v in edges:
        out_deg[u] += 1
        in_deg[v] += 1
    return out_deg, in_deg


def _pick_pair(
    n: int, edges: list[tuple[int, int]], hop: int, rng: random.Random
) -> Optional[tuple[tuple[int, int], tuple[int, int]]]:
    """One (s,t) at shortest-path distance ``hop`` and one strong hard negative, or None."""
    mat = _hop_matrix(n, edges)
    deg = total_degrees(n, edges)
    out_deg, in_deg = _in_out_degrees(n, edges)
    pos = [(s, t) for s in range(n) for t in range(n) if mat[s][t] == hop]
    neg = [
        (s, t)
        for s in range(n)
        for t in range(n)
        if s != t
        and mat[s][t] < 0
        and deg[s] >= 1
        and deg[t] >= 1
        and out_deg[s] >= 1
        and in_deg[t] >= 1
    ]
    if not pos or not neg:
        return None
    p_pair = rng.choice(pos)
    n_pair = rng.choice(neg)
    ok, _ = is_hard_negative(n, edges, n_pair[0], n_pair[1], y=0)
    return (p_pair, n_pair) if ok else None


def _quotas(n_total: int, n_val: int) -> tuple[int, int]:
    """Graphs per hop overall and in val (each graph gives one pos + one neg)."""
    unit = 2 * len(ID_HOPS)
    if n_total % unit or n_val % unit or not 0 < n_val < n_total:
        raise ValueError(
            f"n_total={n_total} and n_val={n_val} must be multiples of {unit}, "
            "with 0 < n_val < n_total"
        )
    return n_total // unit, n_val // unit


def generate_id_disjoint(
    *,
    seed: int = ID_DISJOINT_SEED,
    max_graph_draws: int = 400_000,
    n_total: int = N_TOTAL,
    n_val: int = N_VAL,
) -> tuple[list[ReachabilityExample], dict[str, Any]]:
    """Generate the graph-disjoint paired set; return (examples, report)."""
    graphs_per_hop, graphs_per_hop_val = _quotas(n_total, n_val)
    master = random.Random(seed)
    records: dict[int, list[tuple[int, float, list[tuple[int, int]], tuple[int, int], tuple[int, int]]]] = {
        k: [] for k in ID_HOPS
    }
    seen_graphs: set[str] = set()
    rejects: Counter = Counter()
    draws = 0
    while any(len(records[k]) < graphs_per_hop for k in ID_HOPS):
        if draws >= max_graph_draws:
            raise RuntimeError(
                f"id_disjoint shortfall after {draws} draws: "
                f"{ {k: len(v) for k, v in records.items()} }"
            )
        draws += 1
        hop = min(ID_HOPS, key=lambda k: (len(records[k]), k))  # fill evenly
        n = master.choice(N_SUPPORT)
        p = round(master.choice(MEAN_DEGREE_BY_HOP[hop]) / (n - 1), 4)
        edges = er_digraph(n, p, master)
        eh = compute_edge_hash(edges)
        if eh in seen_graphs:
            rejects["duplicate_graph"] += 1
            continue
        picked = _pick_pair(n, edges, hop, master)
        if picked is None:
            rejects[f"no_pair_hop{hop}"] += 1
            continue
        (ps, pt), (ns, nt) = picked
        if max(_encoding_len(n, edges, ps, pt), _encoding_len(n, edges, ns, nt)) > MAX_TOKENS:
            rejects["overlong"] += 1
            continue
        seen_graphs.add(eh)
        records[hop].append((n, p, edges, (ps, pt), (ns, nt)))

    examples: list[ReachabilityExample] = []
    index = 0
    for k in ID_HOPS:
        random.Random(derive_example_seed(seed, k * 1000)).shuffle(records[k])
        for i, (n, p, edges, (ps, pt), (ns, nt)) in enumerate(records[k]):
            split = "val" if i < graphs_per_hop_val else "train"
            eh = compute_edge_hash(edges)
            for (s, t), y, hop in (((ps, pt), 1, k), ((ns, nt), 0, HOP_UNREACHABLE)):
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
    report = build_report(ordered, seed=seed, graph_draws=draws)
    report["reject_reasons"] = dict(sorted(rejects.items()))
    return ordered, report


def endpoint_rule_accuracy(rows: Sequence[dict[str, Any]]) -> float:
    """Accuracy of 'unreachable iff one endpoint has no edge in the needed direction'."""
    hits = [(0 if row_has_endpoint_cue(r) else 1) == int(r["y"]) for r in rows]
    return sum(hits) / len(hits) if hits else float("nan")


def build_report(
    examples: Sequence[ReachabilityExample] | Sequence[dict[str, Any]],
    *,
    seed: int,
    graph_draws: int,
) -> dict[str, Any]:
    rows = [e.to_dict() if isinstance(e, ReachabilityExample) else dict(e) for e in examples]
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
    negs = [r for r in rows if int(r["y"]) == 0]
    train_graphs = {r["edge_hash"] for r in rows if r["split"] == "train"}
    val_graphs = {r["edge_hash"] for r in rows if r["split"] == "val"}
    return {
        "seed": seed,
        "graph_draws": graph_draws,
        "n_total": len(rows),
        "by_split": by_split,
        "graphs_shared_by_train_and_val": len(train_graphs & val_graphs),
        "n_values_by_label": {
            f"y{y}": {
                str(k): v
                for k, v in sorted(Counter(int(r["n"]) for r in rows if int(r["y"]) == y).items())
            }
            for y in (0, 1)
        },
        "n_values_by_hop": {
            str(k): {
                str(n): c
                for n, c in sorted(
                    Counter(int(r["n"]) for r in rows if int(r["hop_distance"]) == k).items()
                )
            }
            for k in ID_HOPS
        },
        "token_len": {
            "min": min(lens),
            "max": max(lens),
            "mean": statistics.fmean(lens),
            "cap": MAX_TOKENS,
        },
        "negatives_with_endpoint_cue": {
            "count": sum(1 for r in negs if row_has_endpoint_cue(r)),
            "of": len(negs),
        },
        "endpoint_rule_accuracy": {
            split: endpoint_rule_accuracy([r for r in rows if r["split"] == split])
            for split in ("train", "val")
        },
        "science_open": False,
    }


def verify_id_disjoint(
    examples: Sequence[ReachabilityExample] | Sequence[dict[str, Any]] | Path,
    *,
    n_total: int = N_TOTAL,
    n_val: int = N_VAL,
) -> tuple[bool, list[str]]:
    """Verify quotas, pairing, graph-disjointness and the hard-negative rule."""
    graphs_per_hop, graphs_per_hop_val = _quotas(n_total, n_val)
    if isinstance(examples, Path):
        with examples.open(encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        rows = [e.to_dict() if isinstance(e, ReachabilityExample) else dict(e) for e in examples]
    issues: list[str] = []
    train = [r for r in rows if r.get("split") == "train"]
    val = [r for r in rows if r.get("split") == "val"]
    if (len(rows), len(train), len(val)) != (n_total, n_total - n_val, n_val):
        issues.append(f"sizes total/train/val={len(rows)}/{len(train)}/{len(val)}")
    per_graph: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        per_graph.setdefault(r["edge_hash"], []).append(r)
    for eh, rs in per_graph.items():
        if sorted(int(r["y"]) for r in rs) != [0, 1]:
            issues.append(f"graph {eh[:10]} labels {sorted(int(r['y']) for r in rs)} want [0, 1]")
            break
        if len({r["split"] for r in rs}) != 1:
            issues.append(f"graph {eh[:10]} spans splits")
            break
    for name, rs, per_hop in (("train", train, graphs_per_hop - graphs_per_hop_val),
                              ("val", val, graphs_per_hop_val)):
        for k in ID_HOPS:
            c = sum(1 for r in rs if int(r["y"]) == 1 and int(r["hop_distance"]) == k)
            if c != per_hop:
                issues.append(f"{name} hop={k} positives={c} want {per_hop}")
    from reachability_gen.hard_negatives import classify_y0_row

    for r in rows:
        if int(r["y"]) == 0:
            ok, reason = classify_y0_row(r)
            if not ok or int(r["hop_distance"]) != HOP_UNREACHABLE:
                issues.append(f"bad negative ({reason}) graph {r['edge_hash'][:10]}")
                break
            if row_has_endpoint_cue(r):
                issues.append(f"negative decidable from one endpoint, graph {r['edge_hash'][:10]}")
                break
        if len(split_encoding_tokens(r["encoding"])) > MAX_TOKENS:
            issues.append(f"encoding longer than {MAX_TOKENS} tokens")
            break
        if r.get("science_open") is True:
            issues.append("row has science_open=True")
            break
    return not issues, issues


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate a graph-disjoint ID set (default data/id_disjoint_2k.jsonl): "
            "one positive and one strong hard negative per graph, each graph in "
            "one split (MEASURE; no science OPEN)."
        )
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--seed", type=int, default=ID_DISJOINT_SEED)
    p.add_argument("--max-graph-draws", type=int, default=400_000)
    p.add_argument("--n-total", type=int, default=N_TOTAL, help="instances (default 2000)")
    p.add_argument("--n-val", type=int, default=N_VAL, help="validation instances (default 400)")
    p.add_argument("--verify-only", type=Path, default=None)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    sizes = {"n_total": args.n_total, "n_val": args.n_val}
    if args.verify_only is not None:
        ok, issues = verify_id_disjoint(args.verify_only, **sizes)
        print(json.dumps({"ok": ok, "issues": issues}, sort_keys=True))
        return 0 if ok else 1
    examples, report = generate_id_disjoint(
        seed=args.seed, max_graph_draws=args.max_graph_draws, **sizes
    )
    n = write_jsonl(args.out, examples)
    ok, issues = verify_id_disjoint(examples, **sizes)
    report["verify_ok"] = ok
    report["verify_issues"] = issues
    write_report(args.report, report)
    print(
        f"wrote {n} examples to {args.out.as_posix()}; report {args.report.as_posix()}; "
        f"verify_ok={ok}",
        file=sys.stderr,
    )
    if not ok:
        print(f"VERIFY FAIL: {issues}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "n": n, "out": args.out.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

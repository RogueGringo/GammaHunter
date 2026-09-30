# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""One-endpoint reach cues on paired reachability sets (MEASURE).

Pairing each graph's reachable and unreachable query makes query-blind rules,
and the rule that flags an endpoint without edges, score exactly 0.5. The
label can still correlate with how far each endpoint reaches on its own: in
sparse random digraphs the sources of unreachable queries tend to reach fewer
nodes, and their targets tend to have fewer ancestors. None of these features
needs a path between the two endpoints.

For each feature this reports the best single-threshold rule fitted on the set
itself, an optimistic ceiling for a rule on that feature alone, at three
horizons: 1 hop (endpoint degrees), 6 hops (as far as a 6-step arm sees from
one endpoint) and unlimited. It also reports the same ceiling for the distance
between the two endpoints with edge direction ignored: that rule does search,
but not the directed search the label depends on.

Two rules that combine both endpoints' reach, still without a path between them:

* the pigeonhole rule, fitted on nothing: reachable iff
  |desc(s) ∪ {s}| + |anc(t) ∪ {t}| > n. The two sets then share a node, so s
  reaches t: the rule is sound (it never fires on an unreachable pair) and can
  only miss reachable ones;
* a two-threshold AND rule, reachable iff |desc(s)|/n ≥ a and |anc(t)|/n ≥ b,
  fitted on a file's train split and scored on the audited split, with the
  endpoints left out of desc(s) and anc(t) and with them counted (both are
  reported; the definition leaves this open and it moves the result).

``science_open=false`` always.

Usage::

    python -m reachability_gen.reach_cues
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from reachability_gen.encode import parse_instance
from reachability_gen.overfit_ff import load_jsonl

FEATURES: dict[str, str] = {
    "source_reach": "nodes reachable from the source",
    "source_reach_depth": "largest distance from the source to a node it reaches",
    "target_ancestors": "nodes from which the target is reachable",
    "target_ancestor_depth": "largest distance from such a node to the target",
}
HORIZONS: tuple[Optional[int], ...] = (1, 6, None)
DEFAULT_DATA: tuple[str, ...] = (
    "data/id_disjoint_2k.jsonl",
    "data/id_disjoint_20k.jsonl",
    "data/extended_disjoint_2k.jsonl",
    "data/id_crossed_20k.jsonl",
    "data/extended_crossed_2k.jsonl",
)
DEFAULT_OUT = Path("artifacts/reach_cue_audit.json")
# Values reported by an independent review (to reproduce, not to cite); recorded beside the results.
CLAIMS: dict[str, dict[str, str]] = {
    "data/id_disjoint_20k.jsonl": {"pigeonhole": "0.9878 (fires on 1951 of 2000 reachable pairs, precision 1.0)",
                                   "and_rule": "0.976"},
    "data/id_disjoint_2k.jsonl": {"pigeonhole": "0.9925", "and_rule": "0.980"},
}


def horizon_key(limit: Optional[int]) -> str:
    return "unlimited" if limit is None else f"within_{limit}"


def _bfs(adj: dict[int, list[int]], start: int, limit: Optional[int]) -> dict[int, int]:
    """Shortest distances from ``start`` along ``adj``, at most ``limit`` hops."""
    dist = {start: 0}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        if limit is not None and dist[u] >= limit:
            continue
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def endpoint_reach_features(
    edges: Iterable[tuple[int, int]], s: int, t: int, *, limit: Optional[int] = None
) -> dict[str, int]:
    """How far ``s`` reaches forwards and ``t`` backwards, each on its own."""
    fwd: dict[int, list[int]] = {}
    bwd: dict[int, list[int]] = {}
    for u, v in edges:
        fwd.setdefault(u, []).append(v)
        bwd.setdefault(v, []).append(u)
    down = _bfs(fwd, s, limit)
    up = _bfs(bwd, t, limit)
    return {
        "source_reach": len(down) - 1,
        "source_reach_depth": max(down.values()),
        "target_ancestors": len(up) - 1,
        "target_ancestor_depth": max(up.values()),
    }


def undirected_distance(n: int, edges: Iterable[tuple[int, int]], s: int, t: int) -> int:
    """Distance from ``s`` to ``t`` ignoring edge direction (``n`` if disconnected)."""
    adj: dict[int, list[int]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    return _bfs(adj, s, None).get(t, n)


def pigeonhole_fires(n: int, edges: Iterable[tuple[int, int]], s: int, t: int) -> bool:
    """|desc(s) ∪ {s}| + |anc(t) ∪ {t}| > n: the two sets overlap, so s reaches t (sound, no fitting)."""
    fwd: dict[int, list[int]] = {}
    bwd: dict[int, list[int]] = {}
    for u, v in edges:
        fwd.setdefault(u, []).append(v)
        bwd.setdefault(v, []).append(u)
    return len(_bfs(fwd, s, None)) + len(_bfs(bwd, t, None)) > n


def pigeonhole_report(parsed: Sequence[tuple[int, Any, int, int]], labels: Sequence[int]) -> dict[str, Any]:
    """Accuracy of predicting reachable exactly where the pigeonhole rule fires, and where it fires."""
    fires = [pigeonhole_fires(n, edges, s, t) for n, edges, s, t in parsed]
    tp = sum(1 for f, y in zip(fires, labels) if f and y)
    fp = sum(1 for f, y in zip(fires, labels) if f and not y)
    positives = sum(labels)
    return {
        "accuracy": (tp + (len(labels) - positives - fp)) / len(labels),
        "fires_on_reachable": tp,
        "reachable": positives,
        "fires_on_unreachable": fp,
        "precision": tp / (tp + fp) if tp + fp else None,
    }


def reach_fractions(rows: Sequence[dict[str, Any]], *, endpoints: bool = False) -> list[tuple[float, float]]:
    """(|desc(s)|/n, |anc(t)|/n) per row, with s and t themselves counted if ``endpoints``."""
    out = []
    for n, edges, s, t in (parse_instance(str(r["encoding"])) for r in rows):
        f = endpoint_reach_features(edges, s, t)
        out.append(((f["source_reach"] + endpoints) / n, (f["target_ancestors"] + endpoints) / n))
    return out


def and_rule_fit(xs: Sequence[float], ys: Sequence[float], labels: Sequence[int]) -> tuple[float, float, float]:
    """Thresholds (a, b) maximising the accuracy of ``1 iff x >= a and y >= b``; returns (a, b, accuracy).

    Every pair of observed values is tried, plus thresholds above every value (a rule that
    never fires); ties keep the first pair in ascending order.
    """
    ux, uy = sorted(set(xs)), sorted(set(ys))
    ix, iy = {v: i for i, v in enumerate(ux)}, {v: j for j, v in enumerate(uy)}
    pos = [[0] * (len(uy) + 1) for _ in range(len(ux) + 1)]
    neg = [[0] * (len(uy) + 1) for _ in range(len(ux) + 1)]
    for x, y, lab in zip(xs, ys, labels):
        (pos if lab else neg)[ix[x]][iy[y]] += 1
    for i in range(len(ux) - 1, -1, -1):  # suffix sums: counts with x >= ux[i] and y >= uy[j]
        for j in range(len(uy) - 1, -1, -1):
            for table in (pos, neg):
                table[i][j] += table[i + 1][j] + table[i][j + 1] - table[i + 1][j + 1]
    total, n_neg = len(labels), neg[0][0]
    best = (-1.0, float("inf"), float("inf"))
    for i in range(len(ux) + 1):
        for j in range(len(uy) + 1):
            acc = (pos[i][j] + n_neg - neg[i][j]) / total
            if acc > best[0]:
                best = (acc, ux[i] if i < len(ux) else float("inf"), uy[j] if j < len(uy) else float("inf"))
    return best[1], best[2], best[0]


def and_rule_report(train_rows: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """The AND rule fitted on ``train_rows`` and scored on ``rows``, endpoints excluded and counted."""
    labels = [int(r["y"]) for r in rows]
    out: dict[str, Any] = {"train_rows": len(train_rows)}
    for key, endpoints in (("endpoints_excluded", False), ("endpoints_counted", True)):
        tr, ev = reach_fractions(train_rows, endpoints=endpoints), reach_fractions(rows, endpoints=endpoints)
        a, b, train_acc = and_rule_fit([x for x, _ in tr], [y for _, y in tr], [int(r["y"]) for r in train_rows])
        hits = sum(int((x >= a and y >= b) == bool(lab)) for (x, y), lab in zip(ev, labels))
        out[key] = {"source_fraction_at_least": a, "target_fraction_at_least": b, "train_accuracy": train_acc,
                    "accuracy": hits / len(labels)}
    return out


def best_threshold_accuracy(values: Sequence[float], labels: Sequence[int]) -> float:
    """Best accuracy of ``predict 1 iff value >= θ``, or its complement, over all θ.

    Constant rules are included, so the result is at least the majority rate.
    """
    if not values or len(values) != len(labels):
        raise ValueError("need one label per value and at least one value")
    pairs = sorted(zip(values, (int(y) for y in labels)))
    total = len(pairs)
    n_pos = sum(y for _, y in pairs)
    best = max(n_pos, total - n_pos) / total
    pos_below = neg_below = 0
    i = 0
    while i < total:
        value = pairs[i][0]
        acc = (n_pos - pos_below + neg_below) / total
        best = max(best, acc, 1.0 - acc)
        while i < total and pairs[i][0] == value:
            pos_below += pairs[i][1]
            neg_below += 1 - pairs[i][1]
            i += 1
    return best


def reach_cue_report(
    rows: Sequence[dict[str, Any]], *, horizons: Sequence[Optional[int]] = HORIZONS
) -> dict[str, Any]:
    """Ceiling of each single-feature rule per horizon, and the best per horizon."""
    parsed = [parse_instance(str(r["encoding"])) for r in rows]
    labels = [int(r["y"]) for r in rows]
    rules: dict[str, dict[str, float]] = {f: {} for f in FEATURES}
    for limit in horizons:
        feats = [endpoint_reach_features(edges, s, t, limit=limit) for _, edges, s, t in parsed]
        for f in FEATURES:
            rules[f][horizon_key(limit)] = best_threshold_accuracy([x[f] for x in feats], labels)
    return {
        "n": len(rows),
        "positive_rate": sum(labels) / len(labels),
        "rules": rules,
        "max_by_horizon": {
            horizon_key(limit): max(rules[f][horizon_key(limit)] for f in FEATURES)
            for limit in horizons
        },
        "direction_blind_distance": best_threshold_accuracy(
            [undirected_distance(n, edges, s, t) for n, edges, s, t in parsed], labels
        ),
        "pigeonhole": pigeonhole_report(parsed, labels),
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Ceilings for rules that read only one endpoint's reach, on paired sets (MEASURE)."
    )
    p.add_argument("--data", type=Path, nargs="+", default=[Path(d) for d in DEFAULT_DATA])
    p.add_argument("--split", default="val", help="rows audited in each file (default: val)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)

    sets: dict[str, Any] = {}
    missing: list[str] = []
    for path in args.data:
        if not path.exists():
            missing.append(path.as_posix())
            continue
        every = load_jsonl(path)
        rows = [r for r in every if r.get("split") == args.split]
        if rows:
            sets[path.as_posix()] = reach_cue_report(rows)
            train = [r for r in every if r.get("split") == "train"]
            if train and args.split != "train":
                sets[path.as_posix()]["and_rule"] = and_rule_report(train, rows)
            if path.as_posix() in CLAIMS:
                sets[path.as_posix()]["claimed_by_review"] = CLAIMS[path.as_posix()]
    if not sets:
        print(f"FAIL: no {args.split!r} rows audited; missing: {missing}; see docs/USAGE.md", file=sys.stderr)
        return 1
    artifact = {
        "science_open": False,
        "purpose": "ceilings for rules that read only one endpoint's reach (no path between the endpoints)",
        "method": "best single-threshold rule per feature, fitted on the audited rows themselves; the pigeonhole "
                  "rule (no fitting); the two-threshold AND rule, fitted on the train split where a file has one",
        "split": args.split,
        "features": FEATURES,
        "direction_blind_distance": "distance between the two endpoints with edge direction ignored",
        "horizons": [horizon_key(h) for h in HORIZONS],
        "sets": sets,
        "missing": missing,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    for name, rep in sets.items():
        best = " ".join(f"{k}={v:.4f}" for k, v in rep["max_by_horizon"].items())
        pig = rep["pigeonhole"]
        extra = (f"; AND rule {rep['and_rule']['endpoints_excluded']['accuracy']:.4f} "
                 f"(endpoints counted {rep['and_rule']['endpoints_counted']['accuracy']:.4f})") if "and_rule" in rep else ""
        print(f"{name} (n={rep['n']}): best one-endpoint rule {best}; "
              f"direction-blind distance {rep['direction_blind_distance']:.4f}; pigeonhole {pig['accuracy']:.4f} "
              f"(fires on {pig['fires_on_reachable']}/{pig['reachable']} reachable, "
              f"{pig['fires_on_unreachable']} unreachable){extra}"
              + (f"; review claimed {rep['claimed_by_review']}" if "claimed_by_review" in rep else ""), file=sys.stderr)
    print(json.dumps({"ok": True, "out": args.out.as_posix(), "science_open": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

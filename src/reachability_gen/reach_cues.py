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
        rows = [r for r in load_jsonl(path) if r.get("split") == args.split]
        if rows:
            sets[path.as_posix()] = reach_cue_report(rows)
    if not sets:
        print(f"FAIL: no {args.split!r} rows audited; missing: {missing}; see docs/USAGE.md", file=sys.stderr)
        return 1
    artifact = {
        "science_open": False,
        "purpose": "ceilings for rules that read only one endpoint's reach (no path between the endpoints)",
        "method": "best single-threshold rule per feature, fitted on the audited rows themselves",
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
        print(f"{name} (n={rep['n']}): best one-endpoint rule {best}; "
              f"direction-blind distance {rep['direction_blind_distance']:.4f}", file=sys.stderr)
    print(json.dumps({"ok": True, "out": args.out.as_posix(), "science_open": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

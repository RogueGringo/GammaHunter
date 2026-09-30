# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Learned no-search ceilings on the paired and crossed sets (MEASURE; dev-only, needs scikit-learn).

The rules of ``reach_cues`` read one feature or a fixed combination. Here a
gradient-boosted-tree classifier (scikit-learn's ``HistGradientBoostingClassifier``
with default settings) learns the label from features that need no directed path
between the two endpoints, so its accuracy is a ceiling for answers that do not
come from search:

* ``endpoint``: how far the source reaches and the target is reached (count and
  depth, at 1, 6 and unlimited hops) and the graph size, 13 features;
* ``ancestor_side``: the target's side of those and the graph size, 7 features;
* ``endpoint_pair``: ``endpoint`` plus three pair features that ignore edge
  direction: the undirected distance, the number of undirected shortest paths and
  the number of common undirected neighbours;
* ``profile``, ``profile_ancestor``, ``profile_pair``: the same three, extended by
  the number of descendants of the source and of ancestors of the target at each
  distance 1..12, both endpoints' in- and out-degrees and the edge count. These
  were added after the first comparison with the review's numbers, which the
  13-feature sets did not reach; they are reported beside them, not instead.

Files with a train split are fitted on it and scored on their validation split;
files without one are scored by grouped 5-fold cross-validation (a graph's rows
never straddle folds). Five learner seeds each; the seed only matters with at
more than 10,000 training rows, where the learner holds out a seed-dependent split
for early stopping, so smaller sets give the same result for every seed.
Interactions between features are left to the trees (default settings, no
product features), and the graph size is in every feature set. Accuracy is also
reported per graph hop (the hop of the graph's reachable queries). A null
control refits the learner on training labels shuffled with each seed, scored
the same way: the accuracy a learner reaches here without any real signal.
Numbers supplied by an independent review are recorded as claims next to the
values reproduced here.

scikit-learn is an optional extra (``pip install -e ".[cue-ceiling]"``), not a
package dependency. ``science_open=false`` always.

Usage::

    python -m reachability_gen.cue_ceiling
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from reachability_gen.encode import parse_instance
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.reach_cues import FEATURES, HORIZONS, endpoint_reach_features

DEFAULT_OUT = Path("artifacts/cue_ceiling.json")
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
FOLDS: int = 5
FEATURE_SETS: tuple[str, ...] = ("endpoint", "ancestor_side", "endpoint_pair",
                                 "profile", "profile_ancestor", "profile_pair")
PROFILE_DEPTH: int = 12
# (file, feature set): what is measured on it
PLAN: tuple[tuple[str, str], ...] = (
    ("data/id_disjoint_20k.jsonl", "endpoint"),
    ("data/id_disjoint_20k.jsonl", "ancestor_side"),
    ("data/id_disjoint_2k.jsonl", "endpoint"),
    ("data/extended_disjoint_2k.jsonl", "endpoint"),
    ("data/id_crossed_20k.jsonl", "endpoint"),
    ("data/id_crossed_20k.jsonl", "endpoint_pair"),
    ("data/extended_crossed_2k.jsonl", "endpoint_pair"),
    ("data/id_disjoint_20k.jsonl", "profile"),
    ("data/id_disjoint_20k.jsonl", "profile_ancestor"),
    ("data/extended_disjoint_2k.jsonl", "profile"),
    ("data/id_crossed_20k.jsonl", "profile"),
    ("data/id_crossed_20k.jsonl", "profile_pair"),
    ("data/extended_crossed_2k.jsonl", "profile_pair"),
)
# Values reported by the independent review (to reproduce, not to cite).
CLAIMS: dict[str, str] = {
    "data/id_disjoint_20k.jsonl/endpoint": "0.996",
    "data/id_disjoint_20k.jsonl/ancestor_side": "0.934",
    "data/extended_disjoint_2k.jsonl/endpoint": "0.9995 (grouped 5-fold CV)",
    "data/id_crossed_20k.jsonl/endpoint": "0.54",
    "data/id_crossed_20k.jsonl/endpoint_pair": "0.60-0.61 in 4 of 5 learner seeds, 0.54 in one; 0.57-0.67 at every hop",
    "data/extended_crossed_2k.jsonl/endpoint_pair": "about 0.50-0.52 (grouped CV)",
}


def endpoint_features(n: int, edges: Sequence[tuple[int, int]], s: int, t: int) -> list[float]:
    """Graph size, then the four reach features of ``reach_cues`` at each horizon."""
    out: list[float] = [float(n)]
    for limit in HORIZONS:
        f = endpoint_reach_features(edges, s, t, limit=limit)
        out += [float(f[k]) for k in FEATURES]
    return out


def ancestor_side(features: Sequence[float]) -> list[float]:
    """Graph size and the target-side features of ``endpoint_features``."""
    keep = [0] + [1 + h * len(FEATURES) + i for h in range(len(HORIZONS))
                  for i, k in enumerate(FEATURES) if k.startswith("target")]
    return [features[i] for i in keep]


def pair_features(n: int, edges: Iterable[tuple[int, int]], s: int, t: int) -> list[float]:
    """Undirected distance (n if disconnected), number of undirected shortest paths, common neighbours."""
    adj: dict[int, set[int]] = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    dist, paths = {s: 0}, {s: 1}
    queue = deque([s])
    while queue:
        u = queue.popleft()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v], paths[v] = dist[u] + 1, paths[u]
                queue.append(v)
            elif dist[v] == dist[u] + 1:
                paths[v] += paths[u]
    common = len(adj.get(s, set()) & adj.get(t, set()))
    return [float(dist.get(t, n)), float(paths.get(t, 0)), float(common)]


def profile_features(n: int, edges: Sequence[tuple[int, int]], s: int, t: int) -> tuple[list[float], list[float]]:
    """Source side and target side: counts at each distance 1..PROFILE_DEPTH, then out- and in-degree."""
    from reachability_gen.reach_cues import _bfs

    fwd: dict[int, list[int]] = {}
    bwd: dict[int, list[int]] = {}
    outd: dict[int, int] = {}
    ind: dict[int, int] = {}
    for u, v in edges:
        fwd.setdefault(u, []).append(v)
        bwd.setdefault(v, []).append(u)
        outd[u], ind[v] = outd.get(u, 0) + 1, ind.get(v, 0) + 1
    sides = []
    for start, adj in ((s, fwd), (t, bwd)):
        dist = _bfs(adj, start, None)
        counts = [0] * PROFILE_DEPTH
        for d in dist.values():
            if 0 < d <= PROFILE_DEPTH:
                counts[d - 1] += 1
        sides.append([float(c) for c in counts] + [float(outd.get(start, 0)), float(ind.get(start, 0))])
    return sides[0], sides[1]


def featurize(rows: Sequence[dict[str, Any]], feature_set: str) -> list[list[float]]:
    out = []
    for n, edges, s, t in (parse_instance(str(r["encoding"])) for r in rows):
        f = endpoint_features(n, edges, s, t)
        if feature_set.startswith("profile"):
            src, tgt = profile_features(n, edges, s, t)
            if feature_set == "profile_ancestor":
                f = ancestor_side(f) + tgt + [float(len(edges))]
            else:
                f = f + src + tgt + [float(len(edges))]
                if feature_set == "profile_pair":
                    f = f + pair_features(n, edges, s, t)
        elif feature_set == "ancestor_side":
            f = ancestor_side(f)
        elif feature_set == "endpoint_pair":
            f = f + pair_features(n, edges, s, t)
        out.append(f)
    return out


def graph_hops(rows: Sequence[dict[str, Any]]) -> list[int]:
    """Each row's graph hop: the hop of that graph's reachable queries (-1 if it has none)."""
    hop = {r["edge_hash"]: int(r["hop_distance"]) for r in rows if int(r["y"]) == 1}
    return [hop.get(r["edge_hash"], -1) for r in rows]


def by_hop(correct: Sequence[int], hops: Sequence[int]) -> dict[str, float]:
    groups: dict[int, list[int]] = {}
    for c, h in zip(correct, hops):
        groups.setdefault(h, []).append(c)
    return {str(h): sum(v) / len(v) for h, v in sorted(groups.items())}


def score(rows: Sequence[dict[str, Any]], feature_set: str, seeds: Sequence[int] = SEEDS,
          folds: int = FOLDS) -> dict[str, Any]:
    """Train on the train split and score the validation split, or grouped CV if there is no train split."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    train = [r for r in rows if r.get("split") == "train"]
    val = [r for r in rows if r.get("split") == "val"]
    per_seed: dict[str, Any] = {}
    null: list[float] = []
    if train:
        x_tr, x_va = featurize(train, feature_set), featurize(val, feature_set)
        y_tr, y_va = [int(r["y"]) for r in train], [int(r["y"]) for r in val]
        hops = graph_hops(val)
        for seed in seeds:
            clf = HistGradientBoostingClassifier(random_state=seed).fit(x_tr, y_tr)
            correct = [int(p == y) for p, y in zip(clf.predict(x_va), y_va)]
            per_seed[str(seed)] = {"accuracy": sum(correct) / len(correct), "by_graph_hop": by_hop(correct, hops)}
            shuffled = list(y_tr)
            random.Random(seed).shuffle(shuffled)
            pred = HistGradientBoostingClassifier(random_state=seed).fit(x_tr, shuffled).predict(x_va)
            null.append(sum(int(p == y) for p, y in zip(pred, y_va)) / len(y_va))
        mode = "train split -> validation split"
    else:
        x, y = featurize(val, feature_set), [int(r["y"]) for r in val]
        groups, hops = [r["edge_hash"] for r in val], graph_hops(val)
        for seed in seeds:
            pred = [0] * len(y)
            for tr_idx, te_idx in GroupKFold(n_splits=folds).split(x, y, groups):
                clf = HistGradientBoostingClassifier(random_state=seed)
                clf.fit([x[i] for i in tr_idx], [y[i] for i in tr_idx])
                for i, p in zip(te_idx, clf.predict([x[i] for i in te_idx])):
                    pred[i] = int(p)
            correct = [int(p == t) for p, t in zip(pred, y)]
            per_seed[str(seed)] = {"accuracy": sum(correct) / len(correct), "by_graph_hop": by_hop(correct, hops)}
            shuffled = list(y)
            random.Random(seed).shuffle(shuffled)
            null_pred = [0] * len(y)
            for tr_idx, te_idx in GroupKFold(n_splits=folds).split(x, y, groups):
                clf = HistGradientBoostingClassifier(random_state=seed)
                clf.fit([x[i] for i in tr_idx], [shuffled[i] for i in tr_idx])
                for i, q in zip(te_idx, clf.predict([x[i] for i in te_idx])):
                    null_pred[i] = int(q)
            null.append(sum(int(q == t) for q, t in zip(null_pred, y)) / len(y))
        mode = f"grouped {folds}-fold cross-validation (groups: graphs)"
    accs = [v["accuracy"] for v in per_seed.values()]
    hop_keys = next(iter(per_seed.values()))["by_graph_hop"].keys()
    return {
        "mode": mode, "rows_scored": len(val), "train_rows": len(train),
        "seeds": per_seed,
        "accuracy": {"mean": statistics.fmean(accs), "min": min(accs), "max": max(accs)},
        "null_shuffled_labels": {"per_seed": null, "min": min(null), "max": max(null)},
        "by_graph_hop_range": {h: [min(v["by_graph_hop"][h] for v in per_seed.values()),
                                   max(v["by_graph_hop"][h] for v in per_seed.values())] for h in hop_keys},
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Learned no-search ceilings on paired and crossed sets (MEASURE).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = p.parse_args(argv)
    try:
        import sklearn
    except ImportError:
        print('FAIL: needs scikit-learn (pip install -e ".[cue-ceiling]")', file=sys.stderr)
        return 2
    results: dict[str, Any] = {}
    cache: dict[str, list[dict[str, Any]]] = {}
    for path, feature_set in PLAN:
        if not Path(path).exists():
            print(f"FAIL: missing {path}; see docs/USAGE.md", file=sys.stderr)
            return 1
        rows = cache.setdefault(path, load_jsonl(Path(path)))
        key = f"{path}/{feature_set}"
        results[key] = score(rows, feature_set, args.seeds)
        if key in CLAIMS:
            results[key]["claimed_by_review"] = CLAIMS[key]
        a = results[key]["accuracy"]
        print(f"{key}: {a['mean']:.4f} [{a['min']:.4f}, {a['max']:.4f}] ({results[key]['mode']})"
              + (f"; review claimed {CLAIMS[key]}" if key in CLAIMS else ""), file=sys.stderr, flush=True)
    artifact = {
        "science_open": False,
        "purpose": "learned ceilings for answers that need no directed path between the endpoints",
        "learner": f"sklearn.ensemble.HistGradientBoostingClassifier, default settings (scikit-learn "
                   f"{sklearn.__version__})",
        "feature_sets": {
            "endpoint": "graph size; source reach and depth, target ancestors and depth, at 1, 6 and unlimited hops",
            "ancestor_side": "graph size; target ancestors and depth at 1, 6 and unlimited hops",
            "endpoint_pair": "endpoint, plus undirected distance, number of undirected shortest paths and "
                             "number of common undirected neighbours",
            "profile": "endpoint, plus counts of the source's descendants and the target's ancestors at each "
                       "distance 1-12, both endpoints' in- and out-degrees and the edge count (added after the "
                       "first comparison with the review)",
            "profile_ancestor": "ancestor_side, plus the target's ancestor counts at each distance 1-12, its in- "
                                "and out-degree and the edge count (added after the first comparison)",
            "profile_pair": "profile plus the three direction-blind pair features (added after the first comparison)",
        },
        "seeds": args.seeds,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "out": args.out.as_posix(), "science_open": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

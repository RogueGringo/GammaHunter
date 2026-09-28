# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""How much of each graph can reachability answers reveal? (MEASURE, analysis only)

An edge u→v is *implied* when v stays reachable from u without it. Removing it
changes no reachability answer, so no answer, and no number of answers, can
reveal whether it is listed. A reader trained from answers alone can at best
recover a graph with the true reachability closure; it can recover the exact
edge list only for a graph without implied edges. This audit counts implied
edges, and graphs without any, in each set.

``science_open=false`` always.

Usage::

    python -m reachability_gen.reader_identifiability
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.encode import parse_instance
from reachability_gen.overfit_ff import load_jsonl

DEFAULT_OUT = Path("artifacts/reader_identifiability.json")
DEFAULT_SETS: dict[str, tuple[Path, Optional[str]]] = {
    "crossed_train": (Path("data/id_crossed_20k.jsonl"), "train"),
    "crossed_val": (Path("data/id_crossed_20k.jsonl"), "val"),
    "crossed_long": (Path("data/extended_crossed_2k.jsonl"), None),
}


def implied_edges(n: int, edges: Sequence[tuple[int, int]]) -> list[bool]:
    """For each edge u→v, whether v is reachable from u without that edge."""
    out_edges: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        out_edges[u].append(v)
    flags = []
    for u, v in edges:
        seen, stack, found = {u}, [u], False
        while stack and not found:
            a = stack.pop()
            for b in out_edges[a]:
                if a == u and b == v:
                    continue  # the edge itself
                if b == v:
                    found = True
                    break
                if b not in seen:
                    seen.add(b)
                    stack.append(b)
        flags.append(found)
    return flags


def audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Implied edges and graphs without any, over the distinct graphs of ``rows``."""
    graphs: dict[str, str] = {}
    for r in rows:
        graphs.setdefault(r["edge_hash"], r["encoding"])
    edges = implied = clean = 0
    for encoding in graphs.values():
        n, e, _, _ = parse_instance(encoding)
        flags = implied_edges(n, [tuple(x) for x in e])
        edges += len(flags)
        implied += sum(flags)
        clean += not any(flags)
    return {
        "graphs": len(graphs),
        "edges": edges,
        "implied_edges": implied,
        "implied_share": implied / edges if edges else 0.0,
        "graphs_without_implied_edges": clean,
        "exact_graph_rate_attainable_from_answers": clean / len(graphs) if graphs else 0.0,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Implied edges: what reachability answers cannot reveal (MEASURE).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    missing = sorted({str(path) for path, _ in DEFAULT_SETS.values() if not path.exists()})
    if missing:
        print(f"FAIL: missing {missing}", file=sys.stderr)
        return 1
    loaded = {path: load_jsonl(path) for path in {path for path, _ in DEFAULT_SETS.values()}}
    sets = {name: audit([r for r in loaded[path] if split is None or r["split"] == split])
            for name, (path, split) in DEFAULT_SETS.items()}
    artifact = {
        "science_open": False,
        "purpose": "how much of each graph reachability answers can reveal: implied edges and graphs without any",
        "definition": "an edge u->v is implied when v stays reachable from u without it; no reachability "
                      "answer depends on it",
        "sets": sets,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    for name, s in sets.items():
        print(f"{name:14s} implied edges {s['implied_edges']}/{s['edges']} ({s['implied_share']:.1%}); "
              f"graphs without any {s['graphs_without_implied_edges']}/{s['graphs']}", file=sys.stderr)
    print(json.dumps({"ok": True, "out": args.out.as_posix(), "science_open": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

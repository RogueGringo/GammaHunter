# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""NLGraph connectivity → locked-encoding rows, plus an integrity audit.

Source: Hugging Face dataset ``tasksource/nlgraph`` (Wang et al. 2023, "Can
Language Models Solve Graph Problems in Natural Language?", arXiv 2305.10037).
Only the ``connectivity`` task is used: an undirected edge list and "Is there a
path between node s and node t?", answered "The answer is yes/no."

Conversion: every undirected edge is written in both directions, ``n`` is the
largest node id mentioned plus one, ``y`` comes from the answer, and
``hop_distance`` is the undirected shortest-path length (-1 if unreachable).

The audit asks what the benchmark can be answered by *without* path search:
a direct-edge rule, an isolated-endpoint rule, a graph-density threshold that
ignores the query, the positive hop profile, and graph reuse between train
and test.

``science_open=false`` always.

Usage::

    python -m reachability_gen.benchmarks.nlgraph --download
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from reachability_gen.encode import edge_hash as compute_edge_hash
from reachability_gen.encode import encode_instance

SOURCE = "https://huggingface.co/datasets/tasksource/nlgraph"
BASE_URL = SOURCE + "/resolve/main/data/"
FILES = {
    "train": "train-00000-of-00001-32f6e22687881657.parquet",
    "test": "test-00000-of-00001-9be21117f93ba450.parquet",
}
RAW_DIR = Path("data/benchmarks/nlgraph")
OUT_DIR = Path("data/benchmarks")
AUDIT_OUT = Path("artifacts/nlgraph_connectivity_audit.json")

_EDGE = re.compile(r"\((\d+),\s*(\d+)\)")
_QUERY = re.compile(r"between node (\d+) and node (\d+)")


def download(raw_dir: Path = RAW_DIR) -> list[Path]:
    """Fetch the two parquet files (about 1.4 MB) if not already present."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in FILES.values():
        dest = raw_dir / name
        if not dest.exists():
            urllib.request.urlretrieve(BASE_URL + name, dest)
        paths.append(dest)
    return paths


def parse_question(text: str) -> tuple[list[tuple[int, int]], int, int]:
    """Undirected edges and the query pair from one NLGraph connectivity question."""
    graph_part, _, query_part = text.partition("Q:")
    edges = [(int(a), int(b)) for a, b in _EDGE.findall(graph_part)]
    match = _QUERY.search(query_part)
    if match is None:
        raise ValueError(f"no connectivity query in: {text[:120]!r}")
    return edges, int(match.group(1)), int(match.group(2))


def _hop(n: int, directed: Sequence[tuple[int, int]], s: int, t: int) -> int:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in directed:
        adj[u].append(v)
    dist = {s: 0}
    queue = deque([s])
    while queue:
        u = queue.popleft()
        if u == t:
            return dist[u]
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return -1


def to_rows(records: Iterable[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Connectivity records → rows in the project's example schema."""
    rows = []
    for rec in records:
        if rec.get("task") != "connectivity":
            continue
        undirected, s, t = parse_question(rec["question"])
        answer = rec["answer"].strip().lower()
        if "yes" not in answer and "no" not in answer:
            raise ValueError(f"unrecognised answer {rec['answer']!r}")
        y = 1 if "yes" in answer.split(".")[0] else 0
        directed = sorted({(a, b) for a, b in undirected} | {(b, a) for a, b in undirected})
        n = 1 + max([s, t] + [x for e in undirected for x in e])
        hop = _hop(n, directed, s, t)
        if (hop >= 0) != bool(y):
            raise ValueError(f"label {y} disagrees with graph (hop {hop}) for query {s}->{t}")
        rows.append(
            {
                "split": split,
                "source": "nlgraph",
                "difficulty": rec.get("difficulty"),
                "n": n,
                "p": 0.0,
                "edge_hash": compute_edge_hash(directed),
                "s": s,
                "t": t,
                "y": y,
                "hop_distance": hop,
                "is_ood": False,
                "encoding": encode_instance(n, directed, s, t),
            }
        )
    return rows


def _degrees(row: dict[str, Any]) -> tuple[int, int, int, int]:
    """(deg s, deg t, undirected edges, nodes) from a converted row."""
    parts = row["encoding"].split()
    q = parts.index("QUERY")
    directed = [tuple(int(x) for x in a.split(",")) for a in parts[3:q]]
    deg = Counter(u for u, _ in directed)
    return deg[row["s"]], deg[row["t"]], len(directed) // 2, int(row["n"])


def _acc(pairs: Sequence[tuple[int, int]]) -> float:
    return sum(1 for p, y in pairs if p == y) / len(pairs) if pairs else float("nan")


def audit(train: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, Any]:
    """Baselines that need no path search, hop profile, and train/test reuse."""

    def density(r: dict[str, Any]) -> float:
        _, _, m, n = _degrees(r)
        return m / max(1, n * (n - 1) / 2)

    # Query-blind density threshold: pick the cut that best separates train labels.
    cuts = sorted({round(density(r), 4) for r in train})
    best_cut = max(cuts, key=lambda c: _acc([(int(density(r) >= c), int(r["y"])) for r in train]))
    majority = Counter(int(r["y"]) for r in train).most_common(1)[0][0]

    def endpoint(r: dict[str, Any]) -> int:
        ds, dt, _, _ = _degrees(r)
        return 0 if ds == 0 or dt == 0 else 1

    def direct_edge(r: dict[str, Any]) -> int:
        parts = r["encoding"].split()
        return int(f"{r['s']},{r['t']}" in parts[3 : parts.index("QUERY")])

    def block(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(rows),
            "distinct_graphs": len({r["edge_hash"] for r in rows}),
            "yes_share": _acc([(1, int(r["y"])) for r in rows]),
            "majority_label_acc": _acc([(majority, int(r["y"])) for r in rows]),
            "direct_edge_rule_acc": _acc([(direct_edge(r), int(r["y"])) for r in rows]),
            "isolated_endpoint_rule_acc": _acc([(endpoint(r), int(r["y"])) for r in rows]),
            "density_rule_acc": _acc([(int(density(r) >= best_cut), int(r["y"])) for r in rows]),
            "positive_hops": {
                str(k): v
                for k, v in sorted(Counter(int(r["hop_distance"]) for r in rows if int(r["y"]) == 1).items())
            },
            "nodes": {"min": min(int(r["n"]) for r in rows), "max": max(int(r["n"]) for r in rows)},
        }

    by_difficulty = {
        d: block([r for r in test if r["difficulty"] == d])
        for d in sorted({r["difficulty"] for r in test})
    }
    train_graphs = {r["edge_hash"] for r in train}
    return {
        "source": SOURCE,
        "task": "connectivity (undirected)",
        "citation": "Wang et al. 2023, arXiv 2305.10037",
        "train": block(train),
        "test": block(test),
        "test_by_difficulty": by_difficulty,
        "density_cut_fit_on_train": best_cut,
        "test_graphs_seen_in_train": sum(1 for r in test if r["edge_hash"] in train_graphs),
        "test_queries_seen_in_train": len(
            {(r["edge_hash"], r["s"], r["t"]) for r in test}
            & {(r["edge_hash"], r["s"], r["t"]) for r in train}
        ),
        "test_distinct_graphs_seen_in_train": len(
            {r["edge_hash"] for r in test} & train_graphs
        ),
        "rules": {
            "direct_edge_rule": "answer yes iff s and t share an edge; one lookup, no path search",
            "isolated_endpoint_rule": "answer no iff s or t has no edge; never searches a path",
            "density_rule": "answer yes iff edge density >= a cut fit on train; ignores s and t",
        },
        "science_open": False,
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Convert and audit NLGraph connectivity (MEASURE).")
    p.add_argument("--download", action="store_true", help="fetch the parquet files first")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--audit-out", type=Path, default=AUDIT_OUT)
    args = p.parse_args(argv)
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("FAIL: pyarrow required (pip install pyarrow)", file=sys.stderr)
        return 2
    if args.download:
        download(args.raw_dir)
    rows = {}
    for split, name in FILES.items():
        path = args.raw_dir / name
        if not path.exists():
            print(f"FAIL: missing {path}; run with --download", file=sys.stderr)
            return 1
        rows[split] = to_rows(pq.read_table(path).to_pylist(), split)
        out = args.out_dir / f"nlgraph_connectivity_{split}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows[split]))
    report = audit(rows["train"], rows["test"])
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    t = report["test"]
    print(
        f"test n={t['n']} yes={t['yes_share']:.3f} | majority {t['majority_label_acc']:.3f} | "
        f"direct-edge rule {t['direct_edge_rule_acc']:.3f} | "
        f"isolated-endpoint rule {t['isolated_endpoint_rule_acc']:.3f} | "
        f"density rule {t['density_rule_acc']:.3f} | graphs seen in train "
        f"{report['test_graphs_seen_in_train']}",
        file=sys.stderr,
    )
    print(json.dumps({"ok": True, "audit": args.audit_out.as_posix(), "science_open": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

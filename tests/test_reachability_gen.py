# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Unit tests for reachability_gen RESEARCH scaffold."""

from __future__ import annotations

import json
import random
from pathlib import Path

from reachability_gen.encode import canonical_edge_list, edge_hash, encode_instance
from reachability_gen.generate import generate_split, write_jsonl
from reachability_gen.graph import (
    er_digraph,
    has_self_loops,
    reachable_bfs,
    reachable_dfs,
)
from reachability_gen.splits import OOD_N, SEED_TABLE, TRAIN_N, get_split_spec


# --- determinism -----------------------------------------------------------


def test_er_digraph_determinism():
    a = er_digraph(12, 0.25, random.Random(42))
    b = er_digraph(12, 0.25, random.Random(42))
    assert a == b


def test_generate_split_determinism():
    a, _ = generate_split("train", n_per_cell=2, max_rejects=5000)
    b, _ = generate_split("train", n_per_cell=2, max_rejects=5000)
    assert [ex.to_dict() for ex in a] == [ex.to_dict() for ex in b]


def test_edge_hash_stable():
    edges = [(2, 0), (0, 1), (1, 2)]
    assert edge_hash(edges) == edge_hash(list(reversed(edges)))
    assert edge_hash(edges) == edge_hash(canonical_edge_list(edges))


# --- no self-loops ---------------------------------------------------------


def test_no_self_loops_many_draws():
    rng = random.Random(7)
    for n in (8, 12, 16):
        for p in (0.15, 0.25, 0.35):
            for _ in range(20):
                edges = er_digraph(n, p, rng)
                assert not has_self_loops(edges)
                assert all(u != v for u, v in edges)


# --- label correctness on tiny hand graphs ---------------------------------


def test_label_chain():
    # 0 → 1 → 2 → 3
    edges = [(0, 1), (1, 2), (2, 3)]
    n = 4
    assert reachable_bfs(n, edges, 0, 3) is True
    assert reachable_dfs(n, edges, 0, 3) is True
    assert reachable_bfs(n, edges, 3, 0) is False
    assert reachable_dfs(n, edges, 3, 0) is False
    assert reachable_bfs(n, edges, 1, 1) is True  # s == t


def test_label_disconnected():
    edges = [(0, 1), (2, 3)]
    n = 4
    assert reachable_bfs(n, edges, 0, 3) is False
    assert reachable_bfs(n, edges, 2, 3) is True
    assert reachable_bfs(n, edges, 0, 1) is True


def test_label_cycle():
    edges = [(0, 1), (1, 2), (2, 0)]
    n = 3
    assert reachable_bfs(n, edges, 0, 2) is True
    assert reachable_bfs(n, edges, 2, 1) is True


def test_bfs_dfs_agree_random():
    rng = random.Random(99)
    for _ in range(50):
        n = 10
        edges = er_digraph(n, 0.2, rng)
        s, t = rng.randrange(n), rng.randrange(n)
        assert reachable_bfs(n, edges, s, t) == reachable_dfs(n, edges, s, t)


# --- encoding --------------------------------------------------------------


def test_encode_instance_format():
    enc = encode_instance(3, [(1, 0), (0, 2)], 0, 2)
    assert enc == "N 3 EDGES 0,2 1,0 QUERY 0 2"


def test_encode_empty_edges():
    enc = encode_instance(2, [], 1, 0)
    assert enc == "N 2 EDGES QUERY 1 0"


# --- balance & ood n -------------------------------------------------------


def test_approx_balance_in_distribution():
    """Train/val cells are small enough that stratified rejection hits 50/50."""
    examples, reject_rate = generate_split("val", n_per_cell=4, max_rejects=8000)
    ys = [ex.y for ex in examples]
    n_pos = sum(ys)
    n_neg = len(ys) - n_pos
    assert n_pos == n_neg
    assert 0.0 <= reject_rate < 1.0
    assert all(ex.reject_rate == reject_rate for ex in examples)


def _edges_from_encoding(encoding: str) -> list[tuple[int, int]]:
    # "N n EDGES u,v ... QUERY s t" or empty edges.
    parts = encoding.split()
    assert parts[0] == "N" and parts[2] == "EDGES"
    q = parts.index("QUERY")
    edge_toks = parts[3:q]
    return [tuple(map(int, tok.split(","))) for tok in edge_toks]  # type: ignore[misc]


def test_labels_match_bfs():
    examples, _ = generate_split("train", n_per_cell=2, max_rejects=5000)
    for ex in examples:
        edges = _edges_from_encoding(ex.encoding)
        assert reachable_bfs(ex.n, edges, ex.s, ex.t) == bool(ex.y)
        assert edge_hash(edges) == ex.edge_hash


def test_ood_n_values():
    spec = get_split_spec("ood_size")
    assert spec.n_values == OOD_N
    # Low max_rejects: hard cells (large n, high p) fall back rather than hang.
    examples, _ = generate_split("ood_size", n_per_cell=2, max_rejects=50)
    ns = {ex.n for ex in examples}
    assert ns == set(OOD_N)
    assert all(ex.n not in TRAIN_N for ex in examples)
    assert len(examples) == 2 * len(OOD_N) * 3  # n_per_cell * |N| * |P|


def test_train_n_values():
    examples, _ = generate_split("train", n_per_cell=2, max_rejects=5000)
    assert {ex.n for ex in examples} == set(TRAIN_N)


def test_seed_table_fixed():
    assert SEED_TABLE["train"] == 10_001
    assert SEED_TABLE["val"] == 20_002
    assert SEED_TABLE["test"] == 30_003
    assert SEED_TABLE["ood_size"] == 40_004


# --- JSONL I/O -------------------------------------------------------------


def test_write_jsonl_roundtrip(tmp_path: Path):
    examples, _ = generate_split("test", n_per_cell=2, max_rejects=5000)
    out = tmp_path / "test.jsonl"
    n = write_jsonl(out, examples)
    assert n == len(examples)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == n
    required = {
        "split", "seed", "n", "p", "edge_hash", "s", "t", "y",
        "hop_distance", "is_ood",
    }
    for row in rows:
        assert required <= set(row.keys())
        assert row["split"] == "test"
        assert row["y"] in (0, 1)
        if row["y"] == 0:
            assert row["hop_distance"] == -1
            assert row["is_ood"] is False
        else:
            assert row["hop_distance"] >= 0

# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the identifiability audit (analysis only)."""

from __future__ import annotations

from reachability_gen.encode import encode_instance
from reachability_gen.reader_identifiability import audit, implied_edges


def test_implied_edges():
    assert implied_edges(3, [(0, 1), (1, 2), (0, 2)]) == [False, False, True]  # 0->2 also via 0->1->2
    assert implied_edges(2, [(0, 1), (1, 0)]) == [False, False]  # a cycle implies neither edge
    assert implied_edges(4, [(0, 1), (1, 2), (2, 3), (0, 3)]) == [False, False, False, True]


def test_audit_counts_graphs_once():
    a = encode_instance(3, [(0, 1), (1, 2), (0, 2)], 0, 2)
    b = encode_instance(3, [(0, 1), (1, 2)], 0, 2)
    rows = [{"edge_hash": "a", "encoding": a}, {"edge_hash": "a", "encoding": a}, {"edge_hash": "b", "encoding": b}]
    out = audit(rows)
    assert (out["graphs"], out["edges"], out["implied_edges"], out["graphs_without_implied_edges"]) == (2, 5, 1, 1)
    assert out["exact_graph_rate_attainable_from_answers"] == 0.5


def test_identifiability_artifact_contract():
    import json
    from pathlib import Path

    import pytest

    path = Path(__file__).resolve().parents[1] / "artifacts" / "reader_identifiability.json"
    if not path.exists():
        pytest.skip("identifiability artifact not present")
    art = json.loads(path.read_text(encoding="utf-8"))
    assert art["science_open"] is False and set(art["sets"]) == {"crossed_train", "crossed_val", "crossed_long"}
    for s in art["sets"].values():
        assert 0 <= s["implied_edges"] <= s["edges"] and 0 <= s["graphs_without_implied_edges"] <= s["graphs"]
        assert s["exact_graph_rate_attainable_from_answers"] == s["graphs_without_implied_edges"] / s["graphs"]

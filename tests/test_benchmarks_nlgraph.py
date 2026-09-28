# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Offline tests for the NLGraph connectivity adapter and audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reachability_gen.benchmarks.nlgraph import audit, parse_question, to_rows

ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "artifacts" / "nlgraph_connectivity_audit.json"


def _record(edges: str, s: int, t: int, answer: str, difficulty: str = "easy") -> dict:
    return {
        "task": "connectivity",
        "difficulty": difficulty,
        "question": (
            "Determine if there is a path between two nodes in the graph. Note that "
            "(i,j) means that node i and node j are connected with an undirected edge.\n"
            f"Graph: {edges}\nQ: Is there a path between node {s} and node {t}?\nA:"
        ),
        "answer": f"The answer is {answer}.",
    }


def test_parse_question():
    edges, s, t = parse_question(_record("(0,1) (1,2) (3,4)", 2, 0, "yes")["question"])
    assert edges == [(0, 1), (1, 2), (3, 4)] and (s, t) == (2, 0)


def test_to_rows_converts_and_checks_labels():
    recs = [
        _record("(0,1) (1,2) (3,4)", 2, 0, "yes"),
        _record("(0,1) (1,2) (3,4)", 0, 4, "no"),
        {"task": "cycle", "question": "ignored", "answer": "ignored"},
    ]
    rows = to_rows(recs, "test")
    assert len(rows) == 2  # non-connectivity tasks are dropped
    first = rows[0]
    assert first["y"] == 1 and first["hop_distance"] == 2 and first["n"] == 5
    assert "0,1" in first["encoding"] and "1,0" in first["encoding"]  # both directions
    assert first["encoding"].endswith("QUERY 2 0")
    assert rows[1]["y"] == 0 and rows[1]["hop_distance"] == -1
    with pytest.raises(ValueError, match="disagrees"):
        to_rows([_record("(0,1)", 0, 1, "no")], "test")


def test_audit_rules():
    graph = "(0,1) (1,2) (3,4)"
    train = to_rows([_record(graph, 0, 1, "yes"), _record(graph, 0, 3, "no")], "train")
    test = to_rows(
        [
            _record(graph, 1, 2, "yes"),  # direct edge
            _record(graph, 0, 2, "yes"),  # 2 hops: direct-edge rule misses it
            _record(graph + " (5,6)", 0, 5, "no"),
        ],
        "test",
    )
    rep = audit(train, test)
    assert rep["test"]["direct_edge_rule_acc"] == pytest.approx(2 / 3)
    assert rep["test"]["positive_hops"] == {"1": 1, "2": 1}
    assert rep["test_distinct_graphs_seen_in_train"] == 1
    assert rep["science_open"] is False


def test_nlgraph_audit_artifact_if_present():
    if not AUDIT_PATH.exists():
        pytest.skip("artifacts/nlgraph_connectivity_audit.json not written yet")
    rep = json.loads(AUDIT_PATH.read_text())
    assert rep["science_open"] is False
    for key in ("direct_edge_rule_acc", "isolated_endpoint_rule_acc", "density_rule_acc"):
        assert 0.0 <= rep["test"][key] <= 1.0

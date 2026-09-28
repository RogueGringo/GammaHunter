# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the crossed reachability sets."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from reachability_gen.gen_crossed import (
    CROSSED_EXTENDED_SPEC,
    CROSSED_ID_SPEC,
    endpoint_balance,
    generate_crossed,
    min_nodes,
    plant_crossed,
    verify_crossed,
)
from reachability_gen.gen_id_2k import _hop_matrix
from reachability_gen.reach_cues import reach_cue_report

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("k", [2, 4, 6, 9])
def test_planted_paths_are_exact_unique_and_uncrossed(k):
    rng = random.Random(k)
    for _ in range(20):
        n = min_nodes(k) + rng.randrange(6)
        edges, (s1, t1), (s2, t2) = plant_crossed(n, k, 2.0 / (n - 1), rng)
        assert edges == sorted(set(edges)) and all(u != v and 0 <= u < n and 0 <= v < n for u, v in edges)
        mat = _hop_matrix(n, edges)
        assert mat[s1][t1] == k and mat[s2][t2] == k
        assert mat[s1][t2] < 0 and mat[s2][t1] < 0
        for s, t in ((s1, t1), (s2, t2)):
            on_a_path = [v for v in range(n) if mat[s][v] >= 0 and mat[v][t] >= 0]
            assert len(on_a_path) == k + 1  # the planted path is the only one


def test_too_small_graph_is_refused():
    with pytest.raises(ValueError):
        plant_crossed(min_nodes(5) - 1, 5, 0.1, random.Random(0))


def test_small_set_verifies_and_carries_no_one_endpoint_cue():
    examples, report = generate_crossed(seed=5, n_total=200, n_val=40, spec=CROSSED_ID_SPEC)
    ok, issues = verify_crossed(examples, n_total=200, n_val=40, spec=CROSSED_ID_SPEC)
    assert ok, issues
    rows = [e.to_dict() for e in examples]
    assert endpoint_balance(rows) == {"sources_unbalanced": 0, "targets_unbalanced": 0}
    assert report["graphs_shared_by_train_and_val"] == 0
    cues = reach_cue_report(rows)
    assert set(cues["max_by_horizon"].values()) == {0.5}


def test_extended_spec_generates():
    examples, _ = generate_crossed(seed=6, n_total=40, n_val=40, spec=CROSSED_EXTENDED_SPEC)
    ok, issues = verify_crossed(examples, n_total=40, n_val=40, spec=CROSSED_EXTENDED_SPEC)
    assert ok, issues


def test_verify_catches_broken_crossings():
    examples, _ = generate_crossed(seed=5, n_total=200, n_val=40, spec=CROSSED_ID_SPEC)
    rows = [e.to_dict() for e in examples]
    neg = next(r for r in rows if r["y"] == 0)
    rows[rows.index(neg)] = dict(neg, y=1)
    ok, issues = verify_crossed(rows, n_total=200, n_val=40, spec=CROSSED_ID_SPEC)
    assert not ok and issues


@pytest.mark.parametrize(
    "data, report, spec, sizes",
    [
        ("data/id_crossed_20k.jsonl", "artifacts/id_crossed_20k_generation_report.json",
         CROSSED_ID_SPEC, {"n_total": 20000, "n_val": 4000}),
        ("data/extended_crossed_2k.jsonl", "artifacts/extended_crossed_2k_generation_report.json",
         CROSSED_EXTENDED_SPEC, {"n_total": 2000, "n_val": 2000}),
    ],
)
def test_crossed_artifact_contract(data, report, spec, sizes):
    if not (ROOT / report).exists():
        pytest.skip("generation report not present")
    rep = json.loads((ROOT / report).read_text(encoding="utf-8"))
    assert rep["verify_ok"] is True and rep["science_open"] is False
    assert rep["endpoint_balance"] == {"sources_unbalanced": 0, "targets_unbalanced": 0}
    assert rep["graphs_shared_by_train_and_val"] == 0
    if (ROOT / data).exists():
        ok, issues = verify_crossed(ROOT / data, spec=spec, **sizes)
        assert ok, issues

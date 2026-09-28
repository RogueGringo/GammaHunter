# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the one-endpoint reach-cue audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reachability_gen.reach_cues import (
    FEATURES,
    best_threshold_accuracy,
    endpoint_reach_features,
    reach_cue_report,
)

AUDIT = Path(__file__).resolve().parents[1] / "artifacts" / "reach_cue_audit.json"


def test_features_on_a_small_graph():
    edges = [(0, 1), (1, 2), (2, 3), (4, 3)]
    assert endpoint_reach_features(edges, 0, 3) == {
        "source_reach": 3,
        "source_reach_depth": 3,
        "target_ancestors": 4,
        "target_ancestor_depth": 3,
    }
    assert endpoint_reach_features(edges, 0, 3, limit=1) == {
        "source_reach": 1,
        "source_reach_depth": 1,
        "target_ancestors": 2,
        "target_ancestor_depth": 1,
    }


def test_best_threshold_accuracy():
    assert best_threshold_accuracy([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert best_threshold_accuracy([1, 2, 3, 4], [1, 1, 0, 0]) == 1.0
    assert best_threshold_accuracy([5, 5, 5, 5], [0, 1, 0, 1]) == 0.5
    assert best_threshold_accuracy([1, 2, 3], [1, 0, 1]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        best_threshold_accuracy([1, 2], [1])


def test_report_on_generated_paired_set():
    from reachability_gen.gen_id_disjoint import generate_id_disjoint

    rows = [e.to_dict() for e in generate_id_disjoint(seed=3, n_total=100, n_val=50)[0]]
    rep = reach_cue_report(rows)
    assert rep["n"] == 100 and rep["positive_rate"] == 0.5
    assert set(rep["rules"]) == set(FEATURES)
    for per_horizon in rep["rules"].values():
        assert all(0.5 <= acc <= 1.0 for acc in per_horizon.values())
    # Strong hard negatives: every source has an out-edge and every target an
    # in-edge, so one-hop depth is constant and carries no signal.
    assert rep["rules"]["source_reach_depth"]["within_1"] == 0.5
    assert rep["rules"]["target_ancestor_depth"]["within_1"] == 0.5


@pytest.mark.skipif(not AUDIT.exists(), reason="reach-cue audit artifact not present")
def test_reach_cue_artifact_contract():
    art = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert art["science_open"] is False
    assert art["sets"]
    for rep in art["sets"].values():
        assert set(rep["rules"]) == set(FEATURES)
        for key, best in rep["max_by_horizon"].items():
            assert best == max(rep["rules"][f][key] for f in FEATURES)
            assert 0.5 <= best <= 1.0

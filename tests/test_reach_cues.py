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
    undirected_distance,
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


def test_undirected_distance_ignores_direction():
    edges = [(0, 1), (2, 1), (2, 3)]
    assert undirected_distance(5, edges, 0, 3) == 3  # 0→1←2→3
    assert undirected_distance(5, edges, 3, 0) == 3
    assert undirected_distance(5, edges, 0, 4) == 5  # disconnected: n


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
    for name, rep in art["sets"].items():
        assert set(rep["rules"]) == set(FEATURES)
        assert 0.5 <= rep["direction_blind_distance"] <= 1.0
        for key, best in rep["max_by_horizon"].items():
            assert best == max(rep["rules"][f][key] for f in FEATURES)
            assert 0.5 <= best <= 1.0
        if "crossed" in name:  # every endpoint appears once with each label
            assert set(rep["max_by_horizon"].values()) == {0.5}
        assert rep["pigeonhole"]["fires_on_unreachable"] == 0  # the rule is sound
        if "and_rule" in rep:
            assert set(rep["and_rule"]) == {"train_rows", "endpoints_excluded", "endpoints_counted"}


def test_pigeonhole_rule_is_sound_on_random_graphs():
    import random

    from reachability_gen.reach_cues import _bfs, pigeonhole_fires

    rng = random.Random(0)
    fired = 0
    for _ in range(400):
        n = rng.randint(3, 12)
        edges = sorted({(rng.randrange(n), rng.randrange(n)) for _ in range(rng.randint(0, 3 * n))}
                       - {(i, i) for i in range(n)})
        s, t = rng.sample(range(n), 2)
        fwd: dict[int, list[int]] = {}
        for u, v in edges:
            fwd.setdefault(u, []).append(v)
        if pigeonhole_fires(n, edges, s, t):
            fired += 1
            assert t in _bfs(fwd, s, None)  # when it fires, the target is reachable
    assert fired > 0


def test_and_rule_fit_separates_a_toy_set():
    from reachability_gen.reach_cues import and_rule_fit

    xs = [0.1, 0.2, 0.5, 0.6, 0.7, 0.8]
    ys = [0.9, 0.1, 0.6, 0.7, 0.2, 0.9]
    a, b, acc = and_rule_fit(xs, ys, [0, 0, 1, 1, 0, 1])
    assert acc == 1.0 and (a, b) == (0.2, 0.6)  # the first perfect pair in ascending order
    assert and_rule_fit([0.3, 0.3], [0.3, 0.3], [0, 0])[2] == 1.0  # a rule that never fires


def test_and_rule_report_fits_on_train_and_scores_the_given_rows():
    from reachability_gen.gen_id_disjoint import generate_id_disjoint
    from reachability_gen.reach_cues import and_rule_report

    rows = [e.to_dict() for e in generate_id_disjoint(seed=3, n_total=100, n_val=50)[0]]
    train, val = [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"]
    rep = and_rule_report(train, val)
    assert rep["train_rows"] == len(train)
    for key in ("endpoints_excluded", "endpoints_counted"):
        v = rep[key]
        assert 0.0 <= v["accuracy"] <= 1.0 and 0.5 <= v["train_accuracy"] <= 1.0
        labels = [int(r["y"]) for r in val]
        fr = __import__("reachability_gen.reach_cues", fromlist=["reach_fractions"]).reach_fractions(
            val, endpoints=key == "endpoints_counted")
        hits = sum(int((x >= v["source_fraction_at_least"] and y >= v["target_fraction_at_least"]) == bool(lab))
                   for (x, y), lab in zip(fr, labels))
        assert v["accuracy"] == hits / len(labels)  # scored with the thresholds fitted on train

# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the learned no-search ceilings (scikit-learn is an optional extra)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reachability_gen.cue_ceiling import PLAN, ancestor_side, endpoint_features, pair_features, profile_features

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "cue_ceiling.json"


def test_pair_features_ignore_direction_and_count_shortest_paths():
    edges = [(0, 1), (0, 2), (3, 1), (3, 2)]  # undirected: 0-1-3 and 0-2-3
    assert pair_features(5, edges, 0, 3) == [2.0, 2.0, 2.0]
    assert pair_features(5, edges, 0, 4) == [5.0, 0.0, 0.0]  # disconnected: distance n


def test_profile_and_ancestor_side_features():
    edges = [(0, 1), (1, 2), (0, 2), (3, 2)]
    src, tgt = profile_features(4, edges, 0, 2)
    assert src[:2] == [2.0, 0.0] and src[-2:] == [2.0, 0.0]  # 1 and 2 at distance 1; out 2, in 0
    assert tgt[:2] == [3.0, 0.0] and tgt[-2:] == [0.0, 3.0]  # 0, 1 and 3 at distance 1; out 0, in 3
    f = endpoint_features(4, edges, 0, 2)
    assert len(f) == 13 and len(ancestor_side(f)) == 7 and ancestor_side(f)[0] == 4.0


def test_score_smoke_on_generated_sets():
    pytest.importorskip("sklearn")
    from reachability_gen.cue_ceiling import score
    from reachability_gen.gen_crossed import CROSSED_EXTENDED_SPEC, generate_crossed

    rows = [e.to_dict() for e in generate_crossed(seed=21, n_total=40, n_val=20)[0]]
    ext = [e.to_dict() for e in generate_crossed(seed=22, n_total=20, n_val=20, spec=CROSSED_EXTENDED_SPEC)[0]]
    split = score(rows, "endpoint_pair", seeds=(0, 1))
    assert split["mode"].startswith("train split") and 0.0 <= split["accuracy"]["min"] <= 1.0
    cv = score(ext, "profile_pair", seeds=(0,), folds=2)
    assert cv["mode"].startswith("grouped") and set(cv["seeds"]) == {"0"}
    for res in (split, cv):  # the shuffled-label control is recorded for every seed
        assert len(res["null_shuffled_labels"]["per_seed"]) == len(res["seeds"])


@pytest.mark.skipif(not ARTIFACT.exists(), reason="cue-ceiling artifact not present")
def test_cue_ceiling_artifact_contract():
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert art["science_open"] is False
    assert set(art["results"]) == {f"{path}/{fs}" for path, fs in PLAN}
    for res in art["results"].values():
        accs = [s["accuracy"] for s in res["seeds"].values()]
        assert res["accuracy"]["min"] == min(accs) and res["accuracy"]["max"] == max(accs)

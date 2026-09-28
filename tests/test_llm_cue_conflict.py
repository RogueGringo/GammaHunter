# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the LLM cue-conflict test (no model needed)."""

from __future__ import annotations

from reachability_gen.encode import encode_instance
from reachability_gen.llm_cue_conflict import cue_groups, pairwise


def _pair(eh, n, edges, pos, neg):
    return [
        {"edge_hash": eh, "encoding": encode_instance(n, edges, *pos), "y": 1, "s": pos[0], "t": pos[1]},
        {"edge_hash": eh, "encoding": encode_instance(n, edges, *neg), "y": 0, "s": neg[0], "t": neg[1]},
    ]


# g1: the reachable target (3) has 3 ancestors, the unreachable one (5) has 1 -> agrees with the cue.
G1 = _pair("g1", 6, [(0, 1), (1, 2), (2, 3), (4, 5)], (0, 3), (0, 5))
# g2: the reachable target (1) has 1 ancestor, the unreachable one (5) has 3 -> conflicts with it.
G2 = _pair("g2", 6, [(0, 1), (2, 3), (3, 4), (4, 5)], (0, 1), (0, 5))


def test_cue_groups_split_agreeing_and_conflicting_graphs():
    groups = cue_groups(G1 + G2, "target_ancestors", 6)
    assert groups == {"agree": ["g1"], "conflict": ["g2"], "tie": []}


def test_pairwise_scores_each_group():
    # A cue follower: margins rise with the target's ancestor count.
    follower = [3.0, 1.0, 1.0, 3.0]
    out = pairwise(G1 + G2, follower, "target_ancestors", 6)
    assert out["agree"]["ranked_reachable_higher"] == 1.0
    assert out["conflict"]["ranked_reachable_higher"] == 0.0
    # A searcher ranks the reachable question higher in both groups.
    searcher = [2.0, -2.0, 2.0, -2.0]
    out = pairwise(G1 + G2, searcher, "target_ancestors", 6)
    assert out["agree"]["ranked_reachable_higher"] == out["conflict"]["ranked_reachable_higher"] == 1.0


def test_ties_count_half_and_are_left_out_of_the_decided_rate():
    blind = [0.7, 0.7, -1.0, -1.0]  # equal margins within each graph: two ties
    out = pairwise(G1 + G2, blind, "target_ancestors", 6)
    for group in ("agree", "conflict"):
        g = out[group]
        assert (g["wins"], g["ties"], g["losses"]) == (0, 1, 0)
        assert g["ranked_reachable_higher"] == 0.5 and g["decided_win_rate"] is None

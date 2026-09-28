# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the path audit of step-by-step generations (text only)."""

from __future__ import annotations

from reachability_gen.encode import encode_instance
from reachability_gen.llm_cot_paths import audit, audit_text, classify_step, group_of

EDGES = {(4, 9), (3, 9), (4, 8), (9, 5)}


def test_classify_step():
    assert classify_step(4, 9, EDGES) == "listed"
    assert classify_step(9, 3, EDGES) == "reversed"
    assert classify_step(8, 3, EDGES) == "absent"


def test_final_written_path_and_cited_edges():
    text = ("From node 9, we have two outgoing edges: 3->9 and 0->9. So 4->9->3 is a path.\n"
            "Rechecking: 4 -> 9 -> 5.\nAnswer: Yes")
    res = audit_text(text, EDGES, 4, 3)
    assert res["cited"] == ["listed", "absent"]
    assert res["path"] == {"steps": ["listed", "listed"], "from_s_to_t": False}  # the last path counts
    assert audit_text("4→9→3", EDGES, 4, 3)["path"] == {"steps": ["listed", "reversed"], "from_s_to_t": True}
    assert audit_text("No arrows here. Answer: No", EDGES, 4, 3) == {"cited": [], "path": None}


def test_audit_groups_by_label_and_answer():
    enc = encode_instance(10, sorted(EDGES), 4, 3)
    graphs = {("g", 4, 3): enc}
    gens = [
        {"model": "m", "set": "crossed_val", "edge_hash": "g", "s": 4, "t": 3, "y": 0, "answer": 1,
         "text": "4->9->3 Answer: Yes"},
        {"model": "m", "set": "crossed_val", "edge_hash": "g", "s": 4, "t": 3, "y": 0, "answer": None, "text": "?"},
        {"model": "m", "set": "no_graph", "edge_hash": "g", "s": 4, "t": 3, "y": 0, "answer": 0, "text": "skipped"},
    ]
    out = audit(gens, graphs)["m"]
    assert set(out) == {"crossed_val"}
    wrong = out["crossed_val"]["yes_on_unreachable"]
    assert (wrong["generations"], wrong["with_written_path"], wrong["path_with_reversed_step"],
            wrong["path_all_steps_listed"], wrong["path_from_s_to_t"]) == (1, 1, 1, 0, 1)
    assert out["crossed_val"]["unparsed"]["generations"] == 1
    assert group_of(1, 0) == "no_on_reachable"

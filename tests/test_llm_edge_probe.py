# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the edge-lookup probe (question construction; no model needed)."""

from __future__ import annotations

import random

from reachability_gen.encode import parse_instance
from reachability_gen.gen_crossed import generate_crossed
from reachability_gen.run_llm_edge_probe import choose_edge_shots, edge_prompt, edge_questions, probe_rows, summarise


def _rows():
    return [e.to_dict() for e in generate_crossed(seed=5, n_total=200, n_val=40)[0]]


def test_questions_have_the_intended_kinds():
    for r in _rows()[:40]:
        _, edges, _, _ = parse_instance(r["encoding"])
        present = set(edges)
        qs = edge_questions(r["encoding"], random.Random(0))
        assert [q["kind"] for q in qs] == ["listed", "listed", "reversed", "absent"]
        for q in qs:
            pair, back = (q["u"], q["v"]), (q["v"], q["u"])
            if q["kind"] == "listed":
                assert pair in present and q["y"] == 1
            elif q["kind"] == "reversed":
                assert pair not in present and back in present and q["y"] == 0
            else:
                assert pair not in present and back not in present and q["y"] == 0 and q["u"] != q["v"]


def test_probe_rows_and_shots():
    rows = _rows()
    probe = probe_rows([r for r in rows if r["split"] == "val"])
    assert len(probe) == 4 * len({r["edge_hash"] for r in rows if r["split"] == "val"})
    assert sum(r["y"] for r in probe) * 2 == len(probe)  # balanced
    shots = choose_edge_shots([r for r in rows if r["split"] == "train"])
    assert [s["kind"] for s in shots] == ["listed", "reversed", "listed", "absent"]
    assert all(len(s["edge_hash"]) == 64 for s in shots)  # recorded in the artifact
    prompt = edge_prompt(probe[0]["encoding"], probe[0]["u"], probe[0]["v"], shots)
    assert prompt.count("Answer: Yes") == 2 and prompt.count("Answer: No") == 2 and prompt.endswith("Answer:")


def test_summarise_splits_by_kind():
    rows = [{"y": 1, "kind": "listed"}, {"y": 1, "kind": "listed"}, {"y": 0, "kind": "reversed"}, {"y": 0, "kind": "absent"}]
    out = summarise(rows, [1.0, 2.0, 0.5, -1.0])  # reversed wrongly accepted
    assert out["acc"] == 0.75 and out["acc_by_kind"] == {"absent": 1.0, "listed": 1.0, "reversed": 0.0}
    assert out["auroc"] > 0.5

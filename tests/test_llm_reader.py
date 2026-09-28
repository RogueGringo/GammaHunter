# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for language models as the graph reader (parsing and scoring; no language model needed)."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.encode import encode_instance  # noqa: E402
from reachability_gen.run_llm_reader import (  # noqa: E402
    graphs_of,
    parse_successors,
    score,
    step_by_step_reference,
    successor_prompt,
)


def test_prompt_names_the_node_but_no_question():
    q = successor_prompt([(0, 1), (2, 0)], 2)
    assert "0->1, 2->0" in q and "2->v" in q and "path" not in q.lower()


@pytest.mark.parametrize("text, want, raised", [
    ("3, 7", {3, 7}, []),
    ("none", set(), []),
    ("None.", set(), []),
    ("4->3, 4->7", {3, 7}, []),  # the asked node itself is dropped
    ("The nodes are 3 and 7.", {3, 7}, ["extra_words"]),
    ("3, 99", {3}, ["out_of_range"]),
    ("I am not sure.", set(), ["no_answer", "extra_words"]),
])
def test_parse_successors(text, want, raised):
    succ, flags = parse_successors(text, 4, 10)
    assert succ == want and [k for k, v in flags.items() if v] == raised


class ClosureSolver(torch.nn.Module):
    """Answers exactly from the reachability of whatever graph it is given."""

    def forward(self, graph, steps):
        from reachability_gen.models.reader import closure

        reach = closure(graph["adj"].float(), graph["node_mask"])
        hit = reach[torch.arange(len(graph["s"])), graph["s"], graph["t"]].float()
        return torch.stack([1 - hit, hit], dim=1)


def _rows():
    edges = [(0, 1), (1, 2), (3, 4)]
    return [{"edge_hash": "g", "encoding": encode_instance(5, edges, s, t), "s": s, "t": t, "y": y, "hop_distance": h}
            for s, t, y, h in [(0, 2, 1, 2), (3, 4, 1, 1), (0, 4, 0, -1), (3, 2, 0, -1)]]


def test_score_attributes_errors_to_the_graph():
    rows = _rows()
    assert graphs_of(rows) == {"g": (5, [(0, 1), (1, 2), (3, 4)])}
    exact = score(rows, {"g": {(0, 1), (1, 2), (3, 4)}}, [(0, ClosureSolver())], (6,), "cpu")
    assert exact["reader"]["exact_graphs"] == 1.0 and exact["pipeline"]["6"]["accuracy"]["mean"] == 1.0
    missing = score(rows, {"g": {(0, 1), (3, 4)}}, [(0, ClosureSolver())], (6,), "cpu")  # 1->2 not read
    p = missing["pipeline"]["6"]
    assert missing["reader"]["recall"] == pytest.approx(2 / 3) and p["accuracy"]["mean"] == 0.75
    assert p["errors_over_solvers"] == {"graph_caused": 1, "solver_caused": 0} and p["true_graph_accuracy_mean"] == 1.0


def test_step_by_step_reference_requires_the_same_questions(tmp_path):
    rows = _rows()
    cot = {"sets": {"crossed_val": [{"edge_hash": r["edge_hash"], "s": r["s"], "t": r["t"], "y": r["y"]} for r in rows]},
           "models": {"m": {"sets": {"crossed_val": {"acc": 0.5, "parsed_rate": 0.8}}}}}
    path = tmp_path / "cot.json"
    path.write_text(json.dumps(cot))
    ref = step_by_step_reference([path], "m", "crossed_val", rows)
    assert ref["accuracy_among_parsed"] == pytest.approx(0.625)
    assert step_by_step_reference([path], "other", "crossed_val", rows) is None
    with pytest.raises(ValueError):
        step_by_step_reference([path], "m", "crossed_val", rows[:2])

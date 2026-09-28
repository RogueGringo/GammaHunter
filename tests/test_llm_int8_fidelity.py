# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the 8-bit fidelity gate."""

from __future__ import annotations

import json

import pytest

from reachability_gen.llm_int8_fidelity import compare, gate, pearson


def test_pearson():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_compare_passes_close_margins_and_fails_distorted_ones():
    labels = [1, 0, 1, 0, 1, 0]
    full = [2.0, -1.5, 1.0, -0.5, 3.0, -2.0]
    assert compare(full, [m * 1.01 + 0.01 for m in full], labels)["passes"] is True
    assert compare(full, [-m for m in full], labels)["passes"] is False


def test_gate_refuses_different_questions(tmp_path):
    def artifact(rows, margins):
        return {"sets": {"s": rows}, "models": {"m": {"sets": {"s": {"margins": margins}}}}}

    rows = [{"y": 1}, {"y": 0}]
    a, b, c = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "c.json"
    a.write_text(json.dumps(artifact(rows, [1.0, -1.0])))
    b.write_text(json.dumps(artifact(rows, [1.1, -0.9])))
    c.write_text(json.dumps(artifact([{"y": 0}, {"y": 1}], [1.0, -1.0])))
    assert gate("m", [(a, b)])["passes"] is True
    with pytest.raises(ValueError):
        gate("m", [(a, c)])

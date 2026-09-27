# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Unit tests for hard-negative degree / reachability filter."""

from __future__ import annotations

from reachability_gen.encode import encode_instance, parse_instance
from reachability_gen.hard_negatives import (
    classify_y0_row,
    filter_hard_negatives,
    is_hard_negative,
    total_degrees,
)


def test_parse_instance_roundtrip():
    n, edges, s, t = 5, [(0, 1), (2, 3)], 0, 4
    enc = encode_instance(n, edges, s, t)
    n2, edges2, s2, t2 = parse_instance(enc)
    assert (n2, edges2, s2, t2) == (n, list(edges), s, t)


def test_parse_instance_empty_edges():
    enc = encode_instance(3, [], 1, 2)
    n, edges, s, t = parse_instance(enc)
    assert n == 3 and edges == [] and s == 1 and t == 2


def test_total_degrees():
    # 0→1, 1→2: deg = [1, 2, 1]
    deg = total_degrees(3, [(0, 1), (1, 2)])
    assert deg == [1, 2, 1]


def test_hard_negative_accepts_dead_end():
    # 0→1 and 2→3; query 0→3 unreachable; both endpoints have deg>=1.
    edges = [(0, 1), (2, 3)]
    ok, reason = is_hard_negative(4, edges, 0, 3, y=0)
    assert ok and reason == "ok"


def test_hard_negative_rejects_isolated_source():
    # Node 0 isolated; 1→2. Query 0→2: deg(s)=0.
    edges = [(1, 2)]
    ok, reason = is_hard_negative(3, edges, 0, 2, y=0)
    assert not ok and reason == "deg_s0"


def test_hard_negative_rejects_isolated_target():
    edges = [(0, 1)]
    ok, reason = is_hard_negative(3, edges, 0, 2, y=0)
    assert not ok and reason == "deg_t0"


def test_hard_negative_rejects_both_isolated():
    ok, reason = is_hard_negative(3, [], 0, 1, y=0)
    assert not ok and reason == "both_deg0"


def test_hard_negative_rejects_reachable():
    edges = [(0, 1), (1, 2)]
    ok, reason = is_hard_negative(3, edges, 0, 2, y=0)
    assert not ok and reason == "reachable"


def test_hard_negative_rejects_not_y0():
    ok, reason = is_hard_negative(3, [(0, 1)], 0, 2, y=1)
    assert not ok and reason == "not_y0"


def test_filter_hard_negatives_counts_reasons():
    rows = [
        {
            "y": 0,
            "n": 4,
            "s": 0,
            "t": 3,
            "encoding": encode_instance(4, [(0, 1), (2, 3)], 0, 3),
        },
        {
            "y": 0,
            "n": 3,
            "s": 0,
            "t": 2,
            "encoding": encode_instance(3, [(1, 2)], 0, 2),
        },
        {
            "y": 1,
            "n": 3,
            "s": 0,
            "t": 1,
            "encoding": encode_instance(3, [(0, 1)], 0, 1),
        },
    ]
    accepted, reasons = filter_hard_negatives(rows)
    assert len(accepted) == 1
    assert reasons["ok"] == 1
    assert reasons["deg_s0"] == 1
    assert reasons["skipped_not_y0"] == 1


def test_classify_y0_row_uses_encoding():
    row = {
        "y": 0,
        "encoding": encode_instance(4, [(0, 1), (2, 3)], 0, 3),
        "s": 0,
        "t": 3,
        "n": 4,
    }
    ok, reason = classify_y0_row(row)
    assert ok and reason == "ok"

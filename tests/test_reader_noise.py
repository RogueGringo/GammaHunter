# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the noise-tolerance curve (corruptions and the deletion expectation)."""

from __future__ import annotations

import random
import statistics

from reachability_gen.reader_noise import corrupt, deletion_accuracy, matched, random_count

EDGES = [(0, 1), (1, 2), (2, 3)]


def test_corruptions():
    rng = random.Random(0)
    assert corrupt(4, EDGES, "delete", 0.0, rng) == set(EDGES) and corrupt(4, EDGES, "delete", 1.0, rng) == set()
    assert corrupt(4, EDGES, "reverse", 1.0, rng) == {(1, 0), (2, 1), (3, 2)}
    inserted = corrupt(4, EDGES, "insert", 1.0, rng)
    assert set(EDGES) <= inserted and len(inserted) == 6 and all(u != v for u, v in inserted)


def test_random_count_is_unbiased():
    rng = random.Random(1)
    assert abs(statistics.fmean(random_count(0.3, rng) for _ in range(20000)) - 0.3) < 0.02


def test_matched_noise_follows_its_targets():
    rng = random.Random(2)
    assert matched(4, EDGES, recall=1.0, extra_per_edge=0.0, reversed_share=0.0, rng=rng) == set(EDGES)
    noisy = matched(4, EDGES, recall=1.0, extra_per_edge=1.0, reversed_share=1.0, rng=rng)
    assert noisy == set(EDGES) | {(1, 0), (2, 1), (3, 2)}


def test_deletion_expectation_matches_simulation():
    """One positive with a single two-edge path and one negative: accuracy 1/2 + 1/2 (1 - p)^2."""
    rows = [{"y": 1, "hop_distance": 2}, {"y": 0, "hop_distance": -1}]
    p = 0.3
    assert abs(deletion_accuracy(rows, p) - (0.5 + 0.5 * 0.49)) < 1e-12
    rng = random.Random(3)
    hits = 0
    for _ in range(4000):
        kept = corrupt(3, [(0, 1), (1, 2)], "delete", p, rng)
        hits += ((0, 1) in kept and (1, 2) in kept) + 1  # the negative (2 -> 0) stays unreachable
    assert abs(hits / 8000 - deletion_accuracy(rows, p)) < 0.02


def test_noise_artifact_contract():
    import json
    from pathlib import Path

    import pytest

    path = Path(__file__).resolve().parents[1] / "artifacts" / "reader_noise.json"
    if not path.exists():
        pytest.skip("noise artifact not present")
    art = json.loads(path.read_text(encoding="utf-8"))
    assert art["science_open"] is False
    assert all(acc == 1.0 for steps in art["true_graph"].values() for acc in steps.values())
    for set_name, curves in art["curves"].items():  # deletions follow the single-path expectation
        for rate, cell in curves["delete"].items():
            first = next(iter(cell["pipeline"].values()))["accuracy"]["mean"]
            assert abs(first - art["deletion_expectation"][set_name][rate]) < 0.01
    assert len(art["matched_noise"]) == 3

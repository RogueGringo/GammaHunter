# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for id_2k generation constraints and comparison artifact hygiene.

Full 2k regen + dual-arm training is marked ``slow``. Fast tests cover:
  - quota constants / report builder
  - verify_id_2k against synthetic rows
  - existing data/id_2k.jsonl when present
  - comparison JSON science_open=False when present
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reachability_gen.encode import encode_instance
from reachability_gen.gen_id_2k import (
    ID_2K_SEED,
    ID_HOPS,
    N_NEG_TOTAL,
    N_NEG_TRAIN,
    N_NEG_VAL,
    N_POS_TOTAL,
    N_POS_TRAIN,
    N_POS_VAL,
    N_TOTAL,
    N_TRAIN,
    N_VAL,
    POS_PER_HOP_TOTAL,
    POS_PER_HOP_TRAIN,
    POS_PER_HOP_VAL,
    build_generation_report,
    verify_id_2k,
)
from reachability_gen.hard_negatives import is_hard_negative
from reachability_gen.schema import ReachabilityExample

ROOT = Path(__file__).resolve().parents[1]
ID_2K_PATH = ROOT / "data" / "id_2k.jsonl"
COMPARISON_PATH = ROOT / "artifacts" / "id_2k_comparison.json"


def test_id_2k_quota_constants():
    assert N_TOTAL == 2000
    assert N_TRAIN + N_VAL == N_TOTAL
    assert N_POS_TOTAL + N_NEG_TOTAL == N_TOTAL
    assert N_POS_TRAIN + N_NEG_TRAIN == N_TRAIN
    assert N_POS_VAL + N_NEG_VAL == N_VAL
    assert N_POS_TRAIN == 800 and N_NEG_TRAIN == 800
    assert N_POS_VAL == 200 and N_NEG_VAL == 200
    assert len(ID_HOPS) == 5
    assert POS_PER_HOP_TOTAL * len(ID_HOPS) == N_POS_TOTAL
    assert POS_PER_HOP_TRAIN * len(ID_HOPS) == N_POS_TRAIN
    assert POS_PER_HOP_VAL * len(ID_HOPS) == N_POS_VAL
    assert ID_2K_SEED == 42_000


def _synth_row(
    *,
    split: str,
    y: int,
    hop: int,
    n: int = 8,
    s: int = 0,
    t: int = 1,
    edges: list | None = None,
    idx: int = 0,
) -> dict:
    edges = edges if edges is not None else [(0, 2), (2, 1), (3, 4)]
    # For y=0 hard-neg template: unreachable 0→4 with deg>=1 both ends.
    if y == 0:
        edges = [(0, 2), (3, 4)]
        s, t = 0, 4
        n = 5
        hop = -1
    enc = encode_instance(n, edges, s, t)
    return ReachabilityExample(
        split=split,
        seed=idx,
        n=n,
        p=0.15,
        edge_hash=f"h{idx}",
        s=s,
        t=t,
        y=y,
        hop_distance=hop,
        is_ood=False,
        encoding=enc,
    ).to_dict()


def test_verify_id_2k_rejects_wrong_counts():
    rows = [_synth_row(split="train", y=1, hop=2, idx=i) for i in range(10)]
    ok, issues = verify_id_2k(rows, check_hard_neg=False)
    assert not ok
    assert any("n_total" in x for x in issues)


def test_verify_id_2k_accepts_balanced_synthetic():
    """Build a tiny structurally-correct mini set by scaling quotas is heavy;
    instead craft exact full quotas with stub encodings (hard-neg check on)."""
    rows: list[dict] = []
    idx = 0
    # Positives: uniform hops.
    for split, per_hop in (("train", POS_PER_HOP_TRAIN), ("val", POS_PER_HOP_VAL)):
        for k in ID_HOPS:
            for _ in range(per_hop):
                # Chain 0→1→... so hop k is reachable with enough nodes.
                n = k + 2
                edges = [(i, i + 1) for i in range(k)]
                rows.append(
                    _synth_row(
                        split=split,
                        y=1,
                        hop=k,
                        n=n,
                        s=0,
                        t=k,
                        edges=edges,
                        idx=idx,
                    )
                )
                # Fix hop_distance to k (encode may differ but we set field).
                rows[-1]["hop_distance"] = k
                rows[-1]["y"] = 1
                rows[-1]["s"] = 0
                rows[-1]["t"] = k
                idx += 1
    # Hard negatives.
    for split, n_neg in (("train", N_NEG_TRAIN), ("val", N_NEG_VAL)):
        for _ in range(n_neg):
            rows.append(_synth_row(split=split, y=0, hop=-1, idx=idx))
            idx += 1

    assert len(rows) == N_TOTAL
    # Sanity: synthetic y0 is hard.
    ok_hn, reason = is_hard_negative(5, [(0, 2), (3, 4)], 0, 4, y=0)
    assert ok_hn and reason == "ok"

    ok, issues = verify_id_2k(rows, check_hard_neg=True)
    assert ok, issues
    report = build_generation_report(rows, seed=ID_2K_SEED)
    assert report["n_total"] == N_TOTAL
    assert report["science_open"] is False
    assert report["y_counts"]["1"] == N_POS_TOTAL
    assert report["y_counts"]["0"] == N_NEG_TOTAL


def test_id_2k_jsonl_constraints_if_present():
    if not ID_2K_PATH.exists():
        pytest.skip("data/id_2k.jsonl not generated yet")
    ok, issues = verify_id_2k(ID_2K_PATH, check_hard_neg=True)
    assert ok, issues
    # Spot-check file length.
    n_lines = sum(1 for _ in ID_2K_PATH.open())
    assert n_lines == N_TOTAL


def test_comparison_json_science_open_false_if_present():
    if not COMPARISON_PATH.exists():
        pytest.skip("artifacts/id_2k_comparison.json not written yet")
    data = json.loads(COMPARISON_PATH.read_text())
    assert data.get("science_open") is False
    # No nested science_open=True anywhere.
    def _walk(obj):
        if isinstance(obj, dict):
            if obj.get("science_open") is True:
                return False
            return all(_walk(v) for v in obj.values())
        if isinstance(obj, list):
            return all(_walk(v) for v in obj)
        return True
    assert _walk(data)
    # Hop buckets present for FF and Geo.
    for arm in ("ff", "geo"):
        assert arm in data
        by_hop = data[arm]["val_by_hop"]
        for k in ["2", "3", "4", "5", "6", "-1"]:
            assert k in by_hop, f"missing hop {k} in {arm}"
            assert "acc_mean" in by_hop[k]
            assert "loss_mean" in by_hop[k]
    # Geo drift fields.
    geo_h2 = data["geo"]["val_by_hop"]["2"]
    assert "mean_drift_trajectory" in geo_h2
    assert "mean_terminal_drift" in geo_h2
    assert "terminal_drift_damps" in geo_h2 or "terminal_drift_regime" in geo_h2
    assert "drift_summary" in data["geo"]


@pytest.mark.slow
def test_generate_id_2k_end_to_end(tmp_path: Path):
    from reachability_gen.gen_id_2k import generate_id_2k, write_jsonl

    examples, report = generate_id_2k(seed=ID_2K_SEED, max_graph_draws=20_000)
    ok, issues = verify_id_2k(examples)
    assert ok, issues
    out = tmp_path / "id_2k.jsonl"
    write_jsonl(out, examples)
    assert report["science_open"] is False

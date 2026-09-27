# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for OOD-hop generator constraints (Gate 2 MEASURE plumbing).

Full regen is marked ``slow``. Fast tests cover quotas, verify helpers,
synthetic balance / hard-neg / is_ood, and existing data/ood_hops.jsonl
when present. Gate2 artifact hygiene when present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reachability_gen.adr_invariants import (
    HOP_UNREACHABLE,
    K_TRAIN_MAX,
    OOD_HOP_VALUES,
    is_ood_hop,
)
from reachability_gen.encode import encode_instance
from reachability_gen.gen_ood_hops import (
    N_NEG_DEFAULT,
    OOD_HOPS,
    OOD_HOPS_SEED,
    POS_PER_HOP_DEFAULT,
    build_generation_report,
    verify_ood_hops,
)
from reachability_gen.hard_negatives import is_hard_negative
from reachability_gen.schema import ReachabilityExample

ROOT = Path(__file__).resolve().parents[1]
OOD_PATH = ROOT / "data" / "ood_hops.jsonl"
GATE2_PATH = ROOT / "artifacts" / "id_2k_rematch_bound30_gate2_ood.json"
REPORT_PATH = ROOT / "artifacts" / "ood_hops_generation_report.json"


def test_ood_hop_constants_match_adr():
    assert OOD_HOPS == tuple(OOD_HOP_VALUES) == (8, 12, 16)
    assert POS_PER_HOP_DEFAULT >= 40
    assert N_NEG_DEFAULT == POS_PER_HOP_DEFAULT * len(OOD_HOPS)
    assert OOD_HOPS_SEED == 84_000
    for k in OOD_HOPS:
        assert is_ood_hop(k, k_train_max=K_TRAIN_MAX)
    assert not is_ood_hop(HOP_UNREACHABLE)


def test_verify_ood_hops_rejects_wrong_counts():
    rows = [
        ReachabilityExample(
            split="ood_hops",
            seed=i,
            n=20,
            p=0.05,
            edge_hash=f"h{i}",
            s=0,
            t=8,
            y=1,
            hop_distance=8,
            is_ood=True,
            encoding=encode_instance(20, [(j, j + 1) for j in range(8)], 0, 8),
        ).to_dict()
        for i in range(5)
    ]
    ok, issues = verify_ood_hops(rows, pos_per_hop=80, n_neg=240, check_hard_neg=False)
    assert not ok
    assert any("n_total" in x for x in issues)


def _synth_pos(hop: int, idx: int) -> dict:
    n = hop + 2
    edges = [(i, i + 1) for i in range(hop)]
    return ReachabilityExample(
        split="ood_hops",
        seed=idx,
        n=n,
        p=0.05,
        edge_hash=f"pos{idx}",
        s=0,
        t=hop,
        y=1,
        hop_distance=hop,
        is_ood=True,
        encoding=encode_instance(n, edges, 0, hop),
    ).to_dict()


def _synth_neg(idx: int) -> dict:
    edges = [(0, 2), (3, 4)]
    n, s, t = 5, 0, 4
    return ReachabilityExample(
        split="ood_hops",
        seed=idx,
        n=n,
        p=0.05,
        edge_hash=f"neg{idx}",
        s=s,
        t=t,
        y=0,
        hop_distance=HOP_UNREACHABLE,
        is_ood=False,
        encoding=encode_instance(n, edges, s, t),
    ).to_dict()


def test_verify_ood_hops_accepts_balanced_synthetic():
    pos_per = 4  # tiny synthetic quota
    n_neg = pos_per * len(OOD_HOPS)
    rows: list[dict] = []
    idx = 0
    for k in OOD_HOPS:
        for _ in range(pos_per):
            rows.append(_synth_pos(k, idx))
            idx += 1
    for _ in range(n_neg):
        rows.append(_synth_neg(idx))
        idx += 1

    ok_hn, reason = is_hard_negative(5, [(0, 2), (3, 4)], 0, 4, y=0)
    assert ok_hn and reason == "ok"

    ok, issues = verify_ood_hops(
        rows, pos_per_hop=pos_per, n_neg=n_neg, check_hard_neg=True
    )
    assert ok, issues
    report = build_generation_report(
        rows, seed=OOD_HOPS_SEED, pos_per_hop=pos_per, n_neg=n_neg
    )
    assert report["science_open"] is False
    assert report["y_counts"]["1"] == pos_per * len(OOD_HOPS)
    assert report["y_counts"]["0"] == n_neg


def test_verify_rejects_pos_without_is_ood():
    pos_per = 2
    n_neg = pos_per * len(OOD_HOPS)
    rows: list[dict] = []
    idx = 0
    for k in OOD_HOPS:
        for _ in range(pos_per):
            r = _synth_pos(k, idx)
            r["is_ood"] = False  # illegal for OOD pos
            rows.append(r)
            idx += 1
    for _ in range(n_neg):
        rows.append(_synth_neg(idx))
        idx += 1
    ok, issues = verify_ood_hops(
        rows, pos_per_hop=pos_per, n_neg=n_neg, check_hard_neg=False
    )
    assert not ok
    assert any("is_ood" in x for x in issues)


def test_verify_rejects_non_ood_hop():
    pos_per = 2
    n_neg = pos_per * len(OOD_HOPS)
    rows: list[dict] = []
    idx = 0
    for k in OOD_HOPS:
        for _ in range(pos_per):
            rows.append(_synth_pos(k, idx))
            idx += 1
    # Replace one positive hop with ID hop 4
    rows[0]["hop_distance"] = 4
    rows[0]["is_ood"] = False
    for _ in range(n_neg):
        rows.append(_synth_neg(idx))
        idx += 1
    ok, issues = verify_ood_hops(
        rows, pos_per_hop=pos_per, n_neg=n_neg, check_hard_neg=False
    )
    assert not ok


def test_ood_jsonl_constraints_if_present():
    if not OOD_PATH.exists():
        pytest.skip("data/ood_hops.jsonl not generated yet")
    ok, issues = verify_ood_hops(OOD_PATH, check_hard_neg=True)
    assert ok, issues
    n_lines = sum(1 for _ in OOD_PATH.open())
    assert n_lines == POS_PER_HOP_DEFAULT * len(OOD_HOPS) + N_NEG_DEFAULT


def test_ood_generation_report_if_present():
    if not REPORT_PATH.exists():
        pytest.skip("artifacts/ood_hops_generation_report.json not written yet")
    data = json.loads(REPORT_PATH.read_text())
    assert data.get("science_open") is False
    assert data.get("verify_ok") is True
    for k in OOD_HOPS:
        assert data["hop_counts"][str(k)] == POS_PER_HOP_DEFAULT
    assert data["hop_counts"][str(HOP_UNREACHABLE)] == N_NEG_DEFAULT


def test_gate2_artifact_science_open_false_if_present():
    if not GATE2_PATH.exists():
        pytest.skip("gate2 artifact not written yet")
    data = json.loads(GATE2_PATH.read_text())
    assert data.get("science_open") is False

    def _walk(obj):
        if isinstance(obj, dict):
            if obj.get("science_open") is True:
                return False
            return all(_walk(v) for v in obj.values())
        if isinstance(obj, list):
            return all(_walk(v) for v in obj)
        return True

    assert _walk(data)
    assert "fixed_depth" in data
    assert "dynamic_unroll" in data
    assert "prereg" in data or "interpretation_stop" in data
    for arm in ("ff", "geo", "loop"):
        assert arm in data["fixed_depth"]
        by_hop = data["fixed_depth"][arm]["by_hop"]
        for k in ["8", "12", "16", "-1"]:
            assert k in by_hop
    for T in ("8", "12", "16"):
        assert T in data["dynamic_unroll"]["geo"]
        assert T in data["dynamic_unroll"]["loop"]


@pytest.mark.slow
def test_generate_ood_hops_end_to_end(tmp_path: Path):
    from reachability_gen.gen_ood_hops import generate_ood_hops, write_jsonl

    # Smaller quotas for speed but still meet prereg minima shape.
    pos_per = 40
    n_neg = pos_per * len(OOD_HOPS)
    examples, report = generate_ood_hops(
        seed=OOD_HOPS_SEED,
        pos_per_hop=pos_per,
        n_neg=n_neg,
        max_graph_draws=40_000,
    )
    ok, issues = verify_ood_hops(examples, pos_per_hop=pos_per, n_neg=n_neg)
    assert ok, issues
    out = tmp_path / "ood_hops.jsonl"
    write_jsonl(out, examples)
    assert report["science_open"] is False

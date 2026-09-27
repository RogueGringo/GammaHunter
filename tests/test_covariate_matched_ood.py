# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for covariate-matched OOD generator + Gate2-matched eval hygiene.

Covers seq_len band [45,70], hop set {8,12,16}, 50/50 balance, hard-neg,
and artifact science_open=false when present. Full regen marked ``slow``.
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
from reachability_gen.gen_covariate_matched_ood import (
    EXTRA_EDGES_BY_HOP,
    MATCHED_OOD_SEED,
    N_NEG_DEFAULT,
    N_SUPPORT_BY_HOP,
    OOD_HOPS,
    POS_PER_HOP_DEFAULT,
    SEQ_LEN_MAX,
    SEQ_LEN_MIN,
    build_generation_report,
    plant_path_graph,
    verify_covariate_matched_ood,
)
from reachability_gen.hard_negatives import is_hard_negative
from reachability_gen.schema import ReachabilityExample
from reachability_gen.tokenize import split_encoding_tokens

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "covariate_matched_ood.jsonl"
REPORT_PATH = ROOT / "artifacts" / "covariate_matched_ood_generation_report.json"
ARTIFACT_PATH = (
    ROOT / "artifacts" / "id_2k_rematch_bound30_gate2_matched_ood.json"
)


def test_matched_ood_constants():
    assert OOD_HOPS == tuple(OOD_HOP_VALUES) == (8, 12, 16)
    assert POS_PER_HOP_DEFAULT >= 40
    assert N_NEG_DEFAULT == POS_PER_HOP_DEFAULT * len(OOD_HOPS)
    assert MATCHED_OOD_SEED == 126_000
    assert SEQ_LEN_MIN == 45 and SEQ_LEN_MAX == 70
    for k in OOD_HOPS:
        assert is_ood_hop(k, k_train_max=K_TRAIN_MAX)
        assert k in N_SUPPORT_BY_HOP
        assert k in EXTRA_EDGES_BY_HOP
        # Band requires |E| ∈ [13,21]; path has k edges.
        assert min(EXTRA_EDGES_BY_HOP[k]) >= 0
        assert k + max(EXTRA_EDGES_BY_HOP[k]) <= 21 or k == 8
    assert not is_ood_hop(HOP_UNREACHABLE)


def test_plant_path_graph_respects_hop_and_band():
    import random

    rng = random.Random(0)
    for k in OOD_HOPS:
        n = max(N_SUPPORT_BY_HOP[k][0], k + 1)
        extra = min(EXTRA_EDGES_BY_HOP[k][-1], max(0, 21 - k))
        edges, s, t = plant_path_graph(n, k, extra, rng)
        from reachability_gen.graph import hop_distance

        assert hop_distance(n, edges, s, t) == k
        tl = len(split_encoding_tokens(encode_instance(n, edges, s, t)))
        assert SEQ_LEN_MIN <= tl <= SEQ_LEN_MAX, (k, n, len(edges), tl)


def test_verify_rejects_wrong_counts():
    rows = [
        ReachabilityExample(
            split="covariate_matched_ood",
            seed=i,
            n=20,
            p=0.05,
            edge_hash=f"h{i}",
            s=0,
            t=8,
            y=1,
            hop_distance=8,
            is_ood=True,
            encoding=encode_instance(
                20, [(j, j + 1) for j in range(8)], 0, 8
            ),
        ).to_dict()
        for i in range(5)
    ]
    ok, issues = verify_covariate_matched_ood(
        rows, pos_per_hop=80, n_neg=240, check_hard_neg=False, check_band=False
    )
    assert not ok
    assert any("n_total" in x for x in issues)


def _synth_pos(hop: int, idx: int) -> dict:
    """Synthetic positive whose encoding lands in the seq_len band."""
    import random

    from reachability_gen.graph import hop_distance

    rng = random.Random(10_000 + hop * 100 + idx)
    n = hop + 3
    extra = min(max(0, 17 - hop), max(0, 21 - hop))
    edges, s, t = plant_path_graph(n, hop, extra, rng)
    assert hop_distance(n, edges, s, t) == hop
    enc = encode_instance(n, edges, s, t)
    tl = len(split_encoding_tokens(enc))
    assert SEQ_LEN_MIN <= tl <= SEQ_LEN_MAX
    return ReachabilityExample(
        split="covariate_matched_ood",
        seed=idx,
        n=n,
        p=len(edges) / (n * (n - 1)),
        edge_hash=f"pos{idx}",
        s=s,
        t=t,
        y=1,
        hop_distance=hop,
        is_ood=True,
        encoding=enc,
    ).to_dict()


def _synth_neg(idx: int) -> dict:
    """Hard-neg with encoding in band (path + dangling component)."""
    import random

    rng = random.Random(50_000 + idx)
    # Path 0..8 plus a separate edge among high nodes → unreachable hard pair.
    n = 14
    edges, _, _ = plant_path_graph(n, 8, 5, rng, s=0)
    # Ensure a hard-neg pair exists; fall back to known construction.
    from reachability_gen.gen_covariate_matched_ood import (
        _harvest_hard_negs_in_band,
    )

    negs = _harvest_hard_negs_in_band(n, edges)
    if not negs:
        # Force: path on 0..8, isolated edge (10,11); query 0→11.
        edges = [(i, i + 1) for i in range(8)] + [(10, 11), (11, 12), (9, 10)]
        # Pad |E| into band: need 13..21 edges; have 11 → add more among {9..13}
        extras = [(12, 13), (13, 9)]
        edges = sorted(set(edges) | set(extras))
        n = 14
        s, t = 0, 11
        ok, _ = is_hard_negative(n, edges, s, t, y=0)
        assert ok
        enc = encode_instance(n, edges, s, t)
        tl = len(split_encoding_tokens(enc))
        # If still out of band, add more non-connecting edges among {9..13}
        side = [(9, 13), (13, 12), (12, 9), (10, 13), (11, 9)]
        for e in side:
            if SEQ_LEN_MIN <= tl <= SEQ_LEN_MAX:
                break
            edges = sorted(set(edges) | {e})
            enc = encode_instance(n, edges, s, t)
            tl = len(split_encoding_tokens(enc))
        assert SEQ_LEN_MIN <= tl <= SEQ_LEN_MAX, tl
        return ReachabilityExample(
            split="covariate_matched_ood",
            seed=idx,
            n=n,
            p=len(edges) / (n * (n - 1)),
            edge_hash=f"neg{idx}",
            s=s,
            t=t,
            y=0,
            hop_distance=HOP_UNREACHABLE,
            is_ood=False,
            encoding=enc,
        ).to_dict()
    s, t = negs[idx % len(negs)]
    enc = encode_instance(n, edges, s, t)
    return ReachabilityExample(
        split="covariate_matched_ood",
        seed=idx,
        n=n,
        p=len(edges) / (n * (n - 1)),
        edge_hash=f"neg{idx}",
        s=s,
        t=t,
        y=0,
        hop_distance=HOP_UNREACHABLE,
        is_ood=False,
        encoding=enc,
    ).to_dict()


def test_verify_accepts_balanced_synthetic_in_band():
    pos_per = 2
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

    ok, issues = verify_covariate_matched_ood(
        rows, pos_per_hop=pos_per, n_neg=n_neg, check_hard_neg=True, check_band=True
    )
    assert ok, issues
    report = build_generation_report(
        rows, seed=MATCHED_OOD_SEED, pos_per_hop=pos_per, n_neg=n_neg
    )
    assert report["science_open"] is False
    assert report["token_len"]["out_of_band"] == 0
    assert report["token_len"]["min"] >= SEQ_LEN_MIN
    assert report["token_len"]["max"] <= SEQ_LEN_MAX


def test_verify_rejects_out_of_band():
    pos_per = 2
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
    # Blow up first encoding with many fake edges via raw string override.
    # Use a long path encoding outside band.
    long_edges = [(i, i + 1) for i in range(30)]
    rows[0]["encoding"] = encode_instance(32, long_edges, 0, 8)
    rows[0]["n"] = 32
    ok, issues = verify_covariate_matched_ood(
        rows,
        pos_per_hop=pos_per,
        n_neg=n_neg,
        check_hard_neg=False,
        check_band=True,
    )
    assert not ok
    assert any("seq_len" in x for x in issues)


def test_verify_rejects_non_ood_hop():
    pos_per = 2
    n_neg = pos_per * len(OOD_HOPS)
    rows: list[dict] = []
    idx = 0
    for k in OOD_HOPS:
        for _ in range(pos_per):
            rows.append(_synth_pos(k, idx))
            idx += 1
    rows[0]["hop_distance"] = 4
    rows[0]["is_ood"] = False
    for _ in range(n_neg):
        rows.append(_synth_neg(idx))
        idx += 1
    ok, issues = verify_covariate_matched_ood(
        rows, pos_per_hop=pos_per, n_neg=n_neg, check_hard_neg=False, check_band=False
    )
    assert not ok


def test_matched_jsonl_constraints_if_present():
    if not DATA_PATH.exists():
        pytest.skip("data/covariate_matched_ood.jsonl not generated yet")
    ok, issues = verify_covariate_matched_ood(DATA_PATH, check_hard_neg=True)
    assert ok, issues
    n_lines = sum(1 for _ in DATA_PATH.open())
    assert n_lines == POS_PER_HOP_DEFAULT * len(OOD_HOPS) + N_NEG_DEFAULT
    # Spot-check hop set + band on every row.
    with DATA_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            hop = int(r["hop_distance"])
            y = int(r["y"])
            tl = len(split_encoding_tokens(r["encoding"]))
            assert SEQ_LEN_MIN <= tl <= SEQ_LEN_MAX
            if y == 1:
                assert hop in OOD_HOPS
            else:
                assert hop == HOP_UNREACHABLE


def test_matched_generation_report_if_present():
    if not REPORT_PATH.exists():
        pytest.skip("covariate_matched_ood_generation_report.json missing")
    data = json.loads(REPORT_PATH.read_text())
    assert data.get("science_open") is False
    assert data.get("verify_ok") is True
    for k in OOD_HOPS:
        assert data["hop_counts"][str(k)] == POS_PER_HOP_DEFAULT
    assert data["hop_counts"][str(HOP_UNREACHABLE)] == N_NEG_DEFAULT
    tl = data["token_len"]
    assert tl["min"] >= SEQ_LEN_MIN
    assert tl["max"] <= SEQ_LEN_MAX
    assert tl["out_of_band"] == 0
    assert "construction" in data


def test_matched_artifact_science_open_false_if_present():
    if not ARTIFACT_PATH.exists():
        pytest.skip("matched gate2 artifact not written yet")
    data = json.loads(ARTIFACT_PATH.read_text())
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
    assert data["dataset"]["token_len"]["out_of_band"] == 0
    for arm in ("ff", "geo", "loop"):
        assert arm in data["fixed_depth"]
        by_hop = data["fixed_depth"][arm]["by_hop"]
        for k in ["8", "12", "16", "-1"]:
            assert k in by_hop
    for T in ("8", "12", "16"):
        assert T in data["dynamic_unroll"]["geo"]
        assert T in data["dynamic_unroll"]["loop"]


@pytest.mark.slow
def test_generate_matched_ood_end_to_end(tmp_path: Path):
    from reachability_gen.gen_covariate_matched_ood import (
        generate_covariate_matched_ood,
        write_jsonl,
    )

    pos_per = 40
    n_neg = pos_per * len(OOD_HOPS)
    examples, report = generate_covariate_matched_ood(
        seed=MATCHED_OOD_SEED,
        pos_per_hop=pos_per,
        n_neg=n_neg,
        max_graph_draws=40_000,
        run_er_search=False,
    )
    ok, issues = verify_covariate_matched_ood(
        examples, pos_per_hop=pos_per, n_neg=n_neg
    )
    assert ok, issues
    out = tmp_path / "covariate_matched_ood.jsonl"
    write_jsonl(out, examples)
    assert report["science_open"] is False
    assert report["token_len"]["out_of_band"] == 0

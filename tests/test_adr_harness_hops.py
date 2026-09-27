# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for ADR-001 invariants, hop labeling, and harness logging schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reachability_gen.adr_invariants import (
    EPS_SIGMA,
    F_BLOCK_COEFF_M2D,
    F_BLOCK_COEFF_MD2,
    HOP_UNREACHABLE,
    ID_HOP_MAX,
    ID_HOP_MIN,
    K_TRAIN_MAX,
    LEGACY_METRIC_KEYS,
    OOD_HOP_VALUES,
    PARAM_TOL,
    REQUIRED_METRIC_KEYS,
    SCIENCE_OPEN_DEFAULT,
    assert_f_block_matches_implementation,
    f_block_checksum,
    is_ood_hop,
    validate_metric_record,
)
from reachability_gen.arms import (
    EuclideanLoopArm,
    FeedForwardArm,
    GeometricRecurrentArm,
    PARAM_MATCH_TOLERANCE,
)
from reachability_gen.flops import F_block
from reachability_gen.generate import generate_split
from reachability_gen.graph import hop_distance, reachable_bfs
from reachability_gen.harness.runner import (
    ExperimentHarness,
    default_stub_arms,
    eval_jsonl,
    measure_phase_diagnostics,
)
from reachability_gen.schema import RUN_METRIC_REQUIRED_KEYS, RunMetricRecord


# --- ADR constants -----------------------------------------------------------


def test_adr_constants_locked():
    assert PARAM_TOL == pytest.approx(0.05)
    assert PARAM_MATCH_TOLERANCE == PARAM_TOL
    assert K_TRAIN_MAX == 6
    assert ID_HOP_MIN == 2 and ID_HOP_MAX == 6
    assert OOD_HOP_VALUES == (8, 12, 16)
    assert EPS_SIGMA == pytest.approx(1e-4)
    assert HOP_UNREACHABLE == -1
    assert SCIENCE_OPEN_DEFAULT is False
    assert F_BLOCK_COEFF_MD2 == 24
    assert F_BLOCK_COEFF_M2D == 4


def test_required_metric_keys_match_user_typeddict():
    expected = frozenset(
        {
            "run_id",
            "seed",
            "arm",
            "step",
            "epoch",
            "param_count",
            "d_model",
            "seq_len",
            "cycles_or_depth",
            "tokens_decoded",
            "cumulative_flops",
            "hop_distance",
            "is_ood",
            "loss",
            "accuracy",
            "drift_trajectory",
            "terminal_drift",
            "perturbation_delta",
        }
    )
    assert REQUIRED_METRIC_KEYS == expected
    assert RUN_METRIC_REQUIRED_KEYS == REQUIRED_METRIC_KEYS
    # TypedDict annotations include all required keys.
    assert expected <= set(RunMetricRecord.__annotations__)


def test_f_block_checksum_matches_flops():
    assert_f_block_matches_implementation(F_block)
    m, d = 16, 32
    assert F_block(m, d) == f_block_checksum(m, d)
    assert F_block(m, d) == 24 * m * d * d + 4 * m * m * d


def test_is_ood_hop_rule():
    assert is_ood_hop(HOP_UNREACHABLE) is False
    assert is_ood_hop(0) is False
    assert is_ood_hop(6) is False
    assert is_ood_hop(7) is True
    assert is_ood_hop(8) is True
    assert is_ood_hop(16) is True


# --- hop labeling ------------------------------------------------------------


def test_hop_distance_chain():
    edges = [(0, 1), (1, 2), (2, 3)]
    n = 4
    assert hop_distance(n, edges, 0, 3) == 3
    assert hop_distance(n, edges, 0, 0) == 0
    assert hop_distance(n, edges, 3, 0) is None


def test_generate_writes_hop_fields():
    examples, _ = generate_split("train", n_per_cell=2, max_rejects=3000)
    assert examples
    for ex in examples:
        assert hasattr(ex, "hop_distance")
        assert hasattr(ex, "is_ood")
        d = ex.to_dict()
        assert "hop_distance" in d and "is_ood" in d
        if ex.y == 0:
            assert ex.hop_distance == HOP_UNREACHABLE
            assert ex.is_ood is False
        else:
            assert ex.hop_distance >= 0
            assert ex.is_ood == (ex.hop_distance > K_TRAIN_MAX)
            from reachability_gen.encode import edge_hash as eh

            parts = ex.encoding.split()
            q = parts.index("QUERY")
            edge_toks = parts[3:q]
            edges = [tuple(map(int, tok.split(","))) for tok in edge_toks]
            assert reachable_bfs(ex.n, edges, ex.s, ex.t)
            assert hop_distance(ex.n, edges, ex.s, ex.t) == ex.hop_distance
            assert eh(edges) == ex.edge_hash


def test_jsonl_includes_hop_keys(tmp_path: Path):
    from reachability_gen.generate import write_jsonl

    examples, _ = generate_split("val", n_per_cell=2, max_rejects=3000)
    out = tmp_path / "val.jsonl"
    write_jsonl(out, examples)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    for row in rows:
        assert "hop_distance" in row
        assert "is_ood" in row


# --- harness schema ----------------------------------------------------------


def _minimal_valid_record(**overrides):
    rec = {
        "run_id": "test-run",
        "seed": 0,
        "arm": "ff",
        "step": 0,
        "epoch": 0,
        "param_count": 1,
        "d_model": 32,
        "seq_len": 8,
        "cycles_or_depth": 1,
        "tokens_decoded": None,
        "cumulative_flops": 1.0,
        "hop_distance": 2,
        "is_ood": False,
        "loss": 0.0,
        "accuracy": 0.0,
        "drift_trajectory": [],
        "terminal_drift": None,
        "perturbation_delta": None,
    }
    rec.update(overrides)
    return rec


def test_measure_phase_diagnostics_stub():
    diag = measure_phase_diagnostics([{"t": 0}, {"t": 1}, {"t": 2}])
    assert diag["drift_trajectory"] == [0.0, 0.0]
    assert diag["terminal_drift"] == pytest.approx(0.0)
    assert diag["perturbation_delta"] == pytest.approx(EPS_SIGMA)
    assert diag["science_open"] is False
    assert "z_drift_series" not in diag


def test_harness_logs_required_keys(tmp_path: Path):
    examples, _ = generate_split("train", n_per_cell=2, max_rejects=2000)
    ex = examples[0].to_dict()
    arms = [
        GeometricRecurrentArm(T=4, d=32, use_tau=False),
        FeedForwardArm(L=1, d=32),
        EuclideanLoopArm(T=4, d=32),
    ]
    metrics = tmp_path / "metrics.jsonl"
    with ExperimentHarness(
        arms, metrics_path=metrics, assert_parity=True, run_id="unit-test"
    ) as h:
        rec = h.eval_example(ex, arms[0])
    assert set(REQUIRED_METRIC_KEYS) <= set(rec.keys())
    validate_metric_record(rec)
    assert rec["run_id"] == "unit-test"
    assert rec["param_count"] == arms[0].param_count()
    assert rec["d_model"] == 32
    assert rec["cycles_or_depth"] == 4
    assert rec["tokens_decoded"] is None
    assert isinstance(rec["cumulative_flops"], float)
    assert "drift_trajectory" in rec
    assert LEGACY_METRIC_KEYS.isdisjoint(rec.keys())
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert len(rows) == 1
    assert REQUIRED_METRIC_KEYS <= set(rows[0].keys())
    assert LEGACY_METRIC_KEYS.isdisjoint(rows[0].keys())


def test_harness_rejects_science_open_true(tmp_path: Path):
    arms = default_stub_arms(d=32)[:3]  # geo/ff/loop only for parity
    metrics = tmp_path / "m.jsonl"
    with ExperimentHarness(arms, metrics_path=metrics) as h:
        bad = _minimal_valid_record(science_open=True)
        with pytest.raises(ValueError, match="science_open"):
            h.log_eval_step(bad)


def test_harness_rejects_legacy_keys(tmp_path: Path):
    arms = default_stub_arms(d=32)[:3]
    metrics = tmp_path / "m.jsonl"
    with ExperimentHarness(arms, metrics_path=metrics) as h:
        bad = _minimal_valid_record(z_drift_series=[], T_or_L_or_K=1, params=1, flops=1)
        with pytest.raises(ValueError, match="legacy"):
            h.log_eval_step(bad)


def test_validate_metric_record_missing_key():
    rec = _minimal_valid_record()
    del rec["run_id"]
    with pytest.raises(ValueError, match="missing keys"):
        validate_metric_record(rec)


def test_eval_jsonl_demo_path(tmp_path: Path):
    examples, _ = generate_split("train", n_per_cell=2, max_rejects=2000)
    from reachability_gen.generate import write_jsonl

    ex_path = tmp_path / "train.jsonl"
    write_jsonl(ex_path, examples)
    metrics = tmp_path / "metrics.jsonl"
    arms = default_stub_arms(d=32)
    n = eval_jsonl(
        ex_path, metrics, arms, limit=2, assert_parity=True, run_id="demo-run"
    )
    # 2 examples × 4 arms
    assert n == 8
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert len(rows) == 8
    for row in rows:
        validate_metric_record(row)
        assert REQUIRED_METRIC_KEYS <= set(row.keys())
        assert LEGACY_METRIC_KEYS.isdisjoint(row.keys())
        assert row["run_id"] == "demo-run"
        assert "hop_distance" in row
        # CoT has tokens_decoded; others None
        if row["arm"].startswith("cot"):
            assert row["tokens_decoded"] == 4
            assert row["cycles_or_depth"] == 1  # L
        else:
            assert row["tokens_decoded"] is None


def test_every_emitted_dict_passes_validate_metric_record(tmp_path: Path):
    """Success criterion: every harness-emitted dict passes validate_metric_record."""
    examples, _ = generate_split("train", n_per_cell=2, max_rejects=2000)
    from reachability_gen.generate import write_jsonl

    ex_path = tmp_path / "train.jsonl"
    write_jsonl(ex_path, examples)
    metrics = tmp_path / "metrics.jsonl"
    arms = default_stub_arms(d=32)
    eval_jsonl(ex_path, metrics, arms, limit=3, assert_parity=True, run_id="val-all")
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    assert rows
    for row in rows:
        validate_metric_record(row)
        assert not (LEGACY_METRIC_KEYS & set(row.keys()))


def test_forward_return_trajectory():
    arm = GeometricRecurrentArm(T=3, d=16, use_tau=False)
    logits, traj = arm.forward({"batch_size": 2}, return_trajectory=True)
    assert logits == [0.0, 0.0]
    assert len(traj) == 3
    stub = arm.forward({"batch_size": 1}, return_trajectory=False)
    assert isinstance(stub, dict)
    assert stub["placeholder"] is True
    assert "drift_trajectory" in stub
    assert "z_drift_series" not in stub

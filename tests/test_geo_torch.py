# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Torch-required tests for GeometricRecurrent model / trainer / overfit plumbing.

Skipped entirely when torch is not installed so pure schema CI stays green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.arms import GeometricRecurrentArm  # noqa: E402
from reachability_gen.models.geometric import (  # noqa: E402
    GeometricRecurrent,
    drift_from_trajectory,
    trajectory_finite_nonzero,
)
from reachability_gen.overfit_ff import ensure_balanced_batch  # noqa: E402
from reachability_gen.overfit_geo import run_overfit_geo  # noqa: E402
from reachability_gen.tokenize import build_vocab  # noqa: E402
from reachability_gen.train.ff_trainer import examples_to_batch  # noqa: E402
from reachability_gen.train.geo_trainer import GeometricTrainer  # noqa: E402


def test_geometric_forward_shape_and_trajectory():
    vocab = build_vocab()
    T = 6
    model = GeometricRecurrent(
        vocab_size=len(vocab), d=32, T=T, n_heads=4, max_len=64, use_tau=True
    )
    ids = torch.tensor([[1, 2, 3, 4, 0, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
    logits, traj = model(ids, mask, return_trajectory=False)
    assert tuple(logits.shape) == (1, 2)
    assert traj is None
    logits2, traj2 = model(ids, mask, return_trajectory=True)
    assert tuple(logits2.shape) == (1, 2)
    assert traj2 is not None
    # z_0 .. z_T → length T+1
    assert len(traj2) == T + 1
    for z in traj2:
        assert tuple(z.shape) == (1, 32)
    drifts = drift_from_trajectory(traj2)
    assert len(drifts) == T
    ok, reason = trajectory_finite_nonzero(drifts)
    assert ok, reason


def test_geo_trainer_train_step_and_drift():
    vocab = build_vocab()
    model = GeometricRecurrent(
        vocab_size=len(vocab), d=32, T=4, n_heads=4, max_len=64
    )
    trainer = GeometricTrainer(model, lr=1e-2, grad_clip=1.0)
    examples = [
        {"encoding": "N 3 EDGES 0,1 QUERY 0 1", "y": 1},
        {"encoding": "N 3 EDGES QUERY 1 0", "y": 0},
    ]
    token_ids, mask, labels, _ = examples_to_batch(examples, vocab)
    loss, acc = trainer.train_step(token_ids, labels, mask)
    assert isinstance(loss, float) and loss >= 0.0
    assert 0.0 <= acc <= 1.0
    telem = trainer.drift_telemetry(token_ids, mask, eps_sigma=1e-4)
    assert len(telem["drift_trajectory"]) == 4  # T drifts from T+1 states
    ok, reason = trajectory_finite_nonzero(telem["drift_trajectory"])
    assert ok, reason
    assert telem["terminal_drift"] is not None
    assert telem["perturbation_delta"] is not None


def test_geo_arm_real_forward_trajectory():
    arm = GeometricRecurrentArm(T=6, d=32, use_tau=True)
    arm.attach_default_model(seed=0)
    assert arm.has_real_model
    logits, traj = arm.forward(
        {"encoding": ["N 4 EDGES 0,1 1,2 QUERY 0 2"], "y": [1]},
        return_trajectory=True,
    )
    assert isinstance(logits, list) and len(logits) == 1
    assert len(logits[0]) == 2
    assert len(traj) == 7  # T+1
    drifts = drift_from_trajectory(traj)
    assert len(drifts) == 6
    ok, reason = trajectory_finite_nonzero(drifts)
    assert ok, reason


def test_trajectory_finite_nonzero_rejects_bad():
    ok, reason = trajectory_finite_nonzero([])
    assert not ok
    ok, reason = trajectory_finite_nonzero([0.0, 0.0, 0.0])
    assert not ok
    assert "zero" in reason
    ok, reason = trajectory_finite_nonzero([float("nan"), 0.1])
    assert not ok
    ok, reason = trajectory_finite_nonzero([0.1, 0.2, 0.0])
    assert ok


@pytest.mark.slow
def test_balanced_geo_overfit_gate(tmp_path: Path):
    """Balanced 16+16 hard-neg geo overfit; CE < 1e-3, per-class 1.0, drift ok."""
    examples_path = tmp_path / "train.jsonl"
    batch, note, meta = ensure_balanced_batch(
        examples_path, n_pos=8, n_neg=8, regenerate=True
    )
    assert len(batch) == 16, note
    assert sum(1 for e in batch if int(e["y"]) == 1) == 8
    assert sum(1 for e in batch if int(e["y"]) == 0) == 8
    assert meta.get("reject_reasons") is not None
    result = run_overfit_geo(
        batch,
        steps=100,
        d=64,
        T=6,
        lr=5e-3,
        seed=0,
        loss_threshold=1e-3,
        require_per_class=True,
        use_tau=True,
    )
    assert result["ok"], (
        f"balanced geo overfit failed: loss={result['final_loss']:.6f} "
        f"acc={result['final_acc']:.4f} per_class={result.get('per_class')} "
        f"drift={result.get('drift_trajectory')} reason={result.get('drift_reason')} "
        f"note={note}"
    )
    assert result["final_loss"] < 1e-3
    assert result["per_class"]["y0"] == pytest.approx(1.0)
    assert result["per_class"]["y1"] == pytest.approx(1.0)
    drifts = result["drift_trajectory"]
    assert 5 <= len(drifts) <= 7  # T-1 .. T(+1 slack)
    ok, reason = trajectory_finite_nonzero(drifts)
    assert ok, reason

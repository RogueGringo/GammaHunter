# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Torch-required tests for EuclideanLoop model / trainer / arm plumbing.

Skipped entirely when torch is not installed so pure schema CI stays green.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.arms import EuclideanLoopArm  # noqa: E402
from reachability_gen.models.euclidean_loop import EuclideanLoop  # noqa: E402
from reachability_gen.models.geometric import (  # noqa: E402
    drift_from_trajectory,
    trajectory_finite_nonzero,
)
from reachability_gen.tokenize import build_vocab  # noqa: E402
from reachability_gen.train.ff_trainer import examples_to_batch  # noqa: E402
from reachability_gen.train.loop_trainer import LoopTrainer  # noqa: E402


def test_euclidean_loop_no_tau_and_trajectory():
    vocab = build_vocab()
    T = 6
    model = EuclideanLoop(
        vocab_size=len(vocab), d=32, T=T, n_heads=4, max_len=64
    )
    assert model.use_tau is False
    assert model.tau_emb is None
    ids = torch.tensor([[1, 2, 3, 4, 0, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
    logits, traj = model(ids, mask, return_trajectory=False)
    assert tuple(logits.shape) == (1, 2)
    assert traj is None
    logits2, traj2 = model(ids, mask, return_trajectory=True)
    assert len(traj2) == T + 1
    drifts = drift_from_trajectory(traj2)
    assert len(drifts) == T
    ok, reason = trajectory_finite_nonzero(drifts)
    assert ok, reason


def test_loop_trainer_train_step_and_drift():
    vocab = build_vocab()
    model = EuclideanLoop(
        vocab_size=len(vocab), d=32, T=4, n_heads=4, max_len=64
    )
    trainer = LoopTrainer(model, lr=1e-2, grad_clip=1.0)
    examples = [
        {"encoding": "N 3 EDGES 0,1 QUERY 0 1", "y": 1},
        {"encoding": "N 3 EDGES QUERY 1 0", "y": 0},
    ]
    token_ids, mask, labels, _ = examples_to_batch(examples, vocab)
    loss, acc = trainer.train_step(token_ids, labels, mask)
    assert isinstance(loss, float) and loss >= 0.0
    assert 0.0 <= acc <= 1.0
    telem = trainer.drift_telemetry(token_ids, mask, eps_sigma=1e-4)
    assert len(telem["drift_trajectory"]) == 4
    ok, reason = trajectory_finite_nonzero(telem["drift_trajectory"])
    assert ok, reason
    assert telem["terminal_drift"] is not None
    assert telem["perturbation_delta"] is not None


def test_loop_arm_real_forward_trajectory():
    arm = EuclideanLoopArm(T=6, d=32)
    arm.attach_default_model(seed=0)
    assert arm.has_real_model
    logits, traj = arm.forward(
        {"encoding": ["N 4 EDGES 0,1 1,2 QUERY 0 2"], "y": [1]},
        return_trajectory=True,
    )
    assert isinstance(logits, list) and len(logits) == 1
    assert len(logits[0]) == 2
    assert len(traj) == 7
    drifts = drift_from_trajectory(traj)
    ok, reason = trajectory_finite_nonzero(drifts)
    assert ok, reason

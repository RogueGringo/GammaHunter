# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Fast tests for bound30 rematch: RMSNorm state bound / clip-LR / artifact hygiene."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from reachability_gen.run_id_2k_rematch import (
    FF_BASELINE_PARAMS,
    param_window,
    within_5pct,
)

ROOT = Path(__file__).resolve().parents[1]
BOUND30_PATH = ROOT / "artifacts" / "id_2k_rematch_bound30.json"
ID_2K_PATH = ROOT / "data" / "id_2k.jsonl"


def test_rmsnorm_state_bound_keeps_norm_near_sqrt_d():
    torch = pytest.importorskip("torch")
    from reachability_gen.models.geometric import (
        DEFAULT_RESIDUAL_ALPHA,
        GeometricRecurrent,
        drift_from_trajectory,
        mean_z_norms_from_trajectory,
    )
    from reachability_gen.tokenize import build_vocab

    assert DEFAULT_RESIDUAL_ALPHA == 0.5
    vocab = build_vocab()
    model = GeometricRecurrent(
        vocab_size=len(vocab),
        d=64,
        T=6,
        n_heads=4,
        max_len=64,
        use_tau=True,
        apply_cycle_rmsnorm=True,
    )
    assert model.residual_alpha == 0.5
    assert model.apply_cycle_ln is False
    assert model.apply_cycle_rmsnorm is True
    assert model.cycle_ln is None
    assert model.cycle_rmsnorm is not None
    ids = torch.tensor([[1, 2, 3, 4, 0, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
    _, traj = model(ids, mask, return_trajectory=True)
    assert traj is not None and len(traj) == 7
    norms = mean_z_norms_from_trajectory(traj)
    # RMSNorm → ||z|| ~ O(√d)=O(8); pooled may be slightly lower.
    for n in norms:
        assert 0.5 < n < 20.0, f"unexpected ||z||={n} (expected O(√64)=O(8))"
    drifts = drift_from_trajectory(traj)
    drifts_ln = drift_from_trajectory(traj, apply_ln=True)
    drifts_rms = drift_from_trajectory(traj, apply_rmsnorm=True)
    assert len(drifts) == 6 and len(drifts_ln) == 6 and len(drifts_rms) == 6
    assert drifts_rms[-1] < 20.0


def test_rmsnorm_and_ln_mutually_exclusive():
    pytest.importorskip("torch")
    from reachability_gen.models.geometric import GeometricRecurrent
    from reachability_gen.tokenize import build_vocab

    vocab = build_vocab()
    with pytest.raises(ValueError, match="mutually exclusive"):
        GeometricRecurrent(
            vocab_size=len(vocab),
            d=32,
            T=2,
            n_heads=4,
            max_len=64,
            apply_cycle_ln=True,
            apply_cycle_rmsnorm=True,
        )


def test_mlp10_with_rmsnorm_within_5pct():
    torch = pytest.importorskip("torch")
    del torch
    if not ID_2K_PATH.exists():
        pytest.skip("id_2k.jsonl missing")
    from reachability_gen.models.euclidean_loop import EuclideanLoop
    from reachability_gen.models.feedforward import FeedForward
    from reachability_gen.models.geometric import GeometricRecurrent
    from reachability_gen.overfit_ff import load_jsonl
    from reachability_gen.tokenize import build_vocab
    from reachability_gen.train.ff_trainer import examples_to_batch

    rows = load_jsonl(ID_2K_PATH)
    vocab = build_vocab()
    probe = examples_to_batch(rows[:2], vocab)
    max_len = max(int(probe[0].shape[1]) + 8, 64)
    _, _, _, vocab = examples_to_batch(rows, vocab, max_len=max_len)
    ff = FeedForward(
        len(vocab), d=64, L=2, n_heads=4, max_len=max_len, pad_id=vocab.pad_id
    )
    geo = GeometricRecurrent(
        len(vocab),
        d=64,
        T=6,
        n_heads=4,
        max_len=max_len,
        pad_id=vocab.pad_id,
        use_tau=True,
        mlp_expansion=10,
        apply_cycle_rmsnorm=True,
    )
    loop = EuclideanLoop(
        len(vocab),
        d=64,
        T=6,
        n_heads=4,
        max_len=max_len,
        pad_id=vocab.pad_id,
        mlp_expansion=10,
        apply_cycle_rmsnorm=True,
    )
    ff_n = sum(p.numel() for p in ff.parameters() if p.requires_grad)
    geo_n = sum(p.numel() for p in geo.parameters() if p.requires_grad)
    loop_n = sum(p.numel() for p in loop.parameters() if p.requires_grad)
    assert ff_n == FF_BASELINE_PARAMS
    lo, hi = param_window(ff_n)
    assert lo <= geo_n <= hi and within_5pct(geo_n, ff_n)
    assert lo <= loop_n <= hi and within_5pct(loop_n, ff_n)


def test_bound30_phase_a_defaults():
    from reachability_gen.run_id_2k_rematch_bound30 import (
        DEFAULT_FF_LR,
        DEFAULT_REC_LR,
        FF_GRAD_CLIP,
        REC_GRAD_CLIP,
    )

    assert DEFAULT_FF_LR == 3e-3
    assert DEFAULT_REC_LR == 1.5e-3
    assert FF_GRAD_CLIP == 1.0
    assert REC_GRAD_CLIP == 2.5


def test_bound30_artifact_if_present():
    if not BOUND30_PATH.exists():
        pytest.skip("artifacts/id_2k_rematch_bound30.json not written yet")
    data = json.loads(BOUND30_PATH.read_text())
    assert data.get("science_open") is False
    assert data.get("epochs") == 30
    assert data.get("epochs_locked") is True
    assert data.get("early_stopping") is False

    def _walk(obj):
        if isinstance(obj, dict):
            if obj.get("science_open") is True:
                return False
            return all(_walk(v) for v in obj.values())
        if isinstance(obj, list):
            return all(_walk(v) for v in obj)
        return True

    assert _walk(data)
    pm = data["param_match"]
    assert pm["within_5pct"] is True
    assert pm["science_open"] is False
    pa = data["phase_a_clip_lr"]
    assert math.isclose(pa["ff_lr"], 3e-3)
    assert math.isclose(pa["recurrent_lr"], 1.5e-3)
    assert math.isclose(pa["ff_grad_clip"], 1.0)
    assert math.isclose(pa["recurrent_grad_clip"], 2.5)
    assert pa["science_open"] is False
    assert data["drift_audit"]["apply_cycle_rmsnorm"] is True
    assert data["drift_audit"]["state_bound"] == "rmsnorm"
    assert math.isclose(data["drift_audit"]["residual_alpha"], 0.5)
    for arm in ("ff", "geo", "loop"):
        block = data[arm]
        assert block["epochs_run"] == 30
        assert block["early_stop"] is False
        assert "grad_clip_sat_rate" in block
        assert "best_val_acc" in block
        by_hop = block["val_by_hop"]
        for k in ["-1", "2", "3", "4", "5", "6"]:
            assert k in by_hop
            assert "acc_mean" in by_hop[k]
            assert "loss_mean" in by_hop[k]
    # Must still learn (not chance): best val > 0.55
    for arm in ("ff", "geo", "loop"):
        assert data[arm]["best_val_acc"] > 0.55, (
            f"{arm} best_val_acc={data[arm]['best_val_acc']} looks like chance"
        )
    for arm in ("geo", "loop"):
        assert data[arm]["apply_cycle_rmsnorm"] is True
        assert data[arm]["mean_z_norm_by_t"] is not None
        assert len(data[arm]["mean_z_norm_by_t"]) == 7  # T+1
        # RMSNorm → norms near √d=8 (allow slack for pooling / affine)
        for n in data[arm]["mean_z_norm_by_t"]:
            assert n == n and 0.5 < n < 30.0
        h2 = data[arm]["val_by_hop"]["2"]
        assert "mean_drift_trajectory" in h2
        assert "mean_drift_trajectory_ln" in h2
        assert "mean_drift_trajectory_rms" in h2
        assert "mean_terminal_drift_rms" in h2
        td_rms = h2.get("mean_terminal_drift_rms")
        assert td_rms is not None and td_rms < 30.0

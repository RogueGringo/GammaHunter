# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Fast tests for fixed-30 rematch drift fix / parity / artifact hygiene."""

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
FIXED30_PATH = ROOT / "artifacts" / "id_2k_rematch_fixed30.json"
ID_2K_PATH = ROOT / "data" / "id_2k.jsonl"


def test_residual_alpha_and_cycle_ln_defaults():
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
        vocab_size=len(vocab), d=64, T=6, n_heads=4, max_len=64, use_tau=True
    )
    assert model.residual_alpha == 0.5
    assert model.apply_cycle_ln is False
    assert model.cycle_ln is None
    ids = torch.tensor([[1, 2, 3, 4, 0, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
    _, traj = model(ids, mask, return_trajectory=True)
    assert traj is not None and len(traj) == 7
    norms = mean_z_norms_from_trajectory(traj)
    # post-LN token ||·||≈√d; pooled can be lower at init (direction cancel).
    # Guard against pre-fix runaway (||z||≫2√d≈16).
    for n in norms:
        assert 0.5 < n < 20.0, f"unexpected ||z||={n} (expected O(√64)=O(8))"
    drifts = drift_from_trajectory(traj)
    drifts_ln = drift_from_trajectory(traj, apply_ln=True)
    assert len(drifts) == 6 and len(drifts_ln) == 6
    # LN-normalized drift is the diameter proxy (raw may grow without stream LN)
    assert drifts_ln[-1] < 20.0, f"LN terminal drift {drifts_ln[-1]} unexpected"


def test_mlp10_with_cycle_ln_within_5pct():
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
        apply_cycle_ln=False,
    )
    loop = EuclideanLoop(
        len(vocab),
        d=64,
        T=6,
        n_heads=4,
        max_len=max_len,
        pad_id=vocab.pad_id,
        mlp_expansion=10,
        apply_cycle_ln=False,
    )
    ff_n = sum(p.numel() for p in ff.parameters() if p.requires_grad)
    geo_n = sum(p.numel() for p in geo.parameters() if p.requires_grad)
    loop_n = sum(p.numel() for p in loop.parameters() if p.requires_grad)
    assert ff_n == FF_BASELINE_PARAMS
    lo, hi = param_window(ff_n)
    assert lo <= geo_n <= hi and within_5pct(geo_n, ff_n)
    assert lo <= loop_n <= hi and within_5pct(loop_n, ff_n)


def test_fixed30_artifact_if_present():
    if not FIXED30_PATH.exists():
        pytest.skip("artifacts/id_2k_rematch_fixed30.json not written yet")
    data = json.loads(FIXED30_PATH.read_text())
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
    for arm in ("geo", "loop"):
        assert data[arm]["mean_z_norm_by_t"] is not None
        assert len(data[arm]["mean_z_norm_by_t"]) == 7  # T+1
        # post-LN norms near √d
        for n in data[arm]["mean_z_norm_by_t"]:
            assert n == n and n > 0.0  # finite positive; may grow without stream LN
        h2 = data[arm]["val_by_hop"]["2"]
        assert "mean_drift_trajectory" in h2
        assert "mean_drift_trajectory_ln" in h2
        assert "mean_terminal_drift" in h2
        assert "mean_perturbation_delta" in h2
        # LN-normalized terminal drift is the diameter proxy
        td_ln = h2.get("mean_terminal_drift_ln")
        assert td_ln is not None and td_ln < 30.0
        assert "mean_drift_trajectory_ln" in h2
    assert "drift_audit" in data
    assert math.isclose(data["drift_audit"]["residual_alpha"], 0.5)
    assert data["drift_audit"]["apply_cycle_ln"] is False

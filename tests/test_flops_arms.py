# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for FLOP accounting + arm interface stubs (RESEARCH scaffold)."""

from __future__ import annotations

import pytest

from reachability_gen.arms import (
    ChainOfThoughtArm,
    EuclideanLoopArm,
    FeedForwardArm,
    GeometricRecurrentArm,
    SharedArmConfig,
    params_per_block,
)
from reachability_gen.flops import (
    F_block,
    F_decode_step,
    flops_cot,
    flops_euclidean_loop,
    flops_feedforward,
    flops_geometric,
    tau_embed_flops,
)
from reachability_gen.param_match import check_geo_ff_loop, check_param_match


# --- F_block schematic -------------------------------------------------------


def test_F_block_formula_explicit():
    m, d = 16, 32
    # 24 m d^2 + 4 m^2 d  with mlp_expansion=4
    expected = 24 * m * d * d + 4 * m * m * d
    assert F_block(m, d) == expected


def test_F_block_monotonic_in_m_and_d():
    assert F_block(8, 32) < F_block(16, 32)
    assert F_block(16, 32) < F_block(16, 64)


# --- monotonic FLOPs in T / L / K_used ---------------------------------------


def test_ff_flops_monotonic_in_L():
    m, d = 32, 64
    f1 = flops_feedforward(m, d, L=1).flops
    f2 = flops_feedforward(m, d, L=2).flops
    f4 = flops_feedforward(m, d, L=4).flops
    assert f1 < f2 < f4
    assert f2 == 2 * f1
    assert f4 == 4 * f1


def test_geo_loop_flops_monotonic_in_T():
    m, d = 32, 64
    for fn in (flops_euclidean_loop,):
        a = fn(m, d, T=1).flops
        b = fn(m, d, T=2).flops
        c = fn(m, d, T=4).flops
        assert a < b < c
        assert b == 2 * a
        assert c == 4 * a

    # Geo without tau is exactly T * F_block (same as loop).
    g1 = flops_geometric(m, d, T=1, use_tau=False).flops
    g2 = flops_geometric(m, d, T=2, use_tau=False).flops
    assert g1 < g2
    assert g2 == 2 * g1

    # Geo with tau still monotonic (tau term also scales with T).
    gt1 = flops_geometric(m, d, T=1, use_tau=True).flops
    gt2 = flops_geometric(m, d, T=2, use_tau=True).flops
    assert gt1 < gt2
    assert gt2 - gt1 == F_block(m, d) + tau_embed_flops(d, 1)


def test_cot_flops_monotonic_in_K_used():
    m, d = 32, 64
    f0 = flops_cot(m, d, K_used=0, K_cap=64).flops
    f4 = flops_cot(m, d, K_used=4, K_cap=64).flops
    f8 = flops_cot(m, d, K_used=8, K_cap=64).flops
    assert f0 < f4 < f8
    # K_used=0 is pure prefill.
    assert f0 == F_block(m, d)


def test_cot_uses_realized_not_cap():
    m, d = 24, 48
    K_cap = 64
    K_used = 4
    report = flops_cot(m, d, K_used=K_used, K_cap=K_cap, L=1)
    # Same FLOPs as if cap equaled realized — cap must not inflate.
    twin = flops_cot(m, d, K_used=K_used, K_cap=K_used, L=1)
    assert report.flops == twin.flops
    # Changing only the cap leaves FLOPs unchanged.
    bigger_cap = flops_cot(m, d, K_used=K_used, K_cap=K_cap * 2, L=1)
    assert bigger_cap.flops == report.flops
    assert report.extras["K_used"] == K_used
    assert report.extras["K_cap"] == K_cap
    # Prefill + exactly K_used decode steps.
    assert report.extras["prefill_flops"] == F_block(m, d)
    assert len(report.extras["decode_steps"]) == K_used


def test_cot_arm_kwargs_K_used():
    arm = ChainOfThoughtArm(K_cap=32, d=64, L=1, K_used=0)
    r_cap_ignored = arm.inference_flops(16, K_used=5)
    r_via_attr = ChainOfThoughtArm(K_cap=32, d=64, L=1, K_used=5).inference_flops(16)
    assert r_cap_ignored.flops == r_via_attr.flops
    assert r_cap_ignored.extras["K_used"] == 5
    assert r_cap_ignored.extras["K_cap"] == 32


# --- Geo vs Loop same T cost if same block -----------------------------------


def test_geo_vs_loop_same_T_cost_without_tau():
    m, d, T = 40, 64, 8
    geo = flops_geometric(m, d, T, use_tau=False)
    loop = flops_euclidean_loop(m, d, T)
    assert geo.flops == loop.flops == T * F_block(m, d)


def test_geo_with_tau_exceeds_loop_by_tau_constant():
    m, d, T = 40, 64, 8
    geo = flops_geometric(m, d, T, use_tau=True)
    loop = flops_euclidean_loop(m, d, T)
    assert geo.flops == loop.flops + tau_embed_flops(d, T)


def test_arm_geo_loop_same_block_cost():
    cfg = SharedArmConfig(d=64, m=32)
    geo = GeometricRecurrentArm(T=4, d=64, use_tau=False, config=cfg)
    loop = EuclideanLoopArm(T=4, d=64, config=cfg)
    assert geo.inference_flops(32).flops == loop.inference_flops(32).flops


# --- param_match -------------------------------------------------------------


def test_param_match_true_for_intentionally_matched_stubs():
    """L=1 FF, Geo without tau, and Loop share one block → within ±5%."""
    d = 64
    cfg = SharedArmConfig(d=d, m=32)
    geo = GeometricRecurrentArm(T=8, d=d, use_tau=False, config=cfg)
    ff = FeedForwardArm(L=1, d=d, config=cfg)
    loop = EuclideanLoopArm(T=8, d=d, config=cfg)
    result = check_geo_ff_loop(geo, ff, loop)
    assert result.passed is True
    assert result.science_open is False
    assert geo.param_count() == ff.param_count() == loop.param_count()
    assert geo.param_count() == params_per_block(d)
    for ratio in result.ratios.values():
        assert abs(ratio - 1.0) <= 0.05


def test_param_match_fails_when_ff_deeper():
    d = 64
    cfg = SharedArmConfig(d=d)
    geo = GeometricRecurrentArm(T=4, d=d, use_tau=False, config=cfg)
    ff = FeedForwardArm(L=4, d=d, config=cfg)  # 4× params
    loop = EuclideanLoopArm(T=4, d=d, config=cfg)
    result = check_geo_ff_loop(geo, ff, loop)
    assert result.passed is False
    assert result.science_open is False


def test_param_match_never_science_open():
    d = 32
    geo = GeometricRecurrentArm(T=2, d=d, use_tau=False)
    ff = FeedForwardArm(L=1, d=d)
    loop = EuclideanLoopArm(T=2, d=d)
    r = check_param_match([geo, ff, loop])
    assert r.science_open is False
    assert r.to_dict()["science_open"] is False


# --- arm stubs / protocol ----------------------------------------------------


def test_forward_stub_placeholder_logits():
    batch = {"batch_size": 3}
    for arm in (
        FeedForwardArm(L=2, d=32),
        GeometricRecurrentArm(T=2, d=32),
        EuclideanLoopArm(T=2, d=32),
        ChainOfThoughtArm(K_cap=8, d=32, K_used=2),
    ):
        out = arm.forward_stub(batch)
        assert out["placeholder"] is True
        assert out["logits"] == [0.0, 0.0, 0.0]
        assert "arm" in out


def test_shared_config_tolerance_default():
    cfg = SharedArmConfig()
    assert cfg.param_match_tolerance == pytest.approx(0.05)
    assert cfg.d == 64


def test_F_decode_step_grows_with_ctx():
    d = 64
    assert F_decode_step(10, d) < F_decode_step(20, d)

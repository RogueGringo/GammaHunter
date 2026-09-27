# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Schematic inference FLOP accounting for reachability arms (RESEARCH scaffold).

Formulas follow the geometric-recurrence testbed compute table:

  FF:     L * F_block(m, d)
  Geo/Loop: T * F_block(m, d)  (+ optional cheap tau-embed constant)
  CoT:    F_prefill(m) + sum_{k=1}^{K_used} F_decode(m+k)

All math is pure Python (no torch / GPU). Constants are explicit so later
MEASURE runs can swap in profiler-measured costs without changing call sites.

This module does **not** claim empirical accuracy, wall-clock parity, or
hardware-accurate FLOPs — only a documented schematic for fair comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from reachability_gen.adr_invariants import (
    MAC_FLOPS as ADR_MAC_FLOPS,
    MLP_EXPANSION as ADR_MLP_EXPANSION,
    f_block_checksum,
)

# ---------------------------------------------------------------------------
# Explicit constants (document assumptions) — mirrored from adr_invariants
# ---------------------------------------------------------------------------

# MLP expansion ratio inside one transformer-style block (standard GPT-ish).
MLP_EXPANSION: int = ADR_MLP_EXPANSION

# Multiply-accumulate counted as 2 FLOPs (mul + add).
MAC_FLOPS: int = ADR_MAC_FLOPS

# Cheap tau / phase embed: one linear map R^1 -> R^d per recurrence step.
# Cost is tiny vs F_block; included only when use_tau=True for Geo.
TAU_EMBED_FLOPS_PER_STEP: int = MAC_FLOPS  # 2 * 1 * d  — scaled by d at call site


def F_block(m: int, d: int, *, mlp_expansion: int = MLP_EXPANSION) -> int:
    """Schematic FLOPs for one transformer-style block on context length ``m``.

    Assumptions (forward pass only, fp-agnostic counts):
      - Single head for schematic purposes (multi-head has same matmul volume
        when head_dim * n_heads = d).
      - Attention: Q,K,V,O projections + QK^T + attn@V.
      - MLP: two linears with hidden width ``mlp_expansion * d``.
      - No bias / LayerNorm / softmax / activation FLOPs (second-order vs matmuls).
      - Each multiply-accumulate = ``MAC_FLOPS`` (2) FLOPs.

    Derivation
    ----------
    Projections (Q,K,V,O): 4 * (2 * m * d * d) = 8 m d^2
    Scores QK^T:           2 * m * m * d       = 2 m^2 d
    Weighted sum attn@V:   2 * m * m * d       = 2 m^2 d
    MLP up+down (r=4):     2 * (2 * m * d * r d) = 4 r m d^2 = 16 m d^2

    Total::

        F_block(m, d) = (8 + 4 * mlp_expansion) * m * d^2  +  4 * m^2 * d
                      = 24 m d^2 + 4 m^2 d   (when mlp_expansion=4)
    """
    if m < 0 or d < 0:
        raise ValueError(f"m and d must be non-negative, got m={m}, d={d}")
    if mlp_expansion < 1:
        raise ValueError(f"mlp_expansion must be >= 1, got {mlp_expansion}")
    proj = 8 * m * d * d
    scores = 2 * m * m * d
    attn_v = 2 * m * m * d
    mlp = 4 * mlp_expansion * m * d * d
    total = int(proj + scores + attn_v + mlp)
    # ADR-001 immutable: default expansion must match frozen checksum.
    if mlp_expansion == ADR_MLP_EXPANSION:
        expected = f_block_checksum(m, d)
        if total != expected:
            raise AssertionError(
                f"F_block ADR checksum failed: got {total}, expected {expected} "
                f"(m={m}, d={d})"
            )
    return total


def F_decode_step(
    ctx: int,
    d: int,
    *,
    mlp_expansion: int = MLP_EXPANSION,
) -> int:
    """Schematic FLOPs for one autoregressive decode step with KV cache.

    At context length ``ctx`` (tokens already cached + the new token position
    index), one new token is projected and attends over ``ctx`` keys/values.

    Assumptions
    -----------
    - KV cache: K/V projections only for the new token; attention against
      full ``ctx`` cached keys/values.
    - Same MLP expansion and MAC=2 convention as :func:`F_block`.
    - No bias / norm / softmax FLOPs.

    Derivation
    ----------
    Q,K,V for 1 token:  3 * (2 * d * d) = 6 d^2
    O projection:       2 * d * d       = 2 d^2
    Scores vs ctx:      2 * 1 * ctx * d = 2 ctx d
    attn @ V:           2 * 1 * ctx * d = 2 ctx d
    MLP:                4 * mlp_expansion * d^2

    Total per layer::

        F_decode_step(ctx, d) = (8 + 4 * mlp_expansion) d^2 + 4 * ctx * d
                              = 24 d^2 + 4 ctx d   (mlp_expansion=4)
    """
    if ctx < 1:
        raise ValueError(f"ctx must be >= 1 for a decode step, got {ctx}")
    if d < 0:
        raise ValueError(f"d must be non-negative, got {d}")
    proj_qkv_o = 8 * d * d
    attn = 4 * ctx * d
    mlp = 4 * mlp_expansion * d * d
    return int(proj_qkv_o + attn + mlp)


def tau_embed_flops(d: int, T: int) -> int:
    """Cheap tau/phase embed: T steps of (scalar -> d) linear, MAC=2 => 2 T d."""
    if d < 0 or T < 0:
        raise ValueError(f"d and T must be non-negative, got d={d}, T={T}")
    return int(MAC_FLOPS * T * d)


@dataclass(frozen=True)
class FlopReport:
    """Structured FLOP report for one arm at one context configuration."""

    arm: str
    params_estimate: int
    flops: int
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "arm": self.arm,
            "params_estimate": self.params_estimate,
            "flops": self.flops,
            "notes": self.notes,
        }
        if self.extras:
            out["extras"] = dict(self.extras)
        return out


# ---------------------------------------------------------------------------
# Per-arm helpers
# ---------------------------------------------------------------------------


def flops_feedforward(
    m: int,
    d: int,
    L: int,
    *,
    params_estimate: int = 0,
) -> FlopReport:
    """FF arm: L * F_block(m, d)."""
    if L < 0:
        raise ValueError(f"L must be non-negative, got {L}")
    total = L * F_block(m, d)
    return FlopReport(
        arm="feedforward",
        params_estimate=params_estimate,
        flops=total,
        notes=f"L={L} * F_block(m={m}, d={d})",
        extras={"L": L, "m": m, "d": d, "F_block": F_block(m, d)},
    )


def flops_geometric(
    m: int,
    d: int,
    T: int,
    *,
    use_tau: bool = True,
    params_estimate: int = 0,
) -> FlopReport:
    """Geo arm: T * F_block(m, d) + optional cheap tau embed."""
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    block = F_block(m, d)
    total = T * block
    tau = 0
    if use_tau:
        tau = tau_embed_flops(d, T)
        total += tau
    return FlopReport(
        arm="geometric",
        params_estimate=params_estimate,
        flops=total,
        notes=(
            f"T={T} * F_block(m={m}, d={d})"
            + (f" + tau_embed={tau}" if use_tau else " (no tau)")
        ),
        extras={
            "T": T,
            "m": m,
            "d": d,
            "F_block": block,
            "use_tau": use_tau,
            "tau_embed_flops": tau,
        },
    )


def flops_euclidean_loop(
    m: int,
    d: int,
    T: int,
    *,
    params_estimate: int = 0,
) -> FlopReport:
    """Euclidean loop arm: T * F_block(m, d) — no phase / tau regularizers."""
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    block = F_block(m, d)
    total = T * block
    return FlopReport(
        arm="euclidean_loop",
        params_estimate=params_estimate,
        flops=total,
        notes=f"T={T} * F_block(m={m}, d={d}) (no tau / phase)",
        extras={"T": T, "m": m, "d": d, "F_block": block},
    )


def flops_cot(
    m: int,
    d: int,
    K_used: int,
    *,
    L: int = 1,
    K_cap: Optional[int] = None,
    params_estimate: int = 0,
) -> FlopReport:
    """CoT arm: prefill(m) + sum of decode steps for realized K_used (not cap).

    Prefill uses ``L * F_block(m, d)``.
    Each decode step ``k`` (k=1..K_used) uses ``L * F_decode_step(m+k, d)``
    (KV-cache schematic; context grows with generated tokens).

    ``K_cap`` is recorded in extras only; it must not inflate FLOPs.
    """
    if K_used < 0:
        raise ValueError(f"K_used must be non-negative, got {K_used}")
    if L < 1:
        raise ValueError(f"L must be >= 1 for CoT, got {L}")
    if K_cap is not None and K_used > K_cap:
        raise ValueError(f"K_used={K_used} exceeds K_cap={K_cap}")

    prefill = L * F_block(m, d)
    decode_total = 0
    decode_steps: list[int] = []
    for k in range(1, K_used + 1):
        step = L * F_decode_step(m + k, d)
        decode_steps.append(step)
        decode_total += step
    total = prefill + decode_total
    return FlopReport(
        arm="cot",
        params_estimate=params_estimate,
        flops=total,
        notes=(
            f"prefill(m={m})+sum decode k=1..K_used={K_used} "
            f"(K_cap={K_cap}; FLOPs use realized K_used only)"
        ),
        extras={
            "m": m,
            "d": d,
            "L": L,
            "K_used": K_used,
            "K_cap": K_cap,
            "prefill_flops": prefill,
            "decode_flops": decode_total,
            "decode_steps": decode_steps,
        },
    )


def compare_arms(
    reports: Sequence[FlopReport],
) -> list[dict[str, Any]]:
    """Return a list of dict rows suitable for a tiny comparison table."""
    return [r.to_dict() for r in reports]

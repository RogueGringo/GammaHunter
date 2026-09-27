# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Collapse / coherence diagnostics for token-state trajectories (MEASURE plumbing).

Pure functions over tensors and run-history dicts. Used by
:mod:`reachability_gen.audit_checkpoints`; callable from future runners so a
run can flag its own breakdown instead of reporting it as a result.

Why these exist: pooled ``||z_t||`` and pooled drift δ_t cannot tell "the
state is bounded" from "every token has become the same vector". Under an
RMSNorm bound each token has norm ≈√d, and the mean over tokens reaches ≈√d
only when all tokens are parallel — i.e. rank collapse.

No science OPEN claims.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:  # torch stays optional: flag helpers are stdlib-only
    import torch

# Thresholds calibrated on the fixed30 / bound30 best checkpoints
# (artifacts/id_2k_checkpoint_audit.json). Healthy reference: FF L=2
# (val≈0.95) ends at token cosine ≈0.35, pooled ratio ≈0.53; fixed30 Geo sits
# at ≈0.98 / ≈0.99 from cycle 1 on.
TOKEN_COLLAPSE_COS: float = 0.9
# Output concentration: MAD(margin) / SD(margin). Healthy arms ≥0.38;
# fixed30 Geo 0.02 (most examples share one margin).
OUTPUT_CONCENTRATION_RATIO: float = 0.1
# Training breakdown: largest epoch-over-epoch train-acc drop.
TRAIN_ACC_DROP: float = 0.1
# Reported (last-epoch) val acc below best val acc by at least this much.
FINAL_VS_BEST_GAP: float = 0.1


def token_coherence(
    z_seq: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-example token alignment over real (unmasked) tokens.

    Parameters
    ----------
    z_seq :
        ``[B, M, d]`` per-token states.
    attention_mask :
        Optional ``[B, M]`` with 1 = real token.

    Returns
    -------
    cos_mean, pooled_ratio
        Each ``[B]`` (float32).

        * ``cos_mean`` — mean pairwise cosine similarity between distinct real
          tokens: 1 when every token points the same way, ≈0 for unrelated
          directions. NaN when fewer than two real tokens.
        * ``pooled_ratio`` — ``||mean_i z_i|| / mean_i ||z_i||`` in ``[0, 1]``:
          how much per-token norm survives mean-pooling; 1 iff all tokens are
          parallel.
    """
    import torch

    if z_seq.dim() != 3:
        raise ValueError(f"z_seq must be [B, M, d], got {tuple(z_seq.shape)}")
    z = z_seq.float()
    if attention_mask is None:
        m = torch.ones(z.shape[:2], dtype=z.dtype, device=z.device)
    else:
        m = attention_mask.to(dtype=z.dtype)
    m = m.unsqueeze(-1)  # [B, M, 1]
    n_real = m.sum(dim=(1, 2))  # [B]
    norms = torch.linalg.vector_norm(z, dim=-1, keepdim=True)  # [B, M, 1]
    units = z / norms.clamp_min(1e-12) * m
    s = units.sum(dim=1)  # [B, d]
    # ||Σu_i||² = R + Σ_{i≠j} cos_ij  →  mean over ordered pairs i≠j.
    pairs = n_real * (n_real - 1.0)
    cos_mean = ((s * s).sum(dim=-1) - n_real) / pairs.clamp_min(1.0)
    cos_mean = torch.where(pairs > 0, cos_mean, torch.full_like(cos_mean, float("nan")))
    denom = n_real.clamp_min(1.0)
    pooled = (z * m).sum(dim=1) / denom.unsqueeze(-1)
    mean_token_norm = (norms * m).sum(dim=(1, 2)) / denom
    pooled_ratio = torch.linalg.vector_norm(pooled, dim=-1) / mean_token_norm.clamp_min(
        1e-12
    )
    return cos_mean, pooled_ratio


def mean_token_norm(
    z_seq: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-example mean over real tokens of ``||z_i||_2`` → ``[B]``."""
    import torch

    z = z_seq.float()
    norms = torch.linalg.vector_norm(z, dim=-1)  # [B, M]
    if attention_mask is None:
        return norms.mean(dim=1)
    m = attention_mask.to(dtype=z.dtype)
    return (norms * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def perturbation_gain(
    clean: Sequence[torch.Tensor],
    perturbed: Sequence[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    *,
    relative: bool = False,
) -> list[float]:
    """``||Δz_t||_F / ||Δz_0||_F`` over real tokens, batch-mean, for each t.

    ``clean`` / ``perturbed`` are per-token trajectories ``[z_0 … z_T]`` of the
    same input with and without a small context perturbation. Gain < 1 means
    the recurrence contracts the perturbation by cycle t; > 1 amplifies it.
    Unlike ``perturbation_delta`` (measured at z_0 only) this propagates
    through the dynamics.

    ``relative=True`` divides each ``||Δz_t||`` by ``||z_t||`` first, so an
    unbounded state whose norm grows does not read as amplification.
    """
    import torch

    if len(clean) != len(perturbed) or not clean:
        raise ValueError("clean and perturbed trajectories must be same, nonzero length")

    def _fro(x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if attention_mask is not None:
            x = x * attention_mask.to(dtype=x.dtype).unsqueeze(-1)
        return torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=-1)

    def _size(c: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        diff = _fro(p.float() - c.float())
        return diff / _fro(c).clamp_min(1e-30) if relative else diff

    base = _size(clean[0], perturbed[0]).clamp_min(1e-30)
    return [
        float((_size(c, p) / base).mean().item()) for c, p in zip(clean, perturbed)
    ]


def logit_margins(logits: torch.Tensor) -> torch.Tensor:
    """``logit[y=1] - logit[y=0]`` per example → ``[B]`` (float32)."""
    if logits.dim() != 2 or logits.shape[-1] != 2:
        raise ValueError(f"logits must be [B, 2], got {tuple(logits.shape)}")
    return (logits[:, 1] - logits[:, 0]).float()


def median_abs_deviation(xs: Sequence[float]) -> float:
    """``median(|x - median(x)|)``; NaN for empty input."""
    if not xs:
        return float("nan")
    med = statistics.median(xs)
    return float(statistics.median(abs(x - med) for x in xs))


def largest_train_acc_drop(
    train_history: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Largest epoch-over-epoch fall in ``train_acc`` (0.0 if it never falls)."""
    drop, epoch = 0.0, None
    for prev, cur in zip(train_history, train_history[1:]):
        d = float(prev["train_acc"]) - float(cur["train_acc"])
        if d > drop:
            drop, epoch = d, int(cur["epoch"])
    return {"drop": drop, "epoch": epoch}


def collapse_flags(
    *,
    final_token_cos: Optional[float] = None,
    margins: Optional[Sequence[float]] = None,
    train_history: Optional[Sequence[Mapping[str, Any]]] = None,
    best_val_acc: Optional[float] = None,
    last_epoch_val_acc: Optional[float] = None,
) -> dict[str, Any]:
    """Breakdown flags; each is computed only when its inputs are given.

    * ``token_collapse`` — mean token cosine at the last state ≥
      :data:`TOKEN_COLLAPSE_COS`.
    * ``output_concentration`` — MAD/SD of logit margins below
      :data:`OUTPUT_CONCENTRATION_RATIO` (most inputs share one output).
    * ``train_acc_drop`` — some epoch lost ≥ :data:`TRAIN_ACC_DROP` train acc.
    * ``last_epoch_below_best`` — reported last-epoch val acc ≥
      :data:`FINAL_VS_BEST_GAP` below best (reported tables ≠ best model).

    ``any`` is True if any computed flag fired. Never stamps science OPEN.
    """
    flags: dict[str, Any] = {}
    if final_token_cos is not None:
        flags["token_collapse"] = {
            "value": float(final_token_cos),
            "threshold": TOKEN_COLLAPSE_COS,
            "flag": bool(final_token_cos >= TOKEN_COLLAPSE_COS),
        }
    if margins is not None and len(margins) > 1:
        sd = statistics.pstdev(margins)
        ratio = median_abs_deviation(margins) / sd if sd > 0 else 0.0
        flags["output_concentration"] = {
            "value": float(ratio),
            "threshold": OUTPUT_CONCENTRATION_RATIO,
            "flag": bool(ratio < OUTPUT_CONCENTRATION_RATIO),
        }
    if train_history:
        worst = largest_train_acc_drop(train_history)
        flags["train_acc_drop"] = {
            "value": worst["drop"],
            "epoch": worst["epoch"],
            "threshold": TRAIN_ACC_DROP,
            "flag": bool(worst["drop"] >= TRAIN_ACC_DROP),
        }
    if best_val_acc is not None and last_epoch_val_acc is not None:
        gap = float(best_val_acc) - float(last_epoch_val_acc)
        flags["last_epoch_below_best"] = {
            "value": gap,
            "threshold": FINAL_VS_BEST_GAP,
            "flag": bool(gap >= FINAL_VS_BEST_GAP),
        }
    flags["any"] = any(v["flag"] for v in flags.values() if isinstance(v, dict))
    flags["science_open"] = False
    return flags


__all__ = [
    "FINAL_VS_BEST_GAP",
    "OUTPUT_CONCENTRATION_RATIO",
    "TOKEN_COLLAPSE_COS",
    "TRAIN_ACC_DROP",
    "collapse_flags",
    "largest_train_acc_drop",
    "logit_margins",
    "mean_token_norm",
    "median_abs_deviation",
    "perturbation_gain",
    "token_coherence",
]

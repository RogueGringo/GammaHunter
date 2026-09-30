# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Reference torch backends: exact sort-based forwards, analytic backward passes.

Every function normalizes over the last dimension. ``mask`` (bool, broadcastable
to ``z``; True = position may receive weight) is optional; every row must keep at
least one unmasked position. Masked positions get exactly 0.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import nn

from .spec import NORMALIZERS, SSMAX_INIT_SCALE


def _prepare(z: torch.Tensor, mask: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.ones_like(z, dtype=torch.bool) if mask is None else mask.expand_as(z)
    return z.masked_fill(~mask, float("-inf")), mask


def _sorted(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scores sorted descending with masked (-inf) ones set to 0, their validity, and 1..L."""
    xs = torch.sort(x, dim=-1, descending=True).values
    valid = torch.isfinite(xs)
    rho = torch.arange(1, x.shape[-1] + 1, device=x.device, dtype=x.dtype)
    return torch.where(valid, xs, torch.zeros_like(xs)), valid, rho


def sparsemax_forward(z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """p = [z - τ]_+ with Σp = 1 (no autograd graph through the sort)."""
    z, mask = _prepare(z, mask)
    x = z - z.max(dim=-1, keepdim=True).values
    xs, valid, rho = _sorted(x)
    cs = xs.cumsum(-1)
    support = (1 + rho * xs > cs) & valid  # true exactly for the first k positions
    k = support.sum(-1, keepdim=True).clamp(min=1)
    tau = (cs.gather(-1, k - 1) - 1) / k.to(x.dtype)
    return torch.clamp(x - tau, min=0).masked_fill(~mask, 0.0)


def entmax15_forward(z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """p = [z/2 - τ]_+² with Σp = 1 (exact threshold from the sorted scores)."""
    z, mask = _prepare(z, mask)
    x = (z - z.max(dim=-1, keepdim=True).values) / 2
    xs, valid, rho = _sorted(x)
    mean = xs.cumsum(-1) / rho
    mean_sq = (xs * xs).cumsum(-1) / rho
    delta = (1 - rho * (mean_sq - mean * mean)) / rho
    tau = mean - torch.sqrt(delta.clamp(min=0))
    support = (tau <= xs) & valid
    k = support.sum(-1, keepdim=True).clamp(min=1)
    tau_star = tau.gather(-1, k - 1)
    return (torch.clamp(x - tau_star, min=0) ** 2).masked_fill(~mask, 0.0)


class _Sparsemax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, mask):
        p = sparsemax_forward(z, mask)
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, g):
        (p,) = ctx.saved_tensors
        s = (p > 0).to(g.dtype)
        mean = (g * s).sum(-1, keepdim=True) / s.sum(-1, keepdim=True).clamp(min=1)
        return s * (g - mean), None


class _Entmax15(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, mask):
        p = entmax15_forward(z, mask)
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, g):
        (p,) = ctx.saved_tensors
        root = p.sqrt()
        weighted = root * g
        q = weighted.sum(-1, keepdim=True) / root.sum(-1, keepdim=True)
        return weighted - q * root, None


def softmax(z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    z, mask = _prepare(z, mask)
    return z.softmax(dim=-1)  # exp(-inf) is exactly 0 at masked positions


def sparsemax(z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return _Sparsemax.apply(z, mask)


def entmax15(z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    return _Entmax15.apply(z, mask)


def ssmax(z: torch.Tensor, mask: Optional[torch.Tensor], scale: torch.Tensor) -> torch.Tensor:
    """softmax(scale · ln(n) · z), n the unmasked positions of each row; ``scale`` broadcast to ``z[..., :1]``."""
    _, full = _prepare(z, mask)
    n = full.sum(dim=-1, keepdim=True).to(z.dtype)
    return (z * scale * torch.log(n)).masked_fill(~full, float("-inf")).softmax(dim=-1)


BACKENDS: dict[str, Callable[..., torch.Tensor]] = {
    "softmax": softmax,
    "entmax15": entmax15,
    "sparsemax": sparsemax,
    # the battery grades SSMax at a fixed scale; attention learns it per head
    "ssmax": lambda z, mask=None: ssmax(z, mask, torch.ones((), dtype=z.dtype, device=z.device)),
}


class Normalizer(nn.Module):
    """Attention normalizer over the last dimension, with SSMax's per-head scale when chosen."""

    def __init__(self, kind: str = "softmax", heads: int = 1) -> None:
        super().__init__()
        if kind not in NORMALIZERS:
            raise ValueError(f"unknown normalizer {kind!r}; choose from {sorted(NORMALIZERS)}")
        self.kind = kind
        if kind == "ssmax":
            self.scale = nn.Parameter(torch.full((heads,), SSMAX_INIT_SCALE))

    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.kind == "ssmax":  # z is [B, heads, L, L]
            return ssmax(z, mask, self.scale.view(1, -1, 1, 1))
        return BACKENDS[self.kind](z, mask)


class _BrokenSparsemax(torch.autograd.Function):
    """A deliberately wrong sparsemax: threshold divided by k + 1 instead of k; sparsemax's backward."""

    @staticmethod
    def forward(ctx, z, mask):
        z, mask = _prepare(z, mask)
        x = z - z.max(dim=-1, keepdim=True).values
        xs, valid, rho = _sorted(x)
        cs = xs.cumsum(-1)
        k = ((1 + rho * xs > cs) & valid).sum(-1, keepdim=True).clamp(min=1)
        tau = (cs.gather(-1, k - 1) - 1) / (k + 1).to(x.dtype)
        p = torch.clamp(x - tau, min=0).masked_fill(~mask, 0.0)
        ctx.save_for_backward(p)
        return p

    @staticmethod
    def backward(ctx, g):
        (p,) = ctx.saved_tensors
        s = (p > 0).to(g.dtype)
        return s * (g - (g * s).sum(-1, keepdim=True) / s.sum(-1, keepdim=True).clamp(min=1)), None


def broken_sparsemax(z: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The battery must reject this backend."""
    return _BrokenSparsemax.apply(z, mask)

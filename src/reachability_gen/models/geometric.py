# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Real torch geometric recurrent arm: weight-tied Phi(z_t, c; tau_t).

T cycles of one shared transformer block with cycle (tau) embeddings.
Context ``c`` is the tokenized edge-list encoding (same tokenizer as FF).
``forward(..., return_trajectory=True)`` returns ``(logits [B,2], list[z_t])``
for drift telemetry.

Recurrent update (drift-audit)::

    v_t = Phi(h_t) - z_t                # Phi is Pre-LN TransformerBlock
    z_{t+1} = z_t + α · v_t             # α=0.5 (fixed)
    z_{t+1} ← RMSNorm(z_{t+1})          # optional learning-compatible bound

Outer LayerNorm on the residual stream (``z ← LN(z+α·v)``) was tried and
**blocked learning** for this Pre-LN Phi (val stuck at chance). Preferred
state bound is **RMSNorm** after the α-mix (``apply_cycle_rmsnorm=True``),
which keeps ||z|| ~ O(√d) while remaining learnable. With the bound off,
report **raw + LN-normalized** drift (protocol option b); with RMSNorm on,
also report **RMS-normalized** drift.

δ_t computation (documented)::

    raw:         δ_t = mean_batch ||z_{t+1} - z_t||_2
    LN-normed:   δ_t = mean_batch ||LN(z_{t+1}) - LN(z_t)||_2
    RMS-normed:  δ_t = mean_batch ||RMSNorm(z_{t+1}) - RMSNorm(z_t)||_2
                 (feature norm applied only for the metric when not already
                  on the stream; diameter / bound proxy)

RESEARCH / MEASURE plumbing only — no science OPEN claims.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from reachability_gen.models.feedforward import TransformerBlock

# Fixed residual step size for outer recurrent update (not learned).
DEFAULT_RESIDUAL_ALPHA: float = 0.5


class GeometricRecurrent(nn.Module):
    """Weight-tied Phi reused T times → binary reachability logits ``[B, 2]``.

    Parameters
    ----------
    vocab_size :
        Embedding table size (from :func:`tokenize.build_vocab`).
    d :
        Model width.
    T :
        Number of weight-tied recurrence cycles (fixed to 6 for ID-hop gate).
    n_heads :
        Attention heads (must divide ``d``).
    max_len :
        Maximum positional embedding length.
    mlp_expansion :
        MLP width multiplier (default 4, matches FLOP schematic).
    pad_id :
        Padding token id (excluded from mean-pool).
    use_tau :
        If True, add a learned cycle embedding ``tau_t`` each step.
    max_T :
        Capacity of the tau embedding table (default ``max(T, 16)``).
    residual_alpha :
        Outer residual step size α in ``z ← z + α·(Φ-z)`` (default 0.5).
    apply_cycle_ln :
        If True, apply outer LayerNorm after each cycle. Default **False**:
        stream LN blocked learning with this Pre-LN Phi; prefer RMSNorm.
    apply_cycle_rmsnorm :
        If True, apply outer RMSNorm after each α-mix (and on z_0). Preferred
        learning-compatible state bound so ||z|| stays ~O(√d). Default False
        for backward-compatible fixed30 plumbing; bound30 enables it.
    """

    def __init__(
        self,
        vocab_size: int,
        d: int = 64,
        T: int = 6,
        *,
        n_heads: int = 4,
        max_len: int = 256,
        mlp_expansion: int = 4,
        pad_id: int = 0,
        dropout: float = 0.0,
        use_tau: bool = True,
        max_T: Optional[int] = None,
        residual_alpha: float = DEFAULT_RESIDUAL_ALPHA,
        apply_cycle_ln: bool = False,
        apply_cycle_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
        self.vocab_size = int(vocab_size)
        self.d = int(d)
        self.T = int(T)
        self.n_heads = int(n_heads)
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)
        self.mlp_expansion = int(mlp_expansion)
        self.use_tau = bool(use_tau)
        self.max_T = int(max_T) if max_T is not None else max(self.T, 16)
        self.residual_alpha = float(residual_alpha)
        self.apply_cycle_ln = bool(apply_cycle_ln)
        self.apply_cycle_rmsnorm = bool(apply_cycle_rmsnorm)
        if self.apply_cycle_ln and self.apply_cycle_rmsnorm:
            raise ValueError(
                "apply_cycle_ln and apply_cycle_rmsnorm are mutually exclusive; "
                "prefer apply_cycle_rmsnorm=True (learning-compatible bound)"
            )
        if self.T > self.max_T:
            raise ValueError(f"T={self.T} exceeds max_T={self.max_T}")

        self.tok_emb = nn.Embedding(vocab_size, d, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d)
        # Weight-tied Phi: one shared transformer block.
        self.phi = TransformerBlock(
            d, n_heads, mlp_expansion=mlp_expansion, dropout=dropout
        )
        if self.use_tau:
            self.tau_emb = nn.Embedding(self.max_T, d)
        else:
            self.tau_emb = None  # type: ignore[assignment]
        # Outer cycle LN (legacy; blocked learning) or preferred RMSNorm bound.
        self.cycle_ln = nn.LayerNorm(d) if self.apply_cycle_ln else None
        self.cycle_rmsnorm = (
            nn.RMSNorm(d) if self.apply_cycle_rmsnorm else None
        )
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def _pool(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Masked mean-pool → ``[B, d]`` latent used for drift / head."""
        if attention_mask is not None:
            mask = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            return (x * mask).sum(dim=1) / denom
        return x.mean(dim=1)

    def _cycle_update(
        self,
        z_seq: torch.Tensor,
        h: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """One outer recurrent step: ``z ← z + α·(Φ(h)-z)`` (+ optional bound)."""
        phi_out = self.phi(h, key_padding_mask=key_padding_mask)
        # v_t = Phi(h_t) - z_t; z_{t+1} = z_t + α · v_t; then optional bound
        z_new = z_seq + self.residual_alpha * (phi_out - z_seq)
        if self.cycle_rmsnorm is not None:
            z_new = self.cycle_rmsnorm(z_new)
        elif self.cycle_ln is not None:
            z_new = self.cycle_ln(z_new)
        return z_new

    def _resolve_cycles(self, token_ids: torch.Tensor, T: Optional[int]) -> int:
        """Validate ``[B, M]`` input and return the cycle count to run."""
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be [B, M], got {tuple(token_ids.shape)}")
        mlen = token_ids.shape[1]
        if mlen > self.max_len:
            raise ValueError(
                f"sequence length {mlen} exceeds max_len={self.max_len}"
            )
        cycles = int(self.T if T is None else T)
        if cycles < 1:
            raise ValueError(f"T must be >= 1, got {cycles}")
        if self.use_tau and cycles > self.max_T:
            raise ValueError(f"T={cycles} exceeds tau table max_T={self.max_T}")
        return cycles

    def _iter_states(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cycles: int,
        *,
        context_noise: Optional[torch.Tensor] = None,
    ) -> Iterator[torch.Tensor]:
        """Yield per-token states ``z_0, z_1, …, z_T`` (each ``[B, M, d]``).

        Single source of the recurrence for :meth:`forward` and
        :meth:`token_states`, so diagnostics see exactly the trained dynamics.
        """
        bsz, mlen = token_ids.shape
        device = token_ids.device
        pos = torch.arange(mlen, device=device).unsqueeze(0).expand(bsz, -1)
        # Context c from tokenized encoding; z_0 := c (bound z0 if enabled).
        c = self.tok_emb(token_ids) + self.pos_emb(pos)
        if context_noise is not None:
            c = c + context_noise
        if self.cycle_rmsnorm is not None:
            z_seq = self.cycle_rmsnorm(c)
        elif self.cycle_ln is not None:
            z_seq = self.cycle_ln(c)
        else:
            z_seq = c

        key_padding_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        yield z_seq
        for t in range(cycles):
            if self.use_tau and self.tau_emb is not None:
                tau_t = self.tau_emb(
                    torch.tensor(t, device=device, dtype=torch.long)
                )  # [d]
                h = z_seq + tau_t.view(1, 1, -1)
            else:
                h = z_seq
            z_seq = self._cycle_update(z_seq, h, key_padding_mask)
            yield z_seq

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        return_trajectory: bool = False,
        T: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]]]:
        """Forward pass.

        Parameters
        ----------
        token_ids :
            ``LongTensor [B, M]`` (tokenized edge-list + QUERY).
        attention_mask :
            Optional ``[B, M]`` with 1 = real token, 0 = pad.
        return_trajectory :
            If True, also return ``[z_0, z_1, ..., z_T]`` pooled latents
            (length ``T+1``) for drift ``δ_t = ||z_{t+1}-z_t||_2``.
        T :
            Override cycle count (default ``self.T``).

        Returns
        -------
        logits, trajectory
            ``logits`` has shape ``[B, 2]``. ``trajectory`` is a list of
            ``[B, d]`` tensors when ``return_trajectory`` else ``None``.
        """
        cycles = self._resolve_cycles(token_ids, T)
        trajectory: list[torch.Tensor] = []
        z_seq: Optional[torch.Tensor] = None
        for z_seq in self._iter_states(token_ids, attention_mask, cycles):
            if return_trajectory:
                trajectory.append(self._pool(z_seq, attention_mask))
        assert z_seq is not None

        x = self.ln_f(z_seq)
        pooled = self._pool(x, attention_mask)
        logits = self.head(pooled)  # [B, 2]
        if return_trajectory:
            return logits, trajectory
        return logits, None

    def token_states(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        T: Optional[int] = None,
        context_noise: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """Per-token latents ``[z_0, …, z_T]`` (each ``[B, M, d]``).

        Same recurrence as :meth:`forward` (pooling ``z_t`` reproduces its
        trajectory). ``context_noise`` (``[B, M, d]``) is added to
        ``c = tok_emb + pos_emb`` before the initial bound, for propagated
        perturbation diagnostics. Grad mode is left to the caller.
        """
        cycles = self._resolve_cycles(token_ids, T)
        return list(
            self._iter_states(
                token_ids, attention_mask, cycles, context_noise=context_noise
            )
        )

    def non_embedding_param_count(self) -> int:
        """Count params excluding tok/pos embeddings (tau counted as non-emb)."""
        emb_ids = {id(p) for p in self.tok_emb.parameters()}
        emb_ids |= {id(p) for p in self.pos_emb.parameters()}
        total = 0
        for p in self.parameters():
            if id(p) not in emb_ids:
                total += p.numel()
        return int(total)


def drift_from_trajectory(
    trajectory: list[torch.Tensor],
    *,
    apply_ln: bool = False,
    apply_rmsnorm: bool = False,
) -> list[float]:
    """Compute δ_t = mean_batch ||z_{t+1}-z_t||_2 for consecutive latents.

    Parameters
    ----------
    trajectory :
        List of ``[B, d]`` pooled states (length T+1 → T drifts).
    apply_ln :
        If True, LayerNorm each latent (feature-dim) before differencing.
        Use when states lack outer cycle LN so raw drift can explode; when
        cycle LN is already applied this is nearly a no-op (affine re-scale).
    apply_rmsnorm :
        If True, RMSNorm each latent (feature-dim, no affine) before
        differencing. Preferred diameter proxy when ``apply_cycle_rmsnorm``
        is on the stream (near-identity aside from affine weight).

    Returns
    -------
    list[float]
        Length ``len(trajectory)-1``.
    """
    if apply_ln and apply_rmsnorm:
        raise ValueError("apply_ln and apply_rmsnorm are mutually exclusive")
    if len(trajectory) < 2:
        return []
    drifts: list[float] = []
    for t in range(len(trajectory) - 1):
        z0 = trajectory[t]
        z1 = trajectory[t + 1]
        if apply_ln:
            z0 = F.layer_norm(z0.float(), (z0.shape[-1],))
            z1 = F.layer_norm(z1.float(), (z1.shape[-1],))
        elif apply_rmsnorm:
            z0 = F.rms_norm(z0.float(), (z0.shape[-1],))
            z1 = F.rms_norm(z1.float(), (z1.shape[-1],))
        delta = (z1 - z0).float().reshape(z0.shape[0], -1)
        norms = torch.linalg.vector_norm(delta, ord=2, dim=-1)
        drifts.append(float(norms.mean().item()))
    return drifts


def mean_z_norms_from_trajectory(
    trajectory: list[torch.Tensor],
) -> list[float]:
    """Mean over batch of ||z_t||_2 for each cycle index t=0..T."""
    out: list[float] = []
    for z in trajectory:
        flat = z.float().reshape(z.shape[0], -1)
        norms = torch.linalg.vector_norm(flat, ord=2, dim=-1)
        out.append(float(norms.mean().item()))
    return out


def trajectory_finite_nonzero(drifts: list[float]) -> tuple[bool, str]:
    """Gate helper: fail on NaN/Inf or all-zero drifts."""
    if not drifts:
        return False, "empty drift_trajectory"
    for i, d in enumerate(drifts):
        if d != d:  # NaN
            return False, f"NaN at drift[{i}]"
        if d == float("inf") or d == float("-inf"):
            return False, f"Inf at drift[{i}]"
    if all(abs(d) < 1e-12 for d in drifts):
        return False, "all-zero drifts"
    return True, "ok"


__all__ = [
    "DEFAULT_RESIDUAL_ALPHA",
    "GeometricRecurrent",
    "drift_from_trajectory",
    "mean_z_norms_from_trajectory",
    "trajectory_finite_nonzero",
]

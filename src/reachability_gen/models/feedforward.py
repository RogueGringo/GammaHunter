# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Real torch feed-forward transformer for binary reachability (MEASURE plumbing).

L **unshared** transformer blocks → logits ``[B, 2]``. Trajectories are empty
by design for the FF control arm (no recurrent latent to drift); ``forward``
always returns ``(logits, None)``.

No science OPEN claims.
"""

from __future__ import annotations

from typing import Iterator, Optional

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Pre-LN transformer block: MHA + MLP (expansion 4), unshared weights."""

    def __init__(
        self,
        d: int,
        n_heads: int = 4,
        *,
        mlp_expansion: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"d={d} must be divisible by n_heads={n_heads}")
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            d, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d)
        hidden = mlp_expansion * d
        self.mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + self.dropout(a)
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class FeedForward(nn.Module):
    """L unshared transformer blocks → binary reachability logits ``[B, 2]``.

    Parameters
    ----------
    vocab_size :
        Embedding table size (from :func:`tokenize.build_vocab`).
    d :
        Model width.
    L :
        Number of **unshared** blocks (depth).
    n_heads :
        Attention heads (must divide ``d``).
    max_len :
        Maximum positional embedding length.
    mlp_expansion :
        MLP width multiplier (default 4, matches FLOP schematic).
    pad_id :
        Padding token id (excluded from mean-pool).
    """

    def __init__(
        self,
        vocab_size: int,
        d: int = 64,
        L: int = 2,
        *,
        n_heads: int = 4,
        max_len: int = 256,
        mlp_expansion: int = 4,
        pad_id: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if L < 1:
            raise ValueError(f"L must be >= 1, got {L}")
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
        self.vocab_size = int(vocab_size)
        self.d = int(d)
        self.L = int(L)
        self.n_heads = int(n_heads)
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)
        self.mlp_expansion = int(mlp_expansion)

        self.tok_emb = nn.Embedding(vocab_size, d, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d, n_heads, mlp_expansion=mlp_expansion, dropout=dropout
                )
                for _ in range(L)
            ]
        )
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 2)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        return_trajectory: bool = False,
    ) -> tuple[torch.Tensor, None]:
        """Forward pass.

        Parameters
        ----------
        token_ids :
            ``LongTensor [B, M]``.
        attention_mask :
            Optional ``[B, M]`` with 1 = real token, 0 = pad.
        return_trajectory :
            Accepted for arm-interface parity. FF has **no** latent trajectory;
            always returns ``None`` for the second element (empty by design).

        Returns
        -------
        logits, None
            ``logits`` has shape ``[B, 2]`` (binary CE). Trajectory is always
            ``None`` regardless of ``return_trajectory``.
        """
        del return_trajectory  # FF trajectories empty by design
        self._check_input(token_ids)
        x: Optional[torch.Tensor] = None
        for x in self._iter_states(token_ids, attention_mask):
            pass
        assert x is not None
        x = self.ln_f(x)

        if attention_mask is not None:
            mask = attention_mask.to(dtype=x.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (x * mask).sum(dim=1) / denom
        else:
            pooled = x.mean(dim=1)

        logits = self.head(pooled)  # [B, 2]
        return logits, None

    def _check_input(self, token_ids: torch.Tensor) -> None:
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be [B, M], got {tuple(token_ids.shape)}")
        mlen = token_ids.shape[1]
        if mlen > self.max_len:
            raise ValueError(
                f"sequence length {mlen} exceeds max_len={self.max_len}"
            )

    def _iter_states(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Iterator[torch.Tensor]:
        """Yield residual-stream states: embeddings, then after each block."""
        bsz, mlen = token_ids.shape
        device = token_ids.device
        pos = torch.arange(mlen, device=device).unsqueeze(0).expand(bsz, -1)
        x = self.tok_emb(token_ids) + self.pos_emb(pos)

        key_padding_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            # MHA expects True at padded positions.
            key_padding_mask = attention_mask == 0

        yield x
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
            yield x

    def token_states(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> list[torch.Tensor]:
        """Per-token residual states ``[x_0, …, x_L]`` (each ``[B, M, d]``).

        Diagnostics only (depth-wise token coherence as the FF reference for
        the recurrent arms' cycle-wise coherence). Pre-``ln_f``.
        """
        self._check_input(token_ids)
        return list(self._iter_states(token_ids, attention_mask))

    def non_embedding_param_count(self) -> int:
        """Count parameters excluding token/position embeddings (schematic parity)."""
        emb_ids = {id(p) for p in self.tok_emb.parameters()}
        emb_ids |= {id(p) for p in self.pos_emb.parameters()}
        total = 0
        for p in self.parameters():
            if id(p) not in emb_ids:
                total += p.numel()
        return int(total)


__all__ = ["FeedForward", "TransformerBlock"]

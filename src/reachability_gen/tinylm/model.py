# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""A decoder-only transformer written out in plain PyTorch.

Pre-norm blocks with RMSNorm, rotary position embeddings (relative positions,
no learned position table), causal self-attention through
``scaled_dot_product_attention``, a SwiGLU feed-forward layer, and an output
layer tied to the token embedding. Weights start at N(0, 0.02), with the two
projections that write into the residual stream scaled by 1/sqrt(2 · layers).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .tokenizer import EOS, PAD, VOCAB_SIZE


@dataclass(frozen=True)
class TinyConfig:
    vocab_size: int = VOCAB_SIZE
    d: int = 256
    layers: int = 6
    heads: int = 8
    mlp_hidden: int = 704  # about 8/3 · d (SwiGLU keeps the parameter count of a 4 · d MLP), a multiple of 64
    max_len: int = 2048
    rope_base: float = 10_000.0
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.d % self.heads or (self.d // self.heads) % 2:
            raise ValueError("d must split into heads of even size")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def rotary_tables(head_dim: int, max_len: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin of position × frequency, ``[max_len, head_dim / 2]``."""
    inv = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    angles = torch.outer(torch.arange(max_len, dtype=torch.float64), inv)
    return angles.cos().float(), angles.sin().float()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate each consecutive pair of channels of ``x`` (``[..., L, head_dim]``) by its position's angle."""
    x1, x2 = x[..., ::2].float(), x[..., 1::2].float()
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: TinyConfig) -> None:
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.d // cfg.heads
        self.qkv = nn.Linear(cfg.d, 3 * cfg.d, bias=False)
        self.out = nn.Linear(cfg.d, cfg.d, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, length, d = x.shape
        q, k, v = self.qkv(x).view(b, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rotary(q, cos[:length], sin[:length]), apply_rotary(k, cos[:length], sin[:length])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(b, length, d))


class SwiGLU(nn.Module):
    def __init__(self, cfg: TinyConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d, cfg.mlp_hidden, bias=False)
        self.up = nn.Linear(cfg.d, cfg.mlp_hidden, bias=False)
        self.down = nn.Linear(cfg.mlp_hidden, cfg.d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: TinyConfig) -> None:
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.d, cfg.eps), Attention(cfg)
        self.norm2, self.mlp = RMSNorm(cfg.d, cfg.eps), SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class TinyLM(nn.Module):
    def __init__(self, cfg: TinyConfig = TinyConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.d, cfg.eps)
        cos, sin = rotary_tables(cfg.d // cfg.heads, cfg.max_len, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for block in self.blocks:  # the projections that write into the residual stream
            for w in (block.attn.out.weight, block.mlp.down.weight):
                nn.init.normal_(w, std=0.02 / math.sqrt(2 * cfg.layers))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """``[B, L]`` token ids to ``[B, L, vocab]`` next-token logits."""
        if ids.shape[1] > self.cfg.max_len:
            raise ValueError(f"sequence of {ids.shape[1]} tokens exceeds max_len {self.cfg.max_len}")
        x = self.embed(ids)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        return self.norm(x) @ self.embed.weight.T  # tied output layer

    def loss(self, ids: torch.Tensor) -> torch.Tensor:
        """Mean next-token cross-entropy over the non-padding targets of ``[B, L]`` ids."""
        logits = self(ids[:, :-1])
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                               ignore_index=PAD)

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new: int, *, temperature: float = 0.0,
                 generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Extend ``[1, L]`` ids greedily (temperature 0) or by sampling, stopping at EOS."""
        was = self.training
        self.eval()
        for _ in range(max_new):
            logits = self(ids[:, -self.cfg.max_len:])[:, -1].float()
            if temperature > 0:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1, generator=generator)
            else:
                nxt = logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if int(nxt[0, 0]) == EOS:
                break
        self.train(was)
        return ids

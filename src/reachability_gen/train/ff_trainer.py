# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""AdamW + CE trainer for the feed-forward reachability arm (MEASURE plumbing).

Grad clip max-norm 1.0. ``train_step`` returns ``(loss, accuracy)``.
No science OPEN claims.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from reachability_gen.models.feedforward import FeedForward
from reachability_gen.tokenize import (
    OverflowPolicy,
    Vocab,
    batch_encode,
    build_vocab,
)


class FeedForwardTrainer:
    """Single-step AdamW trainer with CE loss and grad clip 1.0."""

    def __init__(
        self,
        model: FeedForward,
        *,
        lr: float = 3e-3,
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.grad_clip = float(grad_clip)
        self.opt = AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.last_pre_clip_grad_norm: float = 0.0

    def train_step(
        self,
        token_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[float, float]:
        """One AdamW step. Returns ``(loss, accuracy)`` as Python floats."""
        self.model.train()
        token_ids = token_ids.to(self.device)
        labels = labels.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        self.opt.zero_grad(set_to_none=True)
        logits, _ = self.model(token_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        loss.backward()
        pre_clip = clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.last_pre_clip_grad_norm = float(
            pre_clip.item() if hasattr(pre_clip, "item") else pre_clip
        )
        self.opt.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean()
        return float(loss.item()), float(acc.item())

    @torch.no_grad()
    def eval_step(
        self,
        token_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[float, float]:
        """Eval-only CE + accuracy (no grad)."""
        self.model.eval()
        token_ids = token_ids.to(self.device)
        labels = labels.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        logits, _ = self.model(token_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        return float(loss.item()), float(acc.item())


def examples_to_batch(
    examples: Sequence[Mapping[str, Any]],
    vocab: Optional[Vocab] = None,
    *,
    max_len: Optional[int] = None,
    device: Optional[torch.device] = None,
    on_overflow: OverflowPolicy = "warn",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Vocab]:
    """Convert JSONL-like example dicts → ``(token_ids, mask, labels, vocab)``.

    ``on_overflow`` is passed to :func:`tokenize.pad_batch` (rows longer than
    ``max_len`` lose their QUERY tokens).
    """
    if vocab is None:
        vocab = build_vocab()
    encodings = [str(ex.get("encoding", "")) for ex in examples]
    labels_list = [int(ex.get("y", 0)) for ex in examples]
    ids, mask = batch_encode(
        encodings, vocab, max_len=max_len, on_overflow=on_overflow
    )
    dev = device or torch.device("cpu")
    token_ids = torch.tensor(ids, dtype=torch.long, device=dev)
    attention_mask = torch.tensor(mask, dtype=torch.long, device=dev)
    labels = torch.tensor(labels_list, dtype=torch.long, device=dev)
    return token_ids, attention_mask, labels, vocab


__all__ = ["FeedForwardTrainer", "examples_to_batch"]

# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""A language-model encoder whose input embeddings and hidden representations can be tuned.

The encoder of the natural-language readers (items 19–20 of the limitations): a
causal language model from the local cache, read at the output of block
``layer``. Here the model is truncated to its first ``layer`` blocks with the
final norm removed, so its last hidden state is exactly the frozen features of
``run_nl_reader.lm_features`` (tested). Its base weights stay frozen; an
``EncoderAdapter`` adds what may be trained:

* embedding deltas: one trainable vector per token id that occurs in the
  training renderings, added to that token's frozen input embedding (tokens
  that never occur in training have none and keep their embedding);
* low-rank representation interventions (LoReFT; Wu et al. 2024, arXiv
  2404.03592): Φ(h) = h + Rᵀ(W h + b − R h) with R of orthonormal rows, applied
  to the output of every block at every position, parameters tied across
  positions within a block.

Two departures from the paper's form, both keeping the same set of
interventions:

* W is written as R + D, so Φ(h) = h + Rᵀ(D h + b). With D and b starting at 0
  the edit is exactly zero and a fresh adapter reproduces the frozen features
  bit for bit (tested; a free W copied from R does not, because the two
  products round differently). The training dynamics differ from a free W: a
  step in R moves W with it, so R's gradient also carries W's.
* R is the orthonormalised rows of a trainable r × d matrix (by QR, with signs
  fixed so that R follows the matrix continuously), instead of a library
  orthogonal parametrization, whose d × d buffers would add some 38 MB to every
  checkpoint.

Gradients pass through the frozen blocks with activation checkpointing
(non-reentrant, so interventions inside checkpointed blocks receive gradients
even when the embeddings do not; tested).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import nn

os.environ.setdefault("HF_HUB_OFFLINE", "1")


@dataclass
class EncodedText:
    """A rendering as token ids, with its node tokens marked (as ``FeatureTokens`` marks them)."""

    ids: torch.Tensor  # [L]
    kinds: torch.Tensor  # [L] NODE on the last token of each node number, 1 elsewhere
    symbol: torch.Tensor  # [L] node number on NODE tokens, -1 elsewhere
    n: int


class LoReFT(nn.Module):
    """Φ(h) = h + Rᵀ(W h + b − R h) with W = R + D, i.e. h + Rᵀ(D h + b); R has orthonormal rows.

    ``learned`` holds D and b, both 0 at initialisation, so the intervention starts
    as the identity exactly. R is the orthonormalised rows of ``basis``.
    """

    def __init__(self, d: int, rank: int, generator: torch.Generator) -> None:
        super().__init__()
        q, _ = torch.linalg.qr(torch.randn(d, rank, generator=generator))  # [d, rank], orthonormal columns
        self.basis = nn.Parameter(q.T.contiguous())
        self.learned = nn.Linear(d, rank)
        with torch.no_grad():
            self.learned.weight.zero_()
            self.learned.bias.zero_()

    def projection(self) -> torch.Tensor:
        """R: the rows of ``basis`` orthonormalised (QR with the signs of R's diagonal fixed positive)."""
        q, r = torch.linalg.qr(self.basis.T)
        signs = torch.where(torch.diagonal(r) < 0, -1.0, 1.0).to(q.dtype)
        return (q * signs).T

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.learned(h) @ self.projection()


class EncoderAdapter(nn.Module):
    """The trainable part of a ``TunedEncoder``: embedding deltas, interventions, both or neither."""

    def __init__(self, *, d: int, layers: int, vocab_size: int, train_ids: Sequence[int], embeddings: bool,
                 interventions: bool, rank: int, seed: int) -> None:
        super().__init__()
        self.train_ids = sorted(set(int(i) for i in train_ids)) if embeddings else []
        slot = torch.full((vocab_size,), -1, dtype=torch.long)
        if embeddings:
            slot[torch.tensor(self.train_ids, dtype=torch.long)] = torch.arange(len(self.train_ids))
            self.delta = nn.Parameter(torch.zeros(len(self.train_ids), d))
        else:
            self.delta = None
        self.register_buffer("slot", slot, persistent=False)  # rebuilt from ``train_ids``, which checkpoints keep
        # its own stream, and the global one left untouched (module initialisers draw from it), so a tuned arm
        # trains on the same batches in the same order as the frozen arm of the same seed
        gen = torch.Generator().manual_seed(10_000 + seed)
        with torch.random.fork_rng(devices=[]):
            self.refts = nn.ModuleList([LoReFT(d, rank, gen) for _ in range(layers)]) if interventions else None

    def embed(self, base: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        if self.delta is None:
            return base
        s = self.slot[ids]
        add = torch.where((s >= 0)[..., None], self.delta[s.clamp(min=0)], torch.zeros((), device=base.device))
        return (base.float() + add).to(base.dtype)

    def groups(self, *, intervention_lr: float, embedding_lr: float) -> list[dict[str, Any]]:
        """Optimiser groups, one per component (each is also clipped on its own)."""
        out = []
        if self.refts is not None:
            out.append({"name": "interventions", "params": list(self.refts.parameters()), "lr": intervention_lr})
        if self.delta is not None:
            out.append({"name": "embeddings", "params": [self.delta], "lr": embedding_lr})
        return out

    def summary(self, embedding_rms: float) -> dict[str, Any]:
        """The size of what training moved: deltas against the embeddings' RMS, and each intervention's D and b."""
        out: dict[str, Any] = {}
        if self.delta is not None:
            d = self.delta.detach().float()
            out["embedding_delta"] = {"rms_over_embedding_rms": float(d.pow(2).mean().sqrt()) / embedding_rms,
                                      "max_abs_over_embedding_rms": float(d.abs().max()) / embedding_rms}
        if self.refts is not None:
            out["interventions"] = [{"d_frobenius": float(ft.learned.weight.detach().norm()),
                                     "b_norm": float(ft.learned.bias.detach().norm())} for ft in self.refts]
        return out


class TunedEncoder(nn.Module):
    """A frozen causal language model read at block ``layer``, with a swappable ``EncoderAdapter``."""

    def __init__(self, name: str, layer: int, device: str, *, checkpointing: bool = True) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModel.from_pretrained(name, local_files_only=True, dtype=torch.bfloat16)
        model.layers = model.layers[:layer]
        model.config.num_hidden_layers = layer
        model.norm = nn.Identity()  # the last hidden state is then the output of block ``layer`` itself
        for p in model.parameters():
            p.requires_grad_(False)
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model = model.to(device)
        self.layer, self.device = layer, device
        self.d = int(model.config.hidden_size)
        self.vocab_size = int(model.embed_tokens.weight.shape[0])
        self.adapter: Optional[EncoderAdapter] = None
        for i, block in enumerate(self.model.layers):
            block.register_forward_hook(self._hook(i))

    def _hook(self, i: int):
        def fn(_module, _inputs, out):
            if self.adapter is None or self.adapter.refts is None:
                return None
            h = out[0] if isinstance(out, tuple) else out
            edited = self.adapter.refts[i](h.float()).to(h.dtype)
            return (edited, *out[1:]) if isinstance(out, tuple) else edited
        return fn

    def embedding_rms(self) -> float:
        return float(self.model.embed_tokens.weight.float().pow(2).mean().sqrt())

    def encode_texts(self, texts: Sequence[str], ns: Sequence[int]) -> list[EncodedText]:
        from reachability_gen.run_nl_reader import node_token_marks

        out = []
        for text, n in zip(texts, ns):
            enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            kinds, symbol = node_token_marks(text, [tuple(map(int, o)) for o in enc["offset_mapping"]])
            out.append(EncodedText(torch.tensor(enc["input_ids"]), torch.tensor(kinds), torch.tensor(symbol), n))
        return out

    def hidden(self, items: Sequence[EncodedText]) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, L, d]`` last hidden states (bf16) and the ``[B, L]`` mask, right-padded."""
        length = max(int(x.ids.numel()) for x in items)
        ids = torch.full((len(items), length), int(self.tokenizer.pad_token_id), dtype=torch.long)
        mask = torch.zeros((len(items), length), dtype=torch.long)
        for b, x in enumerate(items):
            ids[b, : x.ids.numel()], mask[b, : x.ids.numel()] = x.ids, 1
        ids, mask = ids.to(self.device), mask.to(self.device)
        base = self.model.embed_tokens(ids)
        embeds = self.adapter.embed(base, ids) if self.adapter is not None else base
        return self.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state, mask

    def reader_batch(self, items: Sequence[EncodedText]) -> dict[str, torch.Tensor]:
        """The batch layout ``FeatureReader`` reads (as ``collate_features`` pads it), with live features."""
        states, mask = self.hidden(items)
        bsz, length = mask.shape
        kinds = torch.ones((bsz, length), dtype=torch.long)
        node_slot = torch.full((bsz, length), -1, dtype=torch.long)
        for b, x in enumerate(items):
            m = int(x.ids.numel())
            kinds[b, :m], node_slot[b, :m] = x.kinds, x.symbol
        dev = self.device
        return {"features": states, "kinds": kinds.to(dev), "slots": torch.zeros_like(kinds).to(dev),
                "token_mask": mask.bool(), "node_slot": node_slot.to(dev),
                "width": torch.tensor(max(x.n for x in items)).to(dev)}

    @torch.no_grad()
    def feature_tokens(self, items: Sequence[EncodedText], batch: int = 16) -> list[Any]:
        """Features of each rendering as ``FeatureTokens`` held in CPU memory (as ``lm_features`` returns them)."""
        from reachability_gen.models.reader import FeatureTokens

        was = self.model.training
        self.model.eval()
        out = []
        for i in range(0, len(items), batch):
            chunk = items[i : i + batch]
            states, _ = self.hidden(chunk)
            for b, x in enumerate(chunk):
                m = int(x.ids.numel())
                out.append(FeatureTokens(states[b, :m].to("cpu", copy=True), x.kinds, x.symbol, x.n))
        self.model.train(was)
        return out

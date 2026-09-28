# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Graph reader: edge-list tokens → an explicit 0/1 adjacency (MEASURE).

Three constraints keep the reader from carrying anything but the graph:

* **Question-blind.** It reads the edge list only; the query's nodes never
  reach it, so it cannot add an edge that answers the question.
* **Identity-free.** Every node token gets the same embedding. A node is known
  only by which occurrences share its symbol: occurrences are pooled into one
  slot per distinct symbol, so node numbers never seen in training behave
  like any other.
* **Structural positions.** A token knows its slot inside an edge (source,
  separator, target) and, through attention biases, the relative offset of
  other tokens clipped to ±``max_offset``; there is no absolute position, so a
  longer list presents no unseen positions.

For every ordered pair of node occurrences the reader scores "the first is the
source and the second the target of one listed edge"; a slot pair (u, v) is an
edge when any of its occurrence pairs is. The output is a hard 0/1 adjacency
(``hard=True``) whose gradient passes straight through the scores.

``science_open=false`` always.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn

from reachability_gen.encode import parse_instance

NODE, SEP = 0, 1  # token kinds the reader sees
NO_EVIDENCE: float = -1e30  # log z of a node pair with no occurrence pair (finite, so 0 * it stays 0)


def log_softplus(x: torch.Tensor) -> torch.Tensor:
    """log(softplus(x)) without underflow: x itself below -20 (error below 1e-9)."""
    return torch.where(x < -20, x, torch.log(torch.nn.functional.softplus(x.clamp(min=-20))))


@dataclass(frozen=True)
class EdgeListTokens:
    """One graph's edge list as the reader sees it (no query)."""

    kinds: torch.Tensor  # [L] NODE or SEP
    slots: torch.Tensor  # [L] slot within an edge: 0 source, 1 separator, 2 target
    symbol: torch.Tensor  # [L] node number for NODE tokens (grouping only), -1 for SEP
    n: int


def edge_list_tokens(encoding: str) -> EdgeListTokens:
    """The edge list of a locked encoding, without the query."""
    n, edges, _, _ = parse_instance(encoding)
    kinds, slots, symbol = [], [], []
    for u, v in edges:
        kinds += [NODE, SEP, NODE]
        slots += [0, 1, 2]
        symbol += [u, -1, v]
    return EdgeListTokens(torch.tensor(kinds), torch.tensor(slots), torch.tensor(symbol), n)


def collate_tokens(items: Sequence[EdgeListTokens], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Pad a batch; ``node_slot[b, i]`` is the node an occurrence belongs to (-1 for other tokens)."""
    bsz = len(items)
    length = max(1, max(int(x.kinds.numel()) for x in items))
    kinds = torch.full((bsz, length), SEP, dtype=torch.long)
    slots = torch.ones((bsz, length), dtype=torch.long)
    mask = torch.zeros((bsz, length), dtype=torch.bool)
    node_slot = torch.full((bsz, length), -1, dtype=torch.long)
    for b, x in enumerate(items):
        m = int(x.kinds.numel())
        kinds[b, :m], slots[b, :m], mask[b, :m], node_slot[b, :m] = x.kinds, x.slots, True, x.symbol
    width = torch.tensor(max(x.n for x in items))
    return {k: v.to(device) for k, v in {"kinds": kinds, "slots": slots, "token_mask": mask,
                                         "node_slot": node_slot, "width": width}.items()}


class RelativeAttentionLayer(nn.Module):
    """Self-attention with a learned bias per clipped relative offset, then an MLP."""

    def __init__(self, d: int, heads: int, max_offset: int) -> None:
        super().__init__()
        self.heads, self.max_offset = heads, max_offset
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.bias = nn.Embedding(2 * max_offset + 3, heads)  # offsets -max..max, plus "far" both ways
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def offsets(self, length: int, device) -> torch.Tensor:
        pos = torch.arange(length, device=device)
        rel = (pos[None, :] - pos[:, None]).clamp(-self.max_offset - 1, self.max_offset + 1)
        return rel + self.max_offset + 1  # [L, L] bucket ids

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        bsz, length, d = h.shape
        x = self.norm1(h)
        q, k, v = self.qkv(x).view(bsz, length, 3, self.heads, d // self.heads).unbind(2)
        att = torch.einsum("bihd,bjhd->bhij", q, k) / (d // self.heads) ** 0.5
        att = att + self.bias(self.offsets(length, h.device)).permute(2, 0, 1)[None]
        att = att.masked_fill(~mask[:, None, None, :], float("-inf")).softmax(dim=-1)
        h = h + self.out(torch.einsum("bhij,bjhd->bihd", att, v).reshape(bsz, length, d))
        return h + self.mlp(self.norm2(h))


class GraphReader(nn.Module):
    """Edge-list tokens → edge scores for every ordered pair of node slots."""

    def __init__(self, d: int = 64, layers: int = 2, heads: int = 4, max_offset: int = 4) -> None:
        super().__init__()
        self.kind = nn.Embedding(2, d)  # one embedding for every node token, one for the separator
        self.slot = nn.Embedding(3, d)
        self.layers = nn.ModuleList([RelativeAttentionLayer(d, heads, max_offset) for _ in range(layers)])
        self.src, self.dst = nn.Linear(d, d), nn.Linear(d, d)
        self.pair_offset = nn.Embedding(2 * max_offset + 3, 1)  # the scorer also sees the pair's clipped offset
        # One shared starting value for every offset (no offset preferred), low enough
        # that a node pair's noisy-OR starts near the edge density, not near 1.
        nn.init.constant_(self.pair_offset.weight, -3.0)
        self.max_offset = max_offset

    def occurrence_scores(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """``[B, L, L]`` logits: token i is the source and token j the target of one listed edge."""
        h = self.kind(batch["kinds"]) + self.slot(batch["slots"])
        for layer in self.layers:
            h = layer(h, batch["token_mask"])
        length = h.shape[1]
        rel = self.layers[0].offsets(length, h.device) if self.layers else None
        scores = torch.einsum("bid,bjd->bij", self.src(h), self.dst(h)) / h.shape[-1] ** 0.5
        if rel is not None:
            scores = scores + self.pair_offset(rel).squeeze(-1)[None]
        node = (batch["kinds"] == NODE) & batch["token_mask"]
        return scores.masked_fill(~(node[:, :, None] & node[:, None, :]), -1e4)

    def edge_log_evidence(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """``[B, N, N]`` log z per node pair (see ``edge_evidence``), computed in log space.

        log z is the log-sum-exp of log softplus(score) over the occurrence pairs of
        (u, v), scattered into u * width + v (other pairs go to one discarded bin).
        Neither log z nor its gradient underflows when every score is very negative,
        where z itself rounds to 0 in float32 and its gradient vanishes. Pairs
        without occurrence pairs, and the diagonal, get ``NO_EVIDENCE``.
        """
        scores = self.occurrence_scores(batch)
        bsz, width = scores.shape[0], int(batch["width"])
        slot = batch["node_slot"]
        pair = slot[:, :, None] * width + slot[:, None, :]
        pair = pair.masked_fill((slot[:, :, None] < 0) | (slot[:, None, :] < 0), width * width).flatten(1)
        vals = log_softplus(scores).flatten(1)
        bins = width * width + 1
        top = torch.full((bsz, bins), NO_EVIDENCE, device=scores.device, dtype=scores.dtype)
        top = top.scatter_reduce(1, pair, vals.detach(), reduce="amax", include_self=True)
        total = torch.zeros(bsz, bins, device=scores.device, dtype=scores.dtype)
        total = total.scatter_add(1, pair, torch.exp(vals - top.gather(1, pair)))
        log_z = (top + torch.log(total.clamp(min=1e-38)))[:, : width * width].view(bsz, width, width)
        empty = total[:, : width * width].view(bsz, width, width) == 0
        eye = torch.eye(width, device=log_z.device, dtype=torch.bool)
        return log_z.masked_fill(empty | eye[None], NO_EVIDENCE)  # the format lists no self-loops

    def edge_evidence(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """``[B, N, N]`` evidence z ≥ 0 per node pair, with P(edge) = 1 - exp(-z).

        A node pair is an edge unless every one of its occurrence pairs says no
        (noisy-OR): z sums softplus(score) over the occurrence pairs of (u, v).
        Every occurrence pair receives gradient, unlike a max. Losses should use
        ``edge_log_evidence``: z rounds to 0 once every score is very negative.
        """
        return self.edge_log_evidence(batch).exp()

    def forward(self, batch: dict[str, torch.Tensor], *, hard: bool = True, through: str = "probability") -> torch.Tensor:
        """``[B, N, N]`` adjacency (``adj[b, u, v]`` = edge u→v), hard 0/1 with a straight-through gradient.

        ``through="probability"`` passes the gradient through P(edge), which
        vanishes once P saturates at 0 or 1; ``through="logit"`` passes it
        through the edge log-odds, which saturates at neither end.
        """
        log_z = self.edge_log_evidence(batch)
        z = log_z.exp()
        soft = -torch.expm1(-z)  # 1 - exp(-z)
        if not hard:
            return soft
        if through == "probability":
            carrier = soft
        elif through == "logit":
            # log-odds of the noisy-OR, log(expm1(z)): log z + z/2 for small z (taken
            # from log z, so it never underflows), z beyond 20; the clamp keeps the
            # unused branch finite (no inf - inf in the gradient).
            mid = torch.log(torch.expm1(z.clamp(min=1e-4, max=20)))
            carrier = torch.where(log_z < -9, log_z + 0.5 * z, torch.where(z > 20, z, mid))
        else:
            raise ValueError(f"through must be 'probability' or 'logit', got {through!r}")
        # Parenthesised so the correction is exactly zero: non-edges stay exactly 0,
        # which the anchored solver's zero-preservation depends on.
        return (soft > 0.5).to(soft.dtype) + (carrier - carrier.detach())


def edge_metrics(pred: torch.Tensor, gold: torch.Tensor, node_mask: torch.Tensor) -> dict[str, Any]:
    """Edge precision/recall/F1, exact-graph matches and error kinds for a batch of 0/1 adjacencies."""
    valid = (node_mask[:, :, None] & node_mask[:, None, :]) & ~torch.eye(pred.shape[1], dtype=torch.bool,
                                                                          device=pred.device)[None]
    p, g = (pred > 0.5) & valid, (gold > 0.5) & valid
    tp, fp, fn = (p & g).sum().item(), (p & ~g).sum().item(), (~p & g).sum().item()
    reversed_err = (p & ~g & g.transpose(1, 2)).sum().item()  # predicted v→u where only u→v is listed
    exact = ((p == g) | ~valid).flatten(1).all(dim=1)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "edges_true": int(g.sum().item()), "tp": int(tp), "fp": int(fp), "fn": int(fn),
        "reversed_errors": int(reversed_err),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "exact_graphs": int(exact.sum().item()), "graphs": int(exact.numel()),
    }


def closure(adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """``[B, N, N]`` reachability (paths of length ≥ 1) by repeated boolean squaring."""
    a = (adj > 0.5) & node_mask[:, :, None] & node_mask[:, None, :]
    reach = a.clone()
    for _ in range(max(1, int(adj.shape[1]).bit_length())):
        reach = reach | (reach.float() @ reach.float() > 0)
    return reach


__all__ = ["NO_EVIDENCE", "EdgeListTokens", "GraphReader", "closure", "collate_tokens", "edge_list_tokens",
           "edge_metrics", "log_softplus"]

# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Message-passing arms over the graph encoded in each example (MEASURE).

The locked encoding string is parsed into a directed graph. Every node keeps a
state, initialised from two flags (is the query's source, is its target); each
step updates every node from its own state and the maximum of the messages of
its in-neighbours — max aggregation, the choice that matches breadth-first
expansion. The classifier reads the target's and the source's final states.

* ``looped=True``: one shared step applied ``steps`` times; the count can be
  raised at evaluation time.
* ``looped=False``: ``steps`` distinct layers; depth is fixed.

Node identities are never embedded, so the arms are invariant to node
numbering and to graph size.

``science_open=false`` always.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn

from reachability_gen.encode import parse_instance


@dataclass(frozen=True)
class ParsedGraph:
    n: int
    src: torch.Tensor  # [E] edge sources
    dst: torch.Tensor  # [E] edge targets
    s: int
    t: int
    y: int


def parse_rows(rows: Sequence[dict[str, Any]]) -> list[ParsedGraph]:
    """Parse each row's locked encoding once."""
    out = []
    for r in rows:
        n, edges, s, t = parse_instance(str(r["encoding"]))
        e = torch.tensor(edges, dtype=torch.long).reshape(-1, 2)
        out.append(ParsedGraph(n, e[:, 0], e[:, 1], s, t, int(r["y"])))
    return out


def collate(graphs: Sequence[ParsedGraph], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Dense batch: adjacency ``[B,N,N]`` (``adj[b,u,v]`` = edge u→v), masks, flags."""
    bsz, width = len(graphs), max(g.n for g in graphs)
    adj = torch.zeros(bsz, width, width, dtype=torch.bool)
    node_mask = torch.zeros(bsz, width, dtype=torch.bool)
    feats = torch.zeros(bsz, width, 2)
    for i, g in enumerate(graphs):
        adj[i, g.src, g.dst] = True
        node_mask[i, : g.n] = True
        feats[i, g.s, 0] = 1.0
        feats[i, g.t, 1] = 1.0
    return {
        "adj": adj.to(device),
        "node_mask": node_mask.to(device),
        "feats": feats.to(device),
        "s": torch.tensor([g.s for g in graphs], device=device),
        "t": torch.tensor([g.t for g in graphs], device=device),
        "y": torch.tensor([g.y for g in graphs], device=device),
    }


class MPStep(nn.Module):
    """One update: ``h ← LN(h + U([h, max_{u→v} M(h_u)]))``, padded nodes zeroed."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.upd = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        m = self.msg(h)  # [B, U, d]: message sent by each node
        cand = m.unsqueeze(2).expand(-1, -1, adj.shape[2], -1)  # [B, U, V, d]
        cand = cand.masked_fill(~adj.unsqueeze(-1), torch.finfo(m.dtype).min)
        agg = cand.max(dim=1).values  # [B, V, d]: max over in-neighbours
        agg = torch.where(adj.any(dim=1).unsqueeze(-1), agg, torch.zeros_like(agg))
        out = self.norm(h + self.upd(torch.cat([h, agg], dim=-1)))
        return out * node_mask.unsqueeze(-1).to(out.dtype)


class MessagePassing(nn.Module):
    """Looped (weight-tied) or unlooped message passing → logits ``[B, 2]``."""

    def __init__(self, d: int = 64, steps: int = 6, *, looped: bool = True) -> None:
        super().__init__()
        if steps < 1 or d < 1:
            raise ValueError(f"need steps >= 1 and d >= 1, got steps={steps} d={d}")
        self.d, self.steps, self.looped = int(d), int(steps), bool(looped)
        self.inp = nn.Linear(2, d)
        self.layers = nn.ModuleList([MPStep(d) for _ in range(1 if looped else steps)])
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 2))

    def node_states(self, batch: dict[str, torch.Tensor], steps: Optional[int] = None) -> list[torch.Tensor]:
        """Node states ``[h_0, …, h_T]`` (each ``[B, N, d]``)."""
        n_steps = self.steps if steps is None else int(steps)
        if not self.looped and n_steps != self.steps:
            raise ValueError(f"unlooped arm has fixed depth {self.steps}, got steps={n_steps}")
        mask = batch["node_mask"]
        h = self.inp(batch["feats"]) * mask.unsqueeze(-1).to(batch["feats"].dtype)
        states = [h]
        for i in range(n_steps):
            h = self.layers[0 if self.looped else i](h, batch["adj"], mask)
            states.append(h)
        return states

    def forward(self, batch: dict[str, torch.Tensor], steps: Optional[int] = None) -> torch.Tensor:
        h = self.node_states(batch, steps)[-1]
        rows = torch.arange(h.shape[0], device=h.device)
        return self.head(torch.cat([h[rows, batch["t"]], h[rows, batch["s"]]], dim=-1))

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def param_formula(d: int, layers: int) -> int:
    """Closed-form parameter count: ``layers`` steps plus input and head."""
    return layers * (5 * d * d + 6 * d) + 2 * d * d + 6 * d + 2


def match_width(target: int, steps: int, *, looped: bool, lo: int = 8, hi: int = 1024) -> int:
    """Width whose parameter count is closest to ``target``."""
    layers = 1 if looped else steps
    return min(range(lo, hi + 1), key=lambda d: abs(param_formula(d, layers) - target))


__all__ = [
    "MPStep",
    "MessagePassing",
    "ParsedGraph",
    "collate",
    "match_width",
    "param_formula",
    "parse_rows",
]

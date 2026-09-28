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
from typing import Any, Iterator, Optional, Sequence

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


def _max_in_messages(m: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    """``[B, V, d]``: max of the messages ``m`` of each node's in-neighbours (0 if none)."""
    cand = m.unsqueeze(2).expand(-1, -1, adj.shape[2], -1)  # [B, U, V, d]
    cand = cand.masked_fill(~adj.unsqueeze(-1), torch.finfo(m.dtype).min)
    agg = cand.max(dim=1).values  # [B, V, d]: max over in-neighbours
    return torch.where(adj.any(dim=1).unsqueeze(-1), agg, torch.zeros_like(agg))


class MPStep(nn.Module):
    """One update: ``h ← LN(h + U([h, max_{u→v} M(h_u)]))``, padded nodes zeroed."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.upd = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, h: torch.Tensor, adj: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        agg = _max_in_messages(self.msg(h), adj)  # m: message sent by each node
        out = self.norm(h + self.upd(torch.cat([h, agg], dim=-1)))
        return out * node_mask.unsqueeze(-1).to(out.dtype)


class GeoStep(nn.Module):
    """Geometric update: ``h ← B(h + α·(τ + U([N(x), max_{u→v} M(N(x))_u])))``, ``x = h + τ``.

    The branch reads a normalised copy of the state (``N``, LayerNorm); the
    state moves by a fixed fraction ``α`` of it and is then bounded by an RMS
    norm ``B``. ``τ`` is an optional per-step vector (cycle embedding).
    """

    def __init__(self, d: int, *, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.pre = nn.LayerNorm(d)
        self.msg = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.upd = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, d))
        self.bound = nn.RMSNorm(d)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor,
        node_mask: torch.Tensor,
        tau: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.pre(h if tau is None else h + tau)
        branch = self.upd(torch.cat([x, _max_in_messages(self.msg(x), adj)], dim=-1))
        if tau is not None:
            branch = branch + tau
        out = self.bound(h + self.alpha * branch)
        return out * node_mask.unsqueeze(-1).to(out.dtype)


UPDATES: tuple[str, ...] = ("residual", "geo")


class MessagePassing(nn.Module):
    """Looped (weight-tied) or unlooped message passing → logits ``[B, 2]``.

    ``update="residual"`` is the standard post-norm residual step (``MPStep``);
    ``update="geo"`` the geometric step (``GeoStep``), whose initial state is
    also RMS-bounded. ``tau=True`` (geo, looped only) adds one learned vector
    per trained step; steps beyond the trained count reuse the last one.
    """

    def __init__(
        self,
        d: int = 64,
        steps: int = 6,
        *,
        looped: bool = True,
        update: str = "residual",
        tau: bool = False,
    ) -> None:
        super().__init__()
        if steps < 1 or d < 1:
            raise ValueError(f"need steps >= 1 and d >= 1, got steps={steps} d={d}")
        if update not in UPDATES:
            raise ValueError(f"update must be one of {UPDATES}, got {update!r}")
        if tau and (update != "geo" or not looped):
            raise ValueError("tau needs the looped geo update")
        self.d, self.steps, self.looped, self.update = int(d), int(steps), bool(looped), update
        self.inp = nn.Linear(2, d)
        step = MPStep if update == "residual" else GeoStep
        self.layers = nn.ModuleList([step(d) for _ in range(1 if looped else steps)])
        self.init_bound = nn.RMSNorm(d) if update == "geo" else None
        self.tau = nn.Embedding(steps, d) if tau else None
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU(), nn.Linear(d, 2))

    def iter_states(self, batch: dict[str, torch.Tensor], steps: Optional[int] = None) -> Iterator[torch.Tensor]:
        """Yield node states ``h_0, …, h_T`` (each ``[B, N, d]``) one at a time."""
        n_steps = self.steps if steps is None else int(steps)
        if not self.looped and n_steps != self.steps:
            raise ValueError(f"unlooped arm has fixed depth {self.steps}, got steps={n_steps}")
        mask = batch["node_mask"]
        h = self.inp(batch["feats"])
        if self.init_bound is not None:
            h = self.init_bound(h)
        h = h * mask.unsqueeze(-1).to(batch["feats"].dtype)
        yield h
        for i in range(n_steps):
            layer = self.layers[0 if self.looped else i]
            if self.tau is not None:
                h = layer(h, batch["adj"], mask, self.tau.weight[min(i, self.tau.num_embeddings - 1)])
            else:
                h = layer(h, batch["adj"], mask)
            yield h

    def node_states(self, batch: dict[str, torch.Tensor], steps: Optional[int] = None) -> list[torch.Tensor]:
        """Node states ``[h_0, …, h_T]`` (each ``[B, N, d]``)."""
        return list(self.iter_states(batch, steps))

    def readout(self, h: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        rows = torch.arange(h.shape[0], device=h.device)
        return self.head(torch.cat([h[rows, batch["t"]], h[rows, batch["s"]]], dim=-1))

    def forward(self, batch: dict[str, torch.Tensor], steps: Optional[int] = None) -> torch.Tensor:
        h = batch["feats"]
        for h in self.iter_states(batch, steps):
            pass
        return self.readout(h, batch)

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


def param_formula(d: int, layers: int) -> int:
    """Closed-form parameter count: ``layers`` steps plus input and head."""
    return layers * (5 * d * d + 6 * d) + 2 * d * d + 6 * d + 2


def match_width(target: int, steps: int, *, looped: bool, lo: int = 8, hi: int = 1024) -> int:
    """Width whose parameter count is closest to ``target``."""
    layers = 1 if looped else steps
    return min(range(lo, hi + 1), key=lambda d: abs(param_formula(d, layers) - target))


def take(packed: dict[str, torch.Tensor], idx: torch.Tensor) -> dict[str, torch.Tensor]:
    """Rows ``idx`` of a batch built once for a whole set (``collate`` of every graph).

    Padding to the set's largest graph instead of the batch's does not change
    any real node's state: padded nodes send and receive nothing.
    """
    return {k: v[idx] for k, v in packed.items()}


__all__ = [
    "GeoStep",
    "MPStep",
    "MessagePassing",
    "ParsedGraph",
    "UPDATES",
    "collate",
    "match_width",
    "param_formula",
    "parse_rows",
    "take",
]

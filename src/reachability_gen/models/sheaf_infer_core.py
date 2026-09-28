# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary
#
# Ported from the sister line (models/sheaf_infer_core.py at 6f1d4d3), model
# code unchanged; the four names it imported from fractal_core are inlined.

"""SheafInferCore: edge-token → A_hat inference + stalk diffusion (MEASURE).

CYCLE_SHEAF_INFERENCE architecture
----------------------------------
Infer directed adjacency from **edge tokens only** (no hard ``A_ij`` oracle
at eval)::

    E_hat[u→v]  = edge_encoder(token_uv)          # logits on listed edges
    E_hat[other] = absent_bias                    # learned prior (init <0)
    gate_ij = 1{σ(E) > θ}                         # hard gate; STE / Gumbel
    A_hat used as binary aggregation weights
    (self always gated on)

Stalk diffusion (discrete fixed-T, shared bias-free Φ)::

    z_0[i] = stalk_proj(e_s)  if i == s else 0    # local init (s=1, else 0)
    m_t = gate @ W_msg(RMSNorm(z_t))
    φ_t = W_out(MLP_biasfree(RMSNorm(m_t)))
    z_{t+1} = RMSNorm(z_t + α · (φ_t - z_t) + potential)
    logits = head(z_T[t])

Bias-free Φ + RMSNorm preserve zeros: unreachable ``t`` stays ‖h_t‖≈0 when
``gate`` blocks all s⇝t paths (document STE softening if miss).

Training may use gold edges for an **aux reconstruction** BCE on ``E_hat``
only — never as the eval adjacency oracle. Soft ACT / ``c`` broadcast remain
stripped (fail-closed from stalk seal ``b144dac``).

Param parity: ±5% of FF 121218 → window [115157, 127279].

RESEARCH / MEASURE — ``science_open=false`` always. No science OPEN claims.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from reachability_gen.encode import parse_instance
from reachability_gen.models.geometric import DEFAULT_RESIDUAL_ALPHA
from reachability_gen.tokenize import DEFAULT_MAX_NODE_ID

# Inlined from the sister line's models/fractal_core.py (not ported).
DISCRETE_T_VALUES: tuple[int, ...] = (6, 8, 12, 16)

# Aspirational metaphor only — never used as evidence.
MANDELBROT_ANALOGY_NOTE: str = (
    "Mandelbrot-style boundary re-injection is an aspirational analogy for "
    "local stalk/probe potentials; scientific claims require metrics artifacts only."
)

DEFAULT_DISCONNECT_LEAK_ATOL: float = 1e-3


def _verify_param_parity(
    fractal_count: int,
    *,
    ff_baseline: int = 121_218,
    tol: float = 0.05,
) -> dict[str, Any]:
    """Hard check: FractalCore total params within ±tol of FF baseline.

    Always returns ``science_open=False``. Raises ``AssertionError`` on miss
    when used as a gate; callers may also inspect the dict.
    """
    lo = int(ff_baseline * (1.0 - tol))
    hi = int(round(ff_baseline * (1.0 + tol)))
    ok = lo <= int(fractal_count) <= hi
    ratio = fractal_count / ff_baseline if ff_baseline else float("nan")
    section = {
        "ff_baseline_params": int(ff_baseline),
        "fractal_param_count": int(fractal_count),
        "window": [lo, hi],
        "tolerance": tol,
        "fractal_over_ff_ratio": ratio,
        "within_5pct": ok,
        "science_open": False,
        "notes": (
            "Param parity is MEASURE accounting hygiene — never stamps science OPEN."
        ),
    }
    if not ok:
        raise AssertionError(
            f"FractalCore params {fractal_count} outside ±{tol:.0%} of FF "
            f"{ff_baseline} (window [{lo}, {hi}])"
        )
    return section


DEFAULT_GATE_THETA: float = 0.5
DEFAULT_EDGE_RECON_WEIGHT: float = 1.0
DEFAULT_ABSENT_BIAS: float = -4.0


def build_sheaf_batch(
    examples: Sequence[Mapping[str, Any]],
    *,
    max_n: Optional[int] = None,
    max_edges: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> dict[str, torch.Tensor]:
    """Parse encodings → node slots + edge tokens (+ gold adj for aux only).

    Returns dict with ``node_ids``, ``node_mask``, ``edge_index``, ``edge_mask``,
    ``gold_adj`` (TRAIN aux only), ``s_idx``, ``t_idx``, ``n_nodes``,
    optional ``labels``. No oracle ``attn_mask``.
    """
    parsed: list[tuple[int, list[tuple[int, int]], int, int]] = []
    labels_list: list[int] = []
    has_y = True
    for ex in examples:
        enc = ex.get("encoding")
        if enc:
            n, edges, s, t = parse_instance(str(enc))
        else:
            n = int(ex["n"])
            edges = [(int(a), int(b)) for a, b in ex["edges"]]
            s, t = int(ex["s"]), int(ex["t"])
        n = int(ex.get("n", n))
        s = int(ex.get("s", s))
        t = int(ex.get("t", t))
        parsed.append((n, edges, s, t))
        if "y" in ex:
            labels_list.append(int(ex["y"]))
        else:
            has_y = False

    M = int(max_n) if max_n is not None else max(n for n, _, _, _ in parsed)
    if M < 1:
        M = 1
    Emax = (
        int(max_edges)
        if max_edges is not None
        else max((len(edges) for _, edges, _, _ in parsed), default=1)
    )
    if Emax < 1:
        Emax = 1
    B = len(parsed)

    node_ids = torch.zeros(B, M, dtype=torch.long, device=device)
    node_mask = torch.zeros(B, M, dtype=torch.long, device=device)
    edge_index = torch.zeros(B, Emax, 2, dtype=torch.long, device=device)
    edge_mask = torch.zeros(B, Emax, dtype=torch.long, device=device)
    gold_adj = torch.zeros(B, M, M, dtype=torch.float32, device=device)
    s_idx = torch.zeros(B, dtype=torch.long, device=device)
    t_idx = torch.zeros(B, dtype=torch.long, device=device)
    n_nodes = torch.zeros(B, dtype=torch.long, device=device)

    for b, (n, edges, s, t) in enumerate(parsed):
        if n > M:
            raise ValueError(f"n={n} exceeds max_n={M}")
        if not (0 <= s < n and 0 <= t < n):
            raise ValueError(f"query ({s},{t}) out of range for n={n}")
        node_ids[b, :n] = torch.arange(n, device=device)
        node_mask[b, :n] = 1
        for e_i, (u, v) in enumerate(edges):
            if e_i >= Emax:
                break
            ui, vi = int(u), int(v)
            if not (0 <= ui < n and 0 <= vi < n):
                raise ValueError(f"edge ({ui},{vi}) out of range for n={n}")
            edge_index[b, e_i, 0] = ui
            edge_index[b, e_i, 1] = vi
            edge_mask[b, e_i] = 1
            # Aggregation layout: row=query i receives from key j (edge j→i)
            gold_adj[b, vi, ui] = 1.0
        if n > 0:
            idx = torch.arange(n, device=device)
            gold_adj[b, idx, idx] = 1.0
        s_idx[b] = s
        t_idx[b] = t
        n_nodes[b] = n

    out: dict[str, torch.Tensor] = {
        "node_ids": node_ids,
        "node_mask": node_mask,
        "edge_index": edge_index,
        "edge_mask": edge_mask,
        "gold_adj": gold_adj,
        "s_idx": s_idx,
        "t_idx": t_idx,
        "n_nodes": n_nodes,
    }
    if has_y and len(labels_list) == B:
        out["labels"] = torch.tensor(labels_list, dtype=torch.long, device=device)
    return out


def ste_hard_gate(
    logits: torch.Tensor,
    *,
    theta: float = DEFAULT_GATE_THETA,
    training: bool = False,
    mode: str = "ste",
    gumbel_temp: float = 1.0,
) -> torch.Tensor:
    """Hard Bernoulli gate on ``sigmoid(logits)`` with STE or Gumbel-Sigmoid."""
    if mode == "gumbel" and training:
        u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
        noise = torch.log(u) - torch.log(1.0 - u)
        return torch.sigmoid((logits + noise) / max(gumbel_temp, 1e-6))

    probs = torch.sigmoid(logits)
    hard = (probs > theta).to(dtype=logits.dtype)
    if mode == "ste" and training:
        return hard + (probs - probs.detach())
    return hard


class SheafDiffusionPhi(nn.Module):
    """Shared bias-free Φ: gated aggregate + residual MLP (zeros stay zeros).

    Core diffusion is ``gate @ W_msg(h)`` (init W_msg=I) so stalk mass
    propagates along inferred edges from step 0. MLP is a residual
    refinement (init near 0) for capacity / param parity.
    """

    def __init__(self, d: int, *, mlp_expansion: int = 10) -> None:
        super().__init__()
        hidden = int(mlp_expansion) * d
        self.W_msg = nn.Linear(d, d, bias=False)
        self.mlp_up = nn.Linear(d, hidden, bias=False)
        self.mlp_down = nn.Linear(hidden, d, bias=False)
        self.W_out = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.W_msg.weight)
        nn.init.eye_(self.W_out.weight)
        nn.init.zeros_(self.mlp_up.weight)
        nn.init.zeros_(self.mlp_down.weight)

    def forward(self, h: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """``h`` [B,M,d], ``gate`` [B,M,M] in [0,1] (row i ← cols j)."""
        m = torch.bmm(gate, self.W_msg(h))
        # Residual MLP refinement (0 at init ⇒ pure diffusion).
        m = m + self.mlp_down(F.gelu(self.mlp_up(m)))
        return self.W_out(m)


class SheafInferCore(nn.Module):
    """Edge-token sheaf inference + local stalk diffusion (no A oracle at eval)."""

    def __init__(
        self,
        d: int = 64,
        T: int = 6,
        *,
        n_heads: int = 4,  # accepted for API parity; unused (MP Φ)
        mlp_expansion: int = 12,  # ×12 lands in ±5% of FF 121218 with MP Φ
        max_nodes: int = DEFAULT_MAX_NODE_ID,
        max_T: Optional[int] = None,
        dropout: float = 0.0,
        residual_alpha: float = 1.0,  # full φ replace; 0.5 attenuates long paths
        use_tau: bool = False,  # tau broadcast breaks ‖h_t‖=0; keep off
        apply_cycle_rmsnorm: bool = False,  # True@T≥6 → NaN grads through repeated RMSNorm
        gate_theta: float = DEFAULT_GATE_THETA,
        gate_mode: str = "ste",
        gumbel_temp: float = 1.0,
        absent_bias_init: float = DEFAULT_ABSENT_BIAS,
        gate_detach_diffusion: bool = True,
    ) -> None:
        super().__init__()
        del n_heads, dropout  # MP Φ; API compat with FractalCore call sites
        if T < 1:
            raise ValueError(f"T must be >= 1, got {T}")
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}")
        if max_nodes < 1:
            raise ValueError(f"max_nodes must be >= 1, got {max_nodes}")
        if gate_mode not in ("ste", "gumbel"):
            raise ValueError(f"gate_mode must be 'ste' or 'gumbel', got {gate_mode!r}")
        self.d = int(d)
        self.T = int(T)
        self.mlp_expansion = int(mlp_expansion)
        self.max_nodes = int(max_nodes)
        self.max_T = int(max_T) if max_T is not None else max(self.T, max(DISCRETE_T_VALUES))
        self.residual_alpha = float(residual_alpha)
        self.use_tau = bool(use_tau)
        self.apply_cycle_rmsnorm = bool(apply_cycle_rmsnorm)
        self.gate_theta = float(gate_theta)
        self.gate_mode = str(gate_mode)
        self.gumbel_temp = float(gumbel_temp)
        # Detach gate into diffusion by default so CE cannot open phantom edges
        # via STE; encoder is trained by aux edge-recon BCE (train-only).
        # Set False to allow end-to-end STE/Gumbel into Φ (may soften disconnect).
        self.gate_detach_diffusion = bool(gate_detach_diffusion)
        if self.T > self.max_T:
            raise ValueError(f"T={self.T} exceeds max_T={self.max_T}")

        self.node_emb = nn.Embedding(self.max_nodes, d)
        self.src_emb = nn.Parameter(torch.zeros(d))
        self.stalk_proj = nn.Linear(d, d, bias=False)  # bias-free: 0 stays 0
        self.edge_encoder = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.absent_bias = nn.Parameter(torch.tensor(float(absent_bias_init)))
        self.phi = SheafDiffusionPhi(d, mlp_expansion=self.mlp_expansion)
        if self.use_tau:
            self.tau_emb = nn.Embedding(self.max_T, d)
        else:
            self.tau_emb = None  # type: ignore[assignment]
        self.cycle_rmsnorm = nn.RMSNorm(d) if apply_cycle_rmsnorm else None
        self.ln_f = nn.Identity()  # keep stalk magnitude; 0 stays 0
        # Readout: [z_t ; ‖z_t‖] — energy channel makes disconnect (‖h‖=0) linearly separable
        self.head = nn.Linear(d + 1, 2)
        self._init_specials()

    def _init_specials(self) -> None:
        nn.init.normal_(self.src_emb, std=0.02)
        nn.init.eye_(self.stalk_proj.weight)
        last = self.edge_encoder[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, 4.0)
        # Head: energy column separates disconnect (‖h‖=0 → y=0) from reach (‖h‖>0 → y=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.head.weight.data[0, -1] = -1.0  # high energy → not class 0
        self.head.weight.data[1, -1] = 1.0   # high energy → class 1
        # Bias alone must make y=0 (energy=0) confident: σ(2b)>0.999 → b≳3.5
        self.head.bias.data[0] = 5.0
        self.head.bias.data[1] = -5.0

    def _gather_node(self, z: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        bsz = z.shape[0]
        return z[torch.arange(bsz, device=z.device), idx]

    def encode_edge_logits(
        self,
        node_ids: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build ``E_hat`` ``[B,M,M]`` from edge tokens (row i ← col j)."""
        bsz, mlen = node_ids.shape
        device = node_ids.device
        dtype = self.edge_encoder[0].weight.dtype
        E = self.absent_bias.to(dtype=dtype) * torch.ones(
            bsz, mlen, mlen, device=device, dtype=dtype
        )

        ids_clamped = node_ids.clamp(0, self.max_nodes - 1)
        node_e = self.node_emb(ids_clamped)

        em = edge_mask.bool()
        if em.any():
            u = edge_index[..., 0].clamp(0, mlen - 1)
            v = edge_index[..., 1].clamp(0, mlen - 1)
            batch_ix = torch.arange(bsz, device=device).unsqueeze(1).expand_as(u)
            e_u = node_e[batch_ix, u]
            e_v = node_e[batch_ix, v]
            logits = self.edge_encoder(torch.cat([e_u, e_v], dim=-1)).squeeze(-1)
            E = E.clone()
            E[batch_ix[em], v[em], u[em]] = logits[em]

        if mlen > 0:
            eye = torch.eye(mlen, device=device, dtype=torch.bool)
            self_logit = torch.tensor(8.0, device=device, dtype=dtype)
            E = torch.where(eye.unsqueeze(0), self_logit, E)
            pad = node_mask == 0
            E = E.masked_fill(
                pad.unsqueeze(1) | pad.unsqueeze(2),
                float(self.absent_bias.detach().item()),
            )
        return E

    def logits_to_gate(
        self,
        E: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Hard-gate ``E`` → binary (STE) aggregation gate ``[B,M,M]``."""
        gate = ste_hard_gate(
            E,
            theta=self.gate_theta,
            training=self.training,
            mode=self.gate_mode,
            gumbel_temp=self.gumbel_temp,
        )
        pad = node_mask == 0
        gate = gate.masked_fill(pad.unsqueeze(1) | pad.unsqueeze(2), 0.0)
        return gate

    def _local_stalk(
        self,
        node_ids: torch.Tensor,
        s_idx: torch.Tensor,
        *,
        zero_stalk: bool = False,
    ) -> torch.Tensor:
        bsz, mlen = node_ids.shape
        device = node_ids.device
        dtype = self.stalk_proj.weight.dtype
        pot = torch.zeros(bsz, mlen, self.d, device=device, dtype=dtype)
        if zero_stalk:
            return pot
        batch_ix = torch.arange(bsz, device=device)
        s_ids = s_idx.clamp(0, self.max_nodes - 1)
        pot[batch_ix, s_idx] = self.stalk_proj(self.node_emb(s_ids))
        return pot

    def _initial_state(
        self,
        node_ids: torch.Tensor,
        node_mask: torch.Tensor,
        s_idx: torch.Tensor,
        *,
        zero_stalk: bool = False,
    ) -> torch.Tensor:
        bsz, mlen = node_ids.shape
        if mlen > self.max_nodes:
            raise ValueError(
                f"sequence length {mlen} exceeds max_nodes={self.max_nodes}"
            )
        device = node_ids.device
        z = self._local_stalk(node_ids, s_idx, zero_stalk=zero_stalk)
        batch_ix = torch.arange(bsz, device=device)
        if not zero_stalk:
            z = z.clone()
            z[batch_ix, s_idx] = z[batch_ix, s_idx] + self.src_emb
        if self.cycle_rmsnorm is not None:
            z = self.cycle_rmsnorm(z)
        z = z * node_mask.unsqueeze(-1).to(dtype=z.dtype)
        return z

    def _cycle_update(
        self,
        z: torch.Tensor,
        potential: torch.Tensor,
        *,
        gate: torch.Tensor,
        t: int,
    ) -> torch.Tensor:
        device = z.device
        if self.use_tau and self.tau_emb is not None:
            # Tau only on already-nonzero support (preserve ‖h_t‖=0).
            tau_t = self.tau_emb(
                torch.tensor(t, device=device, dtype=torch.long)
            ).view(1, 1, -1)
            support = (z.detach().norm(dim=-1, keepdim=True) > 0).to(dtype=z.dtype)
            h = z + tau_t * support
        else:
            h = z
        phi_out = self.phi(h, gate)
        # Classical diffusion residual: (1-α)z + α φ + potential
        z_new = z + self.residual_alpha * (phi_out - z) + potential
        if self.cycle_rmsnorm is not None:
            z_new = self.cycle_rmsnorm(z_new)
        # Bound state growth (α=1 diffusion can amplify along dense graphs).
        z_new = z_new.clamp(-50.0, 50.0)
        return z_new

    def forward(
        self,
        node_ids: torch.Tensor,
        node_mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        s_idx: torch.Tensor,
        t_idx: torch.Tensor,
        *,
        return_trajectory: bool = False,
        return_halt: bool = False,
        T: Optional[int] = None,
        zero_stalk: bool = False,
        return_states: bool = False,
        return_edge_logits: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]], Optional[dict[str, Any]]]:
        """Discrete T unroll over **inferred** gate (tokens only; no A oracle)."""
        if node_ids.dim() != 2:
            raise ValueError(f"node_ids must be [B, M], got {tuple(node_ids.shape)}")
        cycles = int(self.T if T is None else T)
        if cycles < 1:
            raise ValueError(f"T must be >= 1, got {cycles}")
        if self.use_tau and cycles > self.max_T:
            raise ValueError(f"T={cycles} exceeds tau table max_T={self.max_T}")

        E = self.encode_edge_logits(node_ids, node_mask, edge_index, edge_mask)
        gate = self.logits_to_gate(E, node_mask)
        gate_use = gate.detach() if self.gate_detach_diffusion else gate

        z = self._initial_state(node_ids, node_mask, s_idx, zero_stalk=zero_stalk)
        potential = self._local_stalk(node_ids, s_idx, zero_stalk=zero_stalk)

        trajectory: list[torch.Tensor] = []
        if return_trajectory:
            trajectory.append(self._gather_node(z, t_idx))

        for t in range(cycles):
            z = self._cycle_update(z, potential, gate=gate_use, t=t)
            z = z * node_mask.unsqueeze(-1).to(dtype=z.dtype)
            if return_trajectory:
                trajectory.append(self._gather_node(self.ln_f(z), t_idx))

        tgt = self._gather_node(self.ln_f(z), t_idx)
        energy = tgt.norm(dim=-1, keepdim=True)
        logits = self.head(torch.cat([tgt, energy], dim=-1))

        info: Optional[dict[str, Any]] = None
        if return_halt or return_states or return_edge_logits:
            bsz = node_ids.shape[0]
            info = {
                "adaptive_halt": False,
                "T": cycles,
                "discrete_T": True,
                "mean_halt_step": float(cycles),
                "mean_ponder": float(cycles),
                "local_potential": True,
                "broadcast_c": False,
                "hard_A_oracle": False,
                "gate_theta": self.gate_theta,
                "gate_mode": self.gate_mode,
                "gate_detach_diffusion": self.gate_detach_diffusion,
                "mandelbrot_analogy": MANDELBROT_ANALOGY_NOTE,
                "halt_step": torch.full(
                    (bsz,), float(cycles), device=node_ids.device, dtype=z.dtype
                ),
            }
            if return_states:
                info["final_states"] = z.detach()
                info["target_hidden"] = tgt.detach()
                info["target_l2"] = tgt.detach().norm(dim=-1)
            if return_edge_logits:
                info["edge_logits"] = E
                info["gate"] = gate.detach()

        if return_trajectory and return_halt:
            return logits, trajectory, info
        if return_trajectory:
            return logits, trajectory, info
        if return_halt:
            return logits, None, info
        return logits, None, info

    def edge_recon_accuracy(
        self,
        edge_logits: torch.Tensor,
        gold_adj: torch.Tensor,
        node_mask: torch.Tensor,
        *,
        theta: Optional[float] = None,
    ) -> dict[str, float]:
        th = self.gate_theta if theta is None else float(theta)
        pred = (torch.sigmoid(edge_logits) > th).float()
        real = (node_mask.unsqueeze(1) * node_mask.unsqueeze(2)).float()
        correct = ((pred == gold_adj).float() * real).sum()
        total = real.sum().clamp(min=1.0)
        bsz, mlen, _ = gold_adj.shape
        eye = torch.eye(mlen, device=gold_adj.device, dtype=torch.bool)
        off = real * (~eye).float()
        off_correct = ((pred == gold_adj).float() * off).sum()
        off_total = off.sum().clamp(min=1.0)
        return {
            "acc": float((correct / total).item()),
            "offdiag_acc": float((off_correct / off_total).item()),
            "n_cells": float(total.item()),
            "n_offdiag": float(off_total.item()),
        }

    def edge_recon_loss(
        self,
        edge_logits: torch.Tensor,
        gold_adj: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """BCE aux on ``E_hat`` vs gold adj (TRAIN only — not eval oracle)."""
        real = (node_mask.unsqueeze(1) * node_mask.unsqueeze(2)).float()
        bsz, mlen, _ = gold_adj.shape
        eye = torch.eye(mlen, device=gold_adj.device, dtype=torch.bool)
        weight = real * (~eye).float()
        if weight.sum() < 1:
            return edge_logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(
            edge_logits, gold_adj, weight=weight, reduction="sum"
        ) / weight.sum().clamp(min=1.0)

    def forward_from_examples(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        max_n: Optional[int] = None,
        return_trajectory: bool = False,
        return_halt: bool = False,
        T: Optional[int] = None,
    ) -> tuple[torch.Tensor, Optional[list[torch.Tensor]], Optional[dict[str, Any]]]:
        batch = build_sheaf_batch(
            examples, max_n=max_n or self.max_nodes, device=next(self.parameters()).device
        )
        return self.forward(
            batch["node_ids"],
            batch["node_mask"],
            batch["edge_index"],
            batch["edge_mask"],
            batch["s_idx"],
            batch["t_idx"],
            return_trajectory=return_trajectory,
            return_halt=return_halt,
            T=T,
        )

    @torch.no_grad()
    def disconnected_target_norm(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        T_values: Sequence[int] = DISCRETE_T_VALUES,
        max_n: Optional[int] = None,
        atol: float = DEFAULT_DISCONNECT_LEAK_ATOL,
    ) -> dict[str, Any]:
        """``‖h_t‖`` on y=0 under inferred hard gate (eval). Expect ≈0."""
        negs = [ex for ex in examples if int(ex.get("y", 1)) == 0]
        if not negs:
            return {
                "n_neg": 0,
                "ok": False,
                "reason": "no y=0 examples",
                "atol": atol,
                "science_open": False,
            }
        batch = build_sheaf_batch(
            negs, max_n=max_n or self.max_nodes, device=next(self.parameters()).device
        )
        was_training = self.training
        self.eval()
        by_T: dict[str, Any] = {}
        all_ok = True
        for T in T_values:
            _, _, info = self.forward(
                batch["node_ids"],
                batch["node_mask"],
                batch["edge_index"],
                batch["edge_mask"],
                batch["s_idx"],
                batch["t_idx"],
                T=int(T),
                return_halt=True,
                return_states=True,
                return_edge_logits=True,
            )
            assert info is not None
            l2 = info["target_l2"]
            mean_l2 = float(l2.mean().item())
            max_l2 = float(l2.max().item())
            ok_T = max_l2 <= atol
            all_ok = all_ok and ok_T
            recon = self.edge_recon_accuracy(
                info["edge_logits"], batch["gold_adj"], batch["node_mask"]
            )
            by_T[str(T)] = {
                "mean_l2": mean_l2,
                "max_l2": max_l2,
                "n": int(l2.numel()),
                "ok": bool(ok_T),
                "edge_recon_acc": recon["acc"],
                "edge_recon_offdiag_acc": recon["offdiag_acc"],
            }
        if was_training:
            self.train()
        return {
            "n_neg": len(negs),
            "atol": atol,
            "T_values": list(T_values),
            "by_T": by_T,
            "ok": bool(all_ok),
            "gate_mode": self.gate_mode,
                "gate_detach_diffusion": self.gate_detach_diffusion,
            "gate_theta": self.gate_theta,
            "note": (
                "‖h_t‖ under inferred A_hat/gate (eval hard) for disconnected "
                f"pairs; expect max_l2 <= {atol}. If miss: STE/Gumbel softening "
                "or phantom edges from encoder — document, fail-closed."
            ),
            "science_open": False,
        }

    @torch.no_grad()
    def disconnected_stalk_ablation_leak(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        T_values: Sequence[int] = DISCRETE_T_VALUES,
        max_n: Optional[int] = None,
        atol: float = DEFAULT_DISCONNECT_LEAK_ATOL,
    ) -> dict[str, Any]:
        """Stalk-ablation Δ at target for y=0 (companion invariant)."""
        negs = [ex for ex in examples if int(ex.get("y", 1)) == 0]
        if not negs:
            return {
                "n_neg": 0,
                "ok": False,
                "reason": "no y=0 examples",
                "atol": atol,
                "science_open": False,
            }
        batch = build_sheaf_batch(
            negs, max_n=max_n or self.max_nodes, device=next(self.parameters()).device
        )
        was_training = self.training
        self.eval()
        by_T: dict[str, Any] = {}
        all_ok = True
        for T in T_values:
            _, _, info_full = self.forward(
                batch["node_ids"],
                batch["node_mask"],
                batch["edge_index"],
                batch["edge_mask"],
                batch["s_idx"],
                batch["t_idx"],
                T=int(T),
                return_halt=True,
                return_states=True,
                zero_stalk=False,
            )
            _, _, info_abl = self.forward(
                batch["node_ids"],
                batch["node_mask"],
                batch["edge_index"],
                batch["edge_mask"],
                batch["s_idx"],
                batch["t_idx"],
                T=int(T),
                return_halt=True,
                return_states=True,
                zero_stalk=True,
            )
            assert info_full is not None and info_abl is not None
            delta = (info_full["target_hidden"] - info_abl["target_hidden"]).norm(
                dim=-1
            )
            mean_l2 = float(delta.mean().item())
            max_l2 = float(delta.max().item())
            ok_T = max_l2 <= atol
            all_ok = all_ok and ok_T
            by_T[str(T)] = {
                "mean_l2": mean_l2,
                "max_l2": max_l2,
                "n": int(delta.numel()),
                "ok": bool(ok_T),
            }
        if was_training:
            self.train()
        return {
            "n_neg": len(negs),
            "atol": atol,
            "T_values": list(T_values),
            "by_T": by_T,
            "ok": bool(all_ok),
            "note": (
                "Stalk-ablation L2 at target for disconnected pairs under A_hat; "
                f"expect max_l2 <= {atol}."
            ),
            "science_open": False,
        }

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def non_embedding_param_count(self) -> int:
        emb_ids = {id(p) for p in self.node_emb.parameters()}
        total = 0
        for p in self.parameters():
            if id(p) not in emb_ids:
                total += p.numel()
        return int(total)


__all__ = [
    "DEFAULT_ABSENT_BIAS",
    "DEFAULT_EDGE_RECON_WEIGHT",
    "DEFAULT_GATE_THETA",
    "SheafDiffusionPhi",
    "SheafInferCore",
    "build_sheaf_batch",
    "ste_hard_gate",
    "_verify_param_parity",
    "DISCRETE_T_VALUES",
    "DEFAULT_DISCONNECT_LEAK_ATOL",
]

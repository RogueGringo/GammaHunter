# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""AdamW + CE trainer for the geometric recurrent arm (MEASURE plumbing).

Grad clip max-norm 1.0. Optionally logs drift_trajectory from latent states.
Exposes ``last_pre_clip_grad_norm`` after each ``train_step`` for clip-sat rate.
No science OPEN claims.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from reachability_gen.models.geometric import (
    GeometricRecurrent,
    drift_from_trajectory,
    mean_z_norms_from_trajectory,
)


class GeometricTrainer:
    """Single-step AdamW trainer with CE loss and grad clip 1.0."""

    def __init__(
        self,
        model: GeometricRecurrent,
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
        """One AdamW step. Returns ``(loss, accuracy)`` as Python floats.

        Also sets ``self.last_pre_clip_grad_norm`` (total grad L2 before clip).
        """
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
        *,
        return_drift: bool = False,
    ) -> tuple[float, float] | tuple[float, float, list[float]]:
        """Eval-only CE + accuracy; optionally return drift_trajectory."""
        self.model.eval()
        token_ids = token_ids.to(self.device)
        labels = labels.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        logits, traj = self.model(
            token_ids, attention_mask, return_trajectory=return_drift
        )
        loss = self.loss_fn(logits, labels)
        preds = logits.argmax(dim=-1)
        acc = (preds == labels).float().mean()
        if return_drift:
            drifts = drift_from_trajectory(list(traj or []))
            return float(loss.item()), float(acc.item()), drifts
        return float(loss.item()), float(acc.item())

    @torch.no_grad()
    def drift_telemetry(
        self,
        token_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        eps_sigma: Optional[float] = None,
    ) -> dict:
        """Compute drift (raw + LN-normalized), terminal, z-norms, perturbation."""
        self.model.eval()
        token_ids = token_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        _, traj = self.model(token_ids, attention_mask, return_trajectory=True)
        traj_list = list(traj or [])
        drifts_raw = drift_from_trajectory(traj_list, apply_ln=False)
        drifts_ln = drift_from_trajectory(traj_list, apply_ln=True)
        drifts_rms = drift_from_trajectory(traj_list, apply_rmsnorm=True)
        z_norms = mean_z_norms_from_trajectory(traj_list)
        terminal = drifts_raw[-1] if drifts_raw else None
        terminal_ln = drifts_ln[-1] if drifts_ln else None
        terminal_rms = drifts_rms[-1] if drifts_rms else None
        perturbation_delta = None
        if eps_sigma is not None and traj_list:
            emb = self.model.tok_emb(token_ids)
            noise = torch.randn_like(emb) * float(eps_sigma)
            pos = torch.arange(
                token_ids.shape[1], device=token_ids.device
            ).unsqueeze(0).expand(token_ids.shape[0], -1)
            noisy = emb + noise + self.model.pos_emb(pos)
            if getattr(self.model, "cycle_rmsnorm", None) is not None:
                noisy = self.model.cycle_rmsnorm(noisy)
            elif self.model.cycle_ln is not None:
                noisy = self.model.cycle_ln(noisy)
            clean_z0 = traj_list[0]
            if attention_mask is not None:
                mask = attention_mask.to(dtype=noisy.dtype).unsqueeze(-1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                noisy_z0 = (noisy * mask).sum(dim=1) / denom
            else:
                noisy_z0 = noisy.mean(dim=1)
            delta = (noisy_z0 - clean_z0).float().reshape(clean_z0.shape[0], -1)
            norms = torch.linalg.vector_norm(delta, ord=2, dim=-1)
            perturbation_delta = float(norms.mean().item())
        return {
            "drift_trajectory": drifts_raw,
            "drift_trajectory_ln": drifts_ln,
            "drift_trajectory_rms": drifts_rms,
            "terminal_drift": terminal,
            "terminal_drift_ln": terminal_ln,
            "terminal_drift_rms": terminal_rms,
            "mean_z_norm_by_t": z_norms,
            "perturbation_delta": perturbation_delta,
            "trajectory_len": len(traj_list),
            "residual_alpha": float(getattr(self.model, "residual_alpha", 0.5)),
            "apply_cycle_ln": bool(getattr(self.model, "apply_cycle_ln", False)),
            "apply_cycle_rmsnorm": bool(
                getattr(self.model, "apply_cycle_rmsnorm", False)
            ),
            "drift_formula": (
                "δ_t = mean_batch ||z_{t+1}-z_t||_2 on pooled latents; "
                "z_{t+1}=bound(z_t+α·(Φ(h_t)-z_t)) with α="
                f"{getattr(self.model, 'residual_alpha', 0.5)}; "
                "bound=RMSNorm if apply_cycle_rmsnorm else identity "
                "(LN stream blocked learning); also report LN- and "
                "RMS-normalized drift metrics"
            ),
        }


__all__ = ["GeometricTrainer"]

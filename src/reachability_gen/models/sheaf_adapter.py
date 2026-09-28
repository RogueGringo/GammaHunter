# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""SheafInferCore in GammaHunter's rings: packing, neutral init, one interface (MEASURE).

The ported arm (``models.sheaf_infer_core``) sets several weights by hand at
construction: identity message and output maps, a zeroed residual MLP, every
listed edge gated on (edge bias +4, absent-edge prior -4) and a head that
answers "reachable" exactly when the target's state is non-zero. With those
values it computes T-step reachability before any training.
:func:`neutral_init` replaces each of them with PyTorch's default
initialisation (absent-edge prior 0), so what the arm reaches afterwards has
to be learned.

``science_open=false`` always.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch

from reachability_gen.models.sheaf_infer_core import SheafInferCore, build_sheaf_batch


def neutral_init(model: SheafInferCore) -> SheafInferCore:
    """Replace every hand-set initial value with PyTorch's default initialisation."""
    for layer in (
        model.stalk_proj,
        model.edge_encoder[-1],
        model.phi.W_msg,
        model.phi.mlp_up,
        model.phi.mlp_down,
        model.phi.W_out,
        model.head,
    ):
        layer.reset_parameters()
    with torch.no_grad():
        model.absent_bias.zero_()
    return model


def pack_sheaf(rows: Sequence[dict[str, Any]], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """``build_sheaf_batch`` for a whole set at once (built on the CPU, then moved)."""
    width = max(int(r["n"]) for r in rows)
    batch = build_sheaf_batch(rows, max_n=width)
    return {k: v.to(device) for k, v in batch.items()}


def sheaf_logits(
    model: SheafInferCore,
    batch: dict[str, torch.Tensor],
    steps: int,
    *,
    with_aux: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Logits ``[B, 2]`` and, when asked, the arm's edge-reconstruction loss."""
    logits, _, info = model(
        batch["node_ids"], batch["node_mask"], batch["edge_index"], batch["edge_mask"],
        batch["s_idx"], batch["t_idx"], T=steps, return_edge_logits=with_aux,
    )
    if not with_aux:
        return logits, None
    assert info is not None
    return logits, model.edge_recon_loss(info["edge_logits"], batch["gold_adj"], batch["node_mask"])


def sheaf_target_states(model: SheafInferCore, batch: dict[str, torch.Tensor], steps: int) -> list[torch.Tensor]:
    """The query target's state after each step, ``[z_0[t], …, z_T[t]]``."""
    _, trajectory, _ = model(
        batch["node_ids"], batch["node_mask"], batch["edge_index"], batch["edge_mask"],
        batch["s_idx"], batch["t_idx"], T=steps, return_trajectory=True,
    )
    assert trajectory is not None
    return trajectory


__all__ = ["neutral_init", "pack_sheaf", "sheaf_logits", "sheaf_target_states"]

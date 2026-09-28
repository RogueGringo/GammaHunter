# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Weight-only 8-bit linear layers, so larger reference LLMs fit one GPU (MEASURE plumbing).

Each weight row is scaled by its absolute maximum and stored as int8 (per
output channel); the layer dequantizes to the activation dtype inside every
call, so memory holds one byte per weight while the arithmetic stays in 16-bit
floating point. No quantization library is needed. Fidelity is checked
against the 16-bit model on the same prompts (``run_llm_reference --int8``
records it next to the results).

``science_open=false`` always.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Int8Linear(nn.Module):
    """``nn.Linear`` with int8 weights and one bf16 scale per output channel."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        weight = linear.weight.detach().float()
        scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        self.register_buffer("qweight", torch.round(weight / scale).clamp(-127, 127).to(torch.int8))
        self.register_buffer("scale", scale.to(linear.weight.dtype))
        self.bias: Optional[nn.Parameter] = (
            nn.Parameter(linear.bias.detach().clone(), requires_grad=False) if linear.bias is not None else None
        )
        self.in_features, self.out_features = linear.in_features, linear.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scaled in place: one full-size temporary instead of two (the same
        # multiplication, so the result is identical to qweight * scale).
        weight = self.qweight.to(x.dtype, copy=True).mul_(self.scale.to(x.dtype))
        return F.linear(x, weight, None if self.bias is None else self.bias.to(x.dtype))


def quantize_linears(model: nn.Module) -> int:
    """Replace every ``nn.Linear`` in ``model`` with ``Int8Linear``, in place; return the count."""
    count = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            setattr(model, name, Int8Linear(child))
            count += 1
        else:
            count += quantize_linears(child)
    return count


__all__ = ["Int8Linear", "quantize_linears"]

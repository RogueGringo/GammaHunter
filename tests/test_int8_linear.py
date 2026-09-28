# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the weight-only 8-bit linear layer."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.int8_linear import Int8Linear, quantize_linears  # noqa: E402


def test_int8_linear_tracks_the_float_layer():
    torch.manual_seed(0)
    linear = torch.nn.Linear(64, 32)
    q = Int8Linear(linear)
    x = torch.randn(8, 64)
    ref, out = linear(x), q(x)
    assert q.qweight.dtype == torch.int8 and q.qweight.shape == (32, 64)
    assert (out - ref).abs().max() / ref.abs().max() < 0.01  # per-row absmax: well under 1%


def test_quantize_linears_replaces_nested_layers_in_place():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(),
                                torch.nn.Sequential(torch.nn.Linear(8, 4, bias=False)))
    x = torch.randn(3, 8)
    ref = model(x)
    assert quantize_linears(model) == 2
    assert isinstance(model[0], Int8Linear) and isinstance(model[2][0], Int8Linear)
    assert model[2][0].bias is None
    assert torch.allclose(model(x), ref, atol=0.05)

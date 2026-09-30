# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""What each selection normalizer is, independent of any implementation.

All map scores z (over the unmasked positions of a row) to a probability vector p:

* ``softmax`` (α = 1): p_i = exp(z_i) / Σ_j exp(z_j). Every unmasked position gets
  weight; appending positions dilutes the others.
* ``entmax15`` (α = 1.5): p_i = [z_i/2 − τ]_+², with τ such that Σ p_i = 1.
  Exact zeros below the threshold.
* ``sparsemax`` (α = 2): p_i = [z_i − τ]_+, with τ such that Σ p_i = 1: the
  Euclidean projection of z onto the simplex. Exact zeros below the threshold.
* ``ssmax``: softmax of s · ln(n) · z, where n is the number of unmasked
  positions in the row and s a learned scale per attention head (scalable
  softmax); it is dense like softmax.

Masked positions get exactly 0 under every normalizer. For α > 1, positions whose
score lies at or below the threshold get exactly 0, and appending such positions
changes nothing; the gradient with respect to a zero-weight position is 0.

Analytic Jacobian-vector products (g the incoming gradient, S the support):
sparsemax: s ⊙ (g − mean_S g), with s the support indicator;
entmax15: √p ⊙ g − √p · Σ(√p ⊙ g) / Σ√p.

Sources: sparsemax and its Jacobian, Martins and Astudillo (2016, arXiv
1602.02068); α-entmax, the exact sort-based algorithm for α = 1.5 and the
Jacobian diag(s) − s sᵀ / Σ s with s = p^(2−α) on the support, Peters, Niculae
and Martins (2019, arXiv 1905.05702, Algorithm 2 and Proposition 1); scalable
softmax, Nakanishi (2025, arXiv 2501.19399).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizerSpec:
    name: str
    alpha: float  # Tsallis α; 1 is softmax
    sparse: bool  # exact zeros below a threshold
    learned_scale: bool  # carries a learned per-head parameter
    formula: str


NORMALIZERS: dict[str, NormalizerSpec] = {
    "softmax": NormalizerSpec("softmax", 1.0, False, False, "p_i = exp(z_i) / sum_j exp(z_j)"),
    "entmax15": NormalizerSpec("entmax15", 1.5, True, False, "p_i = [z_i/2 - tau]_+^2, sum_i p_i = 1"),
    "sparsemax": NormalizerSpec("sparsemax", 2.0, True, False, "p_i = [z_i - tau]_+, sum_i p_i = 1"),
    "ssmax": NormalizerSpec("ssmax", 1.0, False, True, "softmax(s * ln(n) * z), n unmasked positions"),
}
# SSMax's scale, one per attention layer and head, starts at 1: the recipe of
# Nakanishi (2025, arXiv 2501.19399) for training with SSMax from the start (that
# paper starts it at about 1 / mean ln n only when replacing softmax after
# training). The paper's optional bias is not used (fixed before any run).
SSMAX_INIT_SCALE: float = 1.0

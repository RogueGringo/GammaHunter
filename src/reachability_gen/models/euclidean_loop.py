# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Real torch Euclidean loop arm: weight-tied Phi(z_t, c) with no τ.

Same architecture as :class:`GeometricRecurrent` but τ disabled / zeroed
(``use_tau=False``). Used as the Euclidean control in the id_2k rematch.
Inherits residual α=0.5 + optional outer RMSNorm/LN bound from GeometricRecurrent.

RESEARCH / MEASURE plumbing only — no science OPEN claims.
"""

from __future__ import annotations

from typing import Optional

from reachability_gen.models.geometric import (
    DEFAULT_RESIDUAL_ALPHA,
    GeometricRecurrent,
    drift_from_trajectory,
    mean_z_norms_from_trajectory,
    trajectory_finite_nonzero,
)


class EuclideanLoop(GeometricRecurrent):
    """Weight-tied Phi reused T times → logits ``[B, 2]``; no cycle τ.

    Parameters mirror :class:`GeometricRecurrent` except ``use_tau`` is always
    forced off (any caller-supplied ``use_tau`` is ignored).
    """

    def __init__(
        self,
        vocab_size: int,
        d: int = 64,
        T: int = 6,
        *,
        n_heads: int = 4,
        max_len: int = 256,
        mlp_expansion: int = 4,
        pad_id: int = 0,
        dropout: float = 0.0,
        max_T: Optional[int] = None,
        use_tau: bool = False,  # accepted for API parity; always forced False
        residual_alpha: float = DEFAULT_RESIDUAL_ALPHA,
        apply_cycle_ln: bool = False,
        apply_cycle_rmsnorm: bool = False,
        readout: str = "mean",
    ) -> None:
        del use_tau  # Euclidean loop never uses τ
        super().__init__(
            vocab_size,
            d=d,
            T=T,
            n_heads=n_heads,
            max_len=max_len,
            mlp_expansion=mlp_expansion,
            pad_id=pad_id,
            dropout=dropout,
            use_tau=False,
            max_T=max_T,
            residual_alpha=residual_alpha,
            apply_cycle_ln=apply_cycle_ln,
            apply_cycle_rmsnorm=apply_cycle_rmsnorm,
            readout=readout,
        )


__all__ = [
    "EuclideanLoop",
    "drift_from_trajectory",
    "mean_z_norms_from_trajectory",
    "trajectory_finite_nonzero",
]

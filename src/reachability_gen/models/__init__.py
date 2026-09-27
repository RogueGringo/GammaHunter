# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Optional torch models (RESEARCH / MEASURE plumbing)."""

from __future__ import annotations

__all__: list[str] = []

try:
    from reachability_gen.models.feedforward import FeedForward, TransformerBlock

    __all__ += ["FeedForward", "TransformerBlock"]
except ImportError:  # pragma: no cover - torch optional
    pass

try:
    from reachability_gen.models.geometric import (
        DEFAULT_RESIDUAL_ALPHA,
        GeometricRecurrent,
        drift_from_trajectory,
        mean_z_norms_from_trajectory,
        trajectory_finite_nonzero,
    )

    __all__ += [
        "DEFAULT_RESIDUAL_ALPHA",
        "GeometricRecurrent",
        "drift_from_trajectory",
        "mean_z_norms_from_trajectory",
        "trajectory_finite_nonzero",
    ]
except ImportError:  # pragma: no cover - torch optional
    pass

try:
    from reachability_gen.models.euclidean_loop import EuclideanLoop

    __all__ += ["EuclideanLoop"]
except ImportError:  # pragma: no cover - torch optional
    pass


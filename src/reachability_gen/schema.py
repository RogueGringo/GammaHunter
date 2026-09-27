# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Logging / example schema for reachability JSONL rows + RunMetricRecord.

``RunMetricRecord`` required keys match the user TypedDict exactly (ADR-001 §7).
Example-identity linkage fields are optional documented extras.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, List, NotRequired, Optional, TypedDict

from reachability_gen.adr_invariants import (
    HOP_UNREACHABLE,
    REQUIRED_METRIC_KEYS,
    validate_metric_record,
)

# Back-compat aliases (prefer REQUIRED_METRIC_KEYS / validate_metric_record).
RUN_METRIC_REQUIRED_KEYS = REQUIRED_METRIC_KEYS
validate_run_metric_record = validate_metric_record


@dataclass
class ReachabilityExample:
    """One labeled reachability instance + logging fields from the v0 / ADR-001 spec.

    Required fields: split, seed, n, p, edge_hash, s, t, y, hop_distance, is_ood.
    Optional arm fields are left empty / None for this scaffold.
    """

    split: str
    seed: int
    n: int
    p: float
    edge_hash: str
    s: int
    t: int
    y: int
    # Shortest-path hop distance; HOP_UNREACHABLE (-1) when y=0 (ADR-001).
    hop_distance: int = HOP_UNREACHABLE
    # hop_distance > K_TRAIN_MAX; False when unreachable.
    is_ood: bool = False
    # Locked v0 encoding string (edge-list + query markers); not a scratchpad.
    encoding: str = ""
    # Optional experiment-arm placeholders (left empty by default).
    arm_id: Optional[str] = None
    arm_meta: Optional[dict[str, Any]] = field(default=None)
    # Diagnostics recorded by the generator (not required for training consumers).
    reject_rate: Optional[float] = None
    n_attempts: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RunMetricRecord(TypedDict):
    """Per-example / per-step eval logging schema locked by ADR-001 §7.

    Required keys = user TypedDict exactly. Optional extras (example linkage,
    hygiene) are NotRequired and documented in the ADR.
    """

    # --- required (user TypedDict) ---
    run_id: str
    seed: int
    arm: str
    step: int
    epoch: int
    param_count: int
    d_model: int
    seq_len: int
    cycles_or_depth: int
    tokens_decoded: Optional[int]
    cumulative_flops: float
    hop_distance: int
    is_ood: bool
    loss: float
    accuracy: float
    drift_trajectory: List[float]
    terminal_drift: Optional[float]
    perturbation_delta: Optional[float]
    # --- optional documented extras (example linkage / hygiene) ---
    split: NotRequired[str]
    n: NotRequired[int]
    p: NotRequired[float]
    edge_hash: NotRequired[str]
    s: NotRequired[int]
    t: NotRequired[int]
    y: NotRequired[int]
    science_open: NotRequired[bool]


# Documented optional keys (not required by validate_metric_record).
OPTIONAL_METRIC_EXTRA_KEYS: frozenset[str] = frozenset(
    {
        "split",
        "n",
        "p",
        "edge_hash",
        "s",
        "t",
        "y",
        "science_open",
    }
)

__all__ = [
    "OPTIONAL_METRIC_EXTRA_KEYS",
    "REQUIRED_METRIC_KEYS",
    "RUN_METRIC_REQUIRED_KEYS",
    "ReachabilityExample",
    "RunMetricRecord",
    "validate_metric_record",
    "validate_run_metric_record",
]

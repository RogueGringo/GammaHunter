# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Non-embedding parameter matching across Geo / FF / Euclidean-loop arms.

Testbed rule: match within ±5% non-embedding params across Geometric,
Feed-forward, and Euclidean-loop. CoT may differ and is excluded from the
match gate (but must still report FLOPs).

This helper returns pass/fail + ratios. It never stamps science OPEN —
matching params is a precondition check for fair MEASURE runs, not a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Union

from reachability_gen.adr_invariants import PARAM_TOL
from reachability_gen.arms import (
    PARAM_MATCH_TOLERANCE,
    Arm,
    EuclideanLoopArm,
    FeedForwardArm,
    GeometricRecurrentArm,
)

# Alias locked ADR constant (same value as PARAM_MATCH_TOLERANCE).
assert abs(PARAM_MATCH_TOLERANCE - PARAM_TOL) < 1e-15

ArmLike = Union[Arm, Any]


@dataclass(frozen=True)
class ParamMatchResult:
    """Outcome of a ±tolerance non-embedding param check."""

    passed: bool
    tolerance: float
    counts: dict[str, int]
    ratios: dict[str, float]  # each arm / reference
    reference_arm: str
    reference_count: int
    notes: str = ""
    # Explicit: this check is scaffolding hygiene, not an experimental claim.
    science_open: bool = field(default=False, init=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "tolerance": self.tolerance,
            "counts": dict(self.counts),
            "ratios": dict(self.ratios),
            "reference_arm": self.reference_arm,
            "reference_count": self.reference_count,
            "notes": self.notes,
            "science_open": self.science_open,  # always False
        }


def _arm_label(arm: ArmLike) -> str:
    name = getattr(arm, "name", None)
    if isinstance(name, str) and name:
        return name
    return type(arm).__name__


def _param_count(arm: ArmLike) -> int:
    if hasattr(arm, "param_count") and callable(arm.param_count):
        return int(arm.param_count())
    raise TypeError(f"object {arm!r} has no param_count()")


def check_param_match(
    arms: Sequence[ArmLike],
    *,
    tolerance: float = PARAM_MATCH_TOLERANCE,
    reference: Optional[ArmLike] = None,
    exclude_names: Optional[Sequence[str]] = None,
) -> ParamMatchResult:
    """Check that non-embedding params lie within ±tolerance of a reference.

    Parameters
    ----------
    arms :
        Arms to compare (typically Geo, FF, Loop). CoT stubs may be passed
        but should be listed in ``exclude_names`` or omitted.
    tolerance :
        Relative half-width (default 0.05 => ±5%).
    reference :
        Arm whose count is the denominator. Defaults to the first arm.
    exclude_names :
        Substring / exact name filters to skip (e.g. ``("cot",)``).

    Returns
    -------
    ParamMatchResult
        ``science_open`` is always False — never self-stamps OPEN.
    """
    if tolerance < 0:
        raise ValueError(f"tolerance must be non-negative, got {tolerance}")
    if not arms:
        return ParamMatchResult(
            passed=False,
            tolerance=tolerance,
            counts={},
            ratios={},
            reference_arm="",
            reference_count=0,
            notes="no arms provided",
        )

    exclude = tuple(exclude_names or ())
    selected: list[ArmLike] = []
    for arm in arms:
        label = _arm_label(arm).lower()
        if any(ex.lower() in label for ex in exclude):
            continue
        selected.append(arm)

    if not selected:
        return ParamMatchResult(
            passed=False,
            tolerance=tolerance,
            counts={},
            ratios={},
            reference_arm="",
            reference_count=0,
            notes="all arms excluded",
        )

    ref = reference if reference is not None else selected[0]
    ref_label = _arm_label(ref)
    ref_count = _param_count(ref)
    if ref_count <= 0:
        return ParamMatchResult(
            passed=False,
            tolerance=tolerance,
            counts={_arm_label(a): _param_count(a) for a in selected},
            ratios={},
            reference_arm=ref_label,
            reference_count=ref_count,
            notes="reference param_count must be > 0",
        )

    counts: dict[str, int] = {}
    ratios: dict[str, float] = {}
    failed: list[str] = []
    for arm in selected:
        label = _arm_label(arm)
        c = _param_count(arm)
        counts[label] = c
        ratio = c / ref_count
        ratios[label] = ratio
        if abs(ratio - 1.0) > tolerance + 1e-15:
            failed.append(f"{label}: ratio={ratio:.6f}")

    passed = len(failed) == 0
    notes = (
        "within ±{:.0%} of {}".format(tolerance, ref_label)
        if passed
        else "mismatch: " + "; ".join(failed)
    )
    return ParamMatchResult(
        passed=passed,
        tolerance=tolerance,
        counts=counts,
        ratios=ratios,
        reference_arm=ref_label,
        reference_count=ref_count,
        notes=notes,
    )


def check_geo_ff_loop(
    geo: GeometricRecurrentArm,
    ff: FeedForwardArm,
    loop: EuclideanLoopArm,
    *,
    tolerance: float = PARAM_MATCH_TOLERANCE,
) -> ParamMatchResult:
    """Convenience: match Geo / FF / Loop with Geo as reference."""
    return check_param_match(
        [geo, ff, loop],
        tolerance=tolerance,
        reference=geo,
    )


def ratios_from_counts(
    counts: Mapping[str, int],
    *,
    reference_key: str,
) -> dict[str, float]:
    """Compute count / reference ratios from a plain dict of param counts."""
    ref = counts[reference_key]
    if ref <= 0:
        raise ValueError(f"reference count must be > 0, got {ref}")
    return {k: v / ref for k, v in counts.items()}

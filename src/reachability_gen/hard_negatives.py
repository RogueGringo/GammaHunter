# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Hard-negative filters for unreachable (y=0) reachability examples.

A *hard negative* is an unreachable query ``(s, t)`` where both endpoints have
total degree ≥ 1 (in + out). Isolated endpoints (deg=0) are discarded — they are
trivial "no edges touch this node" cues rather than directed dead-ends / separate
weakly-relevant components.

RESEARCH / MEASURE plumbing only — no science OPEN claims.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Optional, Sequence

from reachability_gen.encode import parse_instance
from reachability_gen.graph import reachable_bfs


def total_degrees(n: int, edges: Sequence[tuple[int, int]]) -> list[int]:
    """Total degree (in + out) per node; self-loops are forbidden upstream."""
    deg = [0] * n
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u},{v}) out of range for n={n}")
        deg[u] += 1
        deg[v] += 1
    return deg


def is_hard_negative(
    n: int,
    edges: Sequence[tuple[int, int]],
    s: int,
    t: int,
    *,
    y: Optional[int] = None,
) -> tuple[bool, str]:
    """Return ``(ok, reason)`` for hard-negative acceptance.

    Reasons when rejected:
    - ``not_y0`` — label is not unreachable
    - ``reachable`` — s can reach t (should not happen for true y=0)
    - ``deg_s0`` / ``deg_t0`` / ``both_deg0`` — endpoint degree filter
    - ``ok`` — accepted
    """
    if y is not None and int(y) != 0:
        return False, "not_y0"
    if not (0 <= s < n and 0 <= t < n):
        return False, "query_oor"
    if reachable_bfs(n, edges, s, t):
        return False, "reachable"
    deg = total_degrees(n, edges)
    ds, dt = deg[s], deg[t]
    if ds == 0 and dt == 0:
        return False, "both_deg0"
    if ds == 0:
        return False, "deg_s0"
    if dt == 0:
        return False, "deg_t0"
    return True, "ok"


def classify_y0_row(row: Mapping[str, Any]) -> tuple[bool, str]:
    """Classify a JSONL-like row as hard-negative or reject with a reason code."""
    y = int(row.get("y", -1))
    if y != 0:
        return False, "not_y0"
    encoding = row.get("encoding")
    if encoding:
        n, edges, s, t = parse_instance(str(encoding))
    else:
        raise ValueError("row missing encoding; cannot compute degrees")
    # Prefer row s/t if present (should match encoding).
    s = int(row.get("s", s))
    t = int(row.get("t", t))
    n = int(row.get("n", n))
    return is_hard_negative(n, edges, s, t, y=y)


def filter_hard_negatives(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], Counter]:
    """Keep y=0 hard negatives; return ``(accepted, reject_reason_counts)``.

    ``reject_reason_counts`` includes ``ok`` for accepted rows plus reject codes.
    """
    reasons: Counter = Counter()
    accepted: list[dict[str, Any]] = []
    for row in rows:
        ok, reason = classify_y0_row(row) if int(row.get("y", -1)) == 0 else (False, "not_y0")
        # Only tally y=0 rows under degree/reachability reasons; skip non-y0 in
        # the reject log unless caller wants full-pass counts.
        if int(row.get("y", -1)) != 0:
            reasons["skipped_not_y0"] += 1
            continue
        reasons[reason] += 1
        if ok:
            accepted.append(dict(row))
    return accepted, reasons


__all__ = [
    "classify_y0_row",
    "filter_hard_negatives",
    "is_hard_negative",
    "total_degrees",
]

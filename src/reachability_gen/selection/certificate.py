# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Framework-free KKT certificates: grade any kernel's output without trusting any implementation.

Only the standard library is used; inputs are plain sequences of floats. Given
the scores z and a kernel's output p for one row, the support S is read from p
(p_i > 0), the threshold τ that S implies is computed exactly, and the
optimality conditions are checked:

* sparsemax, exactly in ``Fraction``: τ = (Σ_S z_i − 1) / |S|, then z_i > τ on S
  and z_j ≤ τ off S; the exact solution p*_i = z_i − τ sums to 1 by the choice
  of τ, and the kernel's values must match p* within ``tol``;
* entmax-1.5: with x = z/2, τ solves Σ_S (x_i − τ)² = 1 with τ below every x_i
  on S, in ``Fraction`` up to one square root taken in ``Decimal`` at 60 digits;
  then x_i > τ on S and x_j ≤ τ off S (to 1e-40), and the kernel's values must
  match (x_i − τ)² within ``tol``.

For both, the kernel's own output must sum to 1 within ``tol``.

Masked positions must be exactly 0 in p. Every float is converted exactly
(``Fraction(float)``), so the only rounding in a sparsemax certificate is the
kernel's own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Optional, Sequence

DECIMAL_DIGITS: int = 60
BOUNDARY_SLACK = Decimal("1e-40")


@dataclass
class Certificate:
    normalizer: str
    passed: bool
    checks: dict[str, bool]
    support: list[int]
    tau: str
    max_value_error: float
    notes: list[str] = field(default_factory=list)


def _common(z: Sequence[float], p: Sequence[float], mask: Optional[Sequence[bool]]):
    if len(z) != len(p) or (mask is not None and len(mask) != len(z)):
        raise ValueError("z, p and mask must have the same length")
    valid = [True] * len(z) if mask is None else [bool(m) for m in mask]
    checks = {
        "masked_exactly_zero": all(p[i] == 0 for i in range(len(z)) if not valid[i]),
        "nonnegative": all(pi >= 0 for pi in p),
    }
    support = [i for i in range(len(z)) if valid[i] and p[i] > 0]
    return valid, checks, support


def certify_sparsemax(z: Sequence[float], p: Sequence[float], mask: Optional[Sequence[bool]] = None,
                      tol: float = 1e-6) -> Certificate:
    valid, checks, support = _common(z, p, mask)
    if not support:
        return Certificate("sparsemax", False, {**checks, "nonempty_support": False}, [], "undefined", float("inf"))
    zf = [Fraction(float(v)) for v in z]
    tau = (sum(zf[i] for i in support) - 1) / len(support)
    exact = {i: zf[i] - tau for i in support}
    checks["support_above_threshold"] = all(zf[i] > tau for i in support)
    checks["off_support_at_or_below_threshold"] = all(
        zf[j] <= tau for j in range(len(z)) if valid[j] and j not in exact)
    checks["kernel_sums_to_one"] = abs(sum(Fraction(float(p[i])) for i in range(len(z)) if valid[i]) - 1) <= Fraction(tol)
    err = max(abs(Fraction(float(p[i])) - exact[i]) for i in support)
    checks["values_match_exact_solution"] = err <= Fraction(tol)
    return Certificate("sparsemax", all(checks.values()), checks, support, str(tau), float(err))


def certify_entmax15(z: Sequence[float], p: Sequence[float], mask: Optional[Sequence[bool]] = None,
                     tol: float = 1e-6) -> Certificate:
    valid, checks, support = _common(z, p, mask)
    if not support:
        return Certificate("entmax15", False, {**checks, "nonempty_support": False}, [], "undefined", float("inf"))
    x = [Fraction(float(v)) / 2 for v in z]
    k = len(support)
    a = sum(x[i] for i in support)
    b = sum(x[i] * x[i] for i in support)
    disc = a * a - k * (b - 1)
    checks["threshold_exists"] = disc >= 0
    if disc < 0:
        return Certificate("entmax15", False, checks, support, "undefined", float("inf"))
    with localcontext() as ctx:
        ctx.prec = DECIMAL_DIGITS
        root = (Decimal(disc.numerator) / Decimal(disc.denominator)).sqrt()
        tau = (Decimal(a.numerator) / Decimal(a.denominator) - root) / k
        xd = {i: Decimal(x[i].numerator) / Decimal(x[i].denominator) for i in range(len(z))}
        checks["support_above_threshold"] = all(xd[i] - tau > 0 for i in support)
        checks["off_support_at_or_below_threshold"] = all(
            xd[j] <= tau + BOUNDARY_SLACK for j in range(len(z)) if valid[j] and j not in support)
        exact = {i: (xd[i] - tau) ** 2 for i in support}
        checks["kernel_sums_to_one"] = abs(sum(Decimal(float(p[i])) for i in range(len(z)) if valid[i]) - 1) <= Decimal(tol)
        err = max(abs(Decimal(float(p[i])) - exact[i]) for i in support)
        checks["values_match_exact_solution"] = err <= Decimal(tol)
        return Certificate("entmax15", all(checks.values()), checks, support, str(tau), float(err))


CERTIFIERS = {"sparsemax": certify_sparsemax, "entmax15": certify_entmax15}

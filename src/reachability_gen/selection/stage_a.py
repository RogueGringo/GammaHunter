# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Stage A: how much margin one relevant position needs against n distractors (MEASURE, CPU, float64).

For each normalizer, the smallest logit m at which the relevant position keeps
at least half the weight, found by bisection, against n distractors that are
either bounded (all at logit 0) or drawn from N(0, 1) (five seeded draws, every
draw kept). Also: the relevant position's weight when positions far below the
threshold are appended. SSMax is graded at scale 1, so it normalizes with
ln(n + 1) for n distractors and the relevant position. Predictions and
tolerances fixed before any run:

* bounded distractors, exact to 1e-6: softmax ln n; sparsemax (n − 1)/(2n),
  below ½ for every n; entmax-1.5 √2 · (1 − n^(−½)), below √2 for every n;
  SSMax ln n / ln(n + 1), below 1. The bisection itself resolves the margin to
  float64 precision; the tolerance covers the backends' own rounding, which at
  n = 100,000 tied distractors can reach a few times 1e-9 for entmax-1.5 by a
  float64 error estimate (its variance term is a difference of nearly equal
  running sums) and far less for the others;
* N(0, 1) distractors, for n ≥ 1,000: softmax within 0.1 of ln n + ½ (the log of
  the expected sum of exp z); sparsemax within 0.5 of τ + ½, where τ solves
  n · E[(Z − τ)₊] = ½ (half the weight left to the distractors, in
  expectation); smaller n reported, not tested; √(2 ln n), sparsemax's
  large-n rate, reported beside it;
* appending sub-threshold positions changes nothing for α > 1 and dilutes
  softmax.

Figures supplied with the study's specification are recorded as claims beside
the measured values (``claimed``) and count as reproduced when the figure lies
within the range of the five draws widened by the figure's own rounding
(0.05): that range contains the median of the draws' distribution with
probability 1 − 2⁻⁴ = 0.9375, whereas a fixed band around the median of five
draws would be decided mostly by draw noise where the draws spread widely
(sparsemax's margin at n = 100,000 varies by about 0.25 between draws).

Disclosure: the module was smoke-tested at n ≤ 1,000 during development (output
not kept) before the finite-n Gaussian predictions and their tolerances were
written; the test suite runs it at n ≤ 100.

``science_open=false`` always.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch

from .reference import BACKENDS

NS: tuple[int, ...] = (1, 10, 100, 1_000, 10_000, 100_000)
DRAWS: int = 5
SEED: int = 2027
APPENDED: tuple[int, ...] = (10, 100, 1_000)
KINDS: tuple[str, ...] = ("softmax", "ssmax", "entmax15", "sparsemax")
EXACT_TOL: float = 1e-6
GAUSSIAN_MIN_N: int = 1_000
SOFTMAX_GAUSSIAN_TOL: float = 0.1
SPARSEMAX_GAUSSIAN_TOL: float = 0.5
CLAIM_TOL: float = 0.05
# Supplied with the study's specification: the margin needed at n = 100,000 N(0, 1) distractors.
CLAIMS: dict[str, dict[str, float]] = {"gaussian_100000": {"softmax": 12.0, "sparsemax": 4.5}}


def relevant_weight(fn: Callable[..., torch.Tensor], m: float, distractors: torch.Tensor) -> float:
    z = torch.cat([torch.tensor([m], dtype=torch.float64), distractors])
    with torch.no_grad():
        return float(fn(z[None])[0, 0])


def margin_needed(fn: Callable[..., torch.Tensor], distractors: torch.Tensor, target: float = 0.5,
                  lo: float = -10.0, hi: float = 60.0, iters: int = 80) -> float:
    """Smallest m with weight ≥ ``target`` (the weight rises with m), by bisection."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if relevant_weight(fn, mid, distractors) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def claim_reproduced(claim: float, draws: Sequence[float]) -> bool:
    """The claimed figure within the range of the draws, widened by the figure's rounding."""
    return min(draws) - CLAIM_TOL <= claim <= max(draws) + CLAIM_TOL


def gaussian_excess(tau: float) -> float:
    """E[(Z − τ)₊] for Z ~ N(0, 1)."""
    pdf = math.exp(-tau * tau / 2) / math.sqrt(2 * math.pi)
    tail = 0.5 * math.erfc(tau / math.sqrt(2))
    return pdf - tau * tail


def sparsemax_gaussian_prediction(n: int) -> float:
    """τ + ½ with n · E[(Z − τ)₊] = ½, by bisection (the excess falls as τ rises)."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if n * gaussian_excess(mid) > 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2 + 0.5


def bounded_prediction(kind: str, n: int) -> float:
    if kind == "softmax":
        return math.log(n)
    if kind == "sparsemax":
        return (n - 1) / (2 * n)
    if kind == "entmax15":
        return math.sqrt(2) * (1 - n ** -0.5)
    return math.log(n) / math.log(n + 1)  # ssmax at scale 1


def run(ns: Sequence[int] = NS, draws: int = DRAWS, seed: int = SEED) -> dict[str, Any]:
    gen = torch.Generator().manual_seed(seed)
    gaussian = {n: [torch.randn(n, generator=gen, dtype=torch.float64) for _ in range(draws)] for n in ns}
    out: dict[str, Any] = {"ns": list(ns), "draws": draws, "seed": seed, "bounded": {}, "gaussian": {},
                           "appended": {}}
    for kind in KINDS:
        fn = BACKENDS[kind]
        out["bounded"][kind] = {str(n): margin_needed(fn, torch.zeros(n, dtype=torch.float64)) for n in ns}
        out["gaussian"][kind] = {}
        for n in ns:
            margins = [margin_needed(fn, d) for d in gaussian[n]]
            out["gaussian"][kind][str(n)] = {"per_draw": margins, "median": statistics.median(margins),
                                             "min": min(margins), "max": max(margins)}
        base = gaussian[10][0] if 10 in gaussian else torch.randn(10, generator=gen, dtype=torch.float64)
        out["appended"][kind] = {"0": relevant_weight(fn, 2.0, base)}
        for k in APPENDED:
            low = torch.full((k,), -20.0, dtype=torch.float64)  # far below any threshold here
            out["appended"][kind][str(k)] = relevant_weight(fn, 2.0, torch.cat([base, low]))
    out["predictions"] = predictions(out, ns)
    if 100_000 in ns:
        out["claimed"] = {
            key: {kind: {"claimed": v, "measured_median": out["gaussian"][kind]["100000"]["median"],
                         "measured_range": [out["gaussian"][kind]["100000"]["min"],
                                            out["gaussian"][kind]["100000"]["max"]],
                         "reproduced": claim_reproduced(v, out["gaussian"][kind]["100000"]["per_draw"])}
                  for kind, v in claims.items()}
            for key, claims in CLAIMS.items()}
    return out


def predictions(out: dict[str, Any], ns: Sequence[int]) -> dict[str, Any]:
    """The fixed predictions next to the measured values, each with whether it holds."""
    b, g, a = out["bounded"], out["gaussian"], out["appended"]
    pred: dict[str, Any] = {"bounded": {}, "gaussian": {}}
    for kind in KINDS:
        pred["bounded"][kind] = {}
        for n in ns:
            want, got = bounded_prediction(kind, n), b[kind][str(n)]
            pred["bounded"][kind][str(n)] = {"predicted": want, "measured": got, "holds": abs(got - want) <= EXACT_TOL}
    for kind, fn, tol in (("softmax", lambda n: math.log(n) + 0.5, SOFTMAX_GAUSSIAN_TOL),
                          ("sparsemax", sparsemax_gaussian_prediction, SPARSEMAX_GAUSSIAN_TOL)):
        pred["gaussian"][kind] = {}
        for n in ns:
            want, got = fn(n), g[kind][str(n)]["median"]
            entry = {"predicted": want, "measured_median": got, "tolerance": tol}
            if n >= GAUSSIAN_MIN_N:
                entry["holds"] = abs(got - want) <= tol
            if kind == "sparsemax":
                entry["asymptotic_sqrt_2_ln_n"] = math.sqrt(2 * math.log(n)) if n > 1 else 0.0
            pred["gaussian"][kind][str(n)] = entry
    pred["appending_changes_nothing_for_alpha_above_1"] = {
        kind: all(v == a[kind]["0"] for v in a[kind].values()) for kind in ("entmax15", "sparsemax")}
    pred["appending_dilutes_softmax"] = all(a["softmax"][str(k)] < a["softmax"]["0"] for k in APPENDED)
    checks = [e["holds"] for per_n in pred["bounded"].values() for e in per_n.values()]
    checks += [e["holds"] for per_n in pred["gaussian"].values() for e in per_n.values() if "holds" in e]
    checks += list(pred["appending_changes_nothing_for_alpha_above_1"].values()) + [pred["appending_dilutes_softmax"]]
    pred["all_hold"] = all(checks)
    return pred


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Stage A: margin needed against the number of distractors (MEASURE).")
    p.add_argument("--out", type=Path, default=Path("artifacts/selection_stage_a.json"))
    args = p.parse_args(argv)
    result = {"science_open": False, "purpose": "operator-level margin a relevant position needs to keep half the "
                                                  "weight against n distractors", **run()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "all_predictions_hold": result["predictions"]["all_hold"],
                      "out": args.out.as_posix(), "science_open": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

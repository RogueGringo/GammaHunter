# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tiny FLOP comparison demo for one dummy context length.

RESEARCH scaffold only — prints schematic FLOP / param estimates.
Does not train, load weights, or report accuracy.

Usage::

    python -m reachability_gen.flops_demo
    python -m reachability_gen.flops_demo --m 48 --d 64 --T 4 --L 4 --K-used 8
"""

from __future__ import annotations

import argparse
import sys

from reachability_gen.arms import (
    ChainOfThoughtArm,
    EuclideanLoopArm,
    FeedForwardArm,
    GeometricRecurrentArm,
    SharedArmConfig,
)
from reachability_gen.param_match import check_geo_ff_loop


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Print a schematic FLOP comparison table for reachability arms "
            "(RESEARCH scaffold; no training / no accuracy claims)."
        )
    )
    p.add_argument("--m", type=int, default=32, help="context length")
    p.add_argument("--d", type=int, default=64, help="hidden / embed width")
    p.add_argument("--T", type=int, default=4, help="recurrence cycles (Geo/Loop)")
    p.add_argument("--L", type=int, default=1, help="FF / CoT layer count")
    p.add_argument("--K-cap", type=int, default=32, dest="K_cap", help="CoT token cap")
    p.add_argument(
        "--K-used",
        type=int,
        default=8,
        dest="K_used",
        help="realized CoT tokens (FLOPs use this, not K_cap)",
    )
    p.add_argument(
        "--no-tau",
        action="store_true",
        help="disable Geo tau embed in FLOPs / params",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = SharedArmConfig(d=args.d, m=args.m)
    use_tau = not args.no_tau

    ff = FeedForwardArm(L=args.L, d=args.d, config=cfg)
    # For a fair param-match demo when L==1 and use_tau=False, Geo≈Loop≈FF.
    # With default use_tau=True, Geo has a tiny tau-embed param delta.
    geo = GeometricRecurrentArm(
        T=args.T, d=args.d, use_tau=use_tau, config=cfg
    )
    loop = EuclideanLoopArm(T=args.T, d=args.d, config=cfg)
    cot = ChainOfThoughtArm(
        K_cap=args.K_cap,
        d=args.d,
        L=args.L,
        K_used=args.K_used,
        config=cfg,
    )

    reports = [
        ff.inference_flops(args.m),
        geo.inference_flops(args.m),
        loop.inference_flops(args.m),
        cot.inference_flops(args.m, K_used=args.K_used),
    ]

    print("RESEARCH scaffold — schematic FLOP table (not measured hardware).")
    print(f"context m={args.m}  d={args.d}  T={args.T}  L={args.L}  "
          f"K_used={args.K_used} (K_cap={args.K_cap})")
    print()
    header = f"{'arm':<28} {'params':>12} {'flops':>16}  notes"
    print(header)
    print("-" * len(header))
    for r in reports:
        print(
            f"{r.arm:<28} {r.params_estimate:>12,} {r.flops:>16,}  {r.notes}"
        )

    print()
    # Param match for Geo/FF/Loop only (CoT excluded by design).
    match = check_geo_ff_loop(geo, ff, loop)
    print(
        f"param_match Geo/FF/Loop (±{match.tolerance:.0%}): "
        f"{'PASS' if match.passed else 'FAIL'} — {match.notes}"
    )
    print(f"  counts={match.counts}")
    print(f"  ratios={{{', '.join(f'{k}: {v:.4f}' for k, v in match.ratios.items())}}}")
    print(f"  science_open={match.science_open} (never self-stamped)")
    print()
    print("Disclaimer: RESEARCH scaffold only. No accuracy or scaling claims.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

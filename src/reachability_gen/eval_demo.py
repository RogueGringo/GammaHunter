# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tiny fake eval demo: JSONL examples → arm stubs → artifacts/metrics.jsonl.

RESEARCH scaffold only — writes the ADR-001 logging schema with placeholder
predictions. No accuracy claims.

Usage::

    python -m reachability_gen.eval_demo
    python -m reachability_gen.eval_demo --examples data/train_tiny.jsonl \\
        --out artifacts/metrics.jsonl --limit 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from reachability_gen.generate import generate_split, write_jsonl
from reachability_gen.harness.runner import default_stub_arms, eval_jsonl
from reachability_gen.harness.runner import HAS_TORCH


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run a tiny stub eval and write ADR-001 RunMetricRecord JSONL "
            "(no training / no accuracy claims)."
        )
    )
    p.add_argument(
        "--examples",
        type=Path,
        default=Path("data/train_tiny.jsonl"),
        help="Input examples JSONL (default: data/train_tiny.jsonl).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/metrics.jsonl"),
        help="Output metrics JSONL (default: artifacts/metrics.jsonl).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=8,
        help="Max examples to eval (default: 8).",
    )
    p.add_argument(
        "--d",
        type=int,
        default=64,
        help="Stub arm width d.",
    )
    p.add_argument(
        "--T",
        type=int,
        default=4,
        help="Geo/Loop recurrence steps T.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run_id stamped on every metric row.",
    )
    p.add_argument(
        "--epoch",
        type=int,
        default=0,
        help="Epoch stamped on metric rows (stub default 0).",
    )
    p.add_argument(
        "--generate-if-missing",
        action="store_true",
        default=True,
        help="If examples path missing, generate a few train rows (default: on).",
    )
    p.add_argument(
        "--no-generate-if-missing",
        action="store_false",
        dest="generate_if_missing",
        help="Fail if examples path is missing.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    examples_path: Path = args.examples

    if not examples_path.exists():
        if not args.generate_if_missing:
            print(f"missing examples: {examples_path}", file=sys.stderr)
            return 1
        print(
            f"examples missing; generating tiny train set → {examples_path}",
            file=sys.stderr,
        )
        examples, _ = generate_split("train", n_per_cell=2, max_rejects=2000)
        write_jsonl(examples_path, examples)

    arms = default_stub_arms(d=args.d, T=args.T, L=1)
    # Parity among Geo/FF/Loop; CoT excluded inside harness.
    n = eval_jsonl(
        examples_path,
        args.out,
        arms,
        limit=args.limit,
        assert_parity=True,
        run_id=args.run_id,
        epoch=args.epoch,
    )
    print(
        f"wrote {n} metric rows → {args.out} "
        f"(examples_limit={args.limit}, arms={len(arms)}, torch={HAS_TORCH})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

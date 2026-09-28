# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Fidelity gate for 8-bit reference models (MEASURE).

Before an 8-bit model's results are trusted, the same model is scored both
ways on the same seeded questions: its 16-bit margins (from an existing result
file) against its 8-bit margins. The gate passes when, on every set, the
margins correlate at 0.95 or more, the Yes/No predictions agree on 95% or more
of the questions, and the AUROC moves by at most 0.02. The thresholds are
fixed here, before any 8-bit run.

``science_open=false`` always.

Usage::

    python -m reachability_gen.llm_int8_fidelity --model Qwen/Qwen2.5-3B-Instruct \\
        --pair artifacts/llm_reference.json artifacts/llm_reference_int8_3b.json \\
        --pair artifacts/llm_edge_probe.json artifacts/llm_edge_probe_int8_3b.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.run_llm_reference import auroc

MIN_CORRELATION: float = 0.95
MIN_AGREEMENT: float = 0.95
MAX_AUROC_SHIFT: float = 0.02
DEFAULT_OUT = Path("artifacts/llm_int8_fidelity.json")


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def compare(full: Sequence[float], eight: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    """Margin correlation, prediction agreement and AUROC for one set."""
    agree = sum((a > 0) == (b > 0) for a, b in zip(full, eight)) / len(full)
    r = pearson(full, eight)
    a16, a8 = auroc(full, labels), auroc(eight, labels)
    return {
        "n": len(full),
        "margin_correlation": r,
        "prediction_agreement": agree,
        "auroc_16bit": a16,
        "auroc_8bit": a8,
        "auroc_shift": a8 - a16,
        "passes": r >= MIN_CORRELATION and agree >= MIN_AGREEMENT and abs(a8 - a16) <= MAX_AUROC_SHIFT,
    }


def gate(model: str, pairs: Sequence[tuple[Path, Path]]) -> dict[str, Any]:
    sets: dict[str, Any] = {}
    for full_path, eight_path in pairs:
        full, eight = (json.loads(p.read_text(encoding="utf-8")) for p in (full_path, eight_path))
        for set_name, rows in full["sets"].items():
            rows = rows["rows"] if isinstance(rows, dict) else rows
            other = eight["sets"][set_name]
            other = other["rows"] if isinstance(other, dict) else other
            if rows != other:
                raise ValueError(f"{set_name}: the two runs scored different questions")
            labels = [int(r["y"]) for r in rows]
            sets[f"{full_path.stem}/{set_name}"] = compare(
                full["models"][model]["sets"][set_name]["margins"],
                eight["models"][model]["sets"][set_name]["margins"],
                labels,
            )
    return {
        "model": model,
        "thresholds": {"min_margin_correlation": MIN_CORRELATION, "min_prediction_agreement": MIN_AGREEMENT,
                       "max_auroc_shift": MAX_AUROC_SHIFT},
        "sets": sets,
        "passes": all(s["passes"] for s in sets.values()),
        "science_open": False,
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Fidelity gate for 8-bit reference models (MEASURE).")
    p.add_argument("--model", required=True)
    p.add_argument("--pair", nargs=2, type=Path, action="append", required=True, metavar=("SIXTEEN_BIT", "EIGHT_BIT"))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    result = gate(args.model, [tuple(pair) for pair in args.pair])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, s in result["sets"].items():
        print(f"{name:32s} r={s['margin_correlation']:.4f} agree={s['prediction_agreement']:.3f} "
              f"AUROC {s['auroc_16bit']:.3f}->{s['auroc_8bit']:.3f} {'PASS' if s['passes'] else 'FAIL'}", file=sys.stderr)
    print(json.dumps({"passes": result["passes"], "out": args.out.as_posix(), "science_open": False}, sort_keys=True))
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

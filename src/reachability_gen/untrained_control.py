# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Untrained-model control: score each arm at the weights its training starts from.

An accuracy an arm already reaches before any training step cannot be
attributed to learning. For every seed and arm of the message-passing
calibration, this rebuilds the exact initial weights (the runner's own
``init_model``) and evaluates them with the runner's own evaluation, on the
same sets and step counts. When the calibration's result file is present, it
also reports the learned margin (trained minus untrained accuracy) for both
saved checkpoints.

``science_open=false`` always.

Usage::

    python -m reachability_gen.untrained_control --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.models.message_passing import ParsedGraph, parse_rows
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_mp_calibration import (
    ARMS,
    DEFAULT_EXTENDED,
    DEFAULT_OUT as DEFAULT_CALIBRATION,
    DEFAULT_TRAIN,
    TEST_STEPS,
    TRAIN_STEPS,
    _mean,
    evaluate,
    init_model,
)

DEFAULT_OUT = Path("artifacts/mp_calibration_untrained_control.json")


def control_arm(
    kind: str,
    val: Sequence[ParsedGraph],
    val_rows: Sequence[dict[str, Any]],
    extended: Sequence[ParsedGraph],
    extended_rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    d: int,
    device: str,
    test_steps: Sequence[int] = TEST_STEPS,
) -> dict[str, Any]:
    """Evaluate one arm at its initial weights for ``seed``."""
    model = init_model(kind, d, seed, device)
    steps_list = [TRAIN_STEPS] if kind == "unlooped" else sorted({TRAIN_STEPS, *test_steps})
    ev = evaluate(model, val, val_rows, device)
    return {
        "arm": kind,
        "seed": seed,
        "id_val_acc": ev["acc"],
        "id_val_by_graph_hop": ev["by_graph_hop"],
        "extended_by_steps": {
            str(s): evaluate(model, extended, extended_rows, device, s) for s in steps_list
        },
    }


def learned_margin(control: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    """Trained minus untrained accuracy per seed, arm and saved checkpoint."""
    out: dict[str, Any] = {}
    for seed, per_seed in control.items():
        for kind, c in per_seed.items():
            trained = calibration.get("runs", {}).get(seed, {}).get(kind)
            if trained is None:
                continue
            for which in ("best", "final"):
                t = trained[which]
                out[f"seed{seed}/{kind}/{which}"] = {
                    "id_val": t["id_val_acc"] - c["id_val_acc"],
                    "extended_by_steps": {
                        s: t["extended_by_steps"][s]["acc"] - c["extended_by_steps"][s]["acc"]
                        for s in c["extended_by_steps"]
                        if s in t["extended_by_steps"]
                    },
                }
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Untrained-model control for the message-passing calibration (MEASURE)."
    )
    p.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--extended-data", type=Path, default=DEFAULT_EXTENDED)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--d", type=int, default=64, help="unlooped width; looped is width-matched")
    p.add_argument("--test-steps", type=int, nargs="+", default=list(TEST_STEPS))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    try:
        import torch
    except ImportError:
        print("FAIL: torch required", file=sys.stderr)
        return 2
    if args.device == "cuda" and not torch.cuda.is_available():
        print("FAIL: --device cuda, but this torch build sees no CUDA device", file=sys.stderr)
        return 1
    for path in (args.train_data, args.extended_data):
        if not path.exists():
            print(f"FAIL: missing {path}; see docs/USAGE.md", file=sys.stderr)
            return 1
    val_rows = [r for r in load_jsonl(args.train_data) if r["split"] == "val"]
    ext_rows = load_jsonl(args.extended_data)
    val, ext = parse_rows(val_rows), parse_rows(ext_rows)

    control = {
        str(seed): {
            kind: control_arm(
                kind, val, val_rows, ext, ext_rows,
                seed=seed, d=args.d, device=args.device, test_steps=args.test_steps,
            )
            for kind in ARMS
        }
        for seed in args.seeds
    }
    first = str(args.seeds[0])
    aggregate = {
        kind: {
            "id_val_acc_mean": _mean([control[s][kind]["id_val_acc"] for s in control]),
            "extended_acc_by_steps_mean": {
                st: _mean([control[s][kind]["extended_by_steps"][st]["acc"] for s in control])
                for st in control[first][kind]["extended_by_steps"]
            },
        }
        for kind in ARMS
    }
    artifact: dict[str, Any] = {
        "science_open": False,
        "purpose": "untrained-model control: accuracy at the weights training starts from",
        "train_data": args.train_data.as_posix(),
        "extended_data": args.extended_data.as_posix(),
        "seeds": args.seeds,
        "d": args.d,
        "device": args.device,
        "torch_version": torch.__version__,
        "control": control,
        "aggregate": aggregate,
    }
    if args.calibration.exists():
        artifact["calibration"] = args.calibration.as_posix()
        artifact["learned_margin"] = learned_margin(
            control, json.loads(args.calibration.read_text(encoding="utf-8"))
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    for kind, a in aggregate.items():
        ext = " ".join(f"T={s}:{v:.3f}" for s, v in a["extended_acc_by_steps_mean"].items())
        print(f"{kind:9s} untrained ID val {a['id_val_acc_mean']:.4f} | long paths {ext}", file=sys.stderr)
    print(json.dumps({"ok": True, "out": args.out.as_posix(), "science_open": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

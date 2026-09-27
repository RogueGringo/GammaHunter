# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Overfit sanity CLI for the real torch GeometricRecurrent arm (MEASURE plumbing).

Balanced gate (identical construction to FF via shared helpers):
  - exactly 16 y=1 with hop_distance K in [2, 6]
  - exactly 16 y=0 hard negatives (deg(s)>=1, deg(t)>=1, unreachable)
  - Pass: per-class acc = 1.0 (16/16 each) AND CE loss < 1e-3 within <=100 steps
  - Drift: drift_trajectory length T-1 (or T) finite and non-zero across cycles
  - Fail if any NaN/Inf or all-zero drifts

Fixed T=6 (max ID hop). Shares ``ensure_balanced_batch`` with ``overfit_ff``
so the same seed → same examples.

No science OPEN claims.

Usage::

    python -m reachability_gen.overfit_geo --balanced
    python -m reachability_gen.overfit_geo --balanced --steps 100 --T 6
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

from reachability_gen.adr_invariants import EPS_SIGMA, ID_HOP_MAX
from reachability_gen.overfit_ff import (
    BALANCED_LOSS_THRESHOLD,
    BALANCED_N_NEG,
    BALANCED_N_POS,
    _diagnose_failure,
    _per_class_accuracy,
    ensure_balanced_batch,
    ensure_id_hop_batch,
)

LN2 = math.log(2.0)
DEFAULT_T = int(ID_HOP_MAX)  # 6 — max ID hop


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Overfit sanity gate for torch GeometricRecurrent "
            "(MEASURE plumbing; no science OPEN). Default: --balanced, T=6."
        )
    )
    p.add_argument(
        "--examples",
        type=Path,
        default=Path("data/train_tiny.jsonl"),
        help="Train JSONL path (default: data/train_tiny.jsonl).",
    )
    p.add_argument("--steps", type=int, default=100, help="Max train steps (default 100).")
    p.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Max ID-hop examples in legacy mode (default 32).",
    )
    p.add_argument("--d", type=int, default=64, help="Model width d.")
    p.add_argument(
        "--T",
        type=int,
        default=DEFAULT_T,
        help=f"Recurrence cycles T (default {DEFAULT_T} = max ID hop).",
    )
    p.add_argument("--lr", type=float, default=3e-3, help="AdamW learning rate.")
    p.add_argument("--seed", type=int, default=0, help="Torch RNG seed.")
    p.add_argument(
        "--loss-threshold",
        type=float,
        default=None,
        help="Target CE loss (default: 1e-3 balanced, 0.05 legacy).",
    )
    p.add_argument(
        "--no-regenerate",
        action="store_true",
        help="Do not regenerate train JSONL when examples are scarce.",
    )
    p.add_argument(
        "--no-tau",
        action="store_true",
        help="Disable cycle (tau) embeddings.",
    )
    p.add_argument(
        "--perturbation-sigma",
        type=float,
        default=EPS_SIGMA,
        help=f"Optional perturbation_delta noise scale (default {EPS_SIGMA}).",
    )
    bal = p.add_mutually_exclusive_group()
    bal.add_argument(
        "--balanced",
        dest="balanced",
        action="store_true",
        default=True,
        help="Balanced 16+16 hard-negative gate (default).",
    )
    bal.add_argument(
        "--no-balanced",
        dest="balanced",
        action="store_false",
        help="Legacy ID-hop batch gate (loss < 0.05, overall acc 1.0).",
    )
    p.add_argument(
        "--n-pos",
        type=int,
        default=BALANCED_N_POS,
        help=f"Balanced y=1 count (default {BALANCED_N_POS}).",
    )
    p.add_argument(
        "--n-neg",
        type=int,
        default=BALANCED_N_NEG,
        help=f"Balanced y=0 hard-neg count (default {BALANCED_N_NEG}).",
    )
    return p


def run_overfit_geo(
    examples: list[dict[str, Any]],
    *,
    steps: int = 100,
    d: int = 64,
    T: int = DEFAULT_T,
    lr: float = 3e-3,
    seed: int = 0,
    loss_threshold: float = 0.05,
    require_per_class: bool = False,
    use_tau: bool = True,
    perturbation_sigma: Optional[float] = EPS_SIGMA,
) -> dict[str, Any]:
    """Train GeometricRecurrent on a fixed batch; return diagnostics + drift."""
    import torch

    from reachability_gen.models.geometric import (
        GeometricRecurrent,
        trajectory_finite_nonzero,
    )
    from reachability_gen.tokenize import build_vocab
    from reachability_gen.train.ff_trainer import examples_to_batch
    from reachability_gen.train.geo_trainer import GeometricTrainer

    torch.manual_seed(seed)
    vocab = build_vocab()
    token_ids, attention_mask, labels, vocab = examples_to_batch(examples, vocab)
    model = GeometricRecurrent(
        vocab_size=len(vocab),
        d=d,
        T=T,
        n_heads=4 if d % 4 == 0 else 2,
        max_len=max(int(token_ids.shape[1]) + 8, 64),
        pad_id=vocab.pad_id,
        use_tau=use_tau,
    )
    trainer = GeometricTrainer(model, lr=lr, weight_decay=0.01, grad_clip=1.0)

    losses: list[float] = []
    accs: list[float] = []
    per_class_hist: list[dict[str, float]] = []
    passed_at: Optional[int] = None
    last_drifts: list[float] = []
    last_telemetry: dict[str, Any] = {}

    for step in range(1, steps + 1):
        loss, acc = trainer.train_step(token_ids, labels, attention_mask)
        losses.append(loss)
        accs.append(acc)
        eval_out = trainer.eval_step(
            token_ids, labels, attention_mask, return_drift=True
        )
        eval_loss, eval_acc, drifts = eval_out  # type: ignore[misc]
        last_drifts = list(drifts)
        with torch.no_grad():
            logits, _ = trainer.model(token_ids, attention_mask)
            preds = logits.argmax(dim=-1)
            pc = _per_class_accuracy(preds, labels)
        per_class_hist.append(pc)
        gate_loss = eval_loss
        class_ok = (not require_per_class) or (
            pc.get("y0", 0.0) >= 1.0 - 1e-9 and pc.get("y1", 0.0) >= 1.0 - 1e-9
        )
        drift_ok, _drift_reason = trajectory_finite_nonzero(last_drifts)
        if (
            gate_loss < loss_threshold
            and eval_acc >= 1.0 - 1e-9
            and class_ok
            and drift_ok
            and passed_at is None
        ):
            passed_at = step
            losses[-1] = gate_loss
            accs[-1] = eval_acc

    last_telemetry = trainer.drift_telemetry(
        token_ids,
        attention_mask,
        eps_sigma=perturbation_sigma,
    )
    # Prefer telemetry drifts (same as last eval, with optional perturbation).
    if last_telemetry.get("drift_trajectory"):
        last_drifts = list(last_telemetry["drift_trajectory"])

    final_pc = per_class_hist[-1] if per_class_hist else {}
    final_loss, final_acc = losses[-1], accs[-1]
    early_mean = sum(losses[: min(5, len(losses))]) / max(min(5, len(losses)), 1)
    late_mean = sum(losses[-min(5, len(losses)) :]) / max(min(5, len(losses)), 1)
    trends_down = late_mean < early_mean or (passed_at is not None)
    below = final_loss < loss_threshold or (
        passed_at is not None and late_mean < loss_threshold
    )
    acc_ok = final_acc >= 1.0 - 1e-9
    class_ok = (not require_per_class) or (
        final_pc.get("y0", 0.0) >= 1.0 - 1e-9 and final_pc.get("y1", 0.0) >= 1.0 - 1e-9
    )
    drift_ok, drift_reason = trajectory_finite_nonzero(last_drifts)
    # Accept length T-1 (traj T+1 states → T drifts) or T-1 from T states.
    expected_lo, expected_hi = max(T - 1, 1), T
    drift_len_ok = expected_lo <= len(last_drifts) <= expected_hi + 1
    if not drift_len_ok:
        drift_ok = False
        drift_reason = (
            f"drift length {len(last_drifts)} not in [{expected_lo}, {expected_hi + 1}]"
        )

    if require_per_class:
        ok = (passed_at is not None) or (
            below and acc_ok and class_ok and drift_ok
        )
        ok = bool(ok) and drift_ok
    else:
        ok = ((passed_at is not None and acc_ok) or (below and acc_ok and trends_down)) and drift_ok

    return {
        "ok": bool(ok),
        "steps_run": steps,
        "passed_at": passed_at,
        "final_loss": final_loss,
        "final_acc": final_acc,
        "early_mean_loss": early_mean,
        "late_mean_loss": late_mean,
        "losses": losses,
        "accs": accs,
        "per_class": final_pc,
        "n_examples": len(examples),
        "label_counts": {
            "y0": sum(1 for e in examples if int(e["y"]) == 0),
            "y1": sum(1 for e in examples if int(e["y"]) == 1),
        },
        "d": d,
        "T": T,
        "lr": lr,
        "loss_threshold": loss_threshold,
        "require_per_class": require_per_class,
        "use_tau": use_tau,
        "drift_trajectory": last_drifts,
        "terminal_drift": last_telemetry.get("terminal_drift"),
        "perturbation_delta": last_telemetry.get("perturbation_delta"),
        "trajectory_len": last_telemetry.get("trajectory_len"),
        "drift_ok": drift_ok,
        "drift_reason": drift_reason,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "FAIL: torch is required for overfit_geo (pip install torch)",
            file=sys.stderr,
        )
        return 2

    if args.loss_threshold is None:
        loss_threshold = (
            BALANCED_LOSS_THRESHOLD if args.balanced else 0.05
        )
    else:
        loss_threshold = float(args.loss_threshold)

    meta: dict[str, Any] = {}
    if args.balanced:
        batch, note, meta = ensure_balanced_batch(
            args.examples,
            n_pos=args.n_pos,
            n_neg=args.n_neg,
            regenerate=not args.no_regenerate,
        )
        print(f"data[balanced]: {note}", file=sys.stderr)
        need = args.n_pos + args.n_neg
        if len(batch) < need:
            print(
                f"FAIL: need exactly {args.n_pos}+{args.n_neg}={need} balanced "
                f"examples, got {len(batch)} "
                f"(y1={sum(1 for e in batch if int(e['y'])==1)}, "
                f"y0={sum(1 for e in batch if int(e['y'])==0)}). {note}",
                file=sys.stderr,
            )
            return 1
        n1 = sum(1 for e in batch if int(e["y"]) == 1)
        n0 = sum(1 for e in batch if int(e["y"]) == 0)
        if n1 != args.n_pos or n0 != args.n_neg:
            print(
                f"FAIL: balanced counts mismatch y1={n1}/{args.n_pos} "
                f"y0={n0}/{args.n_neg}",
                file=sys.stderr,
            )
            return 1
        require_per_class = True
    else:
        batch, note = ensure_id_hop_batch(
            args.examples,
            target_n=args.batch_size,
            regenerate=not args.no_regenerate,
        )
        print(f"data: {note}", file=sys.stderr)
        if len(batch) < 2:
            print(
                f"FAIL: need ≥2 ID-hop examples, got {len(batch)}. {note}",
                file=sys.stderr,
            )
            return 1
        require_per_class = False

    result = run_overfit_geo(
        batch,
        steps=args.steps,
        d=args.d,
        T=args.T,
        lr=args.lr,
        seed=args.seed,
        loss_threshold=loss_threshold,
        require_per_class=require_per_class,
        use_tau=not args.no_tau,
        perturbation_sigma=args.perturbation_sigma,
    )

    pc = result.get("per_class") or {}
    drifts = result.get("drift_trajectory") or []
    print(
        f"overfit_geo: balanced={args.balanced} n={result['n_examples']} "
        f"labels={result['label_counts']} T={result['T']} "
        f"steps={result['steps_run']} passed_at={result['passed_at']} "
        f"final_loss={result['final_loss']:.6f} final_acc={result['final_acc']:.4f} "
        f"acc_y0={pc.get('y0')} ({int(pc.get('y0_correct', 0))}/{int(pc.get('y0_n', 0))}) "
        f"acc_y1={pc.get('y1')} ({int(pc.get('y1_correct', 0))}/{int(pc.get('y1_n', 0))}) "
        f"drift_ok={result['drift_ok']} terminal_drift={result.get('terminal_drift')} "
        f"early_mean={result['early_mean_loss']:.4f} late_mean={result['late_mean_loss']:.4f}",
        file=sys.stderr,
    )
    print(
        json.dumps(
            {
                "ok": result["ok"],
                "arm": "geometric",
                "balanced": bool(args.balanced),
                "final_loss": result["final_loss"],
                "final_acc": result["final_acc"],
                "per_class": {
                    "y0_acc": pc.get("y0"),
                    "y1_acc": pc.get("y1"),
                    "y0_correct": pc.get("y0_correct"),
                    "y1_correct": pc.get("y1_correct"),
                    "y0_n": pc.get("y0_n"),
                    "y1_n": pc.get("y1_n"),
                },
                "steps": result["steps_run"],
                "passed_at": result["passed_at"],
                "n_examples": result["n_examples"],
                "label_counts": result["label_counts"],
                "loss_threshold": loss_threshold,
                "T": result["T"],
                "drift_trajectory": drifts,
                "terminal_drift": result.get("terminal_drift"),
                "perturbation_delta": result.get("perturbation_delta"),
                "drift_ok": result["drift_ok"],
                "drift_reason": result.get("drift_reason"),
                "reject_reasons": meta.get("reject_reasons"),
            },
            sort_keys=True,
        )
    )

    if result["ok"]:
        print(
            "PASS: geo overfit gate "
            + (
                f"(balanced: loss < {loss_threshold}, per-class acc=1.0, "
                f"drift finite/non-zero len={len(drifts)})"
                if args.balanced
                else f"(loss < threshold, acc=1.0, drift ok len={len(drifts)})"
            ),
            file=sys.stderr,
        )
        return 0

    diag = _diagnose_failure(
        result["losses"],
        result["accs"],
        loss_threshold=loss_threshold,
        per_class={"y0": pc.get("y0", 0.0), "y1": pc.get("y1", 0.0)}
        if require_per_class
        else None,
    )
    if not result["drift_ok"]:
        diag = f"{diag}; drift: {result.get('drift_reason')}"
    print(f"FAIL: geo overfit gate — {diag}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

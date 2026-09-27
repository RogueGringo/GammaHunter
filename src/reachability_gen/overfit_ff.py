# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Overfit sanity CLI for the real torch FeedForward arm (MEASURE plumbing).

Default / CI gate is **balanced**:
  - exactly 16 y=1 with hop_distance K in [2, 6]
  - exactly 16 y=0 hard negatives (deg(s)>=1, deg(t)>=1, unreachable)
  - Pass: per-class acc = 1.0 (16/16 each) AND CE loss < 1e-3 within <=100 steps

Legacy mode (``--no-balanced``): up to 32 ID-hop rows, loss < 0.05, overall acc 1.0.

No science OPEN claims.

Usage::

    python -m reachability_gen.overfit_ff
    python -m reachability_gen.overfit_ff --balanced --steps 100
    python -m reachability_gen.overfit_ff --no-balanced --examples data/train_tiny.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from reachability_gen.adr_invariants import ID_HOP_MAX, ID_HOP_MIN
from reachability_gen.generate import generate_split, write_jsonl
from reachability_gen.hard_negatives import filter_hard_negatives

LN2 = math.log(2.0)  # ~0.693 — CE of uniform binary classifier

BALANCED_N_POS = 16
BALANCED_N_NEG = 16
BALANCED_LOSS_THRESHOLD = 1e-3


def _is_id_hop_pos(row: dict[str, Any]) -> bool:
    y = int(row.get("y", -1))
    if y != 1:
        return False
    hop = int(row.get("hop_distance", -999))
    return ID_HOP_MIN <= hop <= ID_HOP_MAX


def _is_id_hop_row(row: dict[str, Any]) -> bool:
    y = int(row.get("y", -1))
    if y not in (0, 1):
        return False
    hop = int(row.get("hop_distance", -999))
    return ID_HOP_MIN <= hop <= ID_HOP_MAX


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def filter_id_hop(rows: list[dict[str, Any]], *, limit: int = 32) -> list[dict[str, Any]]:
    out = [r for r in rows if _is_id_hop_row(r)]
    return out[:limit]


def filter_id_hop_positives(
    rows: list[dict[str, Any]], *, limit: int = BALANCED_N_POS
) -> list[dict[str, Any]]:
    out = [r for r in rows if _is_id_hop_pos(r)]
    return out[:limit]


def ensure_id_hop_batch(
    examples_path: Path,
    *,
    target_n: int = 32,
    regenerate: bool = True,
) -> tuple[list[dict[str, Any]], str]:
    """Load ≤target_n ID-hop rows; regenerate larger train set if needed.

    Returns (batch, note). Note documents regenerate / fallback decisions.
    """
    notes: list[str] = []
    if examples_path.exists():
        rows = load_jsonl(examples_path)
        batch = filter_id_hop(rows, limit=target_n)
        notes.append(
            f"loaded {len(rows)} rows from {examples_path}; "
            f"ID-hop[ {ID_HOP_MIN},{ID_HOP_MAX} ] candidates={len([r for r in rows if _is_id_hop_row(r)])}"
        )
    else:
        rows = []
        batch = []
        notes.append(f"examples path missing: {examples_path}")

    if len(batch) >= target_n:
        return batch[:target_n], "; ".join(notes)

    if not regenerate:
        notes.append(
            f"fallback: only {len(batch)} ID-hop rows (need {target_n}); "
            f"--no-regenerate set, using available"
        )
        return batch, "; ".join(notes)

    notes.append(
        f"too few ID-hop rows ({len(batch)} < {target_n}); regenerating with hop preference"
    )
    best: list[dict[str, Any]] = list(batch)
    for n_per_cell in (4, 8, 16, 32, 64):
        examples, reject_rate = generate_split(
            "train",
            n_per_cell=n_per_cell,
            max_rejects=5000,
            prefer_id_hops=True,
        )
        write_jsonl(examples_path, examples)
        rows = [ex.to_dict() for ex in examples]
        cand = filter_id_hop(rows, limit=target_n)
        notes.append(
            f"regen n_per_cell={n_per_cell}: total={len(rows)} "
            f"id_hop={len([r for r in rows if _is_id_hop_row(r)])} "
            f"reject_rate={reject_rate:.3f}"
        )
        if len(cand) > len(best):
            best = cand
        if len(cand) >= target_n:
            return cand[:target_n], "; ".join(notes)

    notes.append(
        f"fallback: regenerated but only {len(best)} ID-hop rows "
        f"(target {target_n}); proceeding with available batch"
    )
    return best, "; ".join(notes)


def ensure_balanced_batch(
    examples_path: Path,
    *,
    n_pos: int = BALANCED_N_POS,
    n_neg: int = BALANCED_N_NEG,
    regenerate: bool = True,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    """Collect exactly n_pos ID-hop positives + n_neg hard negatives.

    Regenerates train JSONL with increasing n_per_cell until quotas are met
    (or documents fallback). Logs label counts and y=0 reject reasons.
    """
    notes: list[str] = []
    meta: dict[str, Any] = {
        "n_pos_target": n_pos,
        "n_neg_target": n_neg,
        "reject_reasons": {},
        "label_counts_source": {},
    }

    def _from_rows(rows: list[dict[str, Any]]) -> tuple[list[dict], list[dict], Counter]:
        pos = [r for r in rows if _is_id_hop_pos(r)]
        hard, reasons = filter_hard_negatives(rows)
        return pos, hard, reasons

    rows: list[dict[str, Any]] = []
    if examples_path.exists():
        rows = load_jsonl(examples_path)
        notes.append(f"loaded {len(rows)} rows from {examples_path}")
    else:
        notes.append(f"examples path missing: {examples_path}")

    pos, hard, reasons = _from_rows(rows) if rows else ([], [], Counter())
    meta["reject_reasons"] = dict(reasons)
    meta["label_counts_source"] = {
        "y0": sum(1 for r in rows if int(r.get("y", -1)) == 0),
        "y1": sum(1 for r in rows if int(r.get("y", -1)) == 1),
        "y1_id_hop": len(pos),
        "y0_hard": len(hard),
    }
    notes.append(
        f"candidates y1_id={len(pos)} y0_hard={len(hard)} "
        f"reject_reasons={dict(reasons)}"
    )

    if len(pos) >= n_pos and len(hard) >= n_neg:
        batch = pos[:n_pos] + hard[:n_neg]
        notes.append(f"balanced batch ready: {n_pos}+{n_neg}")
        return batch, "; ".join(notes), meta

    if not regenerate:
        batch = pos[:n_pos] + hard[:n_neg]
        notes.append(
            f"fallback: y1_id={len(pos)}/{n_pos} y0_hard={len(hard)}/{n_neg}; "
            f"--no-regenerate set"
        )
        return batch, "; ".join(notes), meta

    notes.append(
        f"too few balanced examples (y1_id={len(pos)}/{n_pos}, "
        f"y0_hard={len(hard)}/{n_neg}); regenerating"
    )
    best_pos, best_hard = list(pos), list(hard)
    best_reasons = reasons
    for n_per_cell in (4, 8, 16, 32, 64, 128):
        examples, reject_rate = generate_split(
            "train",
            n_per_cell=n_per_cell,
            max_rejects=8000,
            prefer_id_hops=True,
        )
        write_jsonl(examples_path, examples)
        rows = [ex.to_dict() for ex in examples]
        pos, hard, reasons = _from_rows(rows)
        notes.append(
            f"regen n_per_cell={n_per_cell}: total={len(rows)} "
            f"y1_id={len(pos)} y0_hard={len(hard)} "
            f"reject_reasons={dict(reasons)} reject_rate={reject_rate:.3f}"
        )
        if len(pos) > len(best_pos):
            best_pos = pos
        if len(hard) > len(best_hard):
            best_hard = hard
            best_reasons = reasons
        if len(pos) >= n_pos and len(hard) >= n_neg:
            meta["reject_reasons"] = dict(reasons)
            meta["label_counts_source"] = {
                "y0": sum(1 for r in rows if int(r.get("y", -1)) == 0),
                "y1": sum(1 for r in rows if int(r.get("y", -1)) == 1),
                "y1_id_hop": len(pos),
                "y0_hard": len(hard),
            }
            batch = pos[:n_pos] + hard[:n_neg]
            notes.append(f"balanced batch ready after regen: {n_pos}+{n_neg}")
            return batch, "; ".join(notes), meta

    meta["reject_reasons"] = dict(best_reasons)
    meta["label_counts_source"] = {
        "y1_id_hop": len(best_pos),
        "y0_hard": len(best_hard),
    }
    batch = best_pos[:n_pos] + best_hard[:n_neg]
    notes.append(
        f"fallback: regenerated but y1_id={len(best_pos)}/{n_pos} "
        f"y0_hard={len(best_hard)}/{n_neg}; proceeding with {len(batch)} examples"
    )
    return batch, "; ".join(notes), meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Overfit sanity gate for torch FeedForward (MEASURE plumbing; "
            "no science OPEN). Default: --balanced."
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
    p.add_argument("--L", type=int, default=2, help="Unshared FF depth L.")
    p.add_argument("--lr", type=float, default=3e-3, help="AdamW learning rate.")
    p.add_argument("--seed", type=int, default=0, help="Torch / numpy RNG seed.")
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


def _diagnose_failure(
    losses: list[float],
    accs: list[float],
    *,
    loss_threshold: float,
    per_class: Optional[dict[str, float]] = None,
) -> str:
    """Hint ln2 plateau vs vanishing grads vs other."""
    if not losses:
        return "no steps ran"
    final_loss, final_acc = losses[-1], accs[-1]
    start_loss = losses[0]
    hints: list[str] = []
    if abs(final_loss - LN2) < 0.08 and final_acc < 0.99:
        hints.append(
            f"loss≈ln2 ({LN2:.4f}) plateau — model may be stuck at chance "
            f"(check labels, lr, or capacity)"
        )
    if start_loss > 0 and abs(start_loss - final_loss) / max(start_loss, 1e-8) < 0.05:
        hints.append(
            "loss barely moved from init — possible vanishing gradients, "
            "too-small lr, or broken backward"
        )
    if final_loss >= loss_threshold:
        hints.append(
            f"final loss {final_loss:.4f} not below threshold {loss_threshold}"
        )
    if final_acc < 1.0 - 1e-9:
        hints.append(f"final accuracy {final_acc:.4f} != 1.0")
    if per_class is not None:
        for cls, a in per_class.items():
            if a < 1.0 - 1e-9:
                hints.append(f"per-class {cls} acc={a:.4f} != 1.0")
    if len(losses) >= 10:
        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        if late >= early:
            hints.append(
                f"loss did not trend down (early_mean={early:.4f}, late_mean={late:.4f})"
            )
    return "; ".join(hints) if hints else "unknown failure"


def _per_class_accuracy(
    preds, labels
) -> dict[str, float]:
    """Compute accuracy for y=0 and y=1 subsets (empty class → 0.0)."""
    import torch

    out: dict[str, float] = {}
    for cls in (0, 1):
        mask = labels == cls
        n = int(mask.sum().item())
        if n == 0:
            out[f"y{cls}"] = 0.0
            out[f"y{cls}_n"] = 0.0
            out[f"y{cls}_correct"] = 0.0
        else:
            correct = int((preds[mask] == labels[mask]).sum().item())
            out[f"y{cls}"] = correct / n
            out[f"y{cls}_n"] = float(n)
            out[f"y{cls}_correct"] = float(correct)
    return out


def run_overfit(
    examples: list[dict[str, Any]],
    *,
    steps: int = 100,
    d: int = 64,
    L: int = 2,
    lr: float = 3e-3,
    seed: int = 0,
    loss_threshold: float = 0.05,
    require_per_class: bool = False,
) -> dict[str, Any]:
    """Train FF on a fixed batch; return diagnostics dict."""
    import torch

    from reachability_gen.models.feedforward import FeedForward
    from reachability_gen.tokenize import build_vocab
    from reachability_gen.train.ff_trainer import FeedForwardTrainer, examples_to_batch

    torch.manual_seed(seed)
    vocab = build_vocab()
    token_ids, attention_mask, labels, vocab = examples_to_batch(examples, vocab)
    model = FeedForward(
        vocab_size=len(vocab),
        d=d,
        L=L,
        n_heads=4 if d % 4 == 0 else 2,
        max_len=max(int(token_ids.shape[1]) + 8, 64),
        pad_id=vocab.pad_id,
    )
    trainer = FeedForwardTrainer(model, lr=lr, weight_decay=0.01, grad_clip=1.0)

    losses: list[float] = []
    accs: list[float] = []
    per_class_hist: list[dict[str, float]] = []
    passed_at: Optional[int] = None
    for step in range(1, steps + 1):
        loss, acc = trainer.train_step(token_ids, labels, attention_mask)
        losses.append(loss)
        accs.append(acc)
        # Eval preds for per-class (same batch, no extra forward needed after train
        # step — re-run eval for clean metrics).
        eval_loss, eval_acc = trainer.eval_step(token_ids, labels, attention_mask)
        with torch.no_grad():
            logits, _ = trainer.model(token_ids, attention_mask)
            preds = logits.argmax(dim=-1)
            pc = _per_class_accuracy(preds, labels)
        per_class_hist.append(pc)
        # Prefer eval CE for gate (matches "CE loss" wording); train loss is close.
        gate_loss = eval_loss
        class_ok = (not require_per_class) or (
            pc.get("y0", 0.0) >= 1.0 - 1e-9 and pc.get("y1", 0.0) >= 1.0 - 1e-9
        )
        if (
            gate_loss < loss_threshold
            and eval_acc >= 1.0 - 1e-9
            and class_ok
            and passed_at is None
        ):
            passed_at = step
            # Keep final metrics aligned with passing step.
            losses[-1] = gate_loss
            accs[-1] = eval_acc

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
    if require_per_class:
        # Strict: must have observed a passing step (or final satisfies all).
        ok = (passed_at is not None) or (below and acc_ok and class_ok)
    else:
        ok = (passed_at is not None and acc_ok) or (below and acc_ok and trends_down)

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
        "L": L,
        "lr": lr,
        "loss_threshold": loss_threshold,
        "require_per_class": require_per_class,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "FAIL: torch is required for overfit_ff (pip install torch)",
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
        # Enforce exact class counts.
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
        if len(batch) < args.batch_size:
            print(
                f"WARN: using {len(batch)} < {args.batch_size} ID-hop examples "
                f"(documented fallback)",
                file=sys.stderr,
            )
        require_per_class = False

    result = run_overfit(
        batch,
        steps=args.steps,
        d=args.d,
        L=args.L,
        lr=args.lr,
        seed=args.seed,
        loss_threshold=loss_threshold,
        require_per_class=require_per_class,
    )

    pc = result.get("per_class") or {}
    print(
        f"overfit: balanced={args.balanced} n={result['n_examples']} "
        f"labels={result['label_counts']} steps={result['steps_run']} "
        f"passed_at={result['passed_at']} "
        f"final_loss={result['final_loss']:.6f} final_acc={result['final_acc']:.4f} "
        f"acc_y0={pc.get('y0')} ({int(pc.get('y0_correct', 0))}/{int(pc.get('y0_n', 0))}) "
        f"acc_y1={pc.get('y1')} ({int(pc.get('y1_correct', 0))}/{int(pc.get('y1_n', 0))}) "
        f"early_mean={result['early_mean_loss']:.4f} late_mean={result['late_mean_loss']:.4f}",
        file=sys.stderr,
    )
    print(
        json.dumps(
            {
                "ok": result["ok"],
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
                "reject_reasons": meta.get("reject_reasons"),
            },
            sort_keys=True,
        )
    )

    if result["ok"]:
        print(
            "PASS: overfit gate "
            + (
                f"(balanced: loss < {loss_threshold}, per-class acc=1.0)"
                if args.balanced
                else "(loss < threshold, acc=1.0)"
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
    print(f"FAIL: overfit gate — {diag}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

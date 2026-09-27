# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Full ID training for torch FeedForward after balanced overfit PASS.

Trains on ID examples (y=1 with K in [2,6] preferred; y=0 hard negatives
included). Validates with hop-stratified logging (K in {2,3,4,5,6}) via
RunMetricRecord rows. Geo / Loop / CoT remain stubs.

Writes:
  - artifacts/ff_id_train_metrics.jsonl
  - artifacts/ff_id_train_summary.json

MEASURE plumbing only — no science OPEN claims.

Usage::

    python -m reachability_gen.train_ff_id
    python -m reachability_gen.train_ff_id --epochs 8 --examples data/train_tiny.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from reachability_gen.adr_invariants import (
    ID_HOP_MAX,
    ID_HOP_MIN,
    K_TRAIN_MAX,
    SCIENCE_OPEN_DEFAULT,
    validate_metric_record,
)
from reachability_gen.generate import generate_split, write_jsonl
from reachability_gen.hard_negatives import filter_hard_negatives
from reachability_gen.overfit_ff import (
    _is_id_hop_pos,
    load_jsonl,
)


def build_id_train_val(
    examples_path: Path,
    *,
    regenerate: bool = True,
    min_train: int = 64,
    val_per_hop: int = 4,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Build ID train set + hop-stratified val (K in {2..6}).

    Train: y=1 ID-hop positives + y=0 hard negatives (as balanced as possible).
    Val: up to ``val_per_hop`` y=1 examples per hop K in {2,3,4,5,6}, held out
    from train when possible; plus a few hard negatives for completeness.
    """
    notes: list[str] = []
    if examples_path.exists():
        rows = load_jsonl(examples_path)
        notes.append(f"loaded {len(rows)} from {examples_path}")
    else:
        rows = []
        notes.append(f"missing {examples_path}")

    def _split_rows(all_rows: list[dict[str, Any]]):
        pos = [r for r in all_rows if _is_id_hop_pos(r)]
        hard, reasons = filter_hard_negatives(all_rows)
        return pos, hard, reasons

    pos, hard, reasons = _split_rows(rows) if rows else ([], [], {})
    notes.append(
        f"initial y1_id={len(pos)} y0_hard={len(hard)} reject={dict(reasons)}"
    )

    need_pos = max(min_train // 2, val_per_hop * 5 + 16)
    need_neg = max(min_train // 2, 16)
    if regenerate and (len(pos) < need_pos or len(hard) < need_neg):
        for n_per_cell in (8, 16, 32, 64):
            examples, rr = generate_split(
                "train",
                n_per_cell=n_per_cell,
                max_rejects=8000,
                prefer_id_hops=True,
                seed_override=seed + n_per_cell,
            )
            write_jsonl(examples_path, examples)
            rows = [ex.to_dict() for ex in examples]
            pos, hard, reasons = _split_rows(rows)
            notes.append(
                f"regen n_per_cell={n_per_cell}: y1_id={len(pos)} y0_hard={len(hard)} "
                f"reject={dict(reasons)} rr={rr:.3f}"
            )
            if len(pos) >= need_pos and len(hard) >= need_neg:
                break

    # Stratify positives by hop for val hold-out.
    by_hop: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in pos:
        by_hop[int(r["hop_distance"])].append(r)

    val_pos: list[dict[str, Any]] = []
    train_pos: list[dict[str, Any]] = []
    for k in range(ID_HOP_MIN, ID_HOP_MAX + 1):
        bucket = list(by_hop.get(k, []))
        # Deterministic order by seed/edge_hash.
        bucket.sort(key=lambda r: (int(r.get("seed", 0)), str(r.get("edge_hash", ""))))
        take = min(val_per_hop, len(bucket))
        val_pos.extend(bucket[:take])
        train_pos.extend(bucket[take:])

    # Hard-neg train / small val slice.
    hard_sorted = sorted(
        hard, key=lambda r: (int(r.get("seed", 0)), str(r.get("edge_hash", "")))
    )
    val_neg = hard_sorted[: min(val_per_hop, len(hard_sorted))]
    train_neg = hard_sorted[len(val_neg) :]

    # Balance train roughly 50/50 by truncating the larger class.
    n_bal = min(len(train_pos), len(train_neg))
    if n_bal < 8:
        notes.append(
            f"WARN: small train after hold-out (pos={len(train_pos)} neg={len(train_neg)})"
        )
        train = train_pos + train_neg
    else:
        train = train_pos[:n_bal] + train_neg[:n_bal]

    val = val_pos + val_neg
    notes.append(
        f"train={len(train)} (y1={sum(1 for r in train if int(r['y'])==1)}, "
        f"y0={sum(1 for r in train if int(r['y'])==0)}); "
        f"val={len(val)} (pos_by_hop="
        + ",".join(
            f"{k}:{sum(1 for r in val_pos if int(r['hop_distance'])==k)}"
            for k in range(ID_HOP_MIN, ID_HOP_MAX + 1)
        )
        + f", neg={len(val_neg)})"
    )
    return train, val, "; ".join(notes)


def _write_metric_row(fh, record: dict[str, Any]) -> None:
    validate_metric_record(record)
    fh.write(json.dumps(record, sort_keys=True) + "\n")


def run_id_train(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    *,
    epochs: int = 8,
    d: int = 64,
    L: int = 2,
    lr: float = 3e-3,
    seed: int = 0,
    batch_size: int = 32,
    metrics_path: Path = Path("artifacts/ff_id_train_metrics.jsonl"),
    summary_path: Path = Path("artifacts/ff_id_train_summary.json"),
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    """Train FF on ID set; log hop-stratified val RunMetricRecords."""
    import torch

    from reachability_gen.flops import flops_feedforward
    from reachability_gen.models.feedforward import FeedForward
    from reachability_gen.tokenize import build_vocab
    from reachability_gen.train.ff_trainer import FeedForwardTrainer, examples_to_batch

    torch.manual_seed(seed)
    run_id = run_id or f"ff-id-{uuid.uuid4().hex[:10]}"
    vocab = build_vocab()
    # Determine max seq len from train+val.
    probe = examples_to_batch(train[:1] + val[:1] if val else train[:1], vocab)
    max_len = max(int(probe[0].shape[1]) + 8, 64)
    # Re-encode full train for length consistency.
    all_for_len = train + val
    _, _, _, vocab = examples_to_batch(all_for_len, vocab, max_len=max_len)

    model = FeedForward(
        vocab_size=len(vocab),
        d=d,
        L=L,
        n_heads=4 if d % 4 == 0 else 2,
        max_len=max_len,
        pad_id=vocab.pad_id,
    )
    trainer = FeedForwardTrainer(model, lr=lr, weight_decay=0.01, grad_clip=1.0)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Prefer non-embedding schematic for arm-ish logging; use real count.
    arm_name = f"ff-L{L}-d{d}"

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    train_hist: list[dict[str, Any]] = []
    val_by_hop_final: dict[str, Any] = {}

    with metrics_path.open("w", encoding="utf-8") as fh:
        for epoch in range(1, epochs + 1):
            # Shuffle train indices.
            order = torch.randperm(len(train)).tolist()
            epoch_losses: list[float] = []
            epoch_accs: list[float] = []
            for start in range(0, len(train), batch_size):
                idx = order[start : start + batch_size]
                batch = [train[i] for i in idx]
                token_ids, mask, labels, _ = examples_to_batch(
                    batch, vocab, max_len=max_len
                )
                loss, acc = trainer.train_step(token_ids, labels, mask)
                epoch_losses.append(loss)
                epoch_accs.append(acc)
                step += 1

            train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            train_acc = sum(epoch_accs) / max(len(epoch_accs), 1)
            train_hist.append(
                {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc}
            )

            # Hop-stratified val: one RunMetricRecord per val example.
            hop_stats: dict[int, list[tuple[float, float]]] = defaultdict(list)
            for ex in val:
                token_ids, mask, labels, _ = examples_to_batch(
                    [ex], vocab, max_len=max_len
                )
                vloss, vacc = trainer.eval_step(token_ids, labels, mask)
                hop = int(ex.get("hop_distance", -1))
                hop_stats[hop].append((vloss, vacc))
                enc = str(ex.get("encoding", ""))
                seq_len = max(len(enc.split()), 1)
                flop_report = flops_feedforward(seq_len, d=d, L=L)
                record = {
                    "run_id": run_id,
                    "seed": int(ex.get("seed", seed)),
                    "arm": arm_name,
                    "step": int(step),
                    "epoch": int(epoch),
                    "param_count": int(param_count),
                    "d_model": int(d),
                    "seq_len": int(seq_len),
                    "cycles_or_depth": int(L),
                    "tokens_decoded": None,
                    "cumulative_flops": float(flop_report.flops),
                    "hop_distance": hop,
                    "is_ood": bool(ex.get("is_ood", False)),
                    "loss": float(vloss),
                    "accuracy": float(vacc),
                    "drift_trajectory": [],
                    "terminal_drift": None,
                    "perturbation_delta": None,
                    "split": ex.get("split", "train"),
                    "n": int(ex.get("n", 0)),
                    "p": float(ex.get("p", 0.0)),
                    "edge_hash": ex.get("edge_hash", ""),
                    "s": int(ex.get("s", 0)),
                    "t": int(ex.get("t", 0)),
                    "y": int(ex.get("y", 0)),
                    "science_open": SCIENCE_OPEN_DEFAULT,
                }
                _write_metric_row(fh, record)

            # Also log a summary-ish aggregate row per hop (as epoch marker via
            # y=-1 sentinel would break schema; keep only per-example rows).
            val_by_hop: dict[str, Any] = {}
            for k, pairs in sorted(hop_stats.items()):
                losses = [a for a, _ in pairs]
                accs = [b for _, b in pairs]
                val_by_hop[str(k)] = {
                    "n": len(pairs),
                    "loss_mean": sum(losses) / len(losses),
                    "acc_mean": sum(accs) / len(accs),
                }
            val_by_hop_final = val_by_hop
            print(
                f"epoch {epoch}/{epochs}: train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} val_by_hop={val_by_hop}",
                file=sys.stderr,
            )

    summary = {
        "run_id": run_id,
        "science_open": False,
        "arm": arm_name,
        "d": d,
        "L": L,
        "lr": lr,
        "epochs": epochs,
        "seed": seed,
        "n_train": len(train),
        "n_val": len(val),
        "train_label_counts": {
            "y0": sum(1 for e in train if int(e["y"]) == 0),
            "y1": sum(1 for e in train if int(e["y"]) == 1),
        },
        "val_label_counts": {
            "y0": sum(1 for e in val if int(e["y"]) == 0),
            "y1": sum(1 for e in val if int(e["y"]) == 1),
        },
        "train_history": train_hist,
        "val_by_hop": val_by_hop_final,
        "metrics_path": str(metrics_path),
        "param_count": int(param_count),
        "id_hop_range": [ID_HOP_MIN, ID_HOP_MAX],
        "k_train_max": K_TRAIN_MAX,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Full ID FF training with hop-stratified val (no science OPEN)."
    )
    p.add_argument(
        "--examples",
        type=Path,
        default=Path("data/train_tiny.jsonl"),
        help="Train JSONL (regenerated if scarce).",
    )
    p.add_argument("--epochs", type=int, default=8, help="Modest epoch count.")
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--L", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("artifacts/ff_id_train_metrics.jsonl"),
    )
    p.add_argument(
        "--summary-out",
        type=Path,
        default=Path("artifacts/ff_id_train_summary.json"),
    )
    p.add_argument("--no-regenerate", action="store_true")
    p.add_argument("--val-per-hop", type=int, default=4)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("FAIL: torch required for train_ff_id", file=sys.stderr)
        return 2

    train, val, note = build_id_train_val(
        args.examples,
        regenerate=not args.no_regenerate,
        val_per_hop=args.val_per_hop,
        seed=args.seed,
    )
    print(f"data: {note}", file=sys.stderr)
    if len(train) < 4:
        print(f"FAIL: train set too small ({len(train)})", file=sys.stderr)
        return 1
    if not val:
        print("FAIL: empty val set (need hop-stratified examples)", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    summary = run_id_train(
        train,
        val,
        epochs=args.epochs,
        d=args.d,
        L=args.L,
        lr=args.lr,
        seed=args.seed,
        batch_size=args.batch_size,
        metrics_path=args.metrics_out,
        summary_path=args.summary_out,
    )
    elapsed = time.perf_counter() - t0
    print(
        json.dumps(
            {
                "ok": True,
                "elapsed_s": elapsed,
                "n_train": summary["n_train"],
                "n_val": summary["n_val"],
                "val_by_hop": summary["val_by_hop"],
                "summary_path": str(args.summary_out),
                "metrics_path": str(args.metrics_out),
            },
            sort_keys=True,
        )
    )
    print(
        f"PASS: ID train complete → {args.summary_out} / {args.metrics_out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

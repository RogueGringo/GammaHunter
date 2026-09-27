# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Regenerate id_2k + train FF L=2 and Geo T=6 on the same splits (MEASURE).

Writes ``artifacts/id_2k_comparison.json`` with hop-stratified val metrics for
K in {2,3,4,5,6} and negatives (hop=-1). Geo rows include mean drift_trajectory
/ terminal_drift per hop bucket and a damp vs expand/saturate note.

science_open=False always — no OPEN claims.

Usage::

    python -m reachability_gen.run_id_2k_comparison
    python -m reachability_gen.run_id_2k_comparison --epochs 30 --skip-gen
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
    HOP_UNREACHABLE,
    ID_HOP_MAX,
    ID_HOP_MIN,
    SCIENCE_OPEN_DEFAULT,
    validate_metric_record,
)
from reachability_gen.gen_id_2k import (
    ID_2K_SEED,
    generate_id_2k,
    verify_id_2k,
    write_jsonl,
    write_report,
)
from reachability_gen.overfit_ff import load_jsonl

# Default MEASURE epoch budget: modest but serious; early-stop on plateau.
DEFAULT_EPOCHS: int = 30
PLATEAU_PATIENCE: int = 5
PLATEAU_MIN_DELTA: float = 1e-4
DEFAULT_D: int = 64
DEFAULT_L: int = 2
DEFAULT_T: int = 6
DEFAULT_LR: float = 3e-3
DEFAULT_BATCH: int = 32


def _split_train_val(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [r for r in rows if r.get("split") == "train"]
    val = [r for r in rows if r.get("split") == "val"]
    return train, val


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _mean_traj(trajs: list[list[float]]) -> list[float]:
    if not trajs:
        return []
    L = min(len(t) for t in trajs)
    if L == 0:
        return []
    return [_mean([t[i] for t in trajs]) for i in range(L)]


def _drift_regime(mean_traj: list[float]) -> str:
    """Classify terminal drift: damps (δ_T < δ_{T-1}) vs expands/saturates."""
    if len(mean_traj) < 2:
        return "unknown"
    d_tm1, d_t = mean_traj[-2], mean_traj[-1]
    if d_t < d_tm1 - 1e-12:
        return "damps"
    if abs(d_t - d_tm1) <= 1e-6 * max(abs(d_tm1), 1.0):
        return "saturates"
    return "expands"


def _write_metric(fh, record: dict[str, Any]) -> None:
    validate_metric_record(record)
    fh.write(json.dumps(record, sort_keys=True) + "\n")


def train_ff_on_id2k(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    *,
    epochs: int = DEFAULT_EPOCHS,
    d: int = DEFAULT_D,
    L: int = DEFAULT_L,
    lr: float = DEFAULT_LR,
    seed: int = 0,
    batch_size: int = DEFAULT_BATCH,
    metrics_path: Path = Path("artifacts/id_2k_ff_metrics.jsonl"),
    plateau_patience: int = PLATEAU_PATIENCE,
) -> dict[str, Any]:
    """Train FF L=2; log RunMetricRecords on val; return summary dict."""
    import torch

    from reachability_gen.flops import flops_feedforward
    from reachability_gen.models.feedforward import FeedForward
    from reachability_gen.tokenize import build_vocab
    from reachability_gen.train.ff_trainer import FeedForwardTrainer, examples_to_batch

    torch.manual_seed(seed)
    run_id = f"ff-id2k-{uuid.uuid4().hex[:10]}"
    vocab = build_vocab()
    all_rows = train + val
    probe = examples_to_batch(all_rows[:2], vocab)
    max_len = max(int(probe[0].shape[1]) + 8, 64)
    _, _, _, vocab = examples_to_batch(all_rows, vocab, max_len=max_len)

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
    arm_name = f"ff-L{L}-d{d}"

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    train_hist: list[dict[str, Any]] = []
    val_by_hop_final: dict[str, Any] = {}
    best_train_loss = float("inf")
    plateau = 0
    epochs_run = 0

    with metrics_path.open("w", encoding="utf-8") as fh:
        for epoch in range(1, epochs + 1):
            epochs_run = epoch
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

            train_loss = _mean(epoch_losses)
            train_acc = _mean(epoch_accs)
            train_hist.append(
                {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc}
            )

            hop_stats: dict[int, list[tuple[float, float]]] = defaultdict(list)
            for ex in val:
                token_ids, mask, labels, _ = examples_to_batch(
                    [ex], vocab, max_len=max_len
                )
                vloss, vacc = trainer.eval_step(token_ids, labels, mask)
                hop = int(ex.get("hop_distance", HOP_UNREACHABLE))
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
                    "split": ex.get("split", "val"),
                    "n": int(ex.get("n", 0)),
                    "p": float(ex.get("p", 0.0)),
                    "edge_hash": ex.get("edge_hash", ""),
                    "s": int(ex.get("s", 0)),
                    "t": int(ex.get("t", 0)),
                    "y": int(ex.get("y", 0)),
                    "science_open": SCIENCE_OPEN_DEFAULT,
                }
                _write_metric(fh, record)

            val_by_hop: dict[str, Any] = {}
            for k, pairs in sorted(hop_stats.items()):
                losses = [a for a, _ in pairs]
                accs = [b for _, b in pairs]
                val_by_hop[str(k)] = {
                    "n": len(pairs),
                    "loss_mean": _mean(losses),
                    "acc_mean": _mean(accs),
                }
            val_by_hop_final = val_by_hop
            print(
                f"[FF] epoch {epoch}/{epochs}: train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f}",
                file=sys.stderr,
            )

            if train_loss < best_train_loss - PLATEAU_MIN_DELTA:
                best_train_loss = train_loss
                plateau = 0
            else:
                plateau += 1
                if plateau >= plateau_patience and epoch >= 10:
                    print(
                        f"[FF] early stop at epoch {epoch} "
                        f"(train plateau patience={plateau_patience})",
                        file=sys.stderr,
                    )
                    break

    return {
        "run_id": run_id,
        "arm": arm_name,
        "d": d,
        "L": L,
        "lr": lr,
        "epochs_requested": epochs,
        "epochs_run": epochs_run,
        "early_stop": epochs_run < epochs,
        "seed": seed,
        "n_train": len(train),
        "n_val": len(val),
        "param_count": int(param_count),
        "train_history": train_hist,
        "val_by_hop": val_by_hop_final,
        "metrics_path": str(metrics_path),
        "science_open": False,
    }


def train_geo_on_id2k(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    *,
    epochs: int = DEFAULT_EPOCHS,
    d: int = DEFAULT_D,
    T: int = DEFAULT_T,
    lr: float = DEFAULT_LR,
    seed: int = 0,
    batch_size: int = DEFAULT_BATCH,
    use_tau: bool = True,
    metrics_path: Path = Path("artifacts/id_2k_geo_metrics.jsonl"),
    plateau_patience: int = PLATEAU_PATIENCE,
) -> dict[str, Any]:
    """Train Geo T=6 with tau; log val RunMetricRecords with real drift."""
    import torch

    from reachability_gen.flops import flops_geometric
    from reachability_gen.models.geometric import GeometricRecurrent
    from reachability_gen.tokenize import build_vocab
    from reachability_gen.train.ff_trainer import examples_to_batch
    from reachability_gen.train.geo_trainer import GeometricTrainer

    torch.manual_seed(seed)
    run_id = f"geo-id2k-{uuid.uuid4().hex[:10]}"
    vocab = build_vocab()
    all_rows = train + val
    probe = examples_to_batch(all_rows[:2], vocab)
    max_len = max(int(probe[0].shape[1]) + 8, 64)
    _, _, _, vocab = examples_to_batch(all_rows, vocab, max_len=max_len)

    model = GeometricRecurrent(
        vocab_size=len(vocab),
        d=d,
        T=T,
        n_heads=4 if d % 4 == 0 else 2,
        max_len=max_len,
        pad_id=vocab.pad_id,
        use_tau=use_tau,
    )
    trainer = GeometricTrainer(model, lr=lr, weight_decay=0.01, grad_clip=1.0)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    arm_name = f"geo-T{T}-d{d}" + ("-tau" if use_tau else "-notau")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    train_hist: list[dict[str, Any]] = []
    val_by_hop_final: dict[str, Any] = {}
    best_train_loss = float("inf")
    plateau = 0
    epochs_run = 0

    with metrics_path.open("w", encoding="utf-8") as fh:
        for epoch in range(1, epochs + 1):
            epochs_run = epoch
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

            train_loss = _mean(epoch_losses)
            train_acc = _mean(epoch_accs)
            train_hist.append(
                {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc}
            )

            hop_losses: dict[int, list[float]] = defaultdict(list)
            hop_accs: dict[int, list[float]] = defaultdict(list)
            hop_trajs: dict[int, list[list[float]]] = defaultdict(list)
            hop_terminals: dict[int, list[float]] = defaultdict(list)

            for ex in val:
                token_ids, mask, labels, _ = examples_to_batch(
                    [ex], vocab, max_len=max_len
                )
                vloss, vacc, drifts = trainer.eval_step(
                    token_ids, labels, mask, return_drift=True
                )  # type: ignore[misc]
                hop = int(ex.get("hop_distance", HOP_UNREACHABLE))
                hop_losses[hop].append(float(vloss))
                hop_accs[hop].append(float(vacc))
                hop_trajs[hop].append(list(drifts))
                if drifts:
                    hop_terminals[hop].append(float(drifts[-1]))
                enc = str(ex.get("encoding", ""))
                seq_len = max(len(enc.split()), 1)
                flop_report = flops_geometric(seq_len, d=d, T=T, use_tau=use_tau)
                record = {
                    "run_id": run_id,
                    "seed": int(ex.get("seed", seed)),
                    "arm": arm_name,
                    "step": int(step),
                    "epoch": int(epoch),
                    "param_count": int(param_count),
                    "d_model": int(d),
                    "seq_len": int(seq_len),
                    "cycles_or_depth": int(T),
                    "tokens_decoded": None,
                    "cumulative_flops": float(flop_report.flops),
                    "hop_distance": hop,
                    "is_ood": bool(ex.get("is_ood", False)),
                    "loss": float(vloss),
                    "accuracy": float(vacc),
                    "drift_trajectory": list(drifts),
                    "terminal_drift": float(drifts[-1]) if drifts else None,
                    "perturbation_delta": None,
                    "split": ex.get("split", "val"),
                    "n": int(ex.get("n", 0)),
                    "p": float(ex.get("p", 0.0)),
                    "edge_hash": ex.get("edge_hash", ""),
                    "s": int(ex.get("s", 0)),
                    "t": int(ex.get("t", 0)),
                    "y": int(ex.get("y", 0)),
                    "science_open": SCIENCE_OPEN_DEFAULT,
                }
                _write_metric(fh, record)

            val_by_hop: dict[str, Any] = {}
            for k in sorted(set(hop_losses) | set(hop_accs)):
                mean_traj = _mean_traj(hop_trajs.get(k, []))
                terminals = hop_terminals.get(k, [])
                regime = _drift_regime(mean_traj)
                val_by_hop[str(k)] = {
                    "n": len(hop_losses[k]),
                    "loss_mean": _mean(hop_losses[k]),
                    "acc_mean": _mean(hop_accs[k]),
                    "mean_drift_trajectory": mean_traj,
                    "mean_terminal_drift": _mean(terminals) if terminals else None,
                    "terminal_drift_regime": regime,
                    "terminal_drift_damps": regime == "damps",
                }
            val_by_hop_final = val_by_hop
            print(
                f"[Geo] epoch {epoch}/{epochs}: train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f}",
                file=sys.stderr,
            )

            if train_loss < best_train_loss - PLATEAU_MIN_DELTA:
                best_train_loss = train_loss
                plateau = 0
            else:
                plateau += 1
                if plateau >= plateau_patience and epoch >= 10:
                    print(
                        f"[Geo] early stop at epoch {epoch} "
                        f"(train plateau patience={plateau_patience})",
                        file=sys.stderr,
                    )
                    break

    # Drift summary from final val_by_hop.
    regimes = {
        k: v.get("terminal_drift_regime")
        for k, v in val_by_hop_final.items()
    }
    damp_flags = [
        v.get("terminal_drift_damps")
        for v in val_by_hop_final.values()
        if v.get("terminal_drift_damps") is not None
    ]
    drift_summary = {
        "by_hop_regime": regimes,
        "fraction_buckets_damping": (
            sum(1 for x in damp_flags if x) / len(damp_flags) if damp_flags else None
        ),
        "note": (
            "damps means mean δ_T < δ_{T-1} on the hop bucket; "
            "else expands or saturates"
        ),
    }

    return {
        "run_id": run_id,
        "arm": arm_name,
        "d": d,
        "T": T,
        "use_tau": use_tau,
        "lr": lr,
        "epochs_requested": epochs,
        "epochs_run": epochs_run,
        "early_stop": epochs_run < epochs,
        "seed": seed,
        "n_train": len(train),
        "n_val": len(val),
        "param_count": int(param_count),
        "train_history": train_hist,
        "val_by_hop": val_by_hop_final,
        "drift_summary": drift_summary,
        "metrics_path": str(metrics_path),
        "science_open": False,
    }


def build_comparison(
    ff_summary: dict[str, Any],
    geo_summary: dict[str, Any],
    *,
    generation_report: Optional[dict[str, Any]] = None,
    dataset_path: str = "data/id_2k.jsonl",
    epochs_choice: str = "",
) -> dict[str, Any]:
    """Assemble artifacts/id_2k_comparison.json (science_open=False)."""
    ff_pc = int(ff_summary.get("param_count", 0))
    geo_pc = int(geo_summary.get("param_count", 0))
    ref = max(ff_pc, 1)
    ratio = geo_pc / ref
    param_match = {
        "ff_param_count": ff_pc,
        "geo_param_count": geo_pc,
        "geo_over_ff_ratio": ratio,
        "within_5pct": abs(ratio - 1.0) <= 0.05,
        "note": (
            "FF L=2 unshared vs Geo T=6 weight-tied (+ optional tau). "
            "Parity is approximate for MEASURE plumbing; never stamps OPEN."
        ),
        "science_open": False,
    }
    return {
        "science_open": False,
        "dataset": dataset_path,
        "epochs_choice": epochs_choice
        or (
            f"request {DEFAULT_EPOCHS} epochs with early-stop on train-loss "
            f"plateau (patience={PLATEAU_PATIENCE}, min_delta={PLATEAU_MIN_DELTA}, "
            f"min_epochs=10)"
        ),
        "id_hop_range": [ID_HOP_MIN, ID_HOP_MAX],
        "negative_hop": HOP_UNREACHABLE,
        "generation_report": generation_report,
        "param_match": param_match,
        "ff": {
            "arm": ff_summary.get("arm"),
            "param_count": ff_pc,
            "epochs_run": ff_summary.get("epochs_run"),
            "early_stop": ff_summary.get("early_stop"),
            "train_history": ff_summary.get("train_history"),
            "val_by_hop": ff_summary.get("val_by_hop"),
            "metrics_path": ff_summary.get("metrics_path"),
            "science_open": False,
        },
        "geo": {
            "arm": geo_summary.get("arm"),
            "param_count": geo_pc,
            "T": geo_summary.get("T"),
            "use_tau": geo_summary.get("use_tau"),
            "epochs_run": geo_summary.get("epochs_run"),
            "early_stop": geo_summary.get("early_stop"),
            "train_history": geo_summary.get("train_history"),
            "val_by_hop": geo_summary.get("val_by_hop"),
            "drift_summary": geo_summary.get("drift_summary"),
            "metrics_path": geo_summary.get("metrics_path"),
            "science_open": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Regenerate id_2k + train FF L=2 / Geo T=6 comparison "
            "(MEASURE plumbing; science_open=false)."
        )
    )
    p.add_argument(
        "--data-out",
        type=Path,
        default=Path("data/id_2k.jsonl"),
    )
    p.add_argument(
        "--report-out",
        type=Path,
        default=Path("artifacts/id_2k_generation_report.json"),
    )
    p.add_argument(
        "--comparison-out",
        type=Path,
        default=Path("artifacts/id_2k_comparison.json"),
    )
    p.add_argument(
        "--ff-metrics-out",
        type=Path,
        default=Path("artifacts/id_2k_ff_metrics.jsonl"),
    )
    p.add_argument(
        "--geo-metrics-out",
        type=Path,
        default=Path("artifacts/id_2k_geo_metrics.jsonl"),
    )
    p.add_argument("--seed", type=int, default=ID_2K_SEED)
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--d", type=int, default=DEFAULT_D)
    p.add_argument("--L", type=int, default=DEFAULT_L)
    p.add_argument("--T", type=int, default=DEFAULT_T)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--skip-gen", action="store_true", help="Reuse existing id_2k.jsonl")
    p.add_argument("--ff-only", action="store_true")
    p.add_argument("--geo-only", action="store_true")
    p.add_argument("--no-tau", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("FAIL: torch required for id_2k comparison", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    generation_report: Optional[dict[str, Any]] = None

    if args.skip_gen:
        if not args.data_out.exists():
            print(f"FAIL: --skip-gen but missing {args.data_out}", file=sys.stderr)
            return 1
        rows = load_jsonl(args.data_out)
        ok, issues = verify_id_2k(rows)
        if not ok:
            print(f"FAIL: existing id_2k failed verify: {issues}", file=sys.stderr)
            return 1
        if args.report_out.exists():
            generation_report = json.loads(args.report_out.read_text())
        print(f"reusing {args.data_out} ({len(rows)} rows)", file=sys.stderr)
    else:
        examples, generation_report = generate_id_2k(seed=args.seed)
        write_jsonl(args.data_out, examples)
        ok, issues = verify_id_2k(examples)
        generation_report["verify_ok"] = ok
        generation_report["verify_issues"] = issues
        write_report(args.report_out, generation_report)
        if not ok:
            print(f"FAIL: generated id_2k failed verify: {issues}", file=sys.stderr)
            return 1
        rows = [ex.to_dict() for ex in examples]
        print(
            f"generated {len(rows)} → {args.data_out}; report → {args.report_out}",
            file=sys.stderr,
        )

    train, val = _split_train_val(rows)
    print(
        f"splits: train={len(train)} val={len(val)} "
        f"(train y1={sum(1 for r in train if int(r['y'])==1)}, "
        f"val y1={sum(1 for r in val if int(r['y'])==1)})",
        file=sys.stderr,
    )

    do_ff = not args.geo_only
    do_geo = not args.ff_only
    ff_summary: dict[str, Any] = {}
    geo_summary: dict[str, Any] = {}

    if do_ff:
        ff_summary = train_ff_on_id2k(
            train,
            val,
            epochs=args.epochs,
            d=args.d,
            L=args.L,
            lr=args.lr,
            seed=args.train_seed,
            batch_size=args.batch_size,
            metrics_path=args.ff_metrics_out,
        )
    if do_geo:
        geo_summary = train_geo_on_id2k(
            train,
            val,
            epochs=args.epochs,
            d=args.d,
            T=args.T,
            lr=args.lr,
            seed=args.train_seed,
            batch_size=args.batch_size,
            use_tau=not args.no_tau,
            metrics_path=args.geo_metrics_out,
        )

    # If only one arm, still write a partial comparison when possible.
    if do_ff and do_geo:
        comparison = build_comparison(
            ff_summary,
            geo_summary,
            generation_report=generation_report,
            dataset_path=str(args.data_out),
            epochs_choice=(
                f"request {args.epochs} epochs; early-stop on train-loss plateau "
                f"(patience={PLATEAU_PATIENCE}, min_delta={PLATEAU_MIN_DELTA}, "
                f"min_epochs=10); identical train/val from id_2k; "
                f"FF L={args.L} vs Geo T={args.T} tau={not args.no_tau}; "
                f"d={args.d} lr={args.lr} batch={args.batch_size} train_seed={args.train_seed}"
            ),
        )
    else:
        comparison = {
            "science_open": False,
            "dataset": str(args.data_out),
            "ff": ff_summary or None,
            "geo": geo_summary or None,
            "generation_report": generation_report,
            "note": "partial run (--ff-only / --geo-only)",
        }

    # Hard guarantee: never stamp OPEN.
    comparison["science_open"] = False
    if "ff" in comparison and isinstance(comparison["ff"], dict):
        comparison["ff"]["science_open"] = False
    if "geo" in comparison and isinstance(comparison["geo"], dict):
        comparison["geo"]["science_open"] = False

    args.comparison_out.parent.mkdir(parents=True, exist_ok=True)
    args.comparison_out.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    elapsed = time.perf_counter() - t0

    # Print stratified tables for the parent/user.
    def _print_table(label: str, by_hop: dict[str, Any], with_drift: bool) -> None:
        print(f"\n=== {label} val_by_hop ===", file=sys.stderr)
        hdr = f"{'hop':>4} {'n':>5} {'acc_mean':>10} {'loss_mean':>10}"
        if with_drift:
            hdr += f" {'term_drift':>12} {'regime':>10}"
        print(hdr, file=sys.stderr)
        for k in sorted(by_hop.keys(), key=lambda x: int(x)):
            v = by_hop[k]
            line = (
                f"{k:>4} {v.get('n', 0):>5} {v.get('acc_mean', float('nan')):>10.4f} "
                f"{v.get('loss_mean', float('nan')):>10.4f}"
            )
            if with_drift:
                td = v.get("mean_terminal_drift")
                td_s = f"{td:.6f}" if isinstance(td, (int, float)) else "null"
                line += f" {td_s:>12} {str(v.get('terminal_drift_regime', '')):>10}"
            print(line, file=sys.stderr)

    if ff_summary:
        _print_table("FF", ff_summary.get("val_by_hop") or {}, with_drift=False)
    if geo_summary:
        _print_table("Geo", geo_summary.get("val_by_hop") or {}, with_drift=True)
        print(
            f"\nGeo drift_summary: {json.dumps(geo_summary.get('drift_summary'), sort_keys=True)}",
            file=sys.stderr,
        )

    print(
        json.dumps(
            {
                "ok": True,
                "elapsed_s": elapsed,
                "comparison_out": str(args.comparison_out),
                "data_out": str(args.data_out),
                "science_open": False,
                "ff_epochs_run": ff_summary.get("epochs_run"),
                "geo_epochs_run": geo_summary.get("epochs_run"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

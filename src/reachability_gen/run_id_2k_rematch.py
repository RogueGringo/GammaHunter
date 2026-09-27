# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Param-matched rematch: FF L=2 vs Geo T=6 (τ) vs Euclidean loop T=6 (MEASURE).

Uses the exact existing ``data/id_2k.jsonl`` (does NOT regenerate). Scales
Geo / Loop ``d`` and/or MLP expansion so real ``sum(p.numel())`` lands within
±5% of the FF L=2 baseline (~121218 → window [115157, 127279]). Hard-fails
before any recurrent training if either arm is outside that window.

science_open=False always — no OPEN claims.

Usage::

    python -m reachability_gen.run_id_2k_rematch
    python -m reachability_gen.run_id_2k_rematch --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from reachability_gen.adr_invariants import (
    HOP_UNREACHABLE,
    ID_HOP_MAX,
    ID_HOP_MIN,
    PARAM_TOL,
    SCIENCE_OPEN_DEFAULT,
    validate_metric_record,
)
from reachability_gen.gen_id_2k import verify_id_2k
from reachability_gen.overfit_ff import load_jsonl

# Shared with id_2k comparison run.
DEFAULT_EPOCHS: int = 30
PLATEAU_PATIENCE: int = 5
PLATEAU_MIN_DELTA: float = 1e-4
DEFAULT_FF_D: int = 64
DEFAULT_L: int = 2
DEFAULT_T: int = 6
DEFAULT_LR: float = 3e-3
DEFAULT_BATCH: int = 32
FF_BASELINE_PARAMS: int = 121_218  # known FF L=2 on id_2k vocab/max_len
EPS_SIGMA: float = 1e-4


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


def param_window(ref: int, tol: float = PARAM_TOL) -> tuple[int, int]:
    """Inclusive ±tol window around ``ref`` (matches user [115157, 127279])."""
    lo = int(ref * (1.0 - tol))
    hi = int(round(ref * (1.0 + tol)))
    return lo, hi


def within_5pct(count: int, ref: int, tol: float = PARAM_TOL) -> bool:
    """True iff ``count`` is inside the inclusive ±tol window around ``ref``.

    Uses :func:`param_window` (int floor / round) so the published bounds
    ``[115157, 127279]`` for FF=121218 are accepted end-to-end.
    """
    if ref <= 0:
        return False
    lo, hi = param_window(ref, tol)
    return lo <= int(count) <= hi


def _n_heads(d: int) -> int:
    if d % 4 == 0:
        return 4
    if d % 2 == 0:
        return 2
    return 1


@dataclass(frozen=True)
class ScaledConfig:
    d: int
    mlp_expansion: int
    n_heads: int
    param_count: int
    use_tau: bool


def scale_recurrent_to_window(
    *,
    vocab_size: int,
    max_len: int,
    pad_id: int,
    use_tau: bool,
    ref_count: int,
    T: int = DEFAULT_T,
    tol: float = PARAM_TOL,
) -> ScaledConfig:
    """Search d / mlp_expansion so real total params land in ±tol of ``ref_count``.

    Prefers mlp_expansion=4 with larger d, then raises expansion at fixed d.
    """
    import torch  # noqa: F401

    from reachability_gen.models.euclidean_loop import EuclideanLoop
    from reachability_gen.models.geometric import GeometricRecurrent

    lo, hi = param_window(ref_count, tol)
    hits: list[tuple[int, int, int, int]] = []  # |dev|, d, mlp, n

    # Prefer expansion-4 first (user example), then raise mlp / vary d.
    d_candidates = list(range(64, 200, 4))
    mlp_first = [4] + [m for m in range(5, 13) if m != 4]

    for mlp in mlp_first:
        for d in d_candidates:
            heads = _n_heads(d)
            if d % heads != 0:
                continue
            if use_tau:
                model = GeometricRecurrent(
                    vocab_size=vocab_size,
                    d=d,
                    T=T,
                    n_heads=heads,
                    max_len=max_len,
                    pad_id=pad_id,
                    use_tau=True,
                    mlp_expansion=mlp,
                )
            else:
                model = EuclideanLoop(
                    vocab_size=vocab_size,
                    d=d,
                    T=T,
                    n_heads=heads,
                    max_len=max_len,
                    pad_id=pad_id,
                    mlp_expansion=mlp,
                )
            n = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
            if lo <= n <= hi:
                hits.append((abs(n - ref_count), d, mlp, n))

    if not hits:
        raise RuntimeError(
            f"no (d, mlp) lands in [{lo}, {hi}] for use_tau={use_tau} "
            f"vs FF ref={ref_count}"
        )
    hits.sort()
    _, d, mlp, n = hits[0]
    return ScaledConfig(
        d=d,
        mlp_expansion=mlp,
        n_heads=_n_heads(d),
        param_count=n,
        use_tau=use_tau,
    )


def assert_param_parity(
    *,
    ff_count: int,
    geo_count: int,
    loop_count: int,
    tol: float = PARAM_TOL,
) -> dict[str, Any]:
    """Hard-fail helper: raise AssertionError if Geo or Loop outside ±tol of FF."""
    lo, hi = param_window(ff_count, tol)
    geo_ok = within_5pct(geo_count, ff_count, tol)
    loop_ok = within_5pct(loop_count, ff_count, tol)
    section = {
        "ff_param_count": int(ff_count),
        "geo_param_count": int(geo_count),
        "loop_param_count": int(loop_count),
        "window": [lo, hi],
        "tolerance": tol,
        "geo_over_ff_ratio": geo_count / ff_count if ff_count else float("nan"),
        "loop_over_ff_ratio": loop_count / ff_count if ff_count else float("nan"),
        "geo_within_5pct": geo_ok,
        "loop_within_5pct": loop_ok,
        "within_5pct": geo_ok and loop_ok,
        "science_open": False,
    }
    if not geo_ok:
        raise AssertionError(
            f"Geo params {geo_count} outside ±{tol:.0%} of FF {ff_count} "
            f"(window [{lo}, {hi}])"
        )
    if not loop_ok:
        raise AssertionError(
            f"Loop params {loop_count} outside ±{tol:.0%} of FF {ff_count} "
            f"(window [{lo}, {hi}])"
        )
    return section


def train_ff_on_id2k(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    *,
    epochs: int = DEFAULT_EPOCHS,
    d: int = DEFAULT_FF_D,
    L: int = DEFAULT_L,
    lr: float = DEFAULT_LR,
    seed: int = 0,
    batch_size: int = DEFAULT_BATCH,
    metrics_path: Path = Path("artifacts/id_2k_rematch_ff_metrics.jsonl"),
    plateau_patience: int = PLATEAU_PATIENCE,
    max_len: Optional[int] = None,
    vocab: Any = None,
) -> dict[str, Any]:
    """Train FF L=2; log RunMetricRecords on val; return summary dict."""
    import torch

    from reachability_gen.flops import flops_feedforward
    from reachability_gen.models.feedforward import FeedForward
    from reachability_gen.tokenize import build_vocab
    from reachability_gen.train.ff_trainer import FeedForwardTrainer, examples_to_batch

    torch.manual_seed(seed)
    run_id = f"ff-rematch-{uuid.uuid4().hex[:10]}"
    if vocab is None:
        vocab = build_vocab()
    all_rows = train + val
    if max_len is None:
        probe = examples_to_batch(all_rows[:2], vocab)
        max_len = max(int(probe[0].shape[1]) + 8, 64)
    _, _, _, vocab = examples_to_batch(all_rows, vocab, max_len=max_len)

    model = FeedForward(
        vocab_size=len(vocab),
        d=d,
        L=L,
        n_heads=_n_heads(d),
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
        "max_len": int(max_len),
        "vocab_size": len(vocab),
        "science_open": False,
    }


def _train_recurrent_on_id2k(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    *,
    arm_kind: str,  # "geo" | "loop"
    epochs: int,
    d: int,
    T: int,
    mlp_expansion: int,
    lr: float,
    seed: int,
    batch_size: int,
    metrics_path: Path,
    plateau_patience: int,
    max_len: int,
    vocab: Any,
    use_tau: bool,
) -> dict[str, Any]:
    """Shared Geo / Loop training with drift + perturbation telemetry."""
    import torch

    from reachability_gen.flops import flops_euclidean_loop, flops_geometric
    from reachability_gen.models.euclidean_loop import EuclideanLoop
    from reachability_gen.models.geometric import GeometricRecurrent
    from reachability_gen.train.ff_trainer import examples_to_batch
    from reachability_gen.train.geo_trainer import GeometricTrainer
    from reachability_gen.train.loop_trainer import LoopTrainer

    torch.manual_seed(seed)
    run_id = f"{arm_kind}-rematch-{uuid.uuid4().hex[:10]}"
    heads = _n_heads(d)

    if arm_kind == "geo":
        model = GeometricRecurrent(
            vocab_size=len(vocab),
            d=d,
            T=T,
            n_heads=heads,
            max_len=max_len,
            pad_id=vocab.pad_id,
            use_tau=use_tau,
            mlp_expansion=mlp_expansion,
        )
        trainer: Any = GeometricTrainer(
            model, lr=lr, weight_decay=0.01, grad_clip=1.0
        )
        arm_name = f"geo-T{T}-d{d}-mlp{mlp_expansion}" + (
            "-tau" if use_tau else "-notau"
        )
        flop_fn = lambda seq_len: flops_geometric(
            seq_len, d=d, T=T, use_tau=use_tau
        )
    elif arm_kind == "loop":
        model = EuclideanLoop(
            vocab_size=len(vocab),
            d=d,
            T=T,
            n_heads=heads,
            max_len=max_len,
            pad_id=vocab.pad_id,
            mlp_expansion=mlp_expansion,
        )
        trainer = LoopTrainer(model, lr=lr, weight_decay=0.01, grad_clip=1.0)
        arm_name = f"loop-T{T}-d{d}-mlp{mlp_expansion}"
        flop_fn = lambda seq_len: flops_euclidean_loop(seq_len, d=d, T=T)
    else:
        raise ValueError(f"unknown arm_kind={arm_kind!r}")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[{arm_kind}] real param_count={param_count} "
        f"(d={d} mlp={mlp_expansion} T={T} tau={use_tau})",
        file=sys.stderr,
    )

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
            hop_perturbs: dict[int, list[float]] = defaultdict(list)

            for ex in val:
                token_ids, mask, labels, _ = examples_to_batch(
                    [ex], vocab, max_len=max_len
                )
                vloss, vacc, drifts = trainer.eval_step(
                    token_ids, labels, mask, return_drift=True
                )  # type: ignore[misc]
                telem = trainer.drift_telemetry(
                    token_ids, mask, eps_sigma=EPS_SIGMA
                )
                pert = telem.get("perturbation_delta")
                hop = int(ex.get("hop_distance", HOP_UNREACHABLE))
                hop_losses[hop].append(float(vloss))
                hop_accs[hop].append(float(vacc))
                hop_trajs[hop].append(list(drifts))
                if drifts:
                    hop_terminals[hop].append(float(drifts[-1]))
                if pert is not None:
                    hop_perturbs[hop].append(float(pert))
                enc = str(ex.get("encoding", ""))
                seq_len = max(len(enc.split()), 1)
                flop_report = flop_fn(seq_len)
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
                    "perturbation_delta": float(pert) if pert is not None else None,
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
                perturbs = hop_perturbs.get(k, [])
                regime = _drift_regime(mean_traj)
                val_by_hop[str(k)] = {
                    "n": len(hop_losses[k]),
                    "loss_mean": _mean(hop_losses[k]),
                    "acc_mean": _mean(hop_accs[k]),
                    "mean_drift_trajectory": mean_traj,
                    "mean_terminal_drift": _mean(terminals) if terminals else None,
                    "mean_perturbation_delta": (
                        _mean(perturbs) if perturbs else None
                    ),
                    "terminal_drift_regime": regime,
                    "terminal_drift_damps": regime == "damps",
                    "damp_regime": regime,
                }
            val_by_hop_final = val_by_hop
            label = "Geo" if arm_kind == "geo" else "Loop"
            print(
                f"[{label}] epoch {epoch}/{epochs}: train_loss={train_loss:.4f} "
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
                        f"[{label}] early stop at epoch {epoch} "
                        f"(train plateau patience={plateau_patience})",
                        file=sys.stderr,
                    )
                    break

    regimes = {
        k: v.get("terminal_drift_regime") for k, v in val_by_hop_final.items()
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
            "else expands or saturates; perturbation_delta uses "
            f"eps_sigma={EPS_SIGMA}"
        ),
    }

    out: dict[str, Any] = {
        "run_id": run_id,
        "arm": arm_name,
        "d": d,
        "T": T,
        "mlp_expansion": mlp_expansion,
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
    return out


def build_rematch(
    ff_summary: dict[str, Any],
    geo_summary: dict[str, Any],
    loop_summary: dict[str, Any],
    *,
    param_match: dict[str, Any],
    geo_cfg: ScaledConfig,
    loop_cfg: ScaledConfig,
    dataset_path: str = "data/id_2k.jsonl",
    epochs_choice: str = "",
) -> dict[str, Any]:
    """Assemble artifacts/id_2k_rematch.json (science_open=False)."""
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
        "param_match": param_match,
        "scaling": {
            "ff": {"d": ff_summary.get("d"), "L": ff_summary.get("L"), "mlp_expansion": 4},
            "geo": {
                "d": geo_cfg.d,
                "T": DEFAULT_T,
                "mlp_expansion": geo_cfg.mlp_expansion,
                "use_tau": True,
                "param_count": geo_cfg.param_count,
            },
            "loop": {
                "d": loop_cfg.d,
                "T": DEFAULT_T,
                "mlp_expansion": loop_cfg.mlp_expansion,
                "use_tau": False,
                "param_count": loop_cfg.param_count,
            },
            "note": (
                "Geo/Loop d and/or mlp_expansion scaled so real sum(p.numel()) "
                f"falls in ±{PARAM_TOL:.0%} of FF L=2 baseline; "
                "hard-fail before train if outside."
            ),
            "science_open": False,
        },
        "ff": {
            "arm": ff_summary.get("arm"),
            "param_count": ff_summary.get("param_count"),
            "d": ff_summary.get("d"),
            "L": ff_summary.get("L"),
            "epochs_run": ff_summary.get("epochs_run"),
            "early_stop": ff_summary.get("early_stop"),
            "train_history": ff_summary.get("train_history"),
            "val_by_hop": ff_summary.get("val_by_hop"),
            "metrics_path": ff_summary.get("metrics_path"),
            "science_open": False,
        },
        "geo": {
            "arm": geo_summary.get("arm"),
            "param_count": geo_summary.get("param_count"),
            "d": geo_summary.get("d"),
            "T": geo_summary.get("T"),
            "mlp_expansion": geo_summary.get("mlp_expansion"),
            "use_tau": geo_summary.get("use_tau"),
            "epochs_run": geo_summary.get("epochs_run"),
            "early_stop": geo_summary.get("early_stop"),
            "train_history": geo_summary.get("train_history"),
            "val_by_hop": geo_summary.get("val_by_hop"),
            "drift_summary": geo_summary.get("drift_summary"),
            "metrics_path": geo_summary.get("metrics_path"),
            "science_open": False,
        },
        "loop": {
            "arm": loop_summary.get("arm"),
            "param_count": loop_summary.get("param_count"),
            "d": loop_summary.get("d"),
            "T": loop_summary.get("T"),
            "mlp_expansion": loop_summary.get("mlp_expansion"),
            "use_tau": False,
            "epochs_run": loop_summary.get("epochs_run"),
            "early_stop": loop_summary.get("early_stop"),
            "train_history": loop_summary.get("train_history"),
            "val_by_hop": loop_summary.get("val_by_hop"),
            "drift_summary": loop_summary.get("drift_summary"),
            "metrics_path": loop_summary.get("metrics_path"),
            "science_open": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Param-matched rematch FF L=2 / Geo T=6 τ / Loop T=6 on exact "
            "id_2k.jsonl (MEASURE; science_open=false). Does NOT regenerate data."
        )
    )
    p.add_argument(
        "--data",
        type=Path,
        default=Path("data/id_2k.jsonl"),
        help="Existing id_2k.jsonl (never regenerated by this CLI)",
    )
    p.add_argument(
        "--rematch-out",
        type=Path,
        default=Path("artifacts/id_2k_rematch.json"),
    )
    p.add_argument(
        "--ff-metrics-out",
        type=Path,
        default=Path("artifacts/id_2k_rematch_ff_metrics.jsonl"),
    )
    p.add_argument(
        "--geo-metrics-out",
        type=Path,
        default=Path("artifacts/id_2k_rematch_geo_metrics.jsonl"),
    )
    p.add_argument(
        "--loop-metrics-out",
        type=Path,
        default=Path("artifacts/id_2k_rematch_loop_metrics.jsonl"),
    )
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--ff-d", type=int, default=DEFAULT_FF_D)
    p.add_argument("--L", type=int, default=DEFAULT_L)
    p.add_argument("--T", type=int, default=DEFAULT_T)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument(
        "--reuse-ff-from-comparison",
        action="store_true",
        help=(
            "Reuse FF val_by_hop / history from artifacts/id_2k_comparison.json "
            "instead of retraining FF (still trains Geo+Loop)."
        ),
    )
    p.add_argument(
        "--comparison-path",
        type=Path,
        default=Path("artifacts/id_2k_comparison.json"),
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("FAIL: torch required for id_2k rematch", file=sys.stderr)
        return 2

    if not args.data.exists():
        print(f"FAIL: missing {args.data} (do not regenerate)", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    rows = load_jsonl(args.data)
    ok, issues = verify_id_2k(rows)
    if not ok:
        print(f"FAIL: id_2k failed verify: {issues}", file=sys.stderr)
        return 1

    train, val = _split_train_val(rows)
    print(
        f"reusing {args.data} ({len(rows)} rows); "
        f"train={len(train)} val={len(val)}",
        file=sys.stderr,
    )

    from reachability_gen.tokenize import build_vocab
    from reachability_gen.train.ff_trainer import examples_to_batch

    vocab = build_vocab()
    all_rows = train + val
    probe = examples_to_batch(all_rows[:2], vocab)
    max_len = max(int(probe[0].shape[1]) + 8, 64)
    _, _, _, vocab = examples_to_batch(all_rows, vocab, max_len=max_len)

    # --- FF baseline (existing L=2) ---
    if args.reuse_ff_from_comparison and args.comparison_path.exists():
        comp = json.loads(args.comparison_path.read_text())
        ff_block = comp.get("ff") or {}
        ff_summary = {
            "run_id": "ff-reused-from-comparison",
            "arm": ff_block.get("arm", f"ff-L{args.L}-d{args.ff_d}"),
            "d": args.ff_d,
            "L": args.L,
            "lr": args.lr,
            "epochs_requested": args.epochs,
            "epochs_run": ff_block.get("epochs_run"),
            "early_stop": ff_block.get("early_stop"),
            "seed": args.train_seed,
            "n_train": len(train),
            "n_val": len(val),
            "param_count": int(ff_block.get("param_count", FF_BASELINE_PARAMS)),
            "train_history": ff_block.get("train_history"),
            "val_by_hop": ff_block.get("val_by_hop"),
            "metrics_path": ff_block.get("metrics_path"),
            "max_len": max_len,
            "vocab_size": len(vocab),
            "science_open": False,
            "reused_from_comparison": True,
        }
        print(
            f"[FF] reused from {args.comparison_path} "
            f"param_count={ff_summary['param_count']}",
            file=sys.stderr,
        )
    else:
        ff_summary = train_ff_on_id2k(
            train,
            val,
            epochs=args.epochs,
            d=args.ff_d,
            L=args.L,
            lr=args.lr,
            seed=args.train_seed,
            batch_size=args.batch_size,
            metrics_path=args.ff_metrics_out,
            max_len=max_len,
            vocab=vocab,
        )

    ff_count = int(ff_summary["param_count"])
    lo, hi = param_window(ff_count)
    print(
        f"[parity] FF baseline param_count={ff_count}; "
        f"±5% window=[{lo}, {hi}]",
        file=sys.stderr,
    )

    # --- Scale Geo / Loop into window; hard-fail before train ---
    geo_cfg = scale_recurrent_to_window(
        vocab_size=len(vocab),
        max_len=max_len,
        pad_id=vocab.pad_id,
        use_tau=True,
        ref_count=ff_count,
        T=args.T,
    )
    loop_cfg = scale_recurrent_to_window(
        vocab_size=len(vocab),
        max_len=max_len,
        pad_id=vocab.pad_id,
        use_tau=False,
        ref_count=ff_count,
        T=args.T,
    )
    print(
        f"[scale] Geo → d={geo_cfg.d} mlp={geo_cfg.mlp_expansion} "
        f"n={geo_cfg.param_count} "
        f"(dev={geo_cfg.param_count - ff_count:+d}, "
        f"ratio={geo_cfg.param_count / ff_count:.4f})",
        file=sys.stderr,
    )
    print(
        f"[scale] Loop → d={loop_cfg.d} mlp={loop_cfg.mlp_expansion} "
        f"n={loop_cfg.param_count} "
        f"(dev={loop_cfg.param_count - ff_count:+d}, "
        f"ratio={loop_cfg.param_count / ff_count:.4f})",
        file=sys.stderr,
    )

    try:
        param_match = assert_param_parity(
            ff_count=ff_count,
            geo_count=geo_cfg.param_count,
            loop_count=loop_cfg.param_count,
        )
    except AssertionError as exc:
        print(f"FAIL parity assert (hard-fail before train): {exc}", file=sys.stderr)
        return 1

    param_match["note"] = (
        "Hard-asserted before Geo/Loop train; real sum(p.numel()) vs FF L=2. "
        "Never stamps science OPEN."
    )
    print(
        f"[parity] PASS within_5pct geo={param_match['geo_within_5pct']} "
        f"loop={param_match['loop_within_5pct']}",
        file=sys.stderr,
    )

    geo_summary = _train_recurrent_on_id2k(
        train,
        val,
        arm_kind="geo",
        epochs=args.epochs,
        d=geo_cfg.d,
        T=args.T,
        mlp_expansion=geo_cfg.mlp_expansion,
        lr=args.lr,
        seed=args.train_seed,
        batch_size=args.batch_size,
        metrics_path=args.geo_metrics_out,
        plateau_patience=PLATEAU_PATIENCE,
        max_len=max_len,
        vocab=vocab,
        use_tau=True,
    )
    loop_summary = _train_recurrent_on_id2k(
        train,
        val,
        arm_kind="loop",
        epochs=args.epochs,
        d=loop_cfg.d,
        T=args.T,
        mlp_expansion=loop_cfg.mlp_expansion,
        lr=args.lr,
        seed=args.train_seed,
        batch_size=args.batch_size,
        metrics_path=args.loop_metrics_out,
        plateau_patience=PLATEAU_PATIENCE,
        max_len=max_len,
        vocab=vocab,
        use_tau=False,
    )

    # Re-check realized trained counts (should match scale).
    param_match["geo_param_count"] = int(geo_summary["param_count"])
    param_match["loop_param_count"] = int(loop_summary["param_count"])
    param_match["geo_over_ff_ratio"] = param_match["geo_param_count"] / ff_count
    param_match["loop_over_ff_ratio"] = param_match["loop_param_count"] / ff_count
    param_match["geo_within_5pct"] = within_5pct(
        param_match["geo_param_count"], ff_count
    )
    param_match["loop_within_5pct"] = within_5pct(
        param_match["loop_param_count"], ff_count
    )
    param_match["within_5pct"] = (
        param_match["geo_within_5pct"] and param_match["loop_within_5pct"]
    )
    param_match["science_open"] = False

    rematch = build_rematch(
        ff_summary,
        geo_summary,
        loop_summary,
        param_match=param_match,
        geo_cfg=geo_cfg,
        loop_cfg=loop_cfg,
        dataset_path=str(args.data),
        epochs_choice=(
            f"request {args.epochs} epochs; early-stop on train-loss plateau "
            f"(patience={PLATEAU_PATIENCE}, min_delta={PLATEAU_MIN_DELTA}, "
            f"min_epochs=10); exact id_2k (no regen); "
            f"FF L={args.L} d={args.ff_d} vs Geo T={args.T} tau "
            f"d={geo_cfg.d} mlp={geo_cfg.mlp_expansion} vs Loop T={args.T} "
            f"d={loop_cfg.d} mlp={loop_cfg.mlp_expansion}; "
            f"lr={args.lr} batch={args.batch_size} train_seed={args.train_seed}; "
            f"clip=1.0 AdamW"
        ),
    )
    rematch["science_open"] = False

    args.rematch_out.parent.mkdir(parents=True, exist_ok=True)
    args.rematch_out.write_text(
        json.dumps(rematch, indent=2, sort_keys=True) + "\n"
    )
    elapsed = time.perf_counter() - t0

    def _print_table(label: str, by_hop: dict[str, Any], with_drift: bool) -> None:
        print(f"\n=== {label} val_by_hop ===", file=sys.stderr)
        hdr = f"{'hop':>4} {'n':>5} {'acc_mean':>10} {'loss_mean':>10}"
        if with_drift:
            hdr += f" {'term_drift':>12} {'perturb':>10} {'regime':>10}"
        print(hdr, file=sys.stderr)
        for k in sorted(by_hop.keys(), key=lambda x: int(x)):
            v = by_hop[k]
            line = (
                f"{k:>4} {v.get('n', 0):>5} {v.get('acc_mean', float('nan')):>10.4f} "
                f"{v.get('loss_mean', float('nan')):>10.4f}"
            )
            if with_drift:
                td = v.get("mean_terminal_drift")
                pd = v.get("mean_perturbation_delta")
                td_s = f"{td:.6f}" if isinstance(td, (int, float)) else "null"
                pd_s = f"{pd:.6f}" if isinstance(pd, (int, float)) else "null"
                line += (
                    f" {td_s:>12} {pd_s:>10} "
                    f"{str(v.get('terminal_drift_regime', v.get('damp_regime', ''))):>10}"
                )
            print(line, file=sys.stderr)

    _print_table("FF", ff_summary.get("val_by_hop") or {}, with_drift=False)
    _print_table("Geo", geo_summary.get("val_by_hop") or {}, with_drift=True)
    _print_table("Loop", loop_summary.get("val_by_hop") or {}, with_drift=True)

    print("\n=== param counts / deviations ===", file=sys.stderr)
    print(
        f"FF   {ff_count:>8}  (baseline)",
        file=sys.stderr,
    )
    print(
        f"Geo  {param_match['geo_param_count']:>8}  "
        f"dev={param_match['geo_param_count'] - ff_count:+d}  "
        f"ratio={param_match['geo_over_ff_ratio']:.4f}  "
        f"within_5pct={param_match['geo_within_5pct']}",
        file=sys.stderr,
    )
    print(
        f"Loop {param_match['loop_param_count']:>8}  "
        f"dev={param_match['loop_param_count'] - ff_count:+d}  "
        f"ratio={param_match['loop_over_ff_ratio']:.4f}  "
        f"within_5pct={param_match['loop_within_5pct']}",
        file=sys.stderr,
    )

    print(
        json.dumps(
            {
                "ok": True,
                "elapsed_s": elapsed,
                "rematch_out": str(args.rematch_out),
                "data": str(args.data),
                "science_open": False,
                "param_match": {
                    "ff": ff_count,
                    "geo": param_match["geo_param_count"],
                    "loop": param_match["loop_param_count"],
                    "within_5pct": param_match["within_5pct"],
                },
                "ff_epochs_run": ff_summary.get("epochs_run"),
                "geo_epochs_run": geo_summary.get("epochs_run"),
                "loop_epochs_run": loop_summary.get("epochs_run"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

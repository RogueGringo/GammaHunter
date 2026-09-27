# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Fixed 30-epoch rematch with clip/LR match + RMSNorm state bound (MEASURE).

Protocol
--------
- Exact existing ``data/id_2k.jsonl`` (never regenerated).
- Arms: FF L=2 (control), Geo T=6+τ, Loop T=6 no τ.
- Recurrent locked to mlp_expansion=10 (param parity ±5% of FF).
- **epochs=30 locked; NO early stopping.** Track/save best val-acc
  checkpoint but always finish all 30 epochs.
- ``science_open=false`` always.

Phase A — effective LR / clip (chosen values)
--------------------------------------------
From fixed30 under shared clip=1.0 lr=3e-3: FF sat≈7.7%, Geo≈16%, Loop≈21%.
With RMSNorm, lr=3e-3 stuck Geo at chance; 5-epoch probe selected:

* FF control (unchanged): ``lr=3e-3``, ``grad_clip=1.0``
* Recurrent Geo/Loop: ``lr=1.5e-3`` (=0.5×FF), ``grad_clip=2.5`` (=2.5×FF)

so recurrent grad-clip saturation ≈ FF while remaining learnable.

Phase B — learning-compatible state bound
-----------------------------------------
After residual mix α=0.5 apply **RMSNorm** on z (preferred over unit-sphere
proj / stream LN). Stream LN previously blocked learning; RMSNorm keeps
||z|| ~ O(√d) and still learns at the Phase-A lr. Log mean ||z_t||_2 and
raw + LN-normalized + RMS-normalized drift.

Usage::

    python -m reachability_gen.run_id_2k_rematch_bound30
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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
    PARAM_TOL,
    SCIENCE_OPEN_DEFAULT,
    validate_metric_record,
)
from reachability_gen.gen_id_2k import verify_id_2k
from reachability_gen.models.geometric import DEFAULT_RESIDUAL_ALPHA
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_id_2k_rematch import (
    FF_BASELINE_PARAMS,
    assert_param_parity,
    param_window,
    within_5pct,
)

DEFAULT_EPOCHS: int = 30
DEFAULT_FF_D: int = 64
DEFAULT_L: int = 2
DEFAULT_T: int = 6
DEFAULT_FF_LR: float = 3e-3  # FF control unchanged
DEFAULT_REC_LR: float = 1.5e-3  # Phase A: 0.5× FF
DEFAULT_BATCH: int = 32
DEFAULT_MLP_REC: int = 10  # locked mlp×10 for recurrent (parity)
EPS_SIGMA: float = 1e-4
FF_GRAD_CLIP: float = 1.0  # FF control unchanged
REC_GRAD_CLIP: float = 2.5  # Phase A: raised so sat ≈ FF
DEFAULT_LR: float = DEFAULT_FF_LR  # alias
GRAD_CLIP: float = FF_GRAD_CLIP  # alias (FF path)


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


def _n_heads(d: int) -> int:
    if d % 4 == 0:
        return 4
    if d % 2 == 0:
        return 2
    return 1


def _overall_val_acc(val_by_hop: dict[str, Any]) -> float:
    n_tot = 0
    acc_sum = 0.0
    for v in val_by_hop.values():
        n = int(v.get("n", 0))
        a = float(v.get("acc_mean", float("nan")))
        if n > 0 and a == a:
            acc_sum += a * n
            n_tot += n
    return acc_sum / n_tot if n_tot else float("nan")


def _check_diameter(mean_z_norms: list[float], d: int) -> dict[str, Any]:
    """Post-LN expected ||z||≈√d; max pairwise diameter ≲ 2√d."""
    expected = math.sqrt(float(d))
    max_norm = max(mean_z_norms) if mean_z_norms else float("nan")
    min_norm = min(mean_z_norms) if mean_z_norms else float("nan")
    return {
        "expected_norm_sqrt_d": expected,
        "mean_z_norm_by_t": mean_z_norms,
        "min_mean_z_norm": min_norm,
        "max_mean_z_norm": max_norm,
        "implied_max_pairwise_diameter": (
            2.0 * max_norm if max_norm == max_norm else float("nan")
        ),
        "note": (
            f"post-RMSNorm ||z||≈√d=√{d}≈{expected:.3f}; "
            "max pairwise distance on bound manifold ≲ 2√d≈"
            f"{2*expected:.3f}"
        ),
    }


def train_ff_bound30(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    *,
    epochs: int = DEFAULT_EPOCHS,
    d: int = DEFAULT_FF_D,
    L: int = DEFAULT_L,
    lr: float = DEFAULT_FF_LR,
    seed: int = 0,
    batch_size: int = DEFAULT_BATCH,
    grad_clip: float = FF_GRAD_CLIP,
    metrics_path: Path = Path("artifacts/id_2k_rematch_bound30_ff_metrics.jsonl"),
    ckpt_path: Path = Path("artifacts/id_2k_rematch_bound30_ff_best.pt"),
    max_len: int,
    vocab: Any,
) -> dict[str, Any]:
    import torch

    from reachability_gen.flops import flops_feedforward
    from reachability_gen.models.feedforward import FeedForward
    from reachability_gen.train.ff_trainer import FeedForwardTrainer, examples_to_batch

    torch.manual_seed(seed)
    run_id = f"ff-bound30-{uuid.uuid4().hex[:10]}"
    model = FeedForward(
        vocab_size=len(vocab),
        d=d,
        L=L,
        n_heads=_n_heads(d),
        max_len=max_len,
        pad_id=vocab.pad_id,
    )
    trainer = FeedForwardTrainer(
        model, lr=lr, weight_decay=0.01, grad_clip=grad_clip
    )
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    arm_name = f"ff-L{L}-d{d}"

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    train_hist: list[dict[str, Any]] = []
    val_by_hop_final: dict[str, Any] = {}
    epochs_run = 0
    n_sat = 0
    n_steps = 0
    best_val_acc = -1.0
    best_epoch = 0
    best_state: Optional[dict[str, Any]] = None

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
                n_steps += 1
                if trainer.last_pre_clip_grad_norm >= grad_clip:
                    n_sat += 1
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
            ov_acc = _overall_val_acc(val_by_hop)
            if ov_acc > best_val_acc:
                best_val_acc = ov_acc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"[FF] epoch {epoch}/{epochs}: train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} val_acc={ov_acc:.4f} "
                f"(best={best_val_acc:.4f}@ep{best_epoch})",
                file=sys.stderr,
            )

    # NO early stopping — epochs_run must equal epochs.
    assert epochs_run == epochs, f"FF did not finish {epochs} epochs"

    if best_state is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        import torch as _torch

        _torch.save(
            {
                "epoch": best_epoch,
                "val_acc": best_val_acc,
                "state_dict": best_state,
                "arm": arm_name,
                "science_open": False,
            },
            ckpt_path,
        )

    sat_rate = n_sat / n_steps if n_steps else float("nan")
    return {
        "run_id": run_id,
        "arm": arm_name,
        "d": d,
        "L": L,
        "lr": lr,
        "epochs_requested": epochs,
        "epochs_run": epochs_run,
        "early_stop": False,
        "seed": seed,
        "n_train": len(train),
        "n_val": len(val),
        "param_count": int(param_count),
        "train_history": train_hist,
        "val_by_hop": val_by_hop_final,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "checkpoint_path": str(ckpt_path),
        "grad_clip_sat_rate": sat_rate,
        "grad_clip": grad_clip,
        "n_train_steps": n_steps,
        "n_sat_steps": n_sat,
        "mean_z_norm_by_t": None,
        "metrics_path": str(metrics_path),
        "science_open": False,
    }


def _train_recurrent_bound30(
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    *,
    arm_kind: str,
    epochs: int,
    d: int,
    T: int,
    mlp_expansion: int,
    lr: float,
    seed: int,
    batch_size: int,
    metrics_path: Path,
    ckpt_path: Path,
    max_len: int,
    vocab: Any,
    use_tau: bool,
    residual_alpha: float = DEFAULT_RESIDUAL_ALPHA,
    grad_clip: float = REC_GRAD_CLIP,
    apply_cycle_rmsnorm: bool = True,
) -> dict[str, Any]:
    import torch

    from reachability_gen.flops import flops_euclidean_loop, flops_geometric
    from reachability_gen.models.euclidean_loop import EuclideanLoop
    from reachability_gen.models.geometric import GeometricRecurrent
    from reachability_gen.train.ff_trainer import examples_to_batch
    from reachability_gen.train.geo_trainer import GeometricTrainer
    from reachability_gen.train.loop_trainer import LoopTrainer

    torch.manual_seed(seed)
    run_id = f"{arm_kind}-bound30-{uuid.uuid4().hex[:10]}"
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
            residual_alpha=residual_alpha,
            apply_cycle_ln=False,
            apply_cycle_rmsnorm=apply_cycle_rmsnorm,
        )
        trainer: Any = GeometricTrainer(
            model, lr=lr, weight_decay=0.01, grad_clip=grad_clip
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
            residual_alpha=residual_alpha,
            apply_cycle_ln=False,
            apply_cycle_rmsnorm=apply_cycle_rmsnorm,
        )
        trainer = LoopTrainer(
            model, lr=lr, weight_decay=0.01, grad_clip=grad_clip
        )
        arm_name = f"loop-T{T}-d{d}-mlp{mlp_expansion}"
        flop_fn = lambda seq_len: flops_euclidean_loop(seq_len, d=d, T=T)
    else:
        raise ValueError(f"unknown arm_kind={arm_kind!r}")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[{arm_kind}] real param_count={param_count} "
        f"(d={d} mlp={mlp_expansion} T={T} tau={use_tau} "
        f"alpha={residual_alpha} rmsnorm={apply_cycle_rmsnorm} clip={grad_clip})",
        file=sys.stderr,
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    train_hist: list[dict[str, Any]] = []
    val_by_hop_final: dict[str, Any] = {}
    epochs_run = 0
    n_sat = 0
    n_steps = 0
    best_val_acc = -1.0
    best_epoch = 0
    best_state: Optional[dict[str, Any]] = None
    final_mean_z_norms: list[float] = []

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
                n_steps += 1
                if trainer.last_pre_clip_grad_norm >= grad_clip:
                    n_sat += 1
                step += 1

            train_loss = _mean(epoch_losses)
            train_acc = _mean(epoch_accs)
            train_hist.append(
                {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc}
            )

            hop_losses: dict[int, list[float]] = defaultdict(list)
            hop_accs: dict[int, list[float]] = defaultdict(list)
            hop_trajs: dict[int, list[list[float]]] = defaultdict(list)
            hop_trajs_ln: dict[int, list[list[float]]] = defaultdict(list)
            hop_trajs_rms: dict[int, list[list[float]]] = defaultdict(list)
            hop_terminals: dict[int, list[float]] = defaultdict(list)
            hop_terminals_ln: dict[int, list[float]] = defaultdict(list)
            hop_terminals_rms: dict[int, list[float]] = defaultdict(list)
            hop_perturbs: dict[int, list[float]] = defaultdict(list)
            hop_z_norms: dict[int, list[list[float]]] = defaultdict(list)

            for ex in val:
                token_ids, mask, labels, _ = examples_to_batch(
                    [ex], vocab, max_len=max_len
                )
                # Single forward: loss/acc via eval_step, telemetry reuses
                # drift_telemetry (one extra forward for perturbation only).
                # Prefer telem as source of truth for drifts/z-norms.
                vloss, vacc = trainer.eval_step(token_ids, labels, mask)
                telem = trainer.drift_telemetry(
                    token_ids, mask, eps_sigma=EPS_SIGMA
                )
                drifts = list(telem.get("drift_trajectory") or [])
                drifts_ln = list(telem.get("drift_trajectory_ln") or [])
                drifts_rms = list(telem.get("drift_trajectory_rms") or [])
                pert = telem.get("perturbation_delta")
                z_norms = list(telem.get("mean_z_norm_by_t") or [])
                hop = int(ex.get("hop_distance", HOP_UNREACHABLE))
                hop_losses[hop].append(float(vloss))
                hop_accs[hop].append(float(vacc))
                hop_trajs[hop].append(list(drifts))
                hop_trajs_ln[hop].append(drifts_ln)
                hop_trajs_rms[hop].append(drifts_rms)
                if drifts:
                    hop_terminals[hop].append(float(drifts[-1]))
                if drifts_ln:
                    hop_terminals_ln[hop].append(float(drifts_ln[-1]))
                if drifts_rms:
                    hop_terminals_rms[hop].append(float(drifts_rms[-1]))
                if pert is not None:
                    hop_perturbs[hop].append(float(pert))
                if z_norms:
                    hop_z_norms[hop].append(z_norms)
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
            all_z_for_epoch: list[list[float]] = []
            for k in sorted(set(hop_losses) | set(hop_accs)):
                mean_traj = _mean_traj(hop_trajs.get(k, []))
                mean_traj_ln = _mean_traj(hop_trajs_ln.get(k, []))
                mean_traj_rms = _mean_traj(hop_trajs_rms.get(k, []))
                terminals = hop_terminals.get(k, [])
                terminals_ln = hop_terminals_ln.get(k, [])
                terminals_rms = hop_terminals_rms.get(k, [])
                perturbs = hop_perturbs.get(k, [])
                mean_zn = _mean_traj(hop_z_norms.get(k, []))
                if mean_zn:
                    all_z_for_epoch.append(mean_zn)
                regime = _drift_regime(mean_traj)
                val_by_hop[str(k)] = {
                    "n": len(hop_losses[k]),
                    "loss_mean": _mean(hop_losses[k]),
                    "acc_mean": _mean(hop_accs[k]),
                    "mean_drift_trajectory": mean_traj,
                    "mean_drift_trajectory_ln": mean_traj_ln,
                    "mean_drift_trajectory_rms": mean_traj_rms,
                    "mean_terminal_drift": _mean(terminals) if terminals else None,
                    "mean_terminal_drift_ln": (
                        _mean(terminals_ln) if terminals_ln else None
                    ),
                    "mean_terminal_drift_rms": (
                        _mean(terminals_rms) if terminals_rms else None
                    ),
                    "mean_perturbation_delta": (
                        _mean(perturbs) if perturbs else None
                    ),
                    "mean_z_norm_by_t": mean_zn,
                    "terminal_drift_regime": regime,
                    "terminal_drift_damps": regime == "damps",
                    "damp_regime": regime,
                }
            val_by_hop_final = val_by_hop
            final_mean_z_norms = _mean_traj(all_z_for_epoch)
            ov_acc = _overall_val_acc(val_by_hop)
            if ov_acc > best_val_acc:
                best_val_acc = ov_acc
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
            label = "Geo" if arm_kind == "geo" else "Loop"
            print(
                f"[{label}] epoch {epoch}/{epochs}: train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} val_acc={ov_acc:.4f} "
                f"(best={best_val_acc:.4f}@ep{best_epoch})",
                file=sys.stderr,
            )

    assert epochs_run == epochs, f"{arm_kind} did not finish {epochs} epochs"

    if best_state is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": best_epoch,
                "val_acc": best_val_acc,
                "state_dict": best_state,
                "arm": arm_name,
                "science_open": False,
            },
            ckpt_path,
        )

    regimes = {
        k: v.get("terminal_drift_regime") for k, v in val_by_hop_final.items()
    }
    damp_flags = [
        v.get("terminal_drift_damps")
        for v in val_by_hop_final.values()
        if v.get("terminal_drift_damps") is not None
    ]
    diameter = _check_diameter(final_mean_z_norms, d)
    sat_rate = n_sat / n_steps if n_steps else float("nan")
    drift_summary = {
        "by_hop_regime": regimes,
        "fraction_buckets_damping": (
            sum(1 for x in damp_flags if x) / len(damp_flags) if damp_flags else None
        ),
        "drift_formula": (
            "δ_t = mean_batch ||z_{t+1}-z_t||_2 on pooled latents; "
            f"z_{{t+1}}=RMSNorm(z_t+α·(Φ(h_t)-z_t)) with α={residual_alpha}; "
            "also report LN- and RMS-reapplied drift"
        ),
        "residual_alpha": residual_alpha,
        "apply_cycle_ln": False,
        "apply_cycle_rmsnorm": apply_cycle_rmsnorm,
        "diameter_check": diameter,
        "note": (
            "damps means mean δ_T < δ_{T-1} on the hop bucket; "
            f"else expands or saturates; perturbation_delta uses "
            f"eps_sigma={EPS_SIGMA}"
        ),
    }

    return {
        "run_id": run_id,
        "arm": arm_name,
        "d": d,
        "T": T,
        "mlp_expansion": mlp_expansion,
        "use_tau": use_tau,
        "residual_alpha": residual_alpha,
        "apply_cycle_ln": False,
        "apply_cycle_rmsnorm": apply_cycle_rmsnorm,
        "lr": lr,
        "epochs_requested": epochs,
        "epochs_run": epochs_run,
        "early_stop": False,
        "seed": seed,
        "n_train": len(train),
        "n_val": len(val),
        "param_count": int(param_count),
        "train_history": train_hist,
        "val_by_hop": val_by_hop_final,
        "drift_summary": drift_summary,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "checkpoint_path": str(ckpt_path),
        "grad_clip_sat_rate": sat_rate,
        "grad_clip": grad_clip,
        "n_train_steps": n_steps,
        "n_sat_steps": n_sat,
        "mean_z_norm_by_t": final_mean_z_norms,
        "metrics_path": str(metrics_path),
        "science_open": False,
    }


def build_artifact(
    ff: dict[str, Any],
    geo: dict[str, Any],
    loop: dict[str, Any],
    *,
    param_match: dict[str, Any],
    dataset_path: str,
    epochs: int,
    ff_lr: float,
    rec_lr: float,
    ff_clip: float,
    rec_clip: float,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "science_open": False,
        "dataset": dataset_path,
        "epochs": epochs,
        "epochs_locked": True,
        "early_stopping": False,
        "phase_a_clip_lr": {
            "ff_lr": ff_lr,
            "ff_grad_clip": ff_clip,
            "recurrent_lr": rec_lr,
            "recurrent_grad_clip": rec_clip,
            "ff_unchanged_control": True,
            "rationale": (
                "fixed30 shared clip=1.0 lr=3e-3 → FF sat≈7.7% Geo≈16% Loop≈21%; "
                "RMSNorm + lr=3e-3 stuck Geo at chance; chose recurrent "
                f"lr={rec_lr} (=0.5×FF) and clip={rec_clip} (=2.5×FF) so sat≈FF "
                "while remaining learnable (5-epoch probe)"
            ),
            "science_open": False,
        },
        "epochs_choice": (
            f"LOCKED {epochs} epochs; NO early stopping; "
            f"track/save best val-acc checkpoint but always finish {epochs}; "
            f"FF L={DEFAULT_L} d={DEFAULT_FF_D} vs Geo T={DEFAULT_T} tau "
            f"d={geo.get('d')} mlp={geo.get('mlp_expansion')} vs Loop T={DEFAULT_T} "
            f"d={loop.get('d')} mlp={loop.get('mlp_expansion')}; "
            f"ff_lr={ff_lr} rec_lr={rec_lr} batch={batch_size} train_seed={seed}; "
            f"ff_clip={ff_clip} rec_clip={rec_clip} AdamW; "
            f"residual_alpha={DEFAULT_RESIDUAL_ALPHA} "
            f"apply_cycle_rmsnorm=True (LN stream blocked learning)"
        ),
        "id_hop_range": [ID_HOP_MIN, ID_HOP_MAX],
        "negative_hop": HOP_UNREACHABLE,
        "param_match": param_match,
        "drift_audit": {
            "formula_raw": "δ_t = mean_batch ||z_{t+1}-z_t||_2 (pooled)",
            "formula_ln": (
                "δ_t = mean_batch ||LN(z_{t+1})-LN(z_t)||_2 "
                "(feature LN reapplied for metric)"
            ),
            "formula_rms": (
                "δ_t = mean_batch ||RMSNorm(z_{t+1})-RMSNorm(z_t)||_2 "
                "(feature RMSNorm reapplied; near-identity when cycle RMSNorm on)"
            ),
            "recurrent_update": (
                f"z_{{t+1}} = RMSNorm(z_t + α·(Φ(h_t)-z_t)) with α={DEFAULT_RESIDUAL_ALPHA}; "
                "outer stream LN disabled (blocked learning); RMSNorm is the "
                "learning-compatible state bound so ||z||~O(√d); "
                "report raw + LN- + RMS-normalized δ_t"
            ),
            "residual_alpha": DEFAULT_RESIDUAL_ALPHA,
            "apply_cycle_ln": False,
            "apply_cycle_rmsnorm": True,
            "state_bound": "rmsnorm",
            "deviation": (
                "Preferred z←LN(z+α·v) blocked learning (val stuck at chance); "
                "replaced with RMSNorm after α=0.5 residual mix (still learns); "
                "unit-sphere proj not needed"
            ),
            "expected_norm_sqrt_d": math.sqrt(float(geo.get("d") or DEFAULT_FF_D)),
            "science_open": False,
        },
        "scaling": {
            "ff": {"d": ff.get("d"), "L": ff.get("L"), "mlp_expansion": 4},
            "geo": {
                "d": geo.get("d"),
                "T": geo.get("T"),
                "mlp_expansion": geo.get("mlp_expansion"),
                "use_tau": True,
                "param_count": geo.get("param_count"),
                "residual_alpha": geo.get("residual_alpha"),
                "apply_cycle_ln": False,
                "apply_cycle_rmsnorm": True,
                "lr": geo.get("lr"),
                "grad_clip": geo.get("grad_clip"),
            },
            "loop": {
                "d": loop.get("d"),
                "T": loop.get("T"),
                "mlp_expansion": loop.get("mlp_expansion"),
                "use_tau": False,
                "param_count": loop.get("param_count"),
                "residual_alpha": loop.get("residual_alpha"),
                "apply_cycle_ln": False,
                "apply_cycle_rmsnorm": True,
                "lr": loop.get("lr"),
                "grad_clip": loop.get("grad_clip"),
            },
            "note": (
                "Recurrent arms locked mlp_expansion=10 so real sum(p.numel()) "
                f"falls in ±{PARAM_TOL:.0%} of FF L=2; "
                "hard-fail before train if outside window."
            ),
            "science_open": False,
        },
        "ff": {
            "arm": ff.get("arm"),
            "param_count": ff.get("param_count"),
            "d": ff.get("d"),
            "L": ff.get("L"),
            "lr": ff.get("lr"),
            "grad_clip": ff.get("grad_clip"),
            "epochs": epochs,
            "epochs_run": ff.get("epochs_run"),
            "early_stop": False,
            "best_val_acc": ff.get("best_val_acc"),
            "best_epoch": ff.get("best_epoch"),
            "checkpoint_path": ff.get("checkpoint_path"),
            "grad_clip_sat_rate": ff.get("grad_clip_sat_rate"),
            "mean_z_norm_by_t": None,
            "train_history": ff.get("train_history"),
            "val_by_hop": ff.get("val_by_hop"),
            "metrics_path": ff.get("metrics_path"),
            "science_open": False,
        },
        "geo": {
            "arm": geo.get("arm"),
            "param_count": geo.get("param_count"),
            "d": geo.get("d"),
            "T": geo.get("T"),
            "mlp_expansion": geo.get("mlp_expansion"),
            "use_tau": geo.get("use_tau"),
            "residual_alpha": geo.get("residual_alpha"),
            "apply_cycle_ln": False,
            "apply_cycle_rmsnorm": geo.get("apply_cycle_rmsnorm", True),
            "lr": geo.get("lr"),
            "grad_clip": geo.get("grad_clip"),
            "epochs": epochs,
            "epochs_run": geo.get("epochs_run"),
            "early_stop": False,
            "best_val_acc": geo.get("best_val_acc"),
            "best_epoch": geo.get("best_epoch"),
            "checkpoint_path": geo.get("checkpoint_path"),
            "grad_clip_sat_rate": geo.get("grad_clip_sat_rate"),
            "mean_z_norm_by_t": geo.get("mean_z_norm_by_t"),
            "train_history": geo.get("train_history"),
            "val_by_hop": geo.get("val_by_hop"),
            "drift_summary": geo.get("drift_summary"),
            "metrics_path": geo.get("metrics_path"),
            "science_open": False,
        },
        "loop": {
            "arm": loop.get("arm"),
            "param_count": loop.get("param_count"),
            "d": loop.get("d"),
            "T": loop.get("T"),
            "mlp_expansion": loop.get("mlp_expansion"),
            "use_tau": False,
            "residual_alpha": loop.get("residual_alpha"),
            "apply_cycle_ln": False,
            "apply_cycle_rmsnorm": loop.get("apply_cycle_rmsnorm", True),
            "lr": loop.get("lr"),
            "grad_clip": loop.get("grad_clip"),
            "epochs": epochs,
            "epochs_run": loop.get("epochs_run"),
            "early_stop": False,
            "best_val_acc": loop.get("best_val_acc"),
            "best_epoch": loop.get("best_epoch"),
            "checkpoint_path": loop.get("checkpoint_path"),
            "grad_clip_sat_rate": loop.get("grad_clip_sat_rate"),
            "mean_z_norm_by_t": loop.get("mean_z_norm_by_t"),
            "train_history": loop.get("train_history"),
            "val_by_hop": loop.get("val_by_hop"),
            "drift_summary": loop.get("drift_summary"),
            "metrics_path": loop.get("metrics_path"),
            "science_open": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Fixed 30-epoch rematch with Phase-A clip/LR match + Phase-B RMSNorm "
            "state bound: FF L=2 / Geo T=6 τ / Loop T=6 on exact id_2k.jsonl "
            "(MEASURE; science_open=false). No early stop."
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
        default=Path("artifacts/id_2k_rematch_bound30.json"),
    )
    p.add_argument(
        "--ff-metrics-out",
        type=Path,
        default=Path("artifacts/id_2k_rematch_bound30_ff_metrics.jsonl"),
    )
    p.add_argument(
        "--geo-metrics-out",
        type=Path,
        default=Path("artifacts/id_2k_rematch_bound30_geo_metrics.jsonl"),
    )
    p.add_argument(
        "--loop-metrics-out",
        type=Path,
        default=Path("artifacts/id_2k_rematch_bound30_loop_metrics.jsonl"),
    )
    p.add_argument("--train-seed", type=int, default=0)
    p.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Locked to 30 by protocol; override only for smoke tests",
    )
    p.add_argument("--ff-d", type=int, default=DEFAULT_FF_D)
    p.add_argument("--L", type=int, default=DEFAULT_L)
    p.add_argument("--T", type=int, default=DEFAULT_T)
    p.add_argument("--mlp-recurrent", type=int, default=DEFAULT_MLP_REC)
    p.add_argument("--ff-lr", type=float, default=DEFAULT_FF_LR)
    p.add_argument("--rec-lr", type=float, default=DEFAULT_REC_LR)
    p.add_argument("--ff-grad-clip", type=float, default=FF_GRAD_CLIP)
    p.add_argument("--rec-grad-clip", type=float, default=REC_GRAD_CLIP)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument(
        "--residual-alpha",
        type=float,
        default=DEFAULT_RESIDUAL_ALPHA,
        help="Outer residual α in z←RMSNorm(z+α·(Φ-z)) (default 0.5)",
    )
    p.add_argument(
        "--no-rmsnorm",
        action="store_true",
        help="Disable cycle RMSNorm (debug only; protocol enables it)",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch  # noqa: F401
    except ImportError:
        print("FAIL: torch required for bound30 rematch", file=sys.stderr)
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
    apply_rms = not args.no_rmsnorm
    print(
        f"reusing {args.data} ({len(rows)} rows); "
        f"train={len(train)} val={len(val)}; epochs={args.epochs} LOCKED "
        f"(no early stop); residual_alpha={args.residual_alpha}; "
        f"ff_lr={args.ff_lr} rec_lr={args.rec_lr} "
        f"ff_clip={args.ff_grad_clip} rec_clip={args.rec_grad_clip} "
        f"rmsnorm={apply_rms}",
        file=sys.stderr,
    )

    from reachability_gen.models.euclidean_loop import EuclideanLoop
    from reachability_gen.models.geometric import GeometricRecurrent
    from reachability_gen.tokenize import build_vocab
    from reachability_gen.train.ff_trainer import examples_to_batch

    vocab = build_vocab()
    all_rows = train + val
    probe = examples_to_batch(all_rows[:2], vocab)
    max_len = max(int(probe[0].shape[1]) + 8, 64)
    _, _, _, vocab = examples_to_batch(all_rows, vocab, max_len=max_len)

    # --- FF ---
    ff_summary = train_ff_bound30(
        train,
        val,
        epochs=args.epochs,
        d=args.ff_d,
        L=args.L,
        lr=args.ff_lr,
        seed=args.train_seed,
        batch_size=args.batch_size,
        grad_clip=args.ff_grad_clip,
        metrics_path=args.ff_metrics_out,
        ckpt_path=Path("artifacts/id_2k_rematch_bound30_ff_best.pt"),
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

    # --- Lock mlp×10; probe real counts with cycle_ln ---
    geo_probe = GeometricRecurrent(
        vocab_size=len(vocab),
        d=args.ff_d,
        T=args.T,
        n_heads=_n_heads(args.ff_d),
        max_len=max_len,
        pad_id=vocab.pad_id,
        use_tau=True,
        mlp_expansion=args.mlp_recurrent,
        residual_alpha=args.residual_alpha,
        apply_cycle_ln=False,
        apply_cycle_rmsnorm=apply_rms,
    )
    loop_probe = EuclideanLoop(
        vocab_size=len(vocab),
        d=args.ff_d,
        T=args.T,
        n_heads=_n_heads(args.ff_d),
        max_len=max_len,
        pad_id=vocab.pad_id,
        mlp_expansion=args.mlp_recurrent,
        residual_alpha=args.residual_alpha,
        apply_cycle_ln=False,
        apply_cycle_rmsnorm=apply_rms,
    )
    geo_count = int(
        sum(p.numel() for p in geo_probe.parameters() if p.requires_grad)
    )
    loop_count = int(
        sum(p.numel() for p in loop_probe.parameters() if p.requires_grad)
    )
    del geo_probe, loop_probe
    print(
        f"[scale] Geo → d={args.ff_d} mlp={args.mlp_recurrent} "
        f"n={geo_count} (dev={geo_count - ff_count:+d}, "
        f"ratio={geo_count / ff_count:.4f})",
        file=sys.stderr,
    )
    print(
        f"[scale] Loop → d={args.ff_d} mlp={args.mlp_recurrent} "
        f"n={loop_count} (dev={loop_count - ff_count:+d}, "
        f"ratio={loop_count / ff_count:.4f})",
        file=sys.stderr,
    )

    try:
        param_match = assert_param_parity(
            ff_count=ff_count,
            geo_count=geo_count,
            loop_count=loop_count,
        )
    except AssertionError as exc:
        print(f"FAIL parity assert (hard-fail before train): {exc}", file=sys.stderr)
        return 1

    param_match["note"] = (
        "Hard-asserted before Geo/Loop train; real sum(p.numel()) vs FF L=2 "
        f"with mlp={args.mlp_recurrent}. Never stamps science OPEN."
    )
    print(
        f"[parity] PASS within_5pct geo={param_match['geo_within_5pct']} "
        f"loop={param_match['loop_within_5pct']}",
        file=sys.stderr,
    )

    geo_summary = _train_recurrent_bound30(
        train,
        val,
        arm_kind="geo",
        epochs=args.epochs,
        d=args.ff_d,
        T=args.T,
        mlp_expansion=args.mlp_recurrent,
        lr=args.rec_lr,
        seed=args.train_seed,
        batch_size=args.batch_size,
        metrics_path=args.geo_metrics_out,
        ckpt_path=Path("artifacts/id_2k_rematch_bound30_geo_best.pt"),
        max_len=max_len,
        vocab=vocab,
        use_tau=True,
        residual_alpha=args.residual_alpha,
        grad_clip=args.rec_grad_clip,
        apply_cycle_rmsnorm=apply_rms,
    )
    loop_summary = _train_recurrent_bound30(
        train,
        val,
        arm_kind="loop",
        epochs=args.epochs,
        d=args.ff_d,
        T=args.T,
        mlp_expansion=args.mlp_recurrent,
        lr=args.rec_lr,
        seed=args.train_seed,
        batch_size=args.batch_size,
        metrics_path=args.loop_metrics_out,
        ckpt_path=Path("artifacts/id_2k_rematch_bound30_loop_best.pt"),
        max_len=max_len,
        vocab=vocab,
        use_tau=False,
        residual_alpha=args.residual_alpha,
        grad_clip=args.rec_grad_clip,
        apply_cycle_rmsnorm=apply_rms,
    )

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

    # Sanity: all arms finished full epochs.
    for label, summ in (("FF", ff_summary), ("Geo", geo_summary), ("Loop", loop_summary)):
        if int(summ["epochs_run"]) != int(args.epochs):
            print(
                f"FAIL: {label} epochs_run={summ['epochs_run']} != {args.epochs}",
                file=sys.stderr,
            )
            return 1

    artifact = build_artifact(
        ff_summary,
        geo_summary,
        loop_summary,
        param_match=param_match,
        dataset_path=str(args.data),
        epochs=args.epochs,
        ff_lr=args.ff_lr,
        rec_lr=args.rec_lr,
        ff_clip=args.ff_grad_clip,
        rec_clip=args.rec_grad_clip,
        batch_size=args.batch_size,
        seed=args.train_seed,
    )
    args.rematch_out.parent.mkdir(parents=True, exist_ok=True)
    args.rematch_out.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    )
    elapsed = time.perf_counter() - t0

    def _print_table(label: str, by_hop: dict[str, Any], with_drift: bool) -> None:
        print(f"\n=== {label} val_by_hop (epoch {args.epochs}) ===", file=sys.stderr)
        hdr = f"{'hop':>4} {'n':>5} {'acc_mean':>10} {'loss_mean':>10}"
        if with_drift:
            hdr += (
                f" {'term_drift':>12} {'term_ln':>10} {'term_rms':>10} "
                f"{'perturb':>10} {'regime':>10}"
            )
        print(hdr, file=sys.stderr)
        for k in sorted(by_hop.keys(), key=lambda x: int(x)):
            v = by_hop[k]
            line = (
                f"{k:>4} {v.get('n', 0):>5} "
                f"{v.get('acc_mean', float('nan')):>10.4f} "
                f"{v.get('loss_mean', float('nan')):>10.4f}"
            )
            if with_drift:
                td = v.get("mean_terminal_drift")
                tdln = v.get("mean_terminal_drift_ln")
                tdrms = v.get("mean_terminal_drift_rms")
                pd = v.get("mean_perturbation_delta")
                td_s = f"{td:.6f}" if isinstance(td, (int, float)) else "null"
                tdln_s = f"{tdln:.6f}" if isinstance(tdln, (int, float)) else "null"
                tdrms_s = (
                    f"{tdrms:.6f}" if isinstance(tdrms, (int, float)) else "null"
                )
                pd_s = f"{pd:.6f}" if isinstance(pd, (int, float)) else "null"
                line += (
                    f" {td_s:>12} {tdln_s:>10} {tdrms_s:>10} {pd_s:>10} "
                    f"{str(v.get('terminal_drift_regime', '')):>10}"
                )
            print(line, file=sys.stderr)

    _print_table("FF", ff_summary.get("val_by_hop") or {}, with_drift=False)
    _print_table("Geo", geo_summary.get("val_by_hop") or {}, with_drift=True)
    _print_table("Loop", loop_summary.get("val_by_hop") or {}, with_drift=True)

    print("\n=== param counts / deviations ===", file=sys.stderr)
    print(f"FF   {ff_count:>8}  (baseline)", file=sys.stderr)
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

    print("\n=== drift / grad diagnostics ===", file=sys.stderr)
    for label, summ in (("FF", ff_summary), ("Geo", geo_summary), ("Loop", loop_summary)):
        zn = summ.get("mean_z_norm_by_t")
        print(
            f"{label}: epochs_run={summ.get('epochs_run')}/"
            f"{summ.get('epochs_requested')} early_stop={summ.get('early_stop')} "
            f"best_val_acc={summ.get('best_val_acc'):.4f}@ep{summ.get('best_epoch')} "
            f"grad_clip_sat_rate={summ.get('grad_clip_sat_rate'):.4f} "
            f"mean_z_norm_by_t={zn}",
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
                "epochs": args.epochs,
                "early_stopping": False,
                "param_match": {
                    "ff": ff_count,
                    "geo": param_match["geo_param_count"],
                    "loop": param_match["loop_param_count"],
                    "within_5pct": param_match["within_5pct"],
                },
                "ff_epochs_run": ff_summary.get("epochs_run"),
                "geo_epochs_run": geo_summary.get("epochs_run"),
                "loop_epochs_run": loop_summary.get("epochs_run"),
                "grad_clip_sat_rate": {
                    "ff": ff_summary.get("grad_clip_sat_rate"),
                    "geo": geo_summary.get("grad_clip_sat_rate"),
                    "loop": loop_summary.get("grad_clip_sat_rate"),
                },
                "phase_a": {
                    "ff_lr": args.ff_lr,
                    "ff_grad_clip": args.ff_grad_clip,
                    "rec_lr": args.rec_lr,
                    "rec_grad_clip": args.rec_grad_clip,
                },
                "apply_cycle_rmsnorm": apply_rms,
                "mean_z_norm_by_t": {
                    "geo": geo_summary.get("mean_z_norm_by_t"),
                    "loop": loop_summary.get("mean_z_norm_by_t"),
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

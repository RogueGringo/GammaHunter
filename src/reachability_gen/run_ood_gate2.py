# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Gate 2: OOD-hop stress on bound30 best checkpoints (MEASURE; inference-only).

Loads FF / Geo / Loop best ``.pt`` from id_2k rematch bound30 and evaluates on
``data/ood_hops.jsonl`` (K ∈ {8,12,16} + hard negatives):

1. Fixed trained depth: FF L=2, Geo T=6, Loop T=6
2. Dynamic unroll: Geo/Loop at T ∈ {8,12,16}

Writes ``artifacts/id_2k_rematch_bound30_gate2_ood.json`` with
``science_open: false`` always. No retraining. No OPEN claims.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from reachability_gen.adr_invariants import (
    HOP_UNREACHABLE,
    OOD_HOP_VALUES,
    SCIENCE_OPEN_DEFAULT,
    validate_metric_record,
)
from reachability_gen.gen_ood_hops import (
    BOUND30_MAX_LEN,
    OOD_HOPS,
    verify_ood_hops,
)
from reachability_gen.models.geometric import DEFAULT_RESIDUAL_ALPHA
from reachability_gen.overfit_ff import load_jsonl

# Locked bound30 architecture (must match training checkpoints).
BOUND30_D: int = 64
BOUND30_L: int = 2
BOUND30_T_TRAIN: int = 6
BOUND30_MLP_REC: int = 10
BOUND30_RESIDUAL_ALPHA: float = DEFAULT_RESIDUAL_ALPHA  # 0.5
BOUND30_APPLY_CYCLE_RMSNORM: bool = True
BOUND30_APPLY_CYCLE_LN: bool = False
BOUND30_GEO_USE_TAU: bool = True
BOUND30_LOOP_USE_TAU: bool = False
BOUND30_MAX_T: int = 16  # tau table capacity in trained Geo ckpt

DYNAMIC_T_VALUES: tuple[int, ...] = tuple(OOD_HOP_VALUES)  # 8, 12, 16

DEFAULT_FF_CKPT = Path("artifacts/id_2k_rematch_bound30_ff_best.pt")
DEFAULT_GEO_CKPT = Path("artifacts/id_2k_rematch_bound30_geo_best.pt")
DEFAULT_LOOP_CKPT = Path("artifacts/id_2k_rematch_bound30_loop_best.pt")
DEFAULT_OOD_DATA = Path("data/ood_hops.jsonl")
DEFAULT_ID_DATA = Path("data/id_2k.jsonl")
DEFAULT_OUT = Path("artifacts/id_2k_rematch_bound30_gate2_ood.json")
DEFAULT_SUMMARY = Path("artifacts/id_2k_rematch_bound30.json")


def _n_heads(d: int) -> int:
    if d % 4 == 0:
        return 4
    if d % 2 == 0:
        return 2
    return 1


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _summarize_hop_stats(
    hop_stats: dict[int, list[tuple[float, float]]],
) -> dict[str, Any]:
    by_hop: dict[str, Any] = {}
    for k, pairs in sorted(hop_stats.items()):
        losses = [a for a, _ in pairs]
        accs = [b for _, b in pairs]
        by_hop[str(k)] = {
            "n": len(pairs),
            "loss_mean": _mean(losses),
            "acc_mean": _mean(accs),
            "correct": int(round(sum(accs))),
        }
    n_tot = sum(int(v["n"]) for v in by_hop.values())
    acc_sum = sum(float(v["acc_mean"]) * int(v["n"]) for v in by_hop.values())
    loss_sum = sum(float(v["loss_mean"]) * int(v["n"]) for v in by_hop.values())
    return {
        "overall_acc": acc_sum / n_tot if n_tot else float("nan"),
        "overall_loss": loss_sum / n_tot if n_tot else float("nan"),
        "n": n_tot,
        "by_hop": by_hop,
    }


def _load_bound30_hparams(summary_path: Path) -> dict[str, Any]:
    """Read locked hparams from bound30 summary when present."""
    if not summary_path.exists():
        return {
            "d": BOUND30_D,
            "L": BOUND30_L,
            "T": BOUND30_T_TRAIN,
            "mlp_expansion": BOUND30_MLP_REC,
            "residual_alpha": BOUND30_RESIDUAL_ALPHA,
            "apply_cycle_rmsnorm": BOUND30_APPLY_CYCLE_RMSNORM,
            "apply_cycle_ln": BOUND30_APPLY_CYCLE_LN,
            "geo_use_tau": BOUND30_GEO_USE_TAU,
            "loop_use_tau": BOUND30_LOOP_USE_TAU,
            "max_T": BOUND30_MAX_T,
            "source": "defaults",
        }
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    geo = data.get("geo", {})
    loop = data.get("loop", {})
    ff = data.get("ff", {})
    return {
        "d": int(ff.get("d", BOUND30_D)),
        "L": int(ff.get("L", BOUND30_L)),
        "T": int(geo.get("T", BOUND30_T_TRAIN)),
        "mlp_expansion": int(geo.get("mlp_expansion", BOUND30_MLP_REC)),
        "residual_alpha": float(geo.get("residual_alpha", BOUND30_RESIDUAL_ALPHA)),
        "apply_cycle_rmsnorm": bool(
            geo.get("apply_cycle_rmsnorm", BOUND30_APPLY_CYCLE_RMSNORM)
        ),
        "apply_cycle_ln": bool(geo.get("apply_cycle_ln", BOUND30_APPLY_CYCLE_LN)),
        "geo_use_tau": bool(geo.get("use_tau", BOUND30_GEO_USE_TAU)),
        "loop_use_tau": bool(loop.get("use_tau", BOUND30_LOOP_USE_TAU)),
        "max_T": BOUND30_MAX_T,
        "ff_best_epoch": ff.get("best_epoch"),
        "geo_best_epoch": geo.get("best_epoch"),
        "loop_best_epoch": loop.get("best_epoch"),
        "source": str(summary_path),
    }


def _build_ff(vocab_size: int, pad_id: int, max_len: int, hparams: dict[str, Any]):
    from reachability_gen.models.feedforward import FeedForward

    return FeedForward(
        vocab_size=vocab_size,
        d=int(hparams["d"]),
        L=int(hparams["L"]),
        n_heads=_n_heads(int(hparams["d"])),
        max_len=max_len,
        pad_id=pad_id,
        mlp_expansion=4,
    )


def _build_geo(vocab_size: int, pad_id: int, max_len: int, hparams: dict[str, Any]):
    from reachability_gen.models.geometric import GeometricRecurrent

    return GeometricRecurrent(
        vocab_size=vocab_size,
        d=int(hparams["d"]),
        T=int(hparams["T"]),
        n_heads=_n_heads(int(hparams["d"])),
        max_len=max_len,
        pad_id=pad_id,
        use_tau=bool(hparams["geo_use_tau"]),
        max_T=int(hparams["max_T"]),
        mlp_expansion=int(hparams["mlp_expansion"]),
        residual_alpha=float(hparams["residual_alpha"]),
        apply_cycle_ln=bool(hparams["apply_cycle_ln"]),
        apply_cycle_rmsnorm=bool(hparams["apply_cycle_rmsnorm"]),
    )


def _build_loop(vocab_size: int, pad_id: int, max_len: int, hparams: dict[str, Any]):
    from reachability_gen.models.euclidean_loop import EuclideanLoop

    return EuclideanLoop(
        vocab_size=vocab_size,
        d=int(hparams["d"]),
        T=int(hparams["T"]),
        n_heads=_n_heads(int(hparams["d"])),
        max_len=max_len,
        pad_id=pad_id,
        use_tau=False,
        max_T=int(hparams["max_T"]),
        mlp_expansion=int(hparams["mlp_expansion"]),
        residual_alpha=float(hparams["residual_alpha"]),
        apply_cycle_ln=bool(hparams["apply_cycle_ln"]),
        apply_cycle_rmsnorm=bool(hparams["apply_cycle_rmsnorm"]),
    )


def _load_ckpt(model, path: Path) -> dict[str, Any]:
    import torch

    # the checkpoints hold only tensors and plain values, so no unpickling of arbitrary objects is needed
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return {
        "path": str(path),
        "epoch": ckpt.get("epoch"),
        "val_acc": ckpt.get("val_acc"),
        "arm": ckpt.get("arm"),
        "science_open": ckpt.get("science_open", False),
    }


def _eval_ff(
    model,
    rows: list[dict[str, Any]],
    *,
    vocab,
    max_len: int,
    run_id: str,
    arm_name: str,
    param_count: int,
    d: int,
    L: int,
    metrics_fh=None,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    from reachability_gen.flops import flops_feedforward
    from reachability_gen.train.ff_trainer import examples_to_batch

    loss_fn = nn.CrossEntropyLoss()
    hop_stats: dict[int, list[tuple[float, float]]] = defaultdict(list)
    step = 0
    model.eval()
    with torch.no_grad():
        for ex in rows:
            token_ids, mask, labels, _ = examples_to_batch(
                [ex], vocab, max_len=max_len
            )
            logits, _ = model(token_ids, mask)
            loss = float(loss_fn(logits, labels).item())
            pred = int(logits.argmax(dim=-1).item())
            y = int(labels.item())
            acc = 1.0 if pred == y else 0.0
            hop = int(ex.get("hop_distance", HOP_UNREACHABLE))
            hop_stats[hop].append((loss, acc))
            step += 1
            if metrics_fh is not None:
                enc = str(ex.get("encoding", ""))
                seq_len = max(len(enc.split()), 1)
                flop_report = flops_feedforward(seq_len, d=d, L=L)
                record = {
                    "run_id": run_id,
                    "seed": int(ex.get("seed", 0)),
                    "arm": arm_name,
                    "step": step,
                    "epoch": 0,
                    "param_count": int(param_count),
                    "d_model": int(d),
                    "seq_len": int(seq_len),
                    "cycles_or_depth": int(L),
                    "tokens_decoded": None,
                    "cumulative_flops": float(flop_report.flops),
                    "hop_distance": hop,
                    "is_ood": bool(ex.get("is_ood", False)),
                    "loss": loss,
                    "accuracy": acc,
                    "drift_trajectory": [],
                    "terminal_drift": None,
                    "perturbation_delta": None,
                    "split": ex.get("split", "ood_hops"),
                    "n": int(ex.get("n", 0)),
                    "p": float(ex.get("p", 0.0)),
                    "edge_hash": ex.get("edge_hash", ""),
                    "s": int(ex.get("s", 0)),
                    "t": int(ex.get("t", 0)),
                    "y": y,
                    "science_open": SCIENCE_OPEN_DEFAULT,
                }
                validate_metric_record(record)
                metrics_fh.write(json.dumps(record, sort_keys=True) + "\n")
    summary = _summarize_hop_stats(hop_stats)
    summary["arm"] = arm_name
    summary["cycles_or_depth"] = L
    summary["mode"] = "fixed_L"
    return summary


def _eval_recurrent(
    model,
    rows: list[dict[str, Any]],
    *,
    vocab,
    max_len: int,
    T: int,
    run_id: str,
    arm_name: str,
    param_count: int,
    d: int,
    use_tau: bool,
    metrics_fh=None,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    from reachability_gen.flops import flops_euclidean_loop, flops_geometric
    from reachability_gen.train.ff_trainer import examples_to_batch

    loss_fn = nn.CrossEntropyLoss()
    hop_stats: dict[int, list[tuple[float, float]]] = defaultdict(list)
    step = 0
    model.eval()
    with torch.no_grad():
        for ex in rows:
            token_ids, mask, labels, _ = examples_to_batch(
                [ex], vocab, max_len=max_len
            )
            logits, _ = model(token_ids, mask, T=T)
            loss = float(loss_fn(logits, labels).item())
            pred = int(logits.argmax(dim=-1).item())
            y = int(labels.item())
            acc = 1.0 if pred == y else 0.0
            hop = int(ex.get("hop_distance", HOP_UNREACHABLE))
            hop_stats[hop].append((loss, acc))
            step += 1
            if metrics_fh is not None:
                enc = str(ex.get("encoding", ""))
                seq_len = max(len(enc.split()), 1)
                if use_tau:
                    flop_report = flops_geometric(
                        seq_len, d=d, T=T, use_tau=True
                    )
                else:
                    flop_report = flops_euclidean_loop(seq_len, d=d, T=T)
                record = {
                    "run_id": run_id,
                    "seed": int(ex.get("seed", 0)),
                    "arm": arm_name,
                    "step": step,
                    "epoch": 0,
                    "param_count": int(param_count),
                    "d_model": int(d),
                    "seq_len": int(seq_len),
                    "cycles_or_depth": int(T),
                    "tokens_decoded": None,
                    "cumulative_flops": float(flop_report.flops),
                    "hop_distance": hop,
                    "is_ood": bool(ex.get("is_ood", False)),
                    "loss": loss,
                    "accuracy": acc,
                    "drift_trajectory": [],
                    "terminal_drift": None,
                    "perturbation_delta": None,
                    "split": ex.get("split", "ood_hops"),
                    "n": int(ex.get("n", 0)),
                    "p": float(ex.get("p", 0.0)),
                    "edge_hash": ex.get("edge_hash", ""),
                    "s": int(ex.get("s", 0)),
                    "t": int(ex.get("t", 0)),
                    "y": y,
                    "science_open": SCIENCE_OPEN_DEFAULT,
                }
                validate_metric_record(record)
                metrics_fh.write(json.dumps(record, sort_keys=True) + "\n")
    summary = _summarize_hop_stats(hop_stats)
    summary["arm"] = arm_name
    summary["cycles_or_depth"] = T
    summary["mode"] = "fixed_T" if T == BOUND30_T_TRAIN else "dynamic_T"
    return summary


def _id_val_sanity(
    *,
    ff_model,
    geo_model,
    loop_model,
    id_rows: list[dict[str, Any]],
    vocab,
    max_len: int,
    hparams: dict[str, Any],
) -> dict[str, Any]:
    """Brief ID val sanity at best ckpt to confirm load."""
    val = [r for r in id_rows if r.get("split") == "val"]
    if not val:
        return {"note": "no id val rows", "science_open": False}
    ff_pc = sum(p.numel() for p in ff_model.parameters() if p.requires_grad)
    geo_pc = sum(p.numel() for p in geo_model.parameters() if p.requires_grad)
    loop_pc = sum(p.numel() for p in loop_model.parameters() if p.requires_grad)
    ff = _eval_ff(
        ff_model,
        val,
        vocab=vocab,
        max_len=max_len,
        run_id="id-sanity-ff",
        arm_name=f"ff-L{hparams['L']}-d{hparams['d']}",
        param_count=ff_pc,
        d=int(hparams["d"]),
        L=int(hparams["L"]),
    )
    geo = _eval_recurrent(
        geo_model,
        val,
        vocab=vocab,
        max_len=max_len,
        T=int(hparams["T"]),
        run_id="id-sanity-geo",
        arm_name=f"geo-T{hparams['T']}-d{hparams['d']}-mlp{hparams['mlp_expansion']}-tau",
        param_count=geo_pc,
        d=int(hparams["d"]),
        use_tau=True,
    )
    loop = _eval_recurrent(
        loop_model,
        val,
        vocab=vocab,
        max_len=max_len,
        T=int(hparams["T"]),
        run_id="id-sanity-loop",
        arm_name=f"loop-T{hparams['T']}-d{hparams['d']}-mlp{hparams['mlp_expansion']}",
        param_count=loop_pc,
        d=int(hparams["d"]),
        use_tau=False,
    )
    return {
        "n_val": len(val),
        "ff_overall_acc": ff["overall_acc"],
        "geo_overall_acc": geo["overall_acc"],
        "loop_overall_acc": loop["overall_acc"],
        "note": "ID val at best ckpt; confirms load (not Gate-2 claim)",
        "science_open": False,
    }


def _interpret(
    fixed: dict[str, Any],
    dynamic: dict[str, Any],
) -> dict[str, Any]:
    """Prereg interpretation framing — evidence vs aspiration; never OPEN."""

    def _hop_acc(block: dict[str, Any], hop: int) -> float:
        by = block.get("by_hop", {})
        cell = by.get(str(hop), {})
        return float(cell.get("acc_mean", float("nan")))

    ff_fixed = fixed["ff"]
    geo_fixed = fixed["geo"]
    loop_fixed = fixed["loop"]

    ff_ood_pos = [_hop_acc(ff_fixed, k) for k in OOD_HOPS]
    geo_t6 = [_hop_acc(geo_fixed, k) for k in OOD_HOPS]
    loop_t6 = [_hop_acc(loop_fixed, k) for k in OOD_HOPS]
    ff_neg = _hop_acc(ff_fixed, HOP_UNREACHABLE)
    geo_neg = _hop_acc(geo_fixed, HOP_UNREACHABLE)
    loop_neg = _hop_acc(loop_fixed, HOP_UNREACHABLE)

    geo_tk: list[float] = []
    loop_tk: list[float] = []
    for k in OOD_HOPS:
        geo_block = dynamic["geo"].get(str(k), {})
        loop_block = dynamic["loop"].get(str(k), {})
        geo_tk.append(_hop_acc(geo_block, k))
        loop_tk.append(_hop_acc(loop_block, k))

    def _finite_mean(xs: list[float]) -> float:
        good = [x for x in xs if x == x]
        return sum(good) / len(good) if good else float("nan")

    ff_mean = _finite_mean(ff_ood_pos)
    geo_t6_mean = _finite_mean(geo_t6)
    loop_t6_mean = _finite_mean(loop_t6)
    geo_tk_mean = _finite_mean(geo_tk)
    loop_tk_mean = _finite_mean(loop_tk)

    ff_overall = float(ff_fixed.get("overall_acc", float("nan")))
    geo_overall = float(geo_fixed.get("overall_acc", float("nan")))
    loop_overall = float(loop_fixed.get("overall_acc", float("nan")))

    chance = 0.55
    # Hard-neg collapse: all arms ~0 on y=0 OOD → overall near chance.
    neg_collapse_all = (
        ff_neg < 0.05 and geo_neg < 0.05 and loop_neg < 0.05
    )
    overall_near_chance = (
        ff_overall < chance and geo_overall < chance and loop_overall < chance
    )
    ff_collapses_pos = ff_mean < chance
    geo_maintains = geo_tk_mean >= 0.70
    loop_maintains = loop_tk_mean >= 0.70
    geo_t6_beats_tk = geo_t6_mean > geo_tk_mean + 0.05
    loop_t6_beats_tk = loop_t6_mean > loop_tk_mean + 0.05

    # Priority (fail-closed): STOP if discrimination fails for all arms;
    # OPEN candidate only if FF pos collapses while recurrent T≥K holds;
    # else note unroll harm / inconclusive. Never stamp science_open.
    if neg_collapse_all and overall_near_chance:
        fail_closed = (
            "STOP: all arms near chance on OOD overall (hard-neg acc≈0) — "
            "recurrence length generalization / discrimination on this "
            "substrate not supported; positives alone do not license OPEN"
        )
        framing = "STOP"
    elif ff_collapses_pos and (geo_maintains or loop_maintains):
        fail_closed = (
            "OPEN candidate (NOT stamped): FF collapses on K≥8 positives "
            "while Geo/Loop with T≥K maintain connectivity — structural "
            "advantage *candidate* only; science_open remains false"
        )
        framing = "OPEN_CANDIDATE_NOT_STAMPED"
    elif geo_t6_beats_tk or loop_t6_beats_tk:
        fail_closed = (
            "NOTE: Geo/Loop @T=6 beats @T=K on OOD positives — extra "
            "test-time unroll not helping (or harming); not an OPEN claim"
        )
        framing = "UNROLL_NOT_HELPING"
    else:
        fail_closed = (
            "INCONCLUSIVE under prereg thresholds — document tables; "
            "do not stamp science_open"
        )
        framing = "INCONCLUSIVE"

    return {
        "framing": framing,
        "fail_closed_one_liner": fail_closed,
        "metrics_snapshot": {
            "ff_ood_pos_acc_mean": ff_mean,
            "geo_T6_ood_pos_acc_mean": geo_t6_mean,
            "loop_T6_ood_pos_acc_mean": loop_t6_mean,
            "geo_TeqK_ood_pos_acc_mean": geo_tk_mean,
            "loop_TeqK_ood_pos_acc_mean": loop_tk_mean,
            "ff_overall_acc": ff_overall,
            "geo_T6_overall_acc": geo_overall,
            "loop_T6_overall_acc": loop_overall,
            "ff_hardneg_acc": ff_neg,
            "geo_T6_hardneg_acc": geo_neg,
            "loop_T6_hardneg_acc": loop_neg,
            "ff_per_hop": {str(k): v for k, v in zip(OOD_HOPS, ff_ood_pos)},
            "geo_T6_per_hop": {str(k): v for k, v in zip(OOD_HOPS, geo_t6)},
            "loop_T6_per_hop": {str(k): v for k, v in zip(OOD_HOPS, loop_t6)},
            "geo_TeqK_per_hop": {str(k): v for k, v in zip(OOD_HOPS, geo_tk)},
            "loop_TeqK_per_hop": {str(k): v for k, v in zip(OOD_HOPS, loop_tk)},
        },
        "prereg_rules": {
            "ff_collapses_geo_loop_TgeK_maintains": (
                "structural advantage *candidate* (never auto-OPEN)"
            ),
            "all_arms_fail_similarly": (
                "STOP for recurrence length generalization on this substrate"
            ),
            "hardneg_collapse_overall_chance": (
                "STOP — positives alone do not license OPEN"
            ),
            "geo_T6_beats_geo_TK": "extra unroll not helping / harming",
        },
        "label": "evidence vs aspiration — MEASURE plumbing only",
        "science_open": False,
    }



def run_gate2(
    *,
    ood_data: Path = DEFAULT_OOD_DATA,
    id_data: Path = DEFAULT_ID_DATA,
    ff_ckpt: Path = DEFAULT_FF_CKPT,
    geo_ckpt: Path = DEFAULT_GEO_CKPT,
    loop_ckpt: Path = DEFAULT_LOOP_CKPT,
    summary_path: Path = DEFAULT_SUMMARY,
    out_path: Path = DEFAULT_OUT,
    max_len: int = BOUND30_MAX_LEN,
    skip_id_sanity: bool = False,
    write_metrics_jsonl: bool = False,
) -> dict[str, Any]:
    """Run full Gate-2 OOD eval; return artifact dict."""
    import torch

    from reachability_gen.tokenize import build_vocab
    from reachability_gen.train.ff_trainer import examples_to_batch

    if not ood_data.exists():
        raise FileNotFoundError(
            f"OOD dataset missing: {ood_data}. "
            "Run: python -m reachability_gen.gen_ood_hops"
        )
    for p in (ff_ckpt, geo_ckpt, loop_ckpt):
        if not p.exists():
            raise FileNotFoundError(f"checkpoint missing: {p}")

    ood_rows = load_jsonl(ood_data)
    ok, issues = verify_ood_hops(ood_rows)
    if not ok:
        raise RuntimeError(f"ood_hops verify failed: {issues}")

    hparams = _load_bound30_hparams(summary_path)
    vocab = build_vocab()
    # Warm-touch so examples_to_batch path is consistent.
    _ = examples_to_batch(ood_rows[:1], vocab, max_len=max_len)

    ff_model = _build_ff(len(vocab), vocab.pad_id, max_len, hparams)
    geo_model = _build_geo(len(vocab), vocab.pad_id, max_len, hparams)
    loop_model = _build_loop(len(vocab), vocab.pad_id, max_len, hparams)

    ff_meta = _load_ckpt(ff_model, ff_ckpt)
    geo_meta = _load_ckpt(geo_model, geo_ckpt)
    loop_meta = _load_ckpt(loop_model, loop_ckpt)

    # Sanity: pos_emb length
    ff_pos = int(ff_model.pos_emb.weight.shape[0])
    if ff_pos != max_len:
        print(
            f"WARN: ff pos_emb={ff_pos} != requested max_len={max_len}; "
            f"using checkpoint capacity {ff_pos}",
            file=sys.stderr,
        )
        max_len = ff_pos

    ff_pc = sum(p.numel() for p in ff_model.parameters() if p.requires_grad)
    geo_pc = sum(p.numel() for p in geo_model.parameters() if p.requires_grad)
    loop_pc = sum(p.numel() for p in loop_model.parameters() if p.requires_grad)

    run_id = f"gate2-ood-{uuid.uuid4().hex[:10]}"
    metrics_path = Path("artifacts/id_2k_rematch_bound30_gate2_ood_metrics.jsonl")
    metrics_fh = None
    if write_metrics_jsonl:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_fh = metrics_path.open("w", encoding="utf-8")

    print(
        f"[gate2] OOD n={len(ood_rows)}; fixed T={hparams['T']}/L={hparams['L']}; "
        f"dynamic T={list(DYNAMIC_T_VALUES)}; max_len={max_len}",
        file=sys.stderr,
    )

    # --- Fixed depth ---
    ff_fixed = _eval_ff(
        ff_model,
        ood_rows,
        vocab=vocab,
        max_len=max_len,
        run_id=f"{run_id}-ff-fixed",
        arm_name=f"ff-L{hparams['L']}-d{hparams['d']}",
        param_count=ff_pc,
        d=int(hparams["d"]),
        L=int(hparams["L"]),
        metrics_fh=metrics_fh,
    )
    print(
        f"[gate2] FF L={hparams['L']} overall_acc={ff_fixed['overall_acc']:.4f}",
        file=sys.stderr,
    )

    geo_fixed = _eval_recurrent(
        geo_model,
        ood_rows,
        vocab=vocab,
        max_len=max_len,
        T=int(hparams["T"]),
        run_id=f"{run_id}-geo-T{hparams['T']}",
        arm_name=(
            f"geo-T{hparams['T']}-d{hparams['d']}-mlp{hparams['mlp_expansion']}-tau"
        ),
        param_count=geo_pc,
        d=int(hparams["d"]),
        use_tau=True,
        metrics_fh=metrics_fh,
    )
    print(
        f"[gate2] Geo T={hparams['T']} overall_acc={geo_fixed['overall_acc']:.4f}",
        file=sys.stderr,
    )

    loop_fixed = _eval_recurrent(
        loop_model,
        ood_rows,
        vocab=vocab,
        max_len=max_len,
        T=int(hparams["T"]),
        run_id=f"{run_id}-loop-T{hparams['T']}",
        arm_name=f"loop-T{hparams['T']}-d{hparams['d']}-mlp{hparams['mlp_expansion']}",
        param_count=loop_pc,
        d=int(hparams["d"]),
        use_tau=False,
        metrics_fh=metrics_fh,
    )
    print(
        f"[gate2] Loop T={hparams['T']} overall_acc={loop_fixed['overall_acc']:.4f}",
        file=sys.stderr,
    )

    fixed = {"ff": ff_fixed, "geo": geo_fixed, "loop": loop_fixed}

    # --- Dynamic unroll ---
    geo_dyn: dict[str, Any] = {}
    loop_dyn: dict[str, Any] = {}
    for T in DYNAMIC_T_VALUES:
        geo_dyn[str(T)] = _eval_recurrent(
            geo_model,
            ood_rows,
            vocab=vocab,
            max_len=max_len,
            T=T,
            run_id=f"{run_id}-geo-T{T}",
            arm_name=(
                f"geo-T{T}-d{hparams['d']}-mlp{hparams['mlp_expansion']}-tau"
            ),
            param_count=geo_pc,
            d=int(hparams["d"]),
            use_tau=True,
            metrics_fh=metrics_fh,
        )
        print(
            f"[gate2] Geo T={T} overall_acc={geo_dyn[str(T)]['overall_acc']:.4f}",
            file=sys.stderr,
        )
        loop_dyn[str(T)] = _eval_recurrent(
            loop_model,
            ood_rows,
            vocab=vocab,
            max_len=max_len,
            T=T,
            run_id=f"{run_id}-loop-T{T}",
            arm_name=(
                f"loop-T{T}-d{hparams['d']}-mlp{hparams['mlp_expansion']}"
            ),
            param_count=loop_pc,
            d=int(hparams["d"]),
            use_tau=False,
            metrics_fh=metrics_fh,
        )
        print(
            f"[gate2] Loop T={T} overall_acc={loop_dyn[str(T)]['overall_acc']:.4f}",
            file=sys.stderr,
        )

    dynamic = {"geo": geo_dyn, "loop": loop_dyn, "T_values": list(DYNAMIC_T_VALUES)}

    # Hop × T matrices (rows=hops, cols=T) for Geo and Loop
    def _matrix(arm_dyn: dict[str, Any]) -> dict[str, Any]:
        mat: dict[str, dict[str, float]] = {}
        for hop in list(OOD_HOPS) + [HOP_UNREACHABLE]:
            row: dict[str, float] = {}
            for T in DYNAMIC_T_VALUES:
                cell = arm_dyn[str(T)]["by_hop"].get(str(hop), {})
                row[str(T)] = float(cell.get("acc_mean", float("nan")))
            mat[str(hop)] = row
        return mat

    hop_T_acc_geo = _matrix(geo_dyn)
    hop_T_acc_loop = _matrix(loop_dyn)

    id_sanity: dict[str, Any] = {"skipped": True, "science_open": False}
    if not skip_id_sanity and id_data.exists():
        id_rows = load_jsonl(id_data)
        id_sanity = _id_val_sanity(
            ff_model=ff_model,
            geo_model=geo_model,
            loop_model=loop_model,
            id_rows=id_rows,
            vocab=vocab,
            max_len=max_len,
            hparams=hparams,
        )
        print(
            f"[gate2] ID val sanity: "
            f"FF={id_sanity.get('ff_overall_acc')} "
            f"Geo={id_sanity.get('geo_overall_acc')} "
            f"Loop={id_sanity.get('loop_overall_acc')}",
            file=sys.stderr,
        )

    if metrics_fh is not None:
        metrics_fh.close()

    interpretation = _interpret(fixed, dynamic)

    # Dataset composition from OOD rows
    from collections import Counter

    hop_counts = Counter(int(r["hop_distance"]) for r in ood_rows)
    y_counts = Counter(int(r["y"]) for r in ood_rows)
    n_vals = [int(r["n"]) for r in ood_rows]
    p_vals = [float(r["p"]) for r in ood_rows]

    artifact: dict[str, Any] = {
        "gate": "gate2_ood_hop_stress",
        "mode": "MEASURE",
        "science_open": False,
        "dataset": {
            "path": str(ood_data),
            "n_total": len(ood_rows),
            "y_counts": {str(k): v for k, v in sorted(y_counts.items())},
            "hop_counts": {str(k): v for k, v in sorted(hop_counts.items())},
            "n_range": [min(n_vals), max(n_vals)] if n_vals else None,
            "p_range": [min(p_vals), max(p_vals)] if p_vals else None,
            "ood_hops": list(OOD_HOPS),
            "verify_ok": True,
        },
        "checkpoints": {
            "ff": ff_meta,
            "geo": geo_meta,
            "loop": loop_meta,
        },
        "hparams": hparams,
        "param_counts": {
            "ff": ff_pc,
            "geo": geo_pc,
            "loop": loop_pc,
        },
        "max_len": max_len,
        "id_val_sanity": id_sanity,
        "fixed_depth": {
            "note": "FF at trained L=2; Geo/Loop at trained T=6",
            "ff": ff_fixed,
            "geo": geo_fixed,
            "loop": loop_fixed,
        },
        "dynamic_unroll": {
            "note": "Geo/Loop only; T in {8,12,16}; test extra test-time cycles",
            "T_values": list(DYNAMIC_T_VALUES),
            "geo": geo_dyn,
            "loop": loop_dyn,
            "hop_T_acc_matrix_geo": hop_T_acc_geo,
            "hop_T_acc_matrix_loop": hop_T_acc_loop,
        },
        "prereg": interpretation,
        "interpretation_stop": interpretation,
        "metrics_jsonl": str(metrics_path) if write_metrics_jsonl else None,
    }

    # Walk: never allow science_open True anywhere we control.
    def _assert_no_open(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            if obj.get("science_open") is True:
                raise AssertionError(f"science_open=True at {path}")
            for k, v in obj.items():
                _assert_no_open(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _assert_no_open(v, f"{path}[{i}]")

    _assert_no_open(artifact)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"[gate2] wrote {out_path}", file=sys.stderr)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Gate 2 OOD-hop stress on bound30 best ckpts "
            "(MEASURE; inference-only; science_open=false)."
        )
    )
    p.add_argument("--ood-data", type=Path, default=DEFAULT_OOD_DATA)
    p.add_argument("--id-data", type=Path, default=DEFAULT_ID_DATA)
    p.add_argument("--ff-ckpt", type=Path, default=DEFAULT_FF_CKPT)
    p.add_argument("--geo-ckpt", type=Path, default=DEFAULT_GEO_CKPT)
    p.add_argument("--loop-ckpt", type=Path, default=DEFAULT_LOOP_CKPT)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-len", type=int, default=BOUND30_MAX_LEN)
    p.add_argument("--skip-id-sanity", action="store_true")
    p.add_argument(
        "--write-metrics-jsonl",
        action="store_true",
        help="Also write per-example RunMetricRecord JSONL (gitignored).",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        artifact = run_gate2(
            ood_data=args.ood_data,
            id_data=args.id_data,
            ff_ckpt=args.ff_ckpt,
            geo_ckpt=args.geo_ckpt,
            loop_ckpt=args.loop_ckpt,
            summary_path=args.summary,
            out_path=args.out,
            max_len=args.max_len,
            skip_id_sanity=args.skip_id_sanity,
            write_metrics_jsonl=args.write_metrics_jsonl,
        )
    except Exception as exc:
        print(f"gate2 FAILED: {exc}", file=sys.stderr)
        return 1
    framing = artifact.get("prereg", {}).get("framing", "?")
    one_liner = artifact.get("prereg", {}).get("fail_closed_one_liner", "")
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(args.out),
                "framing": framing,
                "fail_closed": one_liner,
                "science_open": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

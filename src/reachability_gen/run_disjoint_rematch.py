# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Multi-seed fixed-budget rematch on the graph-disjoint ID set (MEASURE).

Protocol: the bound30 rematch arms and optimisation settings, unchanged
(imported from :mod:`run_id_2k_rematch_bound30`), trained for a fixed 30
epochs with no early stopping on ``data/id_disjoint_2k.jsonl`` instead of
``id_2k``, over several seeds.

Plumbing differences from the bound30 runner:

* context sized with :func:`tokenize.required_max_len`; overlong input is an
  error, never truncated; batches are padded to their longest row;
* parameter parity checked on non-embedding counts (ADR-001 §4);
* validation is batched and logged every epoch with breakdown diagnostics
  (last-state token cosine on a fixed validation subset, output
  concentration);
* best *and* final checkpoints are saved and both are reported; each is
  re-scored with the same evaluation at the end (self-audit) and scored with
  the checkpoint audit's per-example telemetry.

``science_open=false`` always.

Usage::

    python -m reachability_gen.run_disjoint_rematch
    python -m reachability_gen.run_disjoint_rematch --seeds 0 --epochs 2   # smoke
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.diagnostics import collapse_flags, median_abs_deviation
from reachability_gen.gen_id_disjoint import DEFAULT_OUT as DEFAULT_DATA
from reachability_gen.gen_id_disjoint import verify_id_disjoint
from reachability_gen.models.geometric import DEFAULT_RESIDUAL_ALPHA
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_id_2k_rematch import assert_param_parity
from reachability_gen.run_id_2k_rematch_bound30 import (
    DEFAULT_BATCH,
    DEFAULT_EPOCHS,
    DEFAULT_FF_D,
    DEFAULT_FF_LR,
    DEFAULT_L,
    DEFAULT_MLP_REC,
    DEFAULT_REC_LR,
    DEFAULT_T,
    FF_GRAD_CLIP,
    REC_GRAD_CLIP,
    _n_heads,
    _split_train_val,
)
from reachability_gen.tokenize import Vocab, build_vocab, max_token_len, required_max_len

ARMS: tuple[str, ...] = ("ff", "geo", "loop")
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
DEFAULT_OUT = Path("artifacts/id_disjoint_rematch.json")
DEFAULT_CKPT_DIR = Path("artifacts/id_disjoint_rematch")
EVAL_BATCH: int = 100
DIAG_ROWS: int = 64  # fixed validation subset for per-epoch coherence


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _sd(xs: Sequence[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def build_arm(kind: str, vocab: Vocab, max_len: int) -> tuple[Any, Any]:
    """Model + trainer with the bound30 settings for ``kind``."""
    from reachability_gen.models.euclidean_loop import EuclideanLoop
    from reachability_gen.models.feedforward import FeedForward
    from reachability_gen.models.geometric import GeometricRecurrent
    from reachability_gen.train.ff_trainer import FeedForwardTrainer
    from reachability_gen.train.geo_trainer import GeometricTrainer
    from reachability_gen.train.loop_trainer import LoopTrainer

    common = {"n_heads": _n_heads(DEFAULT_FF_D), "max_len": max_len, "pad_id": vocab.pad_id}
    if kind == "ff":
        model = FeedForward(len(vocab), d=DEFAULT_FF_D, L=DEFAULT_L, **common)
        trainer = FeedForwardTrainer(
            model, lr=DEFAULT_FF_LR, weight_decay=0.01, grad_clip=FF_GRAD_CLIP
        )
        return model, trainer
    rec = dict(
        d=DEFAULT_FF_D,
        T=DEFAULT_T,
        mlp_expansion=DEFAULT_MLP_REC,
        residual_alpha=DEFAULT_RESIDUAL_ALPHA,
        apply_cycle_ln=False,
        apply_cycle_rmsnorm=True,
        **common,
    )
    if kind == "geo":
        model = GeometricRecurrent(len(vocab), use_tau=True, **rec)
        trainer = GeometricTrainer(
            model, lr=DEFAULT_REC_LR, weight_decay=0.01, grad_clip=REC_GRAD_CLIP
        )
    elif kind == "loop":
        model = EuclideanLoop(len(vocab), **rec)
        trainer = LoopTrainer(
            model, lr=DEFAULT_REC_LR, weight_decay=0.01, grad_clip=REC_GRAD_CLIP
        )
    else:
        raise ValueError(f"unknown arm {kind!r}")
    return model, trainer


def evaluate(
    model: Any,
    rows: Sequence[dict[str, Any]],
    vocab: Vocab,
    *,
    batch_size: int = EVAL_BATCH,
    diag_rows: int = DIAG_ROWS,
) -> dict[str, Any]:
    """Batched accuracy / loss by hop plus breakdown diagnostics (deterministic)."""
    import torch
    import torch.nn.functional as F

    from reachability_gen.diagnostics import logit_margins, token_coherence
    from reachability_gen.train.ff_trainer import examples_to_batch

    model.eval()
    correct: list[float] = []
    losses: list[float] = []
    margins: list[float] = []
    hops: list[int] = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            chunk = list(rows[i : i + batch_size])
            ids, mask, labels, _ = examples_to_batch(chunk, vocab, on_overflow="error")
            logits, _ = model(ids, mask)
            losses += F.cross_entropy(logits, labels, reduction="none").tolist()
            correct += (logits.argmax(dim=-1) == labels).float().tolist()
            margins += logit_margins(logits).tolist()
            hops += [int(r["hop_distance"]) for r in chunk]
        ids, mask, _, _ = examples_to_batch(list(rows[:diag_rows]), vocab, on_overflow="error")
        cos_last = float(token_coherence(model.token_states(ids, mask)[-1], mask)[0].mean())
    by_hop = {}
    for k in sorted(set(hops)):
        idx = [j for j, h in enumerate(hops) if h == k]
        by_hop[str(k)] = {
            "n": len(idx),
            "acc": _mean([correct[j] for j in idx]),
            "loss": _mean([losses[j] for j in idx]),
        }
    sd = statistics.pstdev(margins) if len(margins) > 1 else 0.0
    return {
        "acc": _mean(correct),
        "n_correct": int(sum(correct)),
        "loss": _mean(losses),
        "by_hop": by_hop,
        "token_cos_last": cos_last,
        "output_concentration": median_abs_deviation(margins) / sd if sd > 0 else 0.0,
        "margins": margins,
    }


def _telemetry(model: Any, kind: str, val: list[dict[str, Any]], vocab: Vocab) -> dict[str, Any]:
    """Per-example audit telemetry (drift, coherence, perturbation gain) by hop."""
    from reachability_gen.audit_checkpoints import _aggregate, score_model

    records = score_model(model, kind, val, vocab)
    recurrent = kind != "ff"
    overall = _aggregate(records, recurrent)
    keep = ("acc_mean", "token_cos_by_t", "pooled_ratio_by_t", "token_norm_by_t")
    if recurrent:
        keep += ("mean_z_norm_by_t", "perturb_gain_by_t", "perturb_gain_rel_by_t")
    return {
        "per_example_acc": overall["acc_mean"],
        "summary": {k: overall[k] for k in keep},
        "by_hop": {
            str(k): {
                kk: v
                for kk, v in _aggregate([r for r in records if r["hop"] == k], recurrent).items()
                if kk in ("n", "acc_mean", "loss_mean", "pred_pos_rate", "token_cos_by_t",
                          "mean_drift_trajectory", "perturb_gain_by_t", "damp_regime")
            }
            for k in sorted({r["hop"] for r in records})
        },
    }


def train_arm(
    kind: str,
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    vocab: Vocab,
    *,
    seed: int,
    epochs: int,
    max_len: int,
    ckpt_dir: Path,
    telemetry: bool = True,
) -> dict[str, Any]:
    """Train one arm for a fixed budget; save and self-audit best + final."""
    import torch

    from reachability_gen.train.ff_trainer import examples_to_batch

    torch.manual_seed(seed)
    model, trainer = build_arm(kind, vocab, max_len)
    history: list[dict[str, Any]] = []
    best_acc, best_epoch, best_state = -1.0, 0, None
    n_steps = n_sat = 0
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train)).tolist()
        losses: list[float] = []
        accs: list[float] = []
        for start in range(0, len(train), DEFAULT_BATCH):
            batch = [train[i] for i in order[start : start + DEFAULT_BATCH]]
            ids, mask, labels, _ = examples_to_batch(batch, vocab, on_overflow="error")
            loss, acc = trainer.train_step(ids, labels, mask)
            losses.append(loss)
            accs.append(acc)
            n_steps += 1
            n_sat += int(trainer.last_pre_clip_grad_norm >= trainer.grad_clip)
        ev = evaluate(model, val, vocab)
        history.append(
            {
                "epoch": epoch,
                "train_loss": _mean(losses),
                "train_acc": _mean(accs),
                "val_acc": ev["acc"],
                "val_loss": ev["loss"],
                "token_cos_last": ev["token_cos_last"],
                "output_concentration": ev["output_concentration"],
            }
        )
        if ev["acc"] > best_acc:
            best_acc, best_epoch, best_state = ev["acc"], epoch, copy.deepcopy(model.state_dict())
        print(
            f"[seed{seed}/{kind}] epoch {epoch}/{epochs}: train_acc={_mean(accs):.4f} "
            f"val_acc={ev['acc']:.4f} (best {best_acc:.4f}@ep{best_epoch}) "
            f"tok_cos={ev['token_cos_last']:.3f}",
            file=sys.stderr,
        )
    assert best_state is not None
    states = {"best": (best_epoch, best_state), "final": (epochs, copy.deepcopy(model.state_dict()))}
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "arm": kind,
        "seed": seed,
        "epochs": epochs,
        "train_seconds": time.perf_counter() - t0,
        "grad_clip_sat_rate": n_sat / n_steps if n_steps else float("nan"),
        "history": history,
        "science_open": False,
    }
    for which, (epoch, state) in states.items():
        path = ckpt_dir / f"seed{seed}_{kind}_{which}.pt"
        torch.save(
            {"epoch": epoch, "state_dict": state, "arm": kind, "seed": seed, "science_open": False},
            path,
        )
        fresh, _ = build_arm(kind, vocab, max_len)
        fresh.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
        ev = evaluate(fresh, val, vocab)
        recorded = history[epoch - 1]["val_acc"]
        block = {
            "epoch": epoch,
            "checkpoint_path": path.as_posix(),
            "val_acc": ev["acc"],
            "rescore_matches_record": ev["acc"] == recorded,
            "acc_by_hop": {k: v["acc"] for k, v in ev["by_hop"].items()},
            "loss_by_hop": {k: v["loss"] for k, v in ev["by_hop"].items()},
            "checkpoint_flags": collapse_flags(
                final_token_cos=ev["token_cos_last"], margins=ev["margins"]
            ),
        }
        if telemetry:
            block["telemetry"] = _telemetry(fresh, kind, val, vocab)
        out[which] = block
    out["run_flags"] = collapse_flags(
        train_history=history, best_val_acc=best_acc, last_epoch_val_acc=history[-1]["val_acc"]
    )
    return out


def parity_section(vocab: Vocab, max_len: int) -> dict[str, Any]:
    """Non-embedding parity (ADR-001 §4); total counts recorded alongside."""
    models = {kind: build_arm(kind, vocab, max_len)[0] for kind in ARMS}
    non_emb = {k: int(m.non_embedding_param_count()) for k, m in models.items()}
    section = assert_param_parity(
        ff_count=non_emb["ff"], geo_count=non_emb["geo"], loop_count=non_emb["loop"]
    )
    section["counted"] = "non-embedding parameters (ADR-001 §4)"
    section["total_param_counts"] = {
        k: int(sum(p.numel() for p in m.parameters())) for k, m in models.items()
    }
    return section


def aggregate(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Across-seed mean / SD per arm for best and final checkpoints."""
    out: dict[str, Any] = {}
    arms = sorted({kind for per_seed in runs.values() for kind in per_seed})
    for kind in arms:
        blocks = [per_seed[kind] for per_seed in runs.values() if kind in per_seed]
        entry: dict[str, Any] = {"seeds": [b["seed"] for b in blocks]}
        for which in ("best", "final"):
            accs = [b[which]["val_acc"] for b in blocks]
            hops = sorted({h for b in blocks for h in b[which]["acc_by_hop"]}, key=int)
            entry[which] = {
                "val_acc_mean": _mean(accs),
                "val_acc_sd": _sd(accs),
                "val_acc_per_seed": accs,
                "acc_by_hop_mean": {
                    h: _mean([b[which]["acc_by_hop"][h] for b in blocks]) for h in hops
                },
                "acc_by_hop_sd": {
                    h: _sd([b[which]["acc_by_hop"][h] for b in blocks]) for h in hops
                },
                "seeds_with_checkpoint_flags": sum(
                    1 for b in blocks if b[which]["checkpoint_flags"]["any"]
                ),
            }
        entry["seeds_with_run_flags"] = sum(1 for b in blocks if b["run_flags"]["any"])
        out[kind] = entry
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Multi-seed fixed-budget rematch (bound30 arms and settings) on the "
            "graph-disjoint ID set (MEASURE; science_open=false)."
        )
    )
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument(
        "--threads", type=int, default=8, help="torch intra-op threads (recorded)"
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch
    except ImportError:
        print("FAIL: torch required", file=sys.stderr)
        return 2
    if not args.data.exists():
        print(
            f"FAIL: missing {args.data}; run python -m reachability_gen.gen_id_disjoint",
            file=sys.stderr,
        )
        return 1
    torch.set_num_threads(args.threads)
    rows = load_jsonl(args.data)
    ok, issues = verify_id_disjoint(rows)
    if not ok:
        print(f"FAIL: {args.data} failed verify: {issues}", file=sys.stderr)
        return 1
    train, val = _split_train_val(rows)
    vocab = build_vocab()
    max_len = required_max_len((r["encoding"] for r in rows), vocab)
    assert max_token_len((r["encoding"] for r in rows), vocab) <= max_len
    try:
        parity = parity_section(vocab, max_len)
    except AssertionError as exc:
        print(f"FAIL parity (non-embedding): {exc}", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    runs: dict[str, dict[str, Any]] = {}
    for seed in args.seeds:
        runs[str(seed)] = {
            kind: train_arm(
                kind, train, val, vocab,
                seed=seed, epochs=args.epochs, max_len=max_len, ckpt_dir=args.ckpt_dir,
            )
            for kind in args.arms
        }
    agg = aggregate(runs)
    mismatches = [
        f"seed{s}/{k}/{w}"
        for s, per_seed in runs.items()
        for k, b in per_seed.items()
        for w in ("best", "final")
        if not b[w]["rescore_matches_record"]
    ]
    artifact = {
        "science_open": False,
        "dataset": args.data.as_posix(),
        "protocol": {
            "arms_and_settings": "bound30 rematch (imported constants), unchanged",
            "epochs": args.epochs,
            "early_stopping": False,
            "seeds": args.seeds,
            "batch_size": DEFAULT_BATCH,
            "ff": {"lr": DEFAULT_FF_LR, "grad_clip": FF_GRAD_CLIP, "L": DEFAULT_L, "d": DEFAULT_FF_D},
            "recurrent": {
                "lr": DEFAULT_REC_LR,
                "grad_clip": REC_GRAD_CLIP,
                "T": DEFAULT_T,
                "mlp_expansion": DEFAULT_MLP_REC,
                "residual_alpha": DEFAULT_RESIDUAL_ALPHA,
                "state_bound": "rmsnorm",
            },
            "max_len": max_len,
            "torch_threads": args.threads,
            "torch_version": torch.__version__,
        },
        "baselines": {
            "query_blind": 0.5,
            "endpoint_rule": 0.5,
            "note": "both hold by construction of the dataset (see its generation report)",
        },
        "param_match": parity,
        "runs": runs,
        "aggregate": agg,
        "self_audit_mismatches": mismatches,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    print("\n=== val accuracy across seeds (mean ± sd) ===", file=sys.stderr)
    for kind, entry in agg.items():
        for which in ("best", "final"):
            e = entry[which]
            hops = " ".join(
                f"{h}:{e['acc_by_hop_mean'][h]:.3f}" for h in e["acc_by_hop_mean"]
            )
            print(
                f"{kind:5s} {which:5s} {e['val_acc_mean']:.4f} ± {e['val_acc_sd']:.4f} | {hops}",
                file=sys.stderr,
            )
    print(
        json.dumps(
            {
                "ok": not mismatches,
                "out": args.out.as_posix(),
                "self_audit_mismatches": mismatches,
                "science_open": False,
            },
            sort_keys=True,
        )
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())

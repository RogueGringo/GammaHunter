# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Calibration: looped vs unlooped message passing, trained short, tested long.

Positive control for the harness. The neural algorithmic reasoning literature
(e.g. Veličković et al., "Neural Execution of Graph Algorithms", 2019) reports
that message passing over a graph's own edges learns reachability, and that a
processor whose step is shared can run more steps than it was trained with to
reach farther, which a fixed-depth stack cannot. The harness should reproduce
that pattern before it is used to judge anything new.

Protocol:

* train on the graph-disjoint ID set (hops 2–6, 10–20 nodes) with 6 steps;
* unlooped arm: 6 distinct layers of width ``--d`` (default 64);
* looped arm: one shared step, width matched to the unlooped arm's parameter
  count within ±5% (ADR-001 tolerance);
* validate on the ID val split every epoch (accuracy by graph hop);
* after training, evaluate on the long-path set (hops 8–16, 24–48 nodes):
  the unlooped arm at its fixed depth, the looped arm at 6 steps and at each
  ``--test-steps`` count;
* save best and final checkpoints and re-score each (self-audit).

``science_open=false`` always.

Usage::

    python -m reachability_gen.run_mp_calibration --device cuda
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.adr_invariants import PARAM_TOL
from reachability_gen.gen_id_disjoint import EXTENDED_SPEC, verify_id_disjoint
from reachability_gen.models.message_passing import (
    MessagePassing,
    ParsedGraph,
    collate,
    match_width,
    param_formula,
    parse_rows,
)
from reachability_gen.overfit_ff import load_jsonl

DEFAULT_TRAIN = Path("data/id_disjoint_20k.jsonl")
DEFAULT_EXTENDED = Path("data/extended_disjoint_2k.jsonl")
DEFAULT_OUT = Path("artifacts/mp_calibration.json")
DEFAULT_CKPT_DIR = Path("artifacts/mp_calibration")
ARMS: tuple[str, ...] = ("unlooped", "looped")
TRAIN_STEPS: int = 6
TEST_STEPS: tuple[int, ...] = (6, 16, 32, 48)
LR: float = 1e-3
WEIGHT_DECAY: float = 0.01
GRAD_CLIP: float = 1.0
BATCH: int = 32
EVAL_BATCH: int = 100


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _sd(xs: Sequence[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def build(kind: str, d_unlooped: int) -> MessagePassing:
    """Unlooped arm at width ``d_unlooped``; looped arm width-matched to it."""
    if kind == "unlooped":
        return MessagePassing(d_unlooped, TRAIN_STEPS, looped=False)
    if kind == "looped":
        target = param_formula(d_unlooped, TRAIN_STEPS)
        d = match_width(target, TRAIN_STEPS, looped=True)
        return MessagePassing(d, TRAIN_STEPS, looped=True)
    raise ValueError(f"unknown arm {kind!r}")


def evaluate(
    model: MessagePassing,
    graphs: Sequence[ParsedGraph],
    rows: Sequence[dict[str, Any]],
    device: str,
    steps: Optional[int] = None,
) -> dict[str, Any]:
    """Accuracy and loss, overall and by each graph's hop (chance 0.5 per group)."""
    import torch
    import torch.nn.functional as F

    model.eval()
    correct: list[float] = []
    losses: list[float] = []
    with torch.no_grad():
        for i in range(0, len(graphs), EVAL_BATCH):
            batch = collate(graphs[i : i + EVAL_BATCH], device)
            logits = model(batch, steps)
            losses += F.cross_entropy(logits, batch["y"], reduction="none").tolist()
            correct += (logits.argmax(dim=-1) == batch["y"]).float().tolist()
    graph_hop = {r["edge_hash"]: int(r["hop_distance"]) for r in rows if int(r["y"]) == 1}
    ghops = [graph_hop[r["edge_hash"]] for r in rows]
    return {
        "acc": _mean(correct),
        "loss": _mean(losses),
        "by_graph_hop": {
            str(k): _mean([c for c, g in zip(correct, ghops) if g == k]) for k in sorted(set(ghops))
        },
    }


def train_arm(
    kind: str,
    train: Sequence[ParsedGraph],
    val: Sequence[ParsedGraph],
    val_rows: Sequence[dict[str, Any]],
    extended: Sequence[ParsedGraph],
    extended_rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    epochs: int,
    d: int,
    device: str,
    ckpt_dir: Path,
    test_steps: Sequence[int] = TEST_STEPS,
) -> dict[str, Any]:
    """Train one arm; save, self-audit and extrapolation-test best and final."""
    import torch
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_

    torch.manual_seed(seed)
    model = build(kind, d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def cpu_state() -> dict[str, Any]:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    history: list[dict[str, Any]] = []
    best_acc, best_epoch, best_state = -1.0, 0, None
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train)).tolist()
        losses: list[float] = []
        accs: list[float] = []
        for start in range(0, len(order), BATCH):
            batch = collate([train[i] for i in order[start : start + BATCH]], device)
            opt.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch["y"])
            loss.backward()
            clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            losses.append(float(loss.item()))
            accs.append(float((logits.argmax(dim=-1) == batch["y"]).float().mean().item()))
        ev = evaluate(model, val, val_rows, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": _mean(losses),
                "train_acc": _mean(accs),
                "val_acc": ev["acc"],
                "val_acc_by_graph_hop": ev["by_graph_hop"],
            }
        )
        if ev["acc"] > best_acc:
            best_acc, best_epoch, best_state = ev["acc"], epoch, cpu_state()
        print(
            f"[seed{seed}/{kind}] epoch {epoch}/{epochs}: train_acc={_mean(accs):.4f} "
            f"val_acc={ev['acc']:.4f} (best {best_acc:.4f}@ep{best_epoch})",
            file=sys.stderr,
        )
    assert best_state is not None
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "arm": kind,
        "seed": seed,
        "width": model.d,
        "param_count": model.param_count(),
        "train_seconds": time.perf_counter() - t0,
        "history": history,
        "science_open": False,
    }
    for which, (epoch, state) in {"best": (best_epoch, best_state), "final": (epochs, cpu_state())}.items():
        path = ckpt_dir / f"seed{seed}_{kind}_{which}.pt"
        torch.save({"epoch": epoch, "state_dict": state, "arm": kind, "seed": seed, "science_open": False}, path)
        fresh = build(kind, d).to(device)
        fresh.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
        ev = evaluate(fresh, val, val_rows, device)
        steps_list = [TRAIN_STEPS] if kind == "unlooped" else sorted(set([TRAIN_STEPS, *test_steps]))
        out[which] = {
            "epoch": epoch,
            "checkpoint_path": path.as_posix(),
            "id_val_acc": ev["acc"],
            "rescore_matches_record": ev["acc"] == history[epoch - 1]["val_acc"],
            "id_val_by_graph_hop": ev["by_graph_hop"],
            "extended_by_steps": {
                str(s): evaluate(fresh, extended, extended_rows, device, s) for s in steps_list
            },
            "id_val_acc_by_steps": {
                str(s): evaluate(fresh, val, val_rows, device, s)["acc"] for s in steps_list
            },
        }
    return out


def aggregate(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Across-seed mean / SD of ID accuracy and extended accuracy by step count."""
    out: dict[str, Any] = {}
    for kind in ARMS:
        blocks = [per_seed[kind] for per_seed in runs.values() if kind in per_seed]
        if not blocks:
            continue
        entry: dict[str, Any] = {"seeds": [b["seed"] for b in blocks]}
        for which in ("best", "final"):
            ids = [b[which]["id_val_acc"] for b in blocks]
            steps = sorted(blocks[0][which]["extended_by_steps"], key=int)
            entry[which] = {
                "id_val_acc_mean": _mean(ids),
                "id_val_acc_sd": _sd(ids),
                "extended_acc_by_steps": {
                    s: {
                        "mean": _mean([b[which]["extended_by_steps"][s]["acc"] for b in blocks]),
                        "sd": _sd([b[which]["extended_by_steps"][s]["acc"] for b in blocks]),
                    }
                    for s in steps
                },
                "extended_by_graph_hop_at_max_steps": {
                    h: _mean([b[which]["extended_by_steps"][steps[-1]]["by_graph_hop"][h] for b in blocks])
                    for h in sorted(blocks[0][which]["extended_by_steps"][steps[-1]]["by_graph_hop"], key=int)
                },
            }
        out[kind] = entry
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Message-passing calibration: looped vs unlooped, trained short, tested long (MEASURE)."
    )
    p.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--extended-data", type=Path, default=DEFAULT_EXTENDED)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--d", type=int, default=64, help="unlooped width; looped is width-matched")
    p.add_argument("--test-steps", type=int, nargs="+", default=list(TEST_STEPS))
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
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
    rows = load_jsonl(args.train_data)
    ext_rows = load_jsonl(args.extended_data)
    n_val = sum(1 for r in rows if r["split"] == "val")
    checks = [
        verify_id_disjoint(rows, n_total=len(rows), n_val=n_val),
        verify_id_disjoint(ext_rows, n_total=len(ext_rows), n_val=len(ext_rows), spec=EXTENDED_SPEC),
    ]
    issues = [i for ok, found in checks for i in found]
    if issues:
        print(f"FAIL: dataset verify: {issues}", file=sys.stderr)
        return 1
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    train, val, ext = parse_rows(train_rows), parse_rows(val_rows), parse_rows(ext_rows)

    counts = {kind: build(kind, args.d).param_count() for kind in ARMS}
    widths = {kind: build(kind, args.d).d for kind in ARMS}
    ratio = counts["looped"] / counts["unlooped"]
    if abs(ratio - 1.0) > PARAM_TOL:
        print(f"FAIL parity: looped/unlooped = {ratio:.4f}", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    runs = {
        str(seed): {
            kind: train_arm(
                kind, train, val, val_rows, ext, ext_rows,
                seed=seed, epochs=args.epochs, d=args.d, device=args.device,
                ckpt_dir=args.ckpt_dir, test_steps=args.test_steps,
            )
            for kind in args.arms
        }
        for seed in args.seeds
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
        "purpose": "positive control: looped vs unlooped message passing, trained short, tested long",
        "train_data": args.train_data.as_posix(),
        "extended_data": args.extended_data.as_posix(),
        "protocol": {
            "train_steps": TRAIN_STEPS,
            "test_steps": sorted(set([TRAIN_STEPS, *args.test_steps])),
            "epochs": args.epochs,
            "seeds": args.seeds,
            "batch_size": BATCH,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "aggregation": "max over in-neighbours",
            "widths": widths,
            "param_counts": counts,
            "looped_over_unlooped_params": ratio,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "torch_version": torch.__version__,
        },
        "baselines": {"id_val": 0.5, "extended": 0.5, "note": "paired sets; chance by construction"},
        "runs": runs,
        "aggregate": agg,
        "self_audit_mismatches": mismatches,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    print("\n=== mean over seeds (best checkpoint) ===", file=sys.stderr)
    for kind, entry in agg.items():
        e = entry["best"]
        ext = " ".join(f"T={s}:{v['mean']:.3f}" for s, v in e["extended_acc_by_steps"].items())
        print(f"{kind:9s} ID val {e['id_val_acc_mean']:.4f} | long paths {ext}", file=sys.stderr)
    print(json.dumps({"ok": not mismatches, "out": args.out.as_posix(),
                      "self_audit_mismatches": mismatches, "science_open": False}, sort_keys=True))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())

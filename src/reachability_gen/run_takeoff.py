# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Take-off study: how reliably does each looped arm start learning? (MEASURE)

On the crossed sets, where no single-endpoint cue exists, the stability ring
found that the message-passing arms either fit the validation split within
two epochs or stayed at 0.5 for all 30 (13 of 35 runs fitted it). This study
measures that take-off rate with enough seeds to compare arms and training
starts:

* arms (parameter-matched, 6 training steps): ``loop`` (standard step),
  ``geo`` (the recurrence under study), ``anchored`` (a synthesis of this line
  and the sister line: nodes the source has not reached keep an exactly zero
  state, the source is re-injected every step, the answer is read from the
  target's state and its norm, no node identities are embedded);
* starts: ``cold`` (crossed data from the first step), ``paired_warm`` (the
  first epoch on the paired set, whose one-endpoint cues give an easier
  gradient, then crossed data), ``curriculum`` (crossed graphs of hop 2 in
  the first epoch, one more hop per epoch).

Each run trains for 5 epochs (take-off happened within 2 or not at all) and
records the crossed validation accuracy per epoch; a run took off when that
accuracy reached 0.99. The final checkpoint is saved, re-scored and evaluated
on the long-path set at 16, 48 and 192 steps. Checkpoints are not versioned
(their SHA-256 hashes are recorded). Rates come with Wilson 95% intervals and
two-sided Fisher exact tests.

``science_open=false`` always.

Usage::

    python -m reachability_gen.run_takeoff --device cuda
    python -m reachability_gen.run_takeoff --device cuda --seeds 0 1 2 3 --out artifacts/takeoff_part0.json
    python -m reachability_gen.run_takeoff --merge artifacts/takeoff_part*.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.adr_invariants import PARAM_TOL
from reachability_gen.gen_crossed import CROSSED_EXTENDED_SPEC, CROSSED_ID_SPEC, verify_crossed
from reachability_gen.models.message_passing import AnchoredMP, MessagePassing, take
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_mp_calibration import BATCH, GRAD_CLIP, LR, WEIGHT_DECAY
from reachability_gen.run_stability import SetData, mp_width, target_params

DEFAULT_TRAIN = Path("data/id_crossed_20k.jsonl")
DEFAULT_EXTENDED = Path("data/extended_crossed_2k.jsonl")
DEFAULT_PAIRED = Path("data/id_disjoint_20k.jsonl")
DEFAULT_OUT = Path("artifacts/takeoff_study.json")
DEFAULT_CKPT_DIR = Path("artifacts/takeoff")

ARMS: tuple[str, ...] = ("loop", "geo", "anchored")
STARTS: dict[str, str] = {
    "cold": "crossed data from the first step",
    "paired_warm": "first epoch on the paired set, then crossed data",
    "curriculum": "crossed graphs of hop 2 in epoch 1, one more hop per epoch",
}
TRAIN_STEPS: int = 6
EPOCHS: int = 5
TAKEOFF_MIN: float = 0.99
FINAL_STEPS: tuple[int, ...] = (16, 48, 192)
EVAL_BATCH: int = 500


def anchored_width() -> int:
    target = target_params()
    return min(range(64, 256), key=lambda d: abs(AnchoredMP(d, TRAIN_STEPS).param_count() - target))


def build(kind: str):
    if kind == "loop":
        return MessagePassing(mp_width(), TRAIN_STEPS, looped=True, update="residual")
    if kind == "geo":
        return MessagePassing(mp_width(), TRAIN_STEPS, looped=True, update="geo")
    if kind == "anchored":
        return AnchoredMP(anchored_width(), TRAIN_STEPS)
    raise ValueError(f"unknown arm {kind!r}")


def init_model(kind: str, seed: int, device: str = "cpu"):
    import torch

    torch.manual_seed(seed)
    return build(kind).to(device)


def param_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def accuracy(model, data: SetData, steps: int) -> float:
    import torch

    packed = data.packed("mp")
    model.eval()
    hits = []
    with torch.no_grad():
        for start in range(0, len(data.rows), EVAL_BATCH):
            idx = torch.arange(start, min(start + EVAL_BATCH, len(data.rows)), device=data.device)
            batch = take(packed, idx)
            hits.append((model(batch, steps).argmax(dim=-1) == batch["y"]).float())
    return float(torch.cat(hits).mean().item())


def epoch_indices(start: str, epoch: int, crossed: SetData, paired: SetData):
    """The set and row indices one epoch trains on under ``start``."""
    import torch

    if start == "paired_warm" and epoch == 1:
        return paired, torch.arange(len(paired.rows))
    if start == "curriculum":
        cap = min(max(CROSSED_ID_SPEC.hops), 1 + epoch)
        return crossed, torch.tensor([i for i, h in enumerate(crossed.graph_hop) if h <= cap])
    return crossed, torch.arange(len(crossed.rows))


def run(kind: str, start: str, seed: int, *, crossed: SetData, paired: SetData, val: SetData,
        ext: SetData, device: str, ckpt_dir: Path, epochs: int = EPOCHS) -> dict[str, Any]:
    """Train one (arm, start, seed); record take-off and the final checkpoint's reach."""
    import torch
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_

    model = init_model(kind, seed, device)
    untrained = accuracy(model, val, TRAIN_STEPS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        data, rows = epoch_indices(start, epoch, crossed, paired)
        packed = data.packed("mp")
        order = rows[torch.randperm(len(rows))]
        model.train()
        hits = torch.zeros((), device=device)
        for i in range(0, len(order), BATCH):
            batch = take(packed, order[i : i + BATCH].to(device))
            logits = model(batch, TRAIN_STEPS)
            loss = F.cross_entropy(logits, batch["y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            hits += (logits.detach().argmax(dim=-1) == batch["y"]).sum()
        val_acc = accuracy(model, val, TRAIN_STEPS)
        history.append({"epoch": epoch, "rows": len(order), "train_acc": float(hits.item()) / len(order),
                        "val_acc": val_acc})
        print(f"[seed{seed}/{kind}/{start}] epoch {epoch}/{epochs}: train_acc={history[-1]['train_acc']:.4f} "
              f"val_acc={val_acc:.4f}", file=sys.stderr, flush=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"seed{seed}_{kind}_{start}.pt"
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save({"epoch": epochs, "state_dict": state, "arm": kind, "start": start, "seed": seed,
                "science_open": False}, path)
    fresh = build(kind).to(device)
    fresh.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
    final_val = accuracy(fresh, val, TRAIN_STEPS)
    takeoff = next((h["epoch"] for h in history if h["val_acc"] >= TAKEOFF_MIN), None)
    return {
        "arm": kind,
        "start": start,
        "seed": seed,
        "untrained_val_acc": untrained,
        "history": history,
        "took_off": takeoff is not None,
        "takeoff_epoch": takeoff,
        "train_seconds": time.perf_counter() - t0,
        "final": {
            "checkpoint_path": path.as_posix(),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "val_acc": final_val,
            "rescore_matches_record": final_val == history[-1]["val_acc"],
            "val_acc_at_192": accuracy(fresh, val, 192),
            "extended_acc_by_steps": {str(s): accuracy(fresh, ext, s) for s in FINAL_STEPS},
        },
        "science_open": False,
    }


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def fisher_exact(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p-value for k1/n1 vs k2/n2."""
    total_k, total_n = k1 + k2, n1 + n2

    def prob(a: int) -> float:
        return math.comb(n1, a) * math.comb(n2, total_k - a) / math.comb(total_n, total_k)

    lo, hi = max(0, total_k - n2), min(n1, total_k)
    observed = prob(k1)
    return min(1.0, sum(prob(a) for a in range(lo, hi + 1) if prob(a) <= observed * (1 + 1e-9)))


def summarise(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for kind in ARMS:
        for start in STARTS:
            rs = [r for r in runs if r["arm"] == kind and r["start"] == start]
            if not rs:
                continue
            k = sum(r["took_off"] for r in rs)
            flew = [r for r in rs if r["took_off"]]
            cells[f"{kind}/{start}"] = {
                "took_off": k,
                "of": len(rs),
                "rate": k / len(rs),
                "wilson95": wilson(k, len(rs)),
                "takeoff_epochs": sorted(r["takeoff_epoch"] for r in flew),
                "untrained_val_mean": sum(r["untrained_val_acc"] for r in rs) / len(rs),
                "kept_at_final": sum(r["final"]["val_acc"] >= TAKEOFF_MIN for r in flew),
                "long_path_mean_among_took_off": {
                    s: (sum(r["final"]["extended_acc_by_steps"][s] for r in flew) / len(flew)) if flew else None
                    for s in map(str, FINAL_STEPS)
                },
                "stable_to_192": sum(
                    all(r["final"]["extended_acc_by_steps"][s] >= TAKEOFF_MIN for s in map(str, FINAL_STEPS))
                    for r in flew
                ),
            }
    tests: dict[str, Any] = {}
    stable_tests: dict[str, Any] = {}

    def compare(a: str, b: str) -> None:
        if a in cells and b in cells:
            ca, cb = cells[a], cells[b]
            tests[f"{a} vs {b}"] = fisher_exact(ca["took_off"], ca["of"], cb["took_off"], cb["of"])
            stable_tests[f"{a} vs {b}"] = fisher_exact(ca["stable_to_192"], ca["of"], cb["stable_to_192"], cb["of"])

    for start in STARTS:
        compare(f"geo/{start}", f"loop/{start}")
        compare(f"anchored/{start}", f"loop/{start}")
        compare(f"anchored/{start}", f"geo/{start}")
    for kind in ARMS:
        for start in ("paired_warm", "curriculum"):
            compare(f"{kind}/{start}", f"{kind}/cold")
    return {
        "cells": cells,
        "fisher_two_sided": tests,
        "fisher_two_sided_holm": holm(tests),
        "stable_fisher_two_sided": stable_tests,
        "stable_fisher_two_sided_holm": holm(stable_tests),
        "comparisons": len(tests),
    }


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjustment of a family of p-values."""
    order = sorted(pvalues, key=pvalues.__getitem__)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def finish(artifact: dict[str, Any], runs: list[dict[str, Any]], out: Path) -> int:
    mismatches = [f"seed{r['seed']}/{r['arm']}/{r['start']}" for r in runs if not r["final"]["rescore_matches_record"]]
    artifact.update({"runs": runs, "summary": summarise(runs), "self_audit_mismatches": mismatches})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    summary = artifact["summary"]
    print("\n=== take-off (crossed val >= 0.99 within the epoch budget) ===", file=sys.stderr)
    for name, c in summary["cells"].items():
        lo, hi = c["wilson95"]
        lp = " ".join(f"T{s}={v:.3f}" if v is not None else f"T{s}=-" for s, v in c["long_path_mean_among_took_off"].items())
        print(f"{name:22s} {c['took_off']:>2d}/{c['of']:<2d} [{lo:.2f}, {hi:.2f}] epochs {c['takeoff_epochs']} "
              f"| stable to 192: {c['stable_to_192']} | long paths {lp}", file=sys.stderr)
    print("  Fisher two-sided p (Holm-adjusted) — take-off | stable to 192:", file=sys.stderr)
    for name, p in summary["fisher_two_sided"].items():
        print(f"  {p:.2e} ({summary['fisher_two_sided_holm'][name]:.2e}) | "
              f"{summary['stable_fisher_two_sided'][name]:.2e} ({summary['stable_fisher_two_sided_holm'][name]:.2e})  {name}",
              file=sys.stderr)
    print(json.dumps({"ok": not mismatches, "out": out.as_posix(), "self_audit_mismatches": mismatches,
                      "science_open": False}, sort_keys=True))
    return 1 if mismatches else 0


def merge(parts: Sequence[Path], out: Path) -> int:
    loaded = [json.loads(p.read_text(encoding="utf-8")) for p in parts]
    first = loaded[0]
    keys = ("train_data", "extended_data", "paired_data", "arms", "starts")
    for path, art in zip(parts, loaded):
        same_protocol = {k: v for k, v in art["protocol"].items() if k != "seeds"} == {
            k: v for k, v in first["protocol"].items() if k != "seeds"}
        if any(art[k] != first[k] for k in keys) or not same_protocol:
            print(f"FAIL: {path} was run with a different protocol", file=sys.stderr)
            return 1
    runs = [r for art in loaded for r in art["runs"]]
    ids = [(r["seed"], r["arm"], r["start"]) for r in runs]
    if len(ids) != len(set(ids)):
        print("FAIL: a run appears in more than one part", file=sys.stderr)
        return 1
    runs.sort(key=lambda r: (r["seed"], ARMS.index(r["arm"]), list(STARTS).index(r["start"])))
    artifact = {k: first[k] for k in ("science_open", "purpose", *keys)}
    artifact["protocol"] = dict(first["protocol"], seeds=sorted({r["seed"] for r in runs}))
    artifact["merged_from"] = [p.as_posix() for p in parts]
    artifact["elapsed_seconds_by_part"] = [art["elapsed_seconds"] for art in loaded]
    return finish(artifact, runs, out)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Take-off study for looped arms on the crossed sets (MEASURE).")
    p.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--extended-data", type=Path, default=DEFAULT_EXTENDED)
    p.add_argument("--paired-data", type=Path, default=DEFAULT_PAIRED)
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--starts", nargs="+", choices=list(STARTS), default=list(STARTS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--merge", type=Path, nargs="+", default=None)
    p.add_argument("--no-verify", action="store_true", help="skip dataset verification (tests only)")
    args = p.parse_args(argv)
    if args.merge:
        return merge(args.merge, args.out)
    try:
        import torch
    except ImportError:
        print("FAIL: torch required", file=sys.stderr)
        return 2
    if args.device == "cuda" and not torch.cuda.is_available():
        print("FAIL: --device cuda, but this torch build sees no CUDA device", file=sys.stderr)
        return 1
    for path in (args.train_data, args.extended_data, args.paired_data):
        if not path.exists():
            print(f"FAIL: missing {path}; see docs/USAGE.md", file=sys.stderr)
            return 1
    rows, ext_rows, paired_rows = (load_jsonl(p) for p in (args.train_data, args.extended_data, args.paired_data))
    if not args.no_verify:
        n_val = sum(1 for r in rows if r["split"] == "val")
        checks = [
            verify_crossed(rows, n_total=len(rows), n_val=n_val, spec=CROSSED_ID_SPEC),
            verify_crossed(ext_rows, n_total=len(ext_rows), n_val=len(ext_rows), spec=CROSSED_EXTENDED_SPEC),
        ]
        issues = [i for ok, found in checks for i in found]
        if issues:
            print(f"FAIL: dataset verify: {issues}", file=sys.stderr)
            return 1
    target = target_params()
    counts = {kind: param_count(build(kind)) for kind in args.arms}
    ratios = {kind: c / target for kind, c in counts.items()}
    off = {k: r for k, r in ratios.items() if abs(r - 1.0) > PARAM_TOL}
    if off:
        print(f"FAIL parity vs {target}: {off}", file=sys.stderr)
        return 1
    crossed = SetData([r for r in rows if r["split"] == "train"], args.device)
    val = SetData([r for r in rows if r["split"] == "val"], args.device)
    ext = SetData(ext_rows, args.device)
    paired = SetData([r for r in paired_rows if r["split"] == "train"], args.device)
    t0 = time.perf_counter()
    runs = [
        run(kind, start, seed, crossed=crossed, paired=paired, val=val, ext=ext,
            device=args.device, ckpt_dir=args.ckpt_dir, epochs=args.epochs)
        for seed in args.seeds for kind in args.arms for start in args.starts
    ]
    artifact: dict[str, Any] = {
        "science_open": False,
        "purpose": "take-off study: how reliably each looped arm starts learning on cue-free data",
        "train_data": args.train_data.as_posix(),
        "extended_data": args.extended_data.as_posix(),
        "paired_data": args.paired_data.as_posix(),
        "arms": list(args.arms),
        "starts": {k: STARTS[k] for k in args.starts},
        "protocol": {
            "train_steps": TRAIN_STEPS,
            "epochs": args.epochs,
            "takeoff_if_val_at_least": TAKEOFF_MIN,
            "final_steps": list(FINAL_STEPS),
            "seeds": args.seeds,
            "batch_size": BATCH,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "target_params": target,
            "param_counts": counts,
            "param_ratios": ratios,
            "widths": {"loop": mp_width(), "geo": mp_width(), "anchored": anchored_width()},
            "checkpoints_versioned": False,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "torch_version": torch.__version__,
        },
        "elapsed_seconds": time.perf_counter() - t0,
    }
    return finish(artifact, runs, args.out)


if __name__ == "__main__":
    raise SystemExit(main())

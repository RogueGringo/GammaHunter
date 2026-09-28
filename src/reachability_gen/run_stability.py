# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Stability ring: do looped arms keep their answers as the step count grows? (MEASURE)

The message-passing calibration found that a standard looped arm fits the
training distribution in every seed but reaches longer paths with more steps
only in some, and that longer training can lose that reach. This runner
compares, on the crossed sets and at matched parameter counts, update rules
and training protocols for exactly that property:

* ``loop``: the calibration's standard looped step, trained with 6 steps;
* ``loop_r10`` / ``loop_r18``: the same, trained with a random step count per
  batch, drawn from [6, 10] or [6, 18];
* ``geo``: the geometric step (a damped move into an RMS-bounded state);
* ``geo_r10`` / ``geo_r18``: geo trained with random step counts;
* ``geo_tau``: geo with one learned vector per trained step, the last reused later;
* ``sheaf``: the sister line's SheafInferCore with default initialisation;
* ``sheaf_sealed``: the same arm with its original hand-set initialisation.

The planned random range was [6, 18]. In a 3-epoch pilot (seed 0) neither
update started learning with it, while [6, 10] did, so both ranges are run.

Every arm is trained on ``id_crossed_20k`` for each seed with the same
optimiser, batch size and epoch budget, then evaluated at initialisation
(untrained control) and at its best and final checkpoints (both saved and
re-scored) on the validation split and on ``extended_crossed_2k`` at 6 to 192
steps, with the relative change of the query target's state per step. A seed
counts as stable when its long-path accuracy is at least 0.99 at every step
count from 16 to 192.

``science_open=false`` always.

Usage::

    python -m reachability_gen.run_stability --device cuda
    # or one process per seed, then merge:
    python -m reachability_gen.run_stability --device cuda --seeds 0 --out artifacts/stability_ring_seed0.json
    python -m reachability_gen.run_stability --merge artifacts/stability_ring_seed*.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.adr_invariants import PARAM_TOL
from reachability_gen.gen_crossed import CROSSED_EXTENDED_SPEC, CROSSED_ID_SPEC, verify_crossed
from reachability_gen.models.message_passing import (
    MessagePassing,
    collate,
    match_width,
    param_formula,
    parse_rows,
    take,
)
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_mp_calibration import BATCH, GRAD_CLIP, LR, WEIGHT_DECAY

DEFAULT_TRAIN = Path("data/id_crossed_20k.jsonl")
DEFAULT_EXTENDED = Path("data/extended_crossed_2k.jsonl")
DEFAULT_OUT = Path("artifacts/stability_ring.json")
DEFAULT_CKPT_DIR = Path("artifacts/stability_ring")

TRAIN_STEPS: int = 6
EVAL_BATCH: int = 500  # evaluation only; long unrolls dominate otherwise
TEST_STEPS: tuple[int, ...] = (6, 16, 32, 48, 96, 192)
STABLE_STEPS: tuple[int, ...] = (16, 32, 48, 96, 192)
STABLE_MIN: float = 0.99
ID_STEPS: tuple[int, ...] = (6, 16, 48, 192)
PROBE_ROWS: int = 400
PROBE_STEPS: tuple[int, ...] = (16, 48)
DRIFT_ROWS: int = 200
DRIFT_AT: tuple[int, ...] = (6, 16, 32, 64, 128, 192)
UNTRAINED_STEPS: tuple[int, ...] = (6, 16, 192)
REFERENCE_WIDTH: int = 64  # the calibration's unlooped width sets the parameter target
SHEAF_D: int = 64
SHEAF_MLP: int = 13  # ×13 puts the ported arm within ±5% of the target (×12: 0.92)
SHEAF_MAX_NODES: int = 64

FIXED: tuple[int, int] = (TRAIN_STEPS, TRAIN_STEPS)
ARMS: dict[str, dict[str, Any]] = {
    "loop": {"family": "mp", "update": "residual", "tau": False, "train_steps": FIXED},
    "loop_r10": {"family": "mp", "update": "residual", "tau": False, "train_steps": (6, 10)},
    "loop_r18": {"family": "mp", "update": "residual", "tau": False, "train_steps": (6, 18)},
    "geo": {"family": "mp", "update": "geo", "tau": False, "train_steps": FIXED},
    "geo_r10": {"family": "mp", "update": "geo", "tau": False, "train_steps": (6, 10)},
    "geo_r18": {"family": "mp", "update": "geo", "tau": False, "train_steps": (6, 18)},
    "geo_tau": {"family": "mp", "update": "geo", "tau": True, "train_steps": FIXED},
    "sheaf": {"family": "sheaf", "neutral": True, "train_steps": FIXED},
    "sheaf_sealed": {"family": "sheaf", "neutral": False, "train_steps": FIXED},
}


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _sd(xs: Sequence[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def target_params() -> int:
    """The calibration's parameter target: its unlooped arm at the reference width."""
    return param_formula(REFERENCE_WIDTH, TRAIN_STEPS)


def mp_width() -> int:
    return match_width(target_params(), TRAIN_STEPS, looped=True)


def build(kind: str):
    """The arm ``kind`` at its ring size (random weights from the current RNG state)."""
    arm = ARMS[kind]
    if arm["family"] == "mp":
        return MessagePassing(mp_width(), TRAIN_STEPS, looped=True, update=arm["update"], tau=arm["tau"])
    from reachability_gen.models.sheaf_adapter import neutral_init
    from reachability_gen.models.sheaf_infer_core import SheafInferCore

    model = SheafInferCore(
        d=SHEAF_D, T=TRAIN_STEPS, mlp_expansion=SHEAF_MLP, max_nodes=SHEAF_MAX_NODES,
        residual_alpha=1.0, use_tau=False, apply_cycle_rmsnorm=False, gate_detach_diffusion=True,
    )
    return neutral_init(model) if arm["neutral"] else model


def init_model(kind: str, seed: int, device: str = "cpu"):
    """The weights training starts from for ``seed`` (also the untrained control)."""
    import torch

    torch.manual_seed(seed)
    return build(kind).to(device)


def param_count(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


@dataclass
class SetData:
    """One set's rows, each graph's hop per row, and whole-set batches per family."""

    rows: list[dict[str, Any]]
    device: str
    graph_hop: list[int] = field(init=False)
    _packed: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        hop = {r["edge_hash"]: int(r["hop_distance"]) for r in self.rows if int(r["y"]) == 1}
        self.graph_hop = [hop.get(r["edge_hash"], -1) for r in self.rows]  # -1: positive not in rows

    def packed(self, family: str) -> dict[str, Any]:
        if family not in self._packed:
            if family == "mp":
                self._packed[family] = collate(parse_rows(self.rows), self.device)
            else:
                from reachability_gen.models.sheaf_adapter import pack_sheaf

                self._packed[family] = pack_sheaf(self.rows, self.device)
        return self._packed[family]


def _labels(family: str, batch: dict[str, Any]):
    return batch["y"] if family == "mp" else batch["labels"]


def logits_and_aux(kind: str, model, batch: dict[str, Any], steps: int, *, with_aux: bool = False):
    if ARMS[kind]["family"] == "mp":
        return model(batch, steps), None
    from reachability_gen.models.sheaf_adapter import sheaf_logits

    return sheaf_logits(model, batch, steps, with_aux=with_aux)


def evaluate(kind: str, model, data: SetData, steps: int, *, limit: Optional[int] = None) -> dict[str, Any]:
    """Accuracy overall and by each graph's hop, on the first ``limit`` rows (all by default)."""
    import torch

    family = ARMS[kind]["family"]
    packed = data.packed(family)
    n = len(data.rows) if limit is None else min(limit, len(data.rows))
    model.eval()
    hits: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, n, EVAL_BATCH):
            idx = torch.arange(start, min(start + EVAL_BATCH, n), device=data.device)
            batch = take(packed, idx)
            logits, _ = logits_and_aux(kind, model, batch, steps)
            hits.append((logits.argmax(dim=-1) == _labels(family, batch)).float())
    correct = torch.cat(hits).tolist()
    hops = data.graph_hop[:n]
    return {
        "acc": _mean(correct),
        "by_graph_hop": {
            str(k): _mean([c for c, g in zip(correct, hops) if g == k]) for k in sorted(set(hops))
        },
    }


def target_drift(kind: str, model, data: SetData, *, rows: Optional[int] = None, at: Optional[Sequence[int]] = None) -> dict[str, float]:
    """Mean relative change of the query target's state from step t-1 to t.

    ``|x_t - x_{t-1}| / max(|x_{t-1}|, |x_t|)``, in [0, 2]; 0 when the state is
    fixed (including a state that stays exactly zero).
    """
    import torch

    rows = DRIFT_ROWS if rows is None else rows
    at = DRIFT_AT if at is None else at
    family = ARMS[kind]["family"]
    batch = take(data.packed(family), torch.arange(min(rows, len(data.rows)), device=data.device))
    model.eval()
    with torch.no_grad():
        if family == "mp":
            ix = torch.arange(batch["t"].shape[0], device=data.device)
            states = [h[ix, batch["t"]] for h in model.iter_states(batch, max(at))]
        else:
            from reachability_gen.models.sheaf_adapter import sheaf_target_states

            states = sheaf_target_states(model, batch, max(at))
    out = {}
    for t in at:
        prev, cur = states[t - 1], states[t]
        scale = torch.maximum(prev.norm(dim=-1), cur.norm(dim=-1)).clamp(min=1e-8)
        rel = (cur - prev).norm(dim=-1) / scale
        out[str(t)] = float(rel.mean().item())
    return out


def untrained_control(kind: str, model, val: SetData, ext: SetData) -> dict[str, Any]:
    return {
        "id_val_acc": evaluate(kind, model, val, TRAIN_STEPS)["acc"],
        "extended_acc_by_steps": {str(s): evaluate(kind, model, ext, s)["acc"] for s in UNTRAINED_STEPS},
    }


def score_checkpoint(kind: str, model, val: SetData, ext: SetData) -> dict[str, Any]:
    ev = evaluate(kind, model, val, TRAIN_STEPS)
    extended = {str(s): evaluate(kind, model, ext, s) for s in TEST_STEPS}
    return {
        "id_val_acc": ev["acc"],
        "id_val_by_graph_hop": ev["by_graph_hop"],
        "id_val_acc_by_steps": {str(s): evaluate(kind, model, val, s)["acc"] for s in ID_STEPS},
        "extended_by_steps": extended,
        "stable": all(extended[str(s)]["acc"] >= STABLE_MIN for s in STABLE_STEPS),
        "target_drift": target_drift(kind, model, ext),
    }


def train_arm(
    kind: str,
    train: SetData,
    val: SetData,
    ext: SetData,
    *,
    seed: int,
    epochs: int,
    device: str,
    ckpt_dir: Path,
) -> dict[str, Any]:
    """Train one arm for one seed; control, save, re-score and evaluate both checkpoints."""
    import torch
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_

    arm = ARMS[kind]
    family = arm["family"]
    model = init_model(kind, seed, device)
    control = untrained_control(kind, model, val, ext)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    step_rng = random.Random(seed * 1_000_003 + 17)
    packed = train.packed(family)

    def cpu_state() -> dict[str, Any]:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    history: list[dict[str, Any]] = []
    best_acc, best_epoch, best_state = -1.0, 0, None
    t0 = time.perf_counter()
    n_train = len(train.rows)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(n_train)
        loss_sum = torch.zeros((), device=device)
        hits = torch.zeros((), device=device)
        for start in range(0, n_train, BATCH):
            idx = order[start : start + BATCH].to(device)
            steps = step_rng.randint(*arm["train_steps"])
            batch = take(packed, idx)
            logits, aux = logits_and_aux(kind, model, batch, steps, with_aux=family == "sheaf")
            y = _labels(family, batch)
            loss = F.cross_entropy(logits, y)
            if aux is not None:
                loss = loss + aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            loss_sum += loss.detach() * idx.numel()
            hits += (logits.detach().argmax(dim=-1) == y).sum()
        ev = evaluate(kind, model, val, TRAIN_STEPS)
        probe = {str(s): evaluate(kind, model, ext, s, limit=PROBE_ROWS)["acc"] for s in PROBE_STEPS}
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss_sum.item()) / n_train,
                "train_acc": float(hits.item()) / n_train,
                "val_acc": ev["acc"],
                "val_acc_by_graph_hop": ev["by_graph_hop"],
                "long_path_probe": probe,
            }
        )
        if ev["acc"] > best_acc:
            best_acc, best_epoch, best_state = ev["acc"], epoch, cpu_state()
        print(
            f"[seed{seed}/{kind}] epoch {epoch}/{epochs}: train_acc={history[-1]['train_acc']:.4f} "
            f"val_acc={ev['acc']:.4f} long16={probe['16']:.3f} long48={probe['48']:.3f} "
            f"(best {best_acc:.4f}@ep{best_epoch})",
            file=sys.stderr,
            flush=True,
        )
    assert best_state is not None
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "arm": kind,
        "seed": seed,
        "param_count": param_count(model),
        "train_seconds": time.perf_counter() - t0,
        "untrained": control,
        "history": history,
        "science_open": False,
    }
    for which, (epoch, state) in {"best": (best_epoch, best_state), "final": (epochs, cpu_state())}.items():
        path = ckpt_dir / f"seed{seed}_{kind}_{which}.pt"
        torch.save({"epoch": epoch, "state_dict": state, "arm": kind, "seed": seed, "science_open": False}, path)
        fresh = build(kind).to(device)
        fresh.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
        scored = score_checkpoint(kind, fresh, val, ext)
        out[which] = {
            "epoch": epoch,
            "checkpoint_path": path.as_posix(),
            "rescore_matches_record": scored["id_val_acc"] == history[epoch - 1]["val_acc"],
            **scored,
        }
    return out


def aggregate(runs: dict[str, dict[str, Any]], arms: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in arms:
        blocks = [per_seed[kind] for per_seed in runs.values() if kind in per_seed]
        if not blocks:
            continue
        entry: dict[str, Any] = {
            "seeds": [b["seed"] for b in blocks],
            "untrained_id_val_mean": _mean([b["untrained"]["id_val_acc"] for b in blocks]),
            "untrained_extended_mean": {
                s: _mean([b["untrained"]["extended_acc_by_steps"][s] for b in blocks])
                for s in blocks[0]["untrained"]["extended_acc_by_steps"]
            },
        }
        for which in ("best", "final"):
            ids = [b[which]["id_val_acc"] for b in blocks]
            entry[which] = {
                "id_val_acc_mean": _mean(ids),
                "id_val_acc_sd": _sd(ids),
                "id_val_acc_by_steps_mean": {
                    s: _mean([b[which]["id_val_acc_by_steps"][s] for b in blocks]) for s in map(str, ID_STEPS)
                },
                "extended_acc_by_steps": {
                    s: {
                        "mean": _mean([b[which]["extended_by_steps"][s]["acc"] for b in blocks]),
                        "sd": _sd([b[which]["extended_by_steps"][s]["acc"] for b in blocks]),
                        "min": min(b[which]["extended_by_steps"][s]["acc"] for b in blocks),
                    }
                    for s in map(str, TEST_STEPS)
                },
                "stable_seeds": sum(1 for b in blocks if b[which]["stable"]),
                "of_seeds": len(blocks),
                "target_drift_mean": {
                    t: _mean([b[which]["target_drift"][t] for b in blocks]) for t in map(str, DRIFT_AT)
                },
            }
        out[kind] = entry
    return out


def finish(artifact: dict[str, Any], runs: dict[str, dict[str, Any]], arms: Sequence[str], out: Path) -> int:
    """Aggregate, check the self-audit, write the result file and print a summary."""
    agg = aggregate(runs, arms)
    mismatches = [
        f"seed{s}/{k}/{w}"
        for s, per_seed in runs.items()
        for k, b in per_seed.items()
        for w in ("best", "final")
        if not b[w]["rescore_matches_record"]
    ]
    artifact.update({"runs": runs, "aggregate": agg, "self_audit_mismatches": mismatches})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print("\n=== stable seeds (long paths >= 0.99 at every step count 16-192) ===", file=sys.stderr)
    for kind, e in agg.items():
        f = e["final"]
        ext_means = " ".join(f"T={s}:{v['mean']:.3f}" for s, v in f["extended_acc_by_steps"].items())
        print(
            f"{kind:13s} untrained {e['untrained_id_val_mean']:.3f} | stable best {e['best']['stable_seeds']}/"
            f"{e['best']['of_seeds']} final {f['stable_seeds']}/{f['of_seeds']} | final ID {f['id_val_acc_mean']:.4f} | {ext_means}",
            file=sys.stderr,
        )
    print(json.dumps({"ok": not mismatches, "out": out.as_posix(),
                      "self_audit_mismatches": mismatches, "science_open": False}, sort_keys=True))
    return 1 if mismatches else 0


def merge(parts: Sequence[Path], out: Path) -> int:
    """Combine per-seed result files of one protocol into a single result file."""
    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in parts]
    first = loaded[0]
    same = ("train_data", "extended_data", "arms")
    for path, art in zip(parts, loaded):
        protocol = {k: v for k, v in art["protocol"].items() if k != "seeds"}
        if any(art[k] != first[k] for k in same) or protocol != {k: v for k, v in first["protocol"].items() if k != "seeds"}:
            print(f"FAIL: {path} was run with a different protocol", file=sys.stderr)
            return 1
    runs: dict[str, dict[str, Any]] = {}
    for path, art in zip(parts, loaded):
        overlap = set(runs) & set(art["runs"])
        if overlap:
            print(f"FAIL: seeds {sorted(overlap)} appear in more than one part", file=sys.stderr)
            return 1
        runs.update(art["runs"])
    runs = dict(sorted(runs.items(), key=lambda kv: int(kv[0])))
    artifact = {k: first[k] for k in ("science_open", "purpose", "train_data", "extended_data", "arms")}
    artifact["protocol"] = dict(first["protocol"], seeds=[int(s) for s in runs])
    artifact["merged_from"] = [path.as_posix() for path in parts]
    artifact["elapsed_seconds_by_part"] = [art["elapsed_seconds"] for art in loaded]
    return finish(artifact, runs, list(first["arms"]), out)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Stability ring for looped arms on the crossed sets (MEASURE).")
    p.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--extended-data", type=Path, default=DEFAULT_EXTENDED)
    p.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--merge", type=Path, nargs="+", default=None, help="combine per-seed result files into --out")
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
    for path in (args.train_data, args.extended_data):
        if not path.exists():
            print(f"FAIL: missing {path}; see docs/USAGE.md", file=sys.stderr)
            return 1
    rows = load_jsonl(args.train_data)
    ext_rows = load_jsonl(args.extended_data)
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

    train = SetData([r for r in rows if r["split"] == "train"], args.device)
    val = SetData([r for r in rows if r["split"] == "val"], args.device)
    ext = SetData(ext_rows, args.device)
    t0 = time.perf_counter()
    runs = {
        str(seed): {
            kind: train_arm(kind, train, val, ext, seed=seed, epochs=args.epochs,
                            device=args.device, ckpt_dir=args.ckpt_dir)
            for kind in args.arms
        }
        for seed in args.seeds
    }
    artifact: dict[str, Any] = {
        "science_open": False,
        "purpose": "stability ring: which looped arms keep their answers as the step count grows",
        "train_data": args.train_data.as_posix(),
        "extended_data": args.extended_data.as_posix(),
        "arms": {k: ARMS[k] for k in args.arms},
        "protocol": {
            "train_steps": TRAIN_STEPS,
            "test_steps": list(TEST_STEPS),
            "stable_if_at_least": STABLE_MIN,
            "stable_steps": list(STABLE_STEPS),
            "id_steps": list(ID_STEPS),
            "epochs": args.epochs,
            "seeds": args.seeds,
            "batch_size": BATCH,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "target_params": target,
            "param_counts": counts,
            "param_ratios": ratios,
            "mp_width": mp_width(),
            "sheaf": {"d": SHEAF_D, "mlp_expansion": SHEAF_MLP, "max_nodes": SHEAF_MAX_NODES},
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "torch_version": torch.__version__,
        },
        "elapsed_seconds": time.perf_counter() - t0,
    }
    return finish(artifact, runs, args.arms, args.out)


if __name__ == "__main__":
    raise SystemExit(main())

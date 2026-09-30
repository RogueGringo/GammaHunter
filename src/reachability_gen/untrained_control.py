# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Untrained-model control: score each arm at the weights its training starts from.

An accuracy an arm already reaches before any training step cannot be
attributed to learning. For every seed and arm of the message-passing
calibration, this rebuilds the exact initial weights (the runner's own
``init_model``) and evaluates them with the runner's own evaluation, on the
same sets and step counts. When the calibration's result file is present, it
also reports the learned margin (trained minus untrained accuracy) for both
saved checkpoints.

``--anchored`` runs the controls for the anchored arm of the take-off study on
the crossed sets. The arm keeps a node's state exactly zero until a path from the
source reaches it, for any weights, so its search may not be learned at all. Two
controls, per seed at the take-off width:

* a frozen random core read by a fixed rule, no training: reachable iff the
  target's state is non-zero after the given number of steps (4, 5 and 6 on the
  validation split, 16, 48 and 192 on the long-path set);
* the same frozen core with only the readout head trained (take-off protocol:
  crossed training split, cold start), checked to leave the core unchanged.

The zero test is read element by element; the float32 norm of a state can
underflow to 0 while its elements are not, and both readings are recorded.
The same measurements are taken on the take-off study's trained anchored
checkpoints (cold start, verified against their recorded SHA-256), to compare
the size of reached states after training with a random core's.

``science_open=false`` always.

Usage::

    python -m reachability_gen.untrained_control --device cuda
    python -m reachability_gen.untrained_control --anchored --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.models.message_passing import ParsedGraph, parse_rows
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_mp_calibration import (
    ARMS,
    DEFAULT_EXTENDED,
    DEFAULT_OUT as DEFAULT_CALIBRATION,
    DEFAULT_TRAIN,
    TEST_STEPS,
    TRAIN_STEPS,
    _mean,
    evaluate,
    init_model,
)

DEFAULT_OUT = Path("artifacts/mp_calibration_untrained_control.json")
ANCHORED_OUT = Path("artifacts/anchored_untrained_control.json")
CROSSED_TRAIN = Path("data/id_crossed_20k.jsonl")
CROSSED_EXTENDED = Path("data/extended_crossed_2k.jsonl")
ZERO_TEST_VAL_STEPS: tuple[int, ...] = (4, 5, 6)
ZERO_TEST_LONG_STEPS: tuple[int, ...] = (16, 48, 192)
ANCHORED_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


def zero_test(model, data, steps: int) -> dict[str, Any]:
    """The fixed zero test after ``steps``, overall and by graph hop, and how far reached states are from zero.

    ``acc`` reads "reachable iff any element of the target's state is non-zero";
    ``norm_rule_acc`` reads "reachable iff the float32 norm of that state is
    non-zero", which differs where squaring underflows. For reachable targets,
    by graph hop: the largest element magnitude and the norm (median, minimum,
    and how many norms are exactly 0), with no target left out.
    """
    import torch

    from reachability_gen.models.message_passing import take
    from reachability_gen.run_takeoff import EVAL_BATCH

    packed = data.packed("mp")
    model.eval()
    elem_hits, norm_hits, norms, peaks = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(data.rows), EVAL_BATCH):
            idx = torch.arange(start, min(start + EVAL_BATCH, len(data.rows)), device=data.device)
            batch = take(packed, idx)
            for h in model.iter_states(batch, steps):
                pass
            target = h[torch.arange(h.shape[0], device=h.device), batch["t"]]
            norm = target.norm(dim=-1)
            elem_hits.append(((target != 0).any(dim=-1).long() == batch["y"]).long())
            norm_hits.append(((norm > 0).long() == batch["y"]).long())
            norms.append(norm)
            peaks.append(target.abs().amax(dim=-1))
    correct = torch.cat(elem_hits).tolist()
    by_norm = torch.cat(norm_hits).tolist()
    groups: dict[int, list[int]] = {}
    reached: dict[int, list[tuple[float, float]]] = {}
    for c, hop, nm, pk, row in zip(correct, data.graph_hop, torch.cat(norms).tolist(), torch.cat(peaks).tolist(),
                                   data.rows):
        groups.setdefault(hop, []).append(c)
        if int(row["y"]) == 1:
            reached.setdefault(hop, []).append((nm, pk))
    return {
        "acc": sum(correct) / len(correct),
        "norm_rule_acc": sum(by_norm) / len(by_norm),
        "by_graph_hop": {str(h): sum(v) / len(v) for h, v in sorted(groups.items())},
        "reachable_target_state_by_graph_hop": {
            str(h): {"norm_median": statistics.median(n for n, _ in v), "norm_min": min(n for n, _ in v),
                     "norms_exactly_zero": sum(1 for n, _ in v if n == 0),
                     "max_abs_element_median": statistics.median(pk for _, pk in v),
                     "max_abs_element_min": min(pk for _, pk in v), "targets": len(v)}
            for h, v in sorted(reached.items())},
    }


def trained_takeoff_states(seeds, val, ext, *, device: str, study: Path) -> dict[str, Any]:
    """The zero test and reached-state sizes of the take-off study's trained anchored checkpoints (cold start)."""
    import torch

    from reachability_gen.run_takeoff import build

    runs = {r["seed"]: r for r in json.loads(study.read_text(encoding="utf-8"))["runs"]
            if r["arm"] == "anchored" and r["start"] == "cold"}
    out: dict[str, Any] = {}
    for seed in seeds:
        rec = runs[seed]["final"]
        path = Path(rec["checkpoint_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != rec["checkpoint_sha256"]:
            raise RuntimeError(f"{path} does not match the SHA-256 recorded in {study}")
        model = build("anchored").to(device)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
        out[str(seed)] = {
            "checkpoint_sha256": rec["checkpoint_sha256"],
            "val_by_steps": {str(s): zero_test(model, val, s) for s in ZERO_TEST_VAL_STEPS},
            "long_by_steps": {str(s): zero_test(model, ext, s) for s in ZERO_TEST_LONG_STEPS},
        }
    return out


def core_digest(model) -> str:
    """SHA-256 of every parameter outside the readout head."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not name.startswith("head."):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def anchored_control(seed: int, crossed, val, ext, *, device: str, epochs: int) -> dict[str, Any]:
    """Both anchored controls for one seed: the fixed zero test, then a trained head on the frozen core."""
    import torch
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_

    from reachability_gen.models.message_passing import take
    from reachability_gen.run_mp_calibration import BATCH, GRAD_CLIP, LR, WEIGHT_DECAY
    from reachability_gen.run_takeoff import FINAL_STEPS, accuracy
    from reachability_gen.run_takeoff import TRAIN_STEPS as TAKEOFF_STEPS
    from reachability_gen.run_takeoff import init_model as init

    model = init("anchored", seed, device)
    out: dict[str, Any] = {
        "seed": seed,
        "width": model.d,
        "untrained_head_val_acc": accuracy(model, val, TAKEOFF_STEPS),
        "zero_test": {
            "val_by_steps": {str(s): zero_test(model, val, s) for s in ZERO_TEST_VAL_STEPS},
            "long_by_steps": {str(s): zero_test(model, ext, s) for s in ZERO_TEST_LONG_STEPS},
        },
    }
    before = core_digest(model)
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith("head."))
    opt = torch.optim.AdamW(model.head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    packed = crossed.packed("mp")
    history = []
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(crossed.rows))
        model.train()
        for i in range(0, len(order), BATCH):
            batch = take(packed, order[i : i + BATCH].to(device))
            loss = F.cross_entropy(model(batch, TAKEOFF_STEPS), batch["y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.head.parameters(), GRAD_CLIP)
            opt.step()
        history.append({"epoch": epoch, "val_acc": accuracy(model, val, TAKEOFF_STEPS)})
    out["trained_head"] = {
        "epochs": epochs,
        "history": history,
        "core_unchanged": core_digest(model) == before,
        "val_acc": accuracy(model, val, TAKEOFF_STEPS),
        "val_acc_at_192": accuracy(model, val, 192),
        "long_acc_by_steps": {str(s): accuracy(model, ext, s) for s in FINAL_STEPS},
    }
    return out


def anchored_main(args) -> int:
    import torch

    from reachability_gen.run_stability import SetData
    from reachability_gen.run_takeoff import EPOCHS

    for path in (args.crossed_data, args.crossed_extended_data):
        if not path.exists():
            print(f"FAIL: missing {path}; see docs/USAGE.md", file=sys.stderr)
            return 1
    rows = load_jsonl(args.crossed_data)
    crossed = SetData([r for r in rows if r["split"] == "train"], args.device)
    val = SetData([r for r in rows if r["split"] == "val"], args.device)
    ext = SetData(load_jsonl(args.crossed_extended_data), args.device)
    epochs = EPOCHS if args.head_epochs is None else args.head_epochs
    trained = None
    if args.takeoff_study.exists():
        trained = trained_takeoff_states(args.anchored_seeds, val, ext, device=args.device, study=args.takeoff_study)
    runs = []
    for seed in args.anchored_seeds:
        torch.manual_seed(seed)
        run = anchored_control(seed, crossed, val, ext, device=args.device, epochs=epochs)
        runs.append(run)
        zt, th = run["zero_test"], run["trained_head"]
        print(f"[anchored seed {seed}] zero test val "
              + " ".join(f"T={k}:{v['acc']:.4f}" for k, v in zt["val_by_steps"].items())
              + " | long " + " ".join(f"T={k}:{v['acc']:.4f}" for k, v in zt["long_by_steps"].items())
              + f" | trained head val {th['val_acc']:.4f}, long "
              + " ".join(f"T={k}:{v:.4f}" for k, v in th["long_acc_by_steps"].items())
              + f", core unchanged {th['core_unchanged']}", file=sys.stderr, flush=True)
    artifact = {
        "science_open": False,
        "purpose": "whether the anchored arm's search is learned: a frozen random core read by a fixed zero test, "
                   "and the frozen core with only its readout head trained",
        "rule": "reachable iff the target's state is non-zero after the given number of steps",
        "train_data": args.crossed_data.as_posix(),
        "extended_data": args.crossed_extended_data.as_posix(),
        "seeds": list(args.anchored_seeds),
        "head_epochs": epochs,
        "device": args.device,
        "torch_version": torch.__version__,
        "runs": runs,
        "trained_takeoff_checkpoints": trained,
        "takeoff_study": args.takeoff_study.as_posix() if trained is not None else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "out": args.out.as_posix(), "science_open": False}, sort_keys=True))
    return 0


def control_arm(
    kind: str,
    val: Sequence[ParsedGraph],
    val_rows: Sequence[dict[str, Any]],
    extended: Sequence[ParsedGraph],
    extended_rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    d: int,
    device: str,
    test_steps: Sequence[int] = TEST_STEPS,
) -> dict[str, Any]:
    """Evaluate one arm at its initial weights for ``seed``."""
    model = init_model(kind, d, seed, device)
    steps_list = [TRAIN_STEPS] if kind == "unlooped" else sorted({TRAIN_STEPS, *test_steps})
    ev = evaluate(model, val, val_rows, device)
    return {
        "arm": kind,
        "seed": seed,
        "id_val_acc": ev["acc"],
        "id_val_by_graph_hop": ev["by_graph_hop"],
        "extended_by_steps": {
            str(s): evaluate(model, extended, extended_rows, device, s) for s in steps_list
        },
    }


def learned_margin(control: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    """Trained minus untrained accuracy per seed, arm and saved checkpoint."""
    out: dict[str, Any] = {}
    for seed, per_seed in control.items():
        for kind, c in per_seed.items():
            trained = calibration.get("runs", {}).get(seed, {}).get(kind)
            if trained is None:
                continue
            for which in ("best", "final"):
                t = trained[which]
                out[f"seed{seed}/{kind}/{which}"] = {
                    "id_val": t["id_val_acc"] - c["id_val_acc"],
                    "extended_by_steps": {
                        s: t["extended_by_steps"][s]["acc"] - c["extended_by_steps"][s]["acc"]
                        for s in c["extended_by_steps"]
                        if s in t["extended_by_steps"]
                    },
                }
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Untrained-model control for the message-passing calibration (MEASURE)."
    )
    p.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--extended-data", type=Path, default=DEFAULT_EXTENDED)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--d", type=int, default=64, help="unlooped width; looped is width-matched")
    p.add_argument("--test-steps", type=int, nargs="+", default=list(TEST_STEPS))
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--anchored", action="store_true",
                   help="controls for the take-off study's anchored arm on the crossed sets")
    p.add_argument("--crossed-data", type=Path, default=CROSSED_TRAIN)
    p.add_argument("--crossed-extended-data", type=Path, default=CROSSED_EXTENDED)
    p.add_argument("--anchored-seeds", type=int, nargs="+", default=list(ANCHORED_SEEDS))
    p.add_argument("--head-epochs", type=int, default=None, help="default: the take-off study's epochs")
    p.add_argument("--takeoff-study", type=Path, default=Path("artifacts/takeoff_study.json"),
                   help="its anchored cold-start checkpoints are measured too, when present")
    args = p.parse_args(argv)
    args.out = args.out or (ANCHORED_OUT if args.anchored else DEFAULT_OUT)
    try:
        import torch
    except ImportError:
        print("FAIL: torch required", file=sys.stderr)
        return 2
    if args.device == "cuda" and not torch.cuda.is_available():
        print("FAIL: --device cuda, but this torch build sees no CUDA device", file=sys.stderr)
        return 1
    if args.anchored:
        return anchored_main(args)
    for path in (args.train_data, args.extended_data):
        if not path.exists():
            print(f"FAIL: missing {path}; see docs/USAGE.md", file=sys.stderr)
            return 1
    val_rows = [r for r in load_jsonl(args.train_data) if r["split"] == "val"]
    ext_rows = load_jsonl(args.extended_data)
    val, ext = parse_rows(val_rows), parse_rows(ext_rows)

    control = {
        str(seed): {
            kind: control_arm(
                kind, val, val_rows, ext, ext_rows,
                seed=seed, d=args.d, device=args.device, test_steps=args.test_steps,
            )
            for kind in ARMS
        }
        for seed in args.seeds
    }
    first = str(args.seeds[0])
    aggregate = {
        kind: {
            "id_val_acc_mean": _mean([control[s][kind]["id_val_acc"] for s in control]),
            "extended_acc_by_steps_mean": {
                st: _mean([control[s][kind]["extended_by_steps"][st]["acc"] for s in control])
                for st in control[first][kind]["extended_by_steps"]
            },
        }
        for kind in ARMS
    }
    artifact: dict[str, Any] = {
        "science_open": False,
        "purpose": "untrained-model control: accuracy at the weights training starts from",
        "train_data": args.train_data.as_posix(),
        "extended_data": args.extended_data.as_posix(),
        "seeds": args.seeds,
        "d": args.d,
        "device": args.device,
        "torch_version": torch.__version__,
        "control": control,
        "aggregate": aggregate,
    }
    if args.calibration.exists():
        artifact["calibration"] = args.calibration.as_posix()
        artifact["learned_margin"] = learned_margin(
            control, json.loads(args.calibration.read_text(encoding="utf-8"))
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    for kind, a in aggregate.items():
        ext = " ".join(f"T={s}:{v:.3f}" for s, v in a["extended_acc_by_steps_mean"].items())
        print(f"{kind:9s} untrained ID val {a['id_val_acc_mean']:.4f} | long paths {ext}", file=sys.stderr)
    print(json.dumps({"ok": True, "out": args.out.as_posix(), "science_open": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Reader → anchored solver: can the graph come from the text? (MEASURE)

The anchored solver (``AnchoredMP``) learns reachability reliably when it is
given each graph. Here a question-blind, identity-free reader
(``models.reader``) must supply that graph from the edge-list tokens as an
explicit 0/1 adjacency. Three regimes:

* ``supervised``: the reader is trained on the true edges; the solver, trained
  first on true graphs, stays frozen (pipeline check);
* ``answers_frozen``: the reader learns from the reachability answers only,
  through the hard adjacency, with the same frozen solver;
* ``answers_joint``: reader and solver both start untrained and learn from the
  answers only.

Scoring, in three layers per run: the reader (edge precision/recall/F1,
exact-graph rate, reversed-direction errors, agreement of the reachability
closure with the true graph's over all node pairs), the pipeline (accuracy and
AUROC on the crossed validation split at 6 steps and on the long-path set at
16, 48 and 192 steps), and attribution of every wrong answer to the reader
(its graph's closure disagrees with the true one on that pair) or the solver.
Controls: the untrained pipeline, and the solver on the true graph (ceiling).
A gradient probe (measurement only) records, before training and after every
epoch, where the answers loss pushes the reader's graph on a fixed batch of
training rows: on each question's own endpoint pair, on the true edges, or
elsewhere, and whether it pushes towards adding or removing edges.

Pass criteria, fixed in advance: exact-graph rate ≥ 0.99 on the validation
graphs, closure agreement ≥ 0.99 on the long-path graphs, and long-path
accuracy ≥ 0.99 at 16, 48 and 192 steps.

``science_open=false`` always.

Usage::

    python -m reachability_gen.run_reader --device cuda
    # or one process per regime (--regimes ... --out PART.json), then
    python -m reachability_gen.run_reader --merge PART.json [PART.json ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.adr_invariants import PARAM_TOL
from reachability_gen.gen_crossed import CROSSED_EXTENDED_SPEC, CROSSED_ID_SPEC, verify_crossed
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_mp_calibration import BATCH, GRAD_CLIP, LR, WEIGHT_DECAY
from reachability_gen.run_takeoff import wilson

DEFAULT_TRAIN = Path("data/id_crossed_20k.jsonl")
DEFAULT_EXTENDED = Path("data/extended_crossed_2k.jsonl")
DEFAULT_OUT = Path("artifacts/reader_pipeline.json")
DEFAULT_CKPT_DIR = Path("artifacts/reader_pipeline")
REGIMES: dict[str, str] = {
    "supervised": "reader trained on the true edges; solver trained on true graphs first, then frozen",
    "answers_frozen": "reader trained on reachability answers only; solver trained on true graphs first, then frozen",
    "answers_joint": "reader and solver both untrained at the start, trained on reachability answers only",
}
TRAIN_STEPS: int = 6
SOLVER_EPOCHS: int = 2
READER_EPOCHS: int = 5
THROUGH: str = "logit"  # straight-through carrier for the answers-only regimes (pilot: "probability" saturated)
READER_LR: float = 1e-2  # the reader's own rate (solver: shared 1e-3); 1e-3 had not converged in a 4-epoch pilot
LONG_STEPS: tuple[int, ...] = (16, 48, 192)
EVAL_BATCH: int = 250
PASS: dict[str, float] = {"exact_graphs": 0.99, "closure_agreement_long": 0.99, "long_path_accuracy": 0.99}
PROBE_ROWS: int = 256  # fixed training rows on which the gradient probe is taken (measurement only)


def anchored_width() -> int:
    from reachability_gen.run_takeoff import anchored_width as width

    return width()


class Data:
    """Rows plus both views of each graph: the solver batch and the reader tokens."""

    def __init__(self, rows: Sequence[dict[str, Any]], device: str) -> None:
        from reachability_gen.models.message_passing import collate, parse_rows
        from reachability_gen.models.reader import edge_list_tokens

        self.rows = list(rows)
        self.device = device
        self.graphs = parse_rows(self.rows)
        self.tokens = [edge_list_tokens(r["encoding"]) for r in self.rows]
        self._collate = collate
        hop = {r["edge_hash"]: int(r["hop_distance"]) for r in self.rows if int(r["y"]) == 1}
        self.graph_hop = [hop.get(r["edge_hash"], -1) for r in self.rows]

    def batch(self, idx: Sequence[int]) -> tuple[dict[str, Any], dict[str, Any]]:
        from reachability_gen.models.reader import collate_tokens

        return (self._collate([self.graphs[i] for i in idx], self.device),
                collate_tokens([self.tokens[i] for i in idx], self.device))


def build_solver():
    from reachability_gen.models.message_passing import AnchoredMP

    return AnchoredMP(anchored_width(), TRAIN_STEPS)


def build_reader():
    from reachability_gen.models.reader import GraphReader

    return GraphReader()


def train_solver(solver, data: Data, *, epochs: int, device: str) -> None:
    """Train the solver on the true graphs (regimes with a frozen solver)."""
    import torch
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_

    opt = torch.optim.AdamW(solver.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for _ in range(epochs):
        solver.train()
        order = torch.randperm(len(data.rows)).tolist()
        for i in range(0, len(order), BATCH):
            graph, _ = data.batch(order[i : i + BATCH])
            loss = F.cross_entropy(solver(graph, TRAIN_STEPS), graph["y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(solver.parameters(), GRAD_CLIP)
            opt.step()
    for p in solver.parameters():
        p.requires_grad_(False)
    solver.eval()


def topology_bce(z, graph: dict[str, Any]):
    """Sum and count of the per-pair BCE of P(edge) = 1 - exp(-z) against the true edges.

    Over ordered pairs of distinct nodes of each graph; written in z, so there is
    no practical dead zone as P(edge) approaches 0 or 1.
    """
    import torch

    mask = graph["node_mask"]
    valid = (mask[:, :, None] & mask[:, None, :]).float()
    valid = valid * (1 - torch.eye(z.shape[1], device=z.device))[None]
    gold = graph["adj"].float()
    per_pair = gold * -torch.log(-torch.expm1(-z.clamp(min=1e-30))) + (1 - gold) * z
    return (per_pair * valid).sum(), valid.sum()


def pipeline_logits(reader, solver, graph: dict[str, Any], tokens: dict[str, Any], steps: int, *,
                    oracle: bool = False, through: str = "probability"):
    """Solver logits on the reader's adjacency (or on the true one when ``oracle``)."""
    if oracle:
        return solver(graph, steps), graph["adj"].float()
    adj = reader(tokens, hard=True, through=through)
    return solver(dict(graph, adj=adj), steps), adj


def gradient_probe(reader, solver, data: Data, idx: Sequence[int], *, through: str) -> dict[str, float]:
    """Where the answers loss pushes the reader's graph (measurement only; nothing is updated).

    Shares of |dL/dA| (the answers loss's gradient on the hard adjacency) that fall
    on each question's own (source, target) pair, on the other true edges and
    elsewhere, and the net push towards adding edges, -sum(dL/dA) / sum|dL/dA|.
    """
    import torch
    import torch.nn.functional as F

    graph, tokens = data.batch(idx)
    with torch.enable_grad():
        adj = reader(tokens, hard=True, through=through)
        loss = F.cross_entropy(solver(dict(graph, adj=adj), TRAIN_STEPS), graph["y"])
        (grad,) = torch.autograd.grad(loss, adj)
    mass = grad.abs()
    total = mass.sum().item()
    if not total:
        return {"share_query_pair": 0.0, "share_true_edges": 0.0, "share_other": 0.0, "net_push_to_add": 0.0}
    query = torch.zeros_like(mass, dtype=torch.bool)
    query[torch.arange(len(idx), device=mass.device), graph["s"], graph["t"]] = True
    q = mass[query].sum().item() / total
    t = mass[(graph["adj"] > 0.5) & ~query].sum().item() / total
    return {"share_query_pair": q, "share_true_edges": t, "share_other": 1.0 - q - t,
            "net_push_to_add": -grad.sum().item() / total}


def evaluate(reader, solver, data: Data, steps: int, *, oracle: bool = False) -> dict[str, Any]:
    """Pipeline accuracy/AUROC by graph hop, reader metrics and error attribution."""
    import torch

    from reachability_gen.models.reader import closure, edge_metrics
    from reachability_gen.run_llm_reference import auroc

    reader.eval()
    solver.eval()
    margins: list[float] = []
    correct: list[int] = []
    reader_caused = solver_caused = 0
    pair_agree = pair_total = query_agree = 0
    bce_sum = bce_pairs = 0.0
    totals: dict[str, int] = {"edges_true": 0, "tp": 0, "fp": 0, "fn": 0, "reversed_errors": 0,
                              "exact_graphs": 0, "graphs": 0}
    with torch.no_grad():
        for start in range(0, len(data.rows), EVAL_BATCH):
            idx = list(range(start, min(start + EVAL_BATCH, len(data.rows))))
            graph, tokens = data.batch(idx)
            logits, adj = pipeline_logits(reader, solver, graph, tokens, steps, oracle=oracle)
            if not oracle:
                s, c = topology_bce(reader.edge_evidence(tokens), graph)
                bce_sum, bce_pairs = bce_sum + s.item(), bce_pairs + c.item()
            margin = (logits[:, 1] - logits[:, 0]).float()
            pred = margin > 0
            y = graph["y"].bool()
            margins += margin.tolist()
            correct += (pred == y).long().tolist()
            mask = graph["node_mask"]
            m = edge_metrics(adj, graph["adj"].float(), mask)
            for k in totals:
                totals[k] += m[k]
            true_reach, read_reach = closure(graph["adj"].float(), mask), closure(adj, mask)
            valid = mask[:, :, None] & mask[:, None, :]
            pair_agree += ((true_reach == read_reach) & valid).sum().item()
            pair_total += valid.sum().item()
            rows = torch.arange(len(idx), device=adj.device)
            q_agree = true_reach[rows, graph["s"], graph["t"]] == read_reach[rows, graph["s"], graph["t"]]
            query_agree += q_agree.sum().item()
            wrong = pred != y
            reader_caused += (wrong & ~q_agree).sum().item()
            solver_caused += (wrong & q_agree).sum().item()
    labels = [int(r["y"]) for r in data.rows]
    k, n = sum(correct), len(correct)
    by_hop: dict[str, list[int]] = {}
    for c, h in zip(correct, data.graph_hop):
        by_hop.setdefault(str(h), []).append(c)
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "steps": steps,
        "accuracy": k / n,
        "accuracy_wilson95": wilson(k, n),
        "auroc": auroc(margins, labels),
        "accuracy_by_graph_hop": {h: sum(v) / len(v) for h, v in sorted(by_hop.items(), key=lambda kv: int(kv[0]))},
        "reader": {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "reversed_errors": totals["reversed_errors"],
            "topology_bce": bce_sum / bce_pairs if bce_pairs else None,  # None for the true graph (oracle)
            "edges_true_mean": totals["edges_true"] / totals["graphs"],  # per question row (graphs repeat)
            "edges_predicted_mean": (tp + fp) / totals["graphs"],
            "exact_graphs": totals["exact_graphs"] / totals["graphs"],
            "closure_agreement_all_pairs": pair_agree / pair_total,
            "closure_agreement_queried_pairs": query_agree / n,
        },
        "errors": {"reader_caused": reader_caused, "solver_caused": solver_caused},
    }


def run(regime: str, seed: int, train: Data, val: Data, ext: Data, *, device: str, ckpt_dir: Path,
        reader_epochs: int = READER_EPOCHS, solver_epochs: int = SOLVER_EPOCHS,
        reader_lr: float = READER_LR, through: str = THROUGH) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_

    torch.manual_seed(seed)
    reader = build_reader().to(device)
    solver = build_solver().to(device)
    out: dict[str, Any] = {"regime": regime, "seed": seed, "science_open": False}
    t0 = time.perf_counter()
    if regime != "answers_joint":
        train_solver(solver, train, epochs=solver_epochs, device=device)
    probe_idx = list(range(min(PROBE_ROWS, len(train.rows))))
    out["untrained_reader"] = {
        "val": evaluate(reader, solver, val, TRAIN_STEPS),
        "long_16": evaluate(reader, solver, ext, 16),
        "gradient_probe": gradient_probe(reader, solver, train, probe_idx, through=through),
    }
    params = list(reader.parameters()) + ([] if regime != "answers_joint" else list(solver.parameters()))
    groups = [{"params": list(reader.parameters()), "lr": reader_lr}]
    if regime == "answers_joint":
        groups.append({"params": list(solver.parameters()), "lr": LR})
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    history = []
    for epoch in range(1, reader_epochs + 1):
        reader.train()
        solver.train(regime == "answers_joint")
        order = torch.randperm(len(train.rows)).tolist()
        loss_sum, hits, seen = 0.0, 0, 0
        for i in range(0, len(order), BATCH):
            idx = order[i : i + BATCH]
            graph, tokens = train.batch(idx)
            if regime == "supervised":
                bce, pairs = topology_bce(reader.edge_evidence(tokens), graph)
                loss = bce / pairs
                with torch.no_grad():
                    logits, _ = pipeline_logits(reader, solver, graph, tokens, TRAIN_STEPS)
            else:
                logits, _ = pipeline_logits(reader, solver, graph, tokens, TRAIN_STEPS, through=through)
                loss = F.cross_entropy(logits, graph["y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(params, GRAD_CLIP)
            opt.step()
            loss_sum += float(loss.item()) * len(idx)
            hits += int((logits.argmax(dim=-1) == graph["y"]).sum().item())
            seen += len(idx)
        ev = evaluate(reader, solver, val, TRAIN_STEPS)
        probe = gradient_probe(reader, solver, train, probe_idx, through=through)
        history.append({"epoch": epoch, "train_loss": loss_sum / seen, "train_pipeline_acc": hits / seen,
                        "val_accuracy": ev["accuracy"], "val_auroc": ev["auroc"],
                        "val_exact_graphs": ev["reader"]["exact_graphs"], "val_edge_f1": ev["reader"]["f1"],
                        "val_topology_bce": ev["reader"]["topology_bce"],
                        "val_edges_predicted_mean": ev["reader"]["edges_predicted_mean"],
                        "gradient_probe": probe})
        print(f"[{regime}/seed{seed}] epoch {epoch}/{reader_epochs}: loss {loss_sum / seen:.4f} "
              f"val acc {ev['accuracy']:.4f} auroc {ev['auroc']:.4f} | exact graphs {ev['reader']['exact_graphs']:.4f} "
              f"edge F1 {ev['reader']['f1']:.4f} topology BCE {ev['reader']['topology_bce']:.4f} "
              f"edges {ev['reader']['edges_predicted_mean']:.1f}/{ev['reader']['edges_true_mean']:.1f} "
              f"| gradient on query pair {probe['share_query_pair']:.2f}, true edges {probe['share_true_edges']:.2f}",
              file=sys.stderr, flush=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{regime}_seed{seed}.pt"
    torch.save({"reader": reader.state_dict(), "solver": solver.state_dict(), "regime": regime, "seed": seed,
                "science_open": False}, path)
    fresh_reader, fresh_solver = build_reader().to(device), build_solver().to(device)
    state = torch.load(path, map_location="cpu", weights_only=True)
    fresh_reader.load_state_dict(state["reader"])
    fresh_solver.load_state_dict(state["solver"])
    final_val = evaluate(fresh_reader, fresh_solver, val, TRAIN_STEPS)
    out.update({
        "history": history,
        "train_seconds": time.perf_counter() - t0,
        "checkpoint_path": path.as_posix(),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rescore_matches_record": final_val["accuracy"] == history[-1]["val_accuracy"],
        "final": {
            "val": final_val,
            **{f"long_{s}": evaluate(fresh_reader, fresh_solver, ext, s) for s in LONG_STEPS},
        },
        "oracle": {
            "val": evaluate(fresh_reader, fresh_solver, val, TRAIN_STEPS, oracle=True),
            "long_16": evaluate(fresh_reader, fresh_solver, ext, 16, oracle=True),
        },
    })
    f = out["final"]
    out["passes"] = (f["val"]["reader"]["exact_graphs"] >= PASS["exact_graphs"]
                     and f["long_16"]["reader"]["closure_agreement_all_pairs"] >= PASS["closure_agreement_long"]
                     and all(f[f"long_{s}"]["accuracy"] >= PASS["long_path_accuracy"] for s in LONG_STEPS))
    return out


def summarise(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for regime in REGIMES:
        rs = [r for r in runs if r["regime"] == regime]
        if not rs:
            continue
        def mean(key_fn):
            vals = [key_fn(r) for r in rs]
            return {"mean": statistics.fmean(vals), "min": min(vals), "max": max(vals)}
        out[regime] = {
            "seeds": [r["seed"] for r in rs],
            "passes": sum(r["passes"] for r in rs),
            "val_accuracy": mean(lambda r: r["final"]["val"]["accuracy"]),
            "val_auroc": mean(lambda r: r["final"]["val"]["auroc"]),
            "val_exact_graphs": mean(lambda r: r["final"]["val"]["reader"]["exact_graphs"]),
            "val_edge_f1": mean(lambda r: r["final"]["val"]["reader"]["f1"]),
            "val_topology_bce": mean(lambda r: r["final"]["val"]["reader"]["topology_bce"]),
            "val_edges_predicted_mean": mean(lambda r: r["final"]["val"]["reader"]["edges_predicted_mean"]),
            "val_edges_true_mean": mean(lambda r: r["final"]["val"]["reader"]["edges_true_mean"]),
            "long_closure_agreement": mean(lambda r: r["final"]["long_16"]["reader"]["closure_agreement_all_pairs"]),
            **{f"long_{s}_accuracy": mean(lambda r, s=s: r["final"][f"long_{s}"]["accuracy"]) for s in LONG_STEPS},
            **{f"long_{s}_auroc": mean(lambda r, s=s: r["final"][f"long_{s}"]["auroc"]) for s in LONG_STEPS},
            "untrained_val_accuracy": mean(lambda r: r["untrained_reader"]["val"]["accuracy"]),
            **{f"gradient_{k}_untrained": mean(lambda r, k=k: r["untrained_reader"]["gradient_probe"][k])
               for k in ("share_query_pair", "share_true_edges", "net_push_to_add")},
            **{f"gradient_{k}_final": mean(lambda r, k=k: r["history"][-1]["gradient_probe"][k])
               for k in ("share_query_pair", "share_true_edges", "net_push_to_add")},
            "errors_reader_caused_long_16": sum(r["final"]["long_16"]["errors"]["reader_caused"] for r in rs),
            "errors_solver_caused_long_16": sum(r["final"]["long_16"]["errors"]["solver_caused"] for r in rs),
        }
    return out


def merge(parts: Sequence[Path], out: Path) -> int:
    """Combine result files of one protocol (e.g. one per regime) into one."""
    loaded = [json.loads(Path(p).read_text(encoding="utf-8")) for p in parts]

    def shared(art: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in art["protocol"].items() if k != "seeds"}

    for path, art in zip(parts, loaded):
        if shared(art) != shared(loaded[0]) or art["science_open"] is not False:
            print(f"FAIL: {path} was run with a different protocol", file=sys.stderr)
            return 1
    runs = [r for art in loaded for r in art["runs"]]
    ids = [(r["regime"], r["seed"]) for r in runs]
    if len(ids) != len(set(ids)):
        print("FAIL: a run appears in more than one part", file=sys.stderr)
        return 1
    regimes = {k: v for art in loaded for k, v in art["regimes"].items()}
    mismatches = [f"{r['regime']}/seed{r['seed']}" for r in runs if not r["rescore_matches_record"]]
    artifact = {
        "science_open": False,
        "purpose": loaded[0]["purpose"],
        "regimes": {k: regimes[k] for k in REGIMES if k in regimes},
        "protocol": dict(loaded[0]["protocol"], seeds=sorted({r["seed"] for r in runs})),
        "runs": runs,
        "summary": summarise(runs),
        "self_audit_mismatches": mismatches,
        "merged_from": [Path(p).as_posix() for p in parts],
        "elapsed_seconds_by_part": [art["elapsed_seconds"] for art in loaded],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": not mismatches, "out": out.as_posix(), "self_audit_mismatches": mismatches,
                      "science_open": False}, sort_keys=True))
    return 1 if mismatches else 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Reader → anchored solver pipeline (MEASURE).")
    p.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--extended-data", type=Path, default=DEFAULT_EXTENDED)
    p.add_argument("--regimes", nargs="+", choices=list(REGIMES), default=list(REGIMES))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--reader-epochs", type=int, default=READER_EPOCHS)
    p.add_argument("--solver-epochs", type=int, default=SOLVER_EPOCHS)
    p.add_argument("--reader-lr", type=float, default=READER_LR)
    p.add_argument("--through", choices=("probability", "logit"), default=THROUGH)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--no-verify", action="store_true", help="skip dataset verification (tests only)")
    p.add_argument("--merge", type=Path, nargs="+", default=None, metavar="PART",
                   help="combine result files of parallel runs into --out (no training)")
    args = p.parse_args(argv)
    if args.merge:
        return merge(args.merge, args.out)
    try:
        import torch
    except ImportError:
        print("FAIL: torch required", file=sys.stderr)
        return 2
    rows, ext_rows = load_jsonl(args.train_data), load_jsonl(args.extended_data)
    if not args.no_verify:
        n_val = sum(1 for r in rows if r["split"] == "val")
        issues = [i for ok, found in (
            verify_crossed(rows, n_total=len(rows), n_val=n_val, spec=CROSSED_ID_SPEC),
            verify_crossed(ext_rows, n_total=len(ext_rows), n_val=len(ext_rows), spec=CROSSED_EXTENDED_SPEC),
        ) for i in found]
        if issues:
            print(f"FAIL: dataset verify: {issues}", file=sys.stderr)
            return 1
    target = __import__("reachability_gen.run_stability", fromlist=["target_params"]).target_params()
    solver_params = sum(p.numel() for p in build_solver().parameters())
    if abs(solver_params / target - 1.0) > PARAM_TOL:
        print(f"FAIL parity: solver {solver_params} vs {target}", file=sys.stderr)
        return 1
    train = Data([r for r in rows if r["split"] == "train"], args.device)
    val = Data([r for r in rows if r["split"] == "val"], args.device)
    ext = Data(ext_rows, args.device)
    artifact: dict[str, Any] = {
        "science_open": False,
        "purpose": "reader → anchored solver: can the graph come from the edge-list text?",
        "regimes": {k: REGIMES[k] for k in args.regimes},
        "protocol": {
            "reader": "question-blind, identity-free (one embedding for every node token), slot and clipped "
                      "relative-offset positions, hard 0/1 adjacency with a straight-through gradient",
            "reader_params": sum(p.numel() for p in build_reader().parameters()),
            "solver_params": solver_params,
            "solver_width": anchored_width(),
            "train_steps": TRAIN_STEPS,
            "solver_epochs": args.solver_epochs,
            "reader_epochs": args.reader_epochs,
            "reader_lr": args.reader_lr,
            "straight_through_answers_regimes": args.through,
            "solver_lr": LR,
            "long_steps": list(LONG_STEPS),
            "pass_criteria": PASS,
            "gradient_probe_rows": PROBE_ROWS,
            "seeds": args.seeds,
            "batch_size": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "checkpoints_versioned": False,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "torch_version": torch.__version__,
        },
    }
    t0 = time.perf_counter()
    runs = [run(regime, seed, train, val, ext, device=args.device, ckpt_dir=args.ckpt_dir,
                reader_epochs=args.reader_epochs, solver_epochs=args.solver_epochs, reader_lr=args.reader_lr,
                through=args.through)
            for regime in args.regimes for seed in args.seeds]
    mismatches = [f"{r['regime']}/seed{r['seed']}" for r in runs if not r["rescore_matches_record"]]
    artifact.update({"runs": runs, "summary": summarise(runs), "self_audit_mismatches": mismatches,
                     "elapsed_seconds": time.perf_counter() - t0})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    for regime, s in artifact["summary"].items():
        long_acc = "/".join(f"{s[f'long_{k}_accuracy']['mean']:.3f}" for k in LONG_STEPS)
        print(f"{regime:15s} passes {s['passes']}/{len(s['seeds'])} | val acc {s['val_accuracy']['mean']:.4f} "
              f"exact graphs {s['val_exact_graphs']['mean']:.4f} | long closure {s['long_closure_agreement']['mean']:.4f} "
              f"| long acc {'/'.join(map(str, LONG_STEPS))}: {long_acc}", file=sys.stderr)
    print(json.dumps({"ok": not mismatches, "out": args.out.as_posix(), "self_audit_mismatches": mismatches,
                      "science_open": False}, sort_keys=True))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())

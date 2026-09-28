# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Language models as the graph reader for the anchored solver (MEASURE, inference only).

The reader study (``run_reader``) showed that the anchored solver answers
exactly when a reader trained on the true edges supplies the graph. Here the
reader is an instruction-tuned language model used as is. It never sees the
question: shown a graph's edge list, it is asked, node by node, to list the
nodes that node has an edge to, and the listed successors form its graph.
The solvers of the reader study (trained on true graphs, then frozen; each
checkpoint verified against its recorded SHA-256) then search that graph.

Scoring follows the reader study's three layers: the reader (edge precision,
recall and F1, exact-graph rate, reversed edges, agreement of the reachability
closure with the true graph's), the pipeline (accuracy and AUROC on the
crossed validation sample at 6 steps and on the long-path sample at 16, 48
and 192 steps, mean over the solvers), and the attribution of each wrong
answer to the graph or the solver. The samples are those of the step-by-step
runner (``run_llm_cot``), so each model's pipeline answers can be set against
its own step-by-step answers to the same questions. Models load only from the
local Hugging Face cache; every generation is recorded.

``science_open=false`` always.

Usage::

    python -m reachability_gen.run_llm_reader --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.encode import parse_instance
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_llm_cot import BATCH, GRAPHS_PER_HOP, KV_BUDGET, generate
from reachability_gen.run_llm_reference import auroc, format_edges, load_model, sample_graphs
from reachability_gen.run_takeoff import wilson

DEFAULT_MODELS: tuple[str, ...] = (
    "Qwen/Qwen2.5-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "tiiuae/Falcon3-3B-Instruct",
)
DEFAULT_CROSSED = Path("data/id_crossed_20k.jsonl")
DEFAULT_LONG = Path("data/extended_crossed_2k.jsonl")
DEFAULT_STUDY = Path("artifacts/reader_pipeline.json")
DEFAULT_COT = (Path("artifacts/llm_cot.json"), Path("artifacts/llm_cot_7b.json"))
DEFAULT_OUT = Path("artifacts/llm_reader.json")
DEFAULT_GENERATIONS = Path("artifacts/llm_reader_generations.jsonl")
SETS: dict[str, tuple[str, tuple[int, ...]]] = {"crossed_val": ("val", (6,)), "crossed_long": ("long", (16, 48, 192))}
MAX_NEW_TOKENS: int = 64
NUMBER = re.compile(r"\d+")


def successor_prompt(edges: Sequence[tuple[int, int]], node: int) -> str:
    """Ask for one node's successors; the question's endpoints never appear."""
    return (
        f"A directed graph has these edges (u->v means an edge from u to v): {format_edges(edges)}\n"
        f"List every node v such that the edge {node}->v appears in the list above. Reply with the node "
        "numbers separated by commas, or with 'none' if there is no such edge, and nothing else."
    )


def parse_successors(text: str, node: int, n: int) -> tuple[set[int], dict[str, bool]]:
    """Successors named in a reply (numbers in range, other than the node itself), with flags."""
    numbers = [int(x) for x in NUMBER.findall(text)]
    succ = {v for v in numbers if 0 <= v < n and v != node}
    words = re.sub(r"[\d\s,.;:\-\[\]()>]", " ", text.lower()).split()
    return succ, {
        "no_answer": not numbers and "none" not in words,
        "out_of_range": any(v >= n for v in numbers),
        "extra_words": any(w not in ("none", "and") for w in words),
    }


def graphs_of(rows: Sequence[dict[str, Any]]) -> dict[str, tuple[int, list[tuple[int, int]]]]:
    """Each distinct graph of the rows: ``edge_hash -> (n, edges)``."""
    out: dict[str, tuple[int, list[tuple[int, int]]]] = {}
    for r in rows:
        if r["edge_hash"] not in out:
            n, edges, _, _ = parse_instance(r["encoding"])
            out[r["edge_hash"]] = (n, [tuple(e) for e in edges])
    return out


def load_solvers(study_path: Path, device: str) -> list[tuple[int, Any]]:
    """The supervised regime's frozen solvers, each checked against its recorded SHA-256."""
    import torch

    from reachability_gen.run_reader import build_solver

    study = json.loads(study_path.read_text(encoding="utf-8"))
    solvers = []
    for run in sorted((r for r in study["runs"] if r["regime"] == "supervised"), key=lambda r: r["seed"]):
        path = Path(run["checkpoint_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != run["checkpoint_sha256"]:
            raise ValueError(f"{path} does not match its recorded SHA-256")
        solver = build_solver()
        solver.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["solver"])
        solvers.append((int(run["seed"]), solver.to(device).eval()))
    return solvers


def score(rows: Sequence[dict[str, Any]], read: dict[str, set[tuple[int, int]]], solvers: Sequence[tuple[int, Any]],
          steps: Sequence[int], device: str) -> dict[str, Any]:
    """Reader metrics, pipeline accuracy/AUROC per step count (over the solvers) and error attribution."""
    import torch

    from reachability_gen.models.message_passing import collate, parse_rows
    from reachability_gen.models.reader import closure, edge_metrics

    graph = collate(parse_rows(rows), device)
    llm_adj = torch.zeros_like(graph["adj"], dtype=torch.float32)
    for i, r in enumerate(rows):
        for u, v in read[r["edge_hash"]]:
            llm_adj[i, u, v] = 1.0
    gold, mask = graph["adj"].float(), graph["node_mask"]
    m = edge_metrics(llm_adj, gold, mask)
    true_reach, read_reach = closure(gold, mask), closure(llm_adj, mask)
    valid = mask[:, :, None] & mask[:, None, :]
    idx = torch.arange(len(rows), device=gold.device)
    query_agree = (true_reach[idx, graph["s"], graph["t"]] == read_reach[idx, graph["s"], graph["t"]]).cpu()
    labels = [int(r["y"]) for r in rows]
    out: dict[str, Any] = {
        "reader": {
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "reversed_errors": m["reversed_errors"],
            "exact_graphs": m["exact_graphs"] / m["graphs"],  # per question row (four rows per graph)
            "edges_true_mean": m["edges_true"] / m["graphs"], "edges_read_mean": (m["tp"] + m["fp"]) / m["graphs"],
            "closure_agreement_all_pairs": ((true_reach == read_reach) & valid).sum().item() / valid.sum().item(),
            "closure_agreement_queried_pairs": query_agree.float().mean().item(),
        },
        "pipeline": {},
    }
    with torch.no_grad():
        for k in steps:
            accs, aucs, reader_caused, solver_caused, oracle = [], [], 0, 0, []
            for _, solver in solvers:
                margin = solver(dict(graph, adj=llm_adj), k).float()
                margin = (margin[:, 1] - margin[:, 0]).cpu()
                wrong = (margin > 0).long() != torch.tensor(labels)
                accs.append(1 - wrong.float().mean().item())
                aucs.append(auroc(margin.tolist(), labels))
                reader_caused += int((wrong & ~query_agree).sum())
                solver_caused += int((wrong & query_agree).sum())
                truth = solver(graph, k).float()
                oracle.append(((truth[:, 1] > truth[:, 0]).long().cpu() == torch.tensor(labels)).float().mean().item())
            out["pipeline"][str(k)] = {
                "accuracy": {"mean": statistics.fmean(accs), "min": min(accs), "max": max(accs)},
                "accuracy_wilson95": wilson(round(statistics.fmean(accs) * len(labels)), len(labels)),
                "auroc": {"mean": statistics.fmean(aucs), "min": min(aucs), "max": max(aucs)},
                "errors_over_solvers": {"graph_caused": reader_caused, "solver_caused": solver_caused},
                "true_graph_accuracy_mean": statistics.fmean(oracle),
            }
    return out


def step_by_step_reference(cot_paths: Sequence[Path], model: str, set_name: str,
                           rows: Sequence[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """The same model's step-by-step result on exactly these questions, if recorded."""
    key = [(r["edge_hash"], int(r["s"]), int(r["t"])) for r in rows]
    for path in cot_paths:
        if not path.exists():
            continue
        art = json.loads(path.read_text(encoding="utf-8"))
        if model in art.get("models", {}) and set_name in art["models"][model]["sets"]:
            if [(r["edge_hash"], int(r["s"]), int(r["t"])) for r in art["sets"][set_name]] != key:
                raise ValueError(f"{path}: {set_name} sample differs from this run's")
            e = art["models"][model]["sets"][set_name]
            return {"file": path.as_posix(), "accuracy": e["acc"], "parsed_rate": e["parsed_rate"],
                    "accuracy_among_parsed": e["acc"] / e["parsed_rate"] if e["parsed_rate"] else None}
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Language models as the graph reader for the anchored solver (MEASURE).")
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--crossed-data", type=Path, default=DEFAULT_CROSSED)
    p.add_argument("--long-data", type=Path, default=DEFAULT_LONG)
    p.add_argument("--study", type=Path, default=DEFAULT_STUDY, help="reader-study result naming the frozen solvers")
    p.add_argument("--cot", type=Path, nargs="*", default=list(DEFAULT_COT))
    p.add_argument("--graphs-per-hop", type=int, default=None)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--int8", action="store_true", help="store linear weights in 8 bits (int8_linear)")
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--kv-budget-gib", type=float, default=KV_BUDGET / 2**30)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    args = p.parse_args(argv)
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer
    except ImportError:
        print("FAIL: torch and transformers required", file=sys.stderr)
        return 2
    missing = [str(x) for x in (args.crossed_data, args.long_data, args.study) if not x.exists()]
    if missing:
        print(f"FAIL: missing {missing}", file=sys.stderr)
        return 1
    solvers = load_solvers(args.study, args.device)
    crossed_val = [r for r in load_jsonl(args.crossed_data) if r["split"] == "val"]
    per_hop = {k: args.graphs_per_hop or GRAPHS_PER_HOP[k] for k in SETS}
    sets = {"crossed_val": sample_graphs(crossed_val, per_hop["crossed_val"]),
            "crossed_long": sample_graphs(load_jsonl(args.long_data), per_hop["crossed_long"])}
    graphs = {name: graphs_of(rows) for name, rows in sets.items()}
    study = json.loads(args.study.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    results: dict[str, Any] = {}
    artifact: dict[str, Any] = {
        "science_open": False,
        "purpose": "instruction-tuned language models as question-blind graph readers for the anchored solver",
        "protocol": {
            "prompt_example": successor_prompt([(0, 1), (2, 0)], 0),
            "decoding": "greedy, chat template",
            "max_new_tokens": args.max_new_tokens,
            "parse_rule": "every number in the reply that is a node of the graph other than the asked node",
            "graphs_per_hop": per_hop,
            "sample": "as run_llm_cot (same seed), so the questions match the step-by-step runs",
            "solvers": [{"seed": s, "checkpoint_sha256": r["checkpoint_sha256"]}
                        for s, _ in solvers for r in study["runs"] if r["regime"] == "supervised" and r["seed"] == s],
            "solver_steps": {name: list(steps) for name, (_, steps) in SETS.items()},
            "batching": f"at most {args.batch} sequences and {args.kv_budget_gib} GiB of attention cache per batch; "
                        "a batch that runs out of GPU memory is halved and retried",
            "dtype": args.dtype, "int8_weights": args.int8, "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "generations_file": args.generations.as_posix(),
        },
        "sets": {k: [{"edge_hash": r["edge_hash"], "s": r["s"], "t": r["t"], "y": r["y"]} for r in v]
                 for k, v in sets.items()},
        "models": results,
        "complete": False,
    }

    def write() -> None:
        artifact["elapsed_seconds"] = time.perf_counter() - t0
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    args.generations.parent.mkdir(parents=True, exist_ok=True)
    with args.generations.open("w", encoding="utf-8") as gen_file:
        for name in args.models:
            t1 = time.perf_counter()
            tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = load_model(name, args.dtype, args.device, None, args.int8)
            stats: dict[str, int] = {"oom_splits": 0}
            res: dict[str, Any] = {"sets": {}}
            for set_name, rows in sets.items():
                items = [(eh, u) for eh, (n, _) in graphs[set_name].items() for u in range(n)]
                prompts = [successor_prompt(graphs[set_name][eh][1], u) for eh, u in items]
                t2 = time.perf_counter()
                texts, _ = generate(model, tokenizer, prompts, args.device, batch=args.batch,
                                    max_new_tokens=args.max_new_tokens, kv_budget=int(args.kv_budget_gib * 2**30),
                                    stats=stats)
                read: dict[str, set[tuple[int, int]]] = {eh: set() for eh in graphs[set_name]}
                flags = {"no_answer": 0, "out_of_range": 0, "extra_words": 0}
                for (eh, u), text in zip(items, texts):
                    succ, f = parse_successors(text, u, graphs[set_name][eh][0])
                    read[eh] |= {(u, v) for v in succ}
                    for k in flags:
                        flags[k] += f[k]
                    gen_file.write(json.dumps({"model": name, "set": set_name, "edge_hash": eh, "node": u,
                                               "text": text, "successors": sorted(succ)}, ensure_ascii=False) + "\n")
                gen_file.flush()
                entry = score(rows, read, solvers, SETS[set_name][1], args.device)
                entry["reader"]["replies"] = len(items)
                entry["reader"]["reply_flags"] = flags
                entry["reading_seconds"] = time.perf_counter() - t2
                entry["step_by_step_same_questions"] = step_by_step_reference(args.cot, name, set_name, rows)
                res["sets"][set_name] = entry
                pipe = entry["pipeline"]
                print(f"[{name}] {set_name}: edge F1 {entry['reader']['f1']:.3f} exact graphs "
                      f"{entry['reader']['exact_graphs']:.3f} closure {entry['reader']['closure_agreement_all_pairs']:.3f}"
                      " | pipeline " + " ".join(f"{k} steps acc {v['accuracy']['mean']:.3f} auroc {v['auroc']['mean']:.3f}"
                                               for k, v in pipe.items())
                      + f" ({entry['reading_seconds']:.0f}s)", file=sys.stderr, flush=True)
            res["seconds"] = time.perf_counter() - t1
            res["oom_splits"] = stats["oom_splits"]
            results[name] = res
            write()  # after every model, so a later failure keeps what finished
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
    artifact["complete"] = True
    write()
    print(json.dumps({"ok": True, "out": args.out.as_posix(), "science_open": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

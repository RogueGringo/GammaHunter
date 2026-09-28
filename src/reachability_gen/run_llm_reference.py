# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""LLM reference points on the reachability sets (MEASURE, inference only).

Standard open language models, loaded only from the local Hugging Face cache
(``local_files_only``; nothing is downloaded), answer the harness's question:
is there a directed path from s to t? Each prompt lists the graph's edges in
canonical order and asks the question after ``k`` solved examples from the
crossed training split, identical for every question. The model's answer is
whichever of " Yes" and " No" it scores higher at the first token where the
two continuations differ; the score difference also gives a threshold-free
AUROC.

Sets (fixed, seeded samples, stratified by each graph's hop):

* ``crossed_val``: crossed validation split, no one-endpoint cue by construction;
* ``crossed_long``: crossed long-path set (40-48 nodes, 8-16 hops);
* ``paired_val``: paired validation split, whose one-endpoint cues reach 0.78,
  for cue sensitivity;
* ``no_graph``: the ``crossed_val`` questions with the edge list withheld, which
  a model cannot answer above chance without the graph.

``science_open=false`` always.

Usage::

    python -m reachability_gen.run_llm_reference --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.encode import parse_instance
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_takeoff import wilson

DEFAULT_MODELS: tuple[str, ...] = (
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-3B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "allenai/OLMo-2-0425-1B-Instruct",
    "tiiuae/Falcon3-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
)
DEFAULT_CROSSED = Path("data/id_crossed_20k.jsonl")
DEFAULT_LONG = Path("data/extended_crossed_2k.jsonl")
DEFAULT_PAIRED = Path("data/id_disjoint_20k.jsonl")
DEFAULT_OUT = Path("artifacts/llm_reference.json")
SAMPLE_SEED: int = 2026
SHOT_SEED: int = 7
SHOTS: int = 4
GRAPHS_PER_HOP: dict[str, int] = {"crossed_val": 50, "crossed_long": 50, "paired_val": 100}
TOKENS_PER_BATCH: int = 12_000
INSTRUCTION = (
    "Each question gives the edges of a directed graph and asks whether a directed path "
    "leads from one node to another. Answer Yes or No."
)
WITHHELD = "(withheld)"


def format_edges(edges: Sequence[tuple[int, int]]) -> str:
    return ", ".join(f"{u}->{v}" for u, v in edges)


def format_item(encoding: str, *, graph: bool = True) -> str:
    """One question block, without its answer."""
    _, edges, s, t = parse_instance(encoding)
    shown = format_edges(edges) if graph else WITHHELD
    return f"Edges: {shown}\nQuestion: Is there a directed path from node {s} to node {t}?\nAnswer:"


def build_prompt(encoding: str, shots: Sequence[dict[str, Any]], *, graph: bool = True) -> str:
    parts = [INSTRUCTION, ""]
    for shot in shots:
        parts += [format_item(shot["encoding"]) + (" Yes" if int(shot["y"]) == 1 else " No"), ""]
    parts.append(format_item(encoding, graph=graph))
    return "\n".join(parts)


def choose_shots(rows: Sequence[dict[str, Any]], k: int = SHOTS, seed: int = SHOT_SEED) -> list[dict[str, Any]]:
    """``k`` solved examples from distinct small training graphs, labels alternating Yes/No."""
    rng = random.Random(seed)
    train = [r for r in rows if r["split"] == "train"]
    smallest = min(int(r["n"]) for r in train)
    small = [r for r in train if int(r["n"]) == smallest]
    by_graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in small:
        by_graph[r["edge_hash"]].append(r)
    graphs = sorted(by_graph)
    rng.shuffle(graphs)
    shots = []
    for i, eh in enumerate(graphs[:k]):
        want = 1 if i % 2 == 0 else 0
        shots.append(next(r for r in sorted(by_graph[eh], key=lambda r: (r["s"], r["t"])) if int(r["y"]) == want))
    return shots


def sample_graphs(rows: Sequence[dict[str, Any]], per_hop: int, seed: int = SAMPLE_SEED) -> list[dict[str, Any]]:
    """All questions of ``per_hop`` randomly chosen graphs per hop (graph hop = its positives' hop)."""
    by_graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_graph[r["edge_hash"]].append(r)
    hop_of = {eh: next(int(r["hop_distance"]) for r in rs if int(r["y"]) == 1) for eh, rs in by_graph.items()}
    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for hop in sorted(set(hop_of.values())):
        graphs = sorted(eh for eh, h in hop_of.items() if h == hop)
        for eh in rng.sample(graphs, min(per_hop, len(graphs))):
            chosen += sorted(by_graph[eh], key=lambda r: (r["s"], r["t"]))
    return chosen


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve (Mann-Whitney U with average ranks for ties)."""
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    pos = [r for r, y in zip(ranks, labels) if int(y) == 1]
    n_pos, n_neg = len(pos), len(labels) - len(pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (sum(pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def answer_tokens(tokenizer, prompt: str) -> tuple[list[int], int, int]:
    """Common prefix of prompt+" Yes" and prompt+" No", and the first differing token of each."""
    yes = tokenizer(prompt + " Yes", add_special_tokens=True)["input_ids"]
    no = tokenizer(prompt + " No", add_special_tokens=True)["input_ids"]
    i = 0
    while i < min(len(yes), len(no)) and yes[i] == no[i]:
        i += 1
    if i >= min(len(yes), len(no)):
        raise ValueError("the two answers do not diverge at a single token")
    return yes[:i], yes[i], no[i]


def score_prompts(model, tokenizer, prompts: Sequence[str], device: str, tokens_per_batch: int = TOKENS_PER_BATCH) -> list[float]:
    """Score difference (Yes minus No) at the diverging token, for each prompt."""
    import torch

    encoded = [answer_tokens(tokenizer, p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i][0]))
    margins = [0.0] * len(prompts)
    head = model.get_output_embeddings()
    body = model.get_decoder()
    start = 0
    with torch.inference_mode():
        while start < len(order):
            end = start + 1  # prompts are sorted by length: grow while the longest still fits
            while end < len(order) and (end - start + 1) * len(encoded[order[end]][0]) <= tokens_per_batch:
                end += 1
            idx = order[start:end]
            longest = len(encoded[idx[-1]][0])
            ids = torch.full((len(idx), longest), tokenizer.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(idx), longest), dtype=torch.long)
            for row, i in enumerate(idx):  # right padding: real tokens keep positions 0..L-1
                seq = encoded[i][0]
                ids[row, : len(seq)] = torch.tensor(seq)
                mask[row, : len(seq)] = 1
            first = model.get_input_embeddings().weight.device
            ids, mask = ids.to(first), mask.to(first)
            hidden = body(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            last = hidden[torch.arange(len(idx), device=hidden.device), mask.sum(dim=1).to(hidden.device) - 1]
            logits = head(last.to(head.weight.device)).float()
            for row, i in enumerate(idx):
                _, yes_id, no_id = encoded[i]
                margins[i] = float(logits[row, yes_id] - logits[row, no_id])
            start += len(idx)
    return margins


def metrics(rows: Sequence[dict[str, Any]], margins: Sequence[float]) -> dict[str, Any]:
    labels = [int(r["y"]) for r in rows]
    correct = [int((m > 0) == (y == 1)) for m, y in zip(margins, labels)]
    k, n = sum(correct), len(correct)
    graph_hop = {r["edge_hash"]: int(r["hop_distance"]) for r in rows if int(r["y"]) == 1}
    by_hop: dict[str, list[int]] = defaultdict(list)
    for r, c in zip(rows, correct):
        by_hop[str(graph_hop.get(r["edge_hash"], -1))].append(c)  # -1: positive not in rows
    return {
        "n": n,
        "acc": k / n,
        "acc_wilson95": wilson(k, n),
        "yes_rate": sum(m > 0 for m in margins) / n,
        "auroc": auroc(margins, labels),
        "acc_by_graph_hop": {h: sum(v) / len(v) for h, v in sorted(by_hop.items(), key=lambda kv: int(kv[0]))},
    }


def load_model(name: str, dtype: str, device: str, max_memory: Optional[dict[Any, str]] = None):
    """Load from the local cache; with ``max_memory``, spread layers over the listed GPUs."""
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {"local_files_only": True, "dtype": getattr(torch, dtype)}
    if max_memory:
        return AutoModelForCausalLM.from_pretrained(name, device_map="auto", max_memory=max_memory, **kwargs).eval()
    return AutoModelForCausalLM.from_pretrained(name, **kwargs).to(device).eval()


def parse_max_memory(spec: Optional[str]) -> Optional[dict[Any, str]]:
    """``"0=11GiB,1=5GiB"`` → ``{0: "11GiB", 1: "5GiB"}``."""
    if not spec:
        return None
    out: dict[Any, str] = {}
    for part in spec.split(","):
        key, value = part.split("=")
        out[int(key) if key.strip().isdigit() else key.strip()] = value.strip()
    return out


def evaluate_model(name: str, sets: dict[str, list[dict[str, Any]]], shots: list[dict[str, Any]], *,
                   device: str, dtype: str, tokens_per_batch: int,
                   max_memory: Optional[dict[Any, str]] = None) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(name, dtype, device, max_memory)
    out: dict[str, Any] = {"params": int(sum(p.numel() for p in model.parameters())), "sets": {},
                           "dtype": dtype, "max_memory": max_memory}
    for set_name, rows in sets.items():
        graph = set_name != "no_graph"
        prompts = [build_prompt(r["encoding"], shots, graph=graph) for r in rows]
        t1 = time.perf_counter()
        margins = score_prompts(model, tokenizer, prompts, device, tokens_per_batch)
        entry = metrics(rows, margins)
        entry["seconds"] = time.perf_counter() - t1
        entry["mean_prompt_tokens"] = sum(len(tokenizer(p)["input_ids"]) for p in prompts[:50]) / min(50, len(prompts))
        entry["margins"] = [round(m, 4) for m in margins]
        out["sets"][set_name] = entry
        print(f"[{name}] {set_name}: acc {entry['acc']:.3f} auroc {entry['auroc']:.3f} yes_rate {entry['yes_rate']:.2f} "
              f"({entry['seconds']:.0f}s)", file=sys.stderr, flush=True)
    out["seconds"] = time.perf_counter() - t0
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="LLM reference points on the reachability sets (MEASURE, inference only).")
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--crossed-data", type=Path, default=DEFAULT_CROSSED)
    p.add_argument("--long-data", type=Path, default=DEFAULT_LONG)
    p.add_argument("--paired-data", type=Path, default=DEFAULT_PAIRED)
    p.add_argument("--graphs-per-hop", type=int, default=None, help="override every set's graphs per hop")
    p.add_argument("--shots", type=int, default=SHOTS)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--tokens-per-batch", type=int, default=TOKENS_PER_BATCH)
    p.add_argument("--max-memory", default=None, help='spread a model over GPUs, e.g. "0=11GiB,1=5GiB"')
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    try:
        import torch
        import transformers
    except ImportError:
        print("FAIL: torch and transformers required (pip install transformers accelerate)", file=sys.stderr)
        return 2
    if args.device == "cuda" and not torch.cuda.is_available():
        print("FAIL: --device cuda, but this torch build sees no CUDA device", file=sys.stderr)
        return 1
    for path in (args.crossed_data, args.long_data, args.paired_data):
        if not path.exists():
            print(f"FAIL: missing {path}; see docs/USAGE.md", file=sys.stderr)
            return 1
    crossed = load_jsonl(args.crossed_data)
    per_hop = {k: args.graphs_per_hop or v for k, v in GRAPHS_PER_HOP.items()}
    sets = {
        "crossed_val": sample_graphs([r for r in crossed if r["split"] == "val"], per_hop["crossed_val"]),
        "crossed_long": sample_graphs(load_jsonl(args.long_data), per_hop["crossed_long"]),
        "paired_val": sample_graphs([r for r in load_jsonl(args.paired_data) if r["split"] == "val"], per_hop["paired_val"]),
    }
    sets["no_graph"] = sets["crossed_val"]
    shots = choose_shots(crossed, args.shots)
    t0 = time.perf_counter()
    results = {
        name: evaluate_model(name, sets, shots, device=args.device, dtype=args.dtype,
                             tokens_per_batch=args.tokens_per_batch,
                             max_memory=parse_max_memory(args.max_memory))
        for name in args.models
    }
    artifact = {
        "science_open": False,
        "purpose": "reference points: open LLMs on the reachability sets, inference only, local cache only",
        "protocol": {
            "shots": args.shots,
            "shot_rows": [{"edge_hash": s["edge_hash"], "s": s["s"], "t": s["t"], "y": s["y"]} for s in shots],
            "instruction": INSTRUCTION,
            "answer": "first diverging token of ' Yes' vs ' No' after the shared prefix; margin = score(Yes) - score(No)",
            "graphs_per_hop": per_hop,
            "sample_seed": SAMPLE_SEED,
            "dtype": args.dtype,
            "device": args.device,
            "max_memory": args.max_memory,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "sets": {
            name: {
                "source": {"crossed_val": args.crossed_data, "crossed_long": args.long_data,
                           "paired_val": args.paired_data, "no_graph": args.crossed_data}[name].as_posix(),
                "rows": [{"edge_hash": r["edge_hash"], "s": r["s"], "t": r["t"], "y": r["y"]} for r in rows],
            }
            for name, rows in sets.items()
        },
        "models": results,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print("\n=== accuracy (AUROC) per set ===", file=sys.stderr)
    for name, res in results.items():
        cells = " ".join(f"{s}={v['acc']:.3f}({v['auroc']:.3f})" for s, v in res["sets"].items())
        print(f"{name:38s} {cells}", file=sys.stderr)
    print(json.dumps({"ok": True, "out": args.out.as_posix(), "science_open": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

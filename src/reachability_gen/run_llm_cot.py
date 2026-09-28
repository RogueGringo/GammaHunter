# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""LLM reference points with step-by-step reasoning (MEASURE, inference only).

The direct protocol (``run_llm_reference``) asks for an immediate Yes/No. This
one lets instruction-tuned models reason first: each question is put to the
model through its own chat template with the request to think step by step
and finish with "Answer: Yes" or "Answer: No"; decoding is greedy, so every
answer is reproducible. The last "Answer: Yes/No" in the output is the
model's answer; outputs without one count as wrong and are reported as
unparsed. Every generation is written to a JSONL file for audit. Models load
only from the local Hugging Face cache.

``science_open=false`` always.

Usage::

    python -m reachability_gen.run_llm_cot --device cuda
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.encode import parse_instance
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_llm_reference import WITHHELD, format_edges, load_model, parse_max_memory, sample_graphs
from reachability_gen.run_takeoff import wilson

DEFAULT_MODELS: tuple[str, ...] = (
    "Qwen/Qwen2.5-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "tiiuae/Falcon3-3B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "allenai/OLMo-2-0425-1B-Instruct",
)
DEFAULT_CROSSED = Path("data/id_crossed_20k.jsonl")
DEFAULT_LONG = Path("data/extended_crossed_2k.jsonl")
DEFAULT_OUT = Path("artifacts/llm_cot.json")
DEFAULT_GENERATIONS = Path("artifacts/llm_cot_generations.jsonl")
GRAPHS_PER_HOP: dict[str, int] = {"crossed_val": 20, "crossed_long": 20, "no_graph": 5}
MAX_NEW_TOKENS: int = 768
BATCH: int = 24
KV_BUDGET: int = 3 * 2**30  # bytes of attention cache per generation batch
ANSWER = re.compile(r"answer[^a-z0-9]{0,8}(yes|no)\b")


def question(encoding: str, *, graph: bool = True) -> str:
    _, edges, s, t = parse_instance(encoding)
    shown = format_edges(edges) if graph else WITHHELD
    return (
        f"A directed graph has these edges (u->v means an edge from u to v): {shown}\n"
        f"Is there a directed path from node {s} to node {t}? Think step by step, then finish "
        "with a final line that reads exactly 'Answer: Yes' or 'Answer: No'."
    )


def parse_answer(text: str) -> Optional[int]:
    """1 for a final 'Answer: Yes', 0 for 'Answer: No', None when neither appears."""
    found = ANSWER.findall(text.lower())
    if not found:
        return None
    return 1 if found[-1] == "yes" else 0


def summarise(rows: Sequence[dict[str, Any]], answers: Sequence[Optional[int]], lengths: Sequence[int],
              limit: int = MAX_NEW_TOKENS) -> dict[str, Any]:
    n = len(rows)
    correct = [int(a is not None and a == int(r["y"])) for r, a in zip(rows, answers)]
    parsed = [a for a in answers if a is not None]
    graph_hop = {r["edge_hash"]: int(r["hop_distance"]) for r in rows if int(r["y"]) == 1}
    by_hop: dict[str, list[int]] = defaultdict(list)
    for r, c in zip(rows, correct):
        by_hop[str(graph_hop.get(r["edge_hash"], -1))].append(c)
    return {
        "n": n,
        "acc": sum(correct) / n,
        "acc_wilson95": wilson(sum(correct), n),
        "parsed_rate": len(parsed) / n,
        "yes_rate_among_parsed": (sum(parsed) / len(parsed)) if parsed else None,
        "acc_by_graph_hop": {h: sum(v) / len(v) for h, v in sorted(by_hop.items(), key=lambda kv: int(kv[0]))},
        "mean_generated_tokens": sum(lengths) / n,
        "hit_token_limit": sum(1 for x in lengths if x >= limit),
    }


def kv_bytes_per_token(model) -> int:
    """Bytes of attention cache one token costs (keys and values, every layer)."""
    cfg = model.config
    heads = getattr(cfg, "num_attention_heads", 1)
    kv_heads = getattr(cfg, "num_key_value_heads", None) or heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // heads
    return 2 * cfg.num_hidden_layers * kv_heads * head_dim * next(model.parameters()).element_size()


def generate(model, tokenizer, prompts: Sequence[str], device: str, *, batch: int, max_new_tokens: int,
             kv_budget: int = KV_BUDGET, stats: Optional[dict[str, int]] = None) -> tuple[list[str], list[int]]:
    """Greedy generations; each batch stays within ``kv_budget`` bytes of attention cache.

    A batch that runs out of GPU memory is split in half and retried, and later
    batches start no larger than that half. Splits are counted in
    ``stats["oom_splits"]`` and the lowered size kept in ``stats["batch_cap"]``
    (also across calls); a single sequence that does not fit still raises.
    """
    import torch

    tokenizer.padding_side = "left"
    chats = [tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True,
                                           tokenize=False) for p in prompts]
    sizes = [len(tokenizer(c, add_special_tokens=False)["input_ids"]) for c in chats]
    per_token = kv_bytes_per_token(model)
    texts: list[str] = [""] * len(prompts)
    lengths: list[int] = [0] * len(prompts)
    order = sorted(range(len(prompts)), key=lambda i: sizes[i])
    first = model.get_input_embeddings().weight.device
    limit = batch if stats is None else min(batch, stats.get("batch_cap", batch))

    def run(idx: list[int]) -> None:
        nonlocal limit
        enc = tokenizer([chats[i] for i in idx], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(first)
        try:
            out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                 pad_token_id=tokenizer.pad_token_id)
        except torch.OutOfMemoryError:
            if len(idx) == 1:
                raise
            out = None  # recover outside the handler, once the failed attempt's tensors are released
        if out is None:
            del enc
            gc.collect()  # the failed attempt's tensors can sit in reference cycles until collected
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            limit = min(limit, max(1, len(idx) // 2))
            if stats is not None:
                stats["oom_splits"] = stats.get("oom_splits", 0) + 1
                stats["batch_cap"] = limit
            run(idx[: len(idx) // 2])
            run(idx[len(idx) // 2 :])
            return
        new = out[:, enc["input_ids"].shape[1]:]
        for row, i in enumerate(idx):
            ids = new[row]
            keep = ids[ids != tokenizer.pad_token_id]
            lengths[i] = int(keep.numel())
            texts[i] = tokenizer.decode(keep, skip_special_tokens=True)

    start = 0
    with torch.inference_mode():
        while start < len(order):
            end = start + 1  # sorted by length: grow while the longest still fits the cache budget
            while (end < len(order) and end - start < limit
                   and (end - start + 1) * (sizes[order[end]] + max_new_tokens) * per_token <= kv_budget):
                end += 1
            run(order[start:end])
            start = end
    return texts, lengths


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="LLM reference points with step-by-step reasoning (MEASURE).")
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--crossed-data", type=Path, default=DEFAULT_CROSSED)
    p.add_argument("--long-data", type=Path, default=DEFAULT_LONG)
    p.add_argument("--graphs-per-hop", type=int, default=None)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--batch", type=int, default=BATCH)
    p.add_argument("--kv-budget-gib", type=float, default=KV_BUDGET / 2**30)
    p.add_argument("--max-memory", default=None, help='spread a model over GPUs, e.g. "0=11GiB,1=5GiB"')
    p.add_argument("--int8", action="store_true", help="store linear weights in 8 bits (int8_linear)")
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
    per_hop = {k: args.graphs_per_hop or v for k, v in GRAPHS_PER_HOP.items()}
    crossed_val = [r for r in load_jsonl(args.crossed_data) if r["split"] == "val"]
    sets = {
        "crossed_val": sample_graphs(crossed_val, per_hop["crossed_val"]),
        "crossed_long": sample_graphs(load_jsonl(args.long_data), per_hop["crossed_long"]),
        "no_graph": sample_graphs(crossed_val, per_hop["no_graph"]),
    }
    results: dict[str, Any] = {}
    t0 = time.perf_counter()
    artifact: dict[str, Any] = {
        "science_open": False,
        "purpose": "reference points with step-by-step reasoning: instruction-tuned LLMs, greedy decoding, local cache only",
        "protocol": {
            "prompt_example": question(sets["crossed_val"][0]["encoding"]),
            "decoding": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "answer_rule": "last 'Answer: Yes/No' in the output; none counts as wrong",
            "graphs_per_hop": per_hop,
            "batching": f"at most {args.batch} sequences and {args.kv_budget_gib} GiB of attention cache per batch; "
                        "a batch that runs out of GPU memory is halved and retried, and later batches start "
                        "no larger than that half (counted per model)",
            "dtype": args.dtype,
            "int8_weights": args.int8,
            "device": args.device,
            "max_memory": args.max_memory,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
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
            model = load_model(name, args.dtype, args.device, parse_max_memory(args.max_memory), args.int8)
            res: dict[str, Any] = {"sets": {}, "kv_bytes_per_token": kv_bytes_per_token(model)}
            stats: dict[str, int] = {"oom_splits": 0}
            for set_name, rows in sets.items():
                prompts = [question(r["encoding"], graph=set_name != "no_graph") for r in rows]
                t2 = time.perf_counter()
                texts, lengths = generate(model, tokenizer, prompts, args.device, batch=args.batch,
                                          max_new_tokens=args.max_new_tokens,
                                          kv_budget=int(args.kv_budget_gib * 2**30), stats=stats)
                answers = [parse_answer(t) for t in texts]
                entry = summarise(rows, answers, lengths, args.max_new_tokens)
                entry["seconds"] = time.perf_counter() - t2
                res["sets"][set_name] = entry
                for r, text, ans in zip(rows, texts, answers):
                    gen_file.write(json.dumps({"model": name, "set": set_name, "edge_hash": r["edge_hash"],
                                               "s": r["s"], "t": r["t"], "y": r["y"], "answer": ans,
                                               "text": text}, ensure_ascii=False) + "\n")
                gen_file.flush()
                print(f"[{name}] {set_name}: acc {entry['acc']:.3f} parsed {entry['parsed_rate']:.2f} "
                      f"tokens {entry['mean_generated_tokens']:.0f} ({entry['seconds']:.0f}s)", file=sys.stderr, flush=True)
            res["seconds"] = time.perf_counter() - t1
            res["oom_splits"] = stats["oom_splits"]  # batches halved after running out of GPU memory
            res["batch_cap"] = stats.get("batch_cap", args.batch)
            results[name] = res
            write()  # after every model, so a later failure keeps what finished
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()
    artifact["complete"] = True
    write()
    print("\n=== accuracy (parsed rate) per set ===", file=sys.stderr)
    for name, res in results.items():
        print(f"{name:38s} " + " ".join(f"{s}={v['acc']:.3f}({v['parsed_rate']:.2f})" for s, v in res["sets"].items()),
              file=sys.stderr)
    print(json.dumps({"ok": True, "out": args.out.as_posix(), "science_open": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

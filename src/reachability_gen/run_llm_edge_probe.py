# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Edge-lookup probe: can the reference LLMs read the edge list? (MEASURE, inference only)

The reachability questions need two abilities, reading the edge list and
searching it; the reference points (``run_llm_reference``) measured only both
together. This probe asks the same models about single edges ("is u->v
listed?") on the same graphs, with the same scoring (first diverging token of
" Yes" vs " No", AUROC from the score difference). Each graph contributes two
listed edges and two absent ones: the reverse of a listed edge (whose reverse
is absent), which only a direction-aware reading gets right, and a pair of
nodes with no edge in either direction. A model that reads the list scores
near 1.0 here even if it fails the reachability questions; then its failure is
search, not reading.

``science_open=false`` always.

Usage::

    python -m reachability_gen.run_llm_edge_probe --device cuda
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.encode import parse_instance
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_llm_reference import (
    DEFAULT_MODELS,
    auroc,
    format_edges,
    load_model,
    parse_max_memory,
    sample_graphs,
    score_prompts,
)
from reachability_gen.run_takeoff import wilson

DEFAULT_CROSSED = Path("data/id_crossed_20k.jsonl")
DEFAULT_LONG = Path("data/extended_crossed_2k.jsonl")
DEFAULT_OUT = Path("artifacts/llm_edge_probe.json")
PROBE_SEED: int = 31
GRAPHS_PER_HOP: int = 50
INSTRUCTION = (
    "Each question gives the edges of a directed graph and asks whether one particular "
    "edge is in the list. Answer Yes or No."
)
KINDS: tuple[str, ...] = ("listed", "reversed", "absent")


def edge_questions(encoding: str, rng: random.Random) -> list[dict[str, Any]]:
    """Two listed edges, one reversed listed edge and one absent pair, for one graph."""
    n, edges, _, _ = parse_instance(encoding)
    present = set(edges)
    listed = sorted(present)
    one_way = [(u, v) for u, v in listed if (v, u) not in present]
    unlinked = [(u, v) for u in range(n) for v in range(n)
                if u != v and (u, v) not in present and (v, u) not in present]
    if len(listed) < 2 or not one_way or not unlinked:
        return []
    out = [{"u": u, "v": v, "y": 1, "kind": "listed"} for u, v in rng.sample(listed, 2)]
    u, v = rng.choice(one_way)
    out.append({"u": v, "v": u, "y": 0, "kind": "reversed"})
    u, v = rng.choice(unlinked)
    out.append({"u": u, "v": v, "y": 0, "kind": "absent"})
    return out


def edge_item(encoding: str, u: int, v: int) -> str:
    _, edges, _, _ = parse_instance(encoding)
    return f"Edges: {format_edges(edges)}\nQuestion: Is the edge {u}->{v} in the list?\nAnswer:"


def edge_prompt(encoding: str, u: int, v: int, shots: Sequence[dict[str, Any]]) -> str:
    parts = [INSTRUCTION, ""]
    for shot in shots:
        parts += [edge_item(shot["encoding"], shot["u"], shot["v"]) + (" Yes" if shot["y"] == 1 else " No"), ""]
    parts.append(edge_item(encoding, u, v))
    return "\n".join(parts)


def probe_rows(graph_rows: Sequence[dict[str, Any]], seed: int = PROBE_SEED) -> list[dict[str, Any]]:
    """Four edge questions for each distinct graph among ``graph_rows``."""
    rng = random.Random(seed)
    seen: dict[str, str] = {}
    for r in graph_rows:
        seen.setdefault(r["edge_hash"], r["encoding"])
    rows = []
    for eh in sorted(seen):
        for q in edge_questions(seen[eh], rng):
            rows.append({"edge_hash": eh, "encoding": seen[eh], **q})
    return rows


def choose_edge_shots(train_rows: Sequence[dict[str, Any]], seed: int = PROBE_SEED + 1) -> list[dict[str, Any]]:
    """Four solved examples from small training graphs: listed, reversed, listed, absent."""
    rng = random.Random(seed)
    smallest = min(int(r["n"]) for r in train_rows)
    graphs = sorted({r["edge_hash"]: r["encoding"] for r in train_rows if int(r["n"]) == smallest}.items())
    rng.shuffle(graphs)
    order = ("listed", "reversed", "listed", "absent")
    shots = []
    for (eh, enc), kind in zip(graphs, order):
        q = next(x for x in edge_questions(enc, rng) if x["kind"] == kind)
        shots.append({"edge_hash": eh, "encoding": enc, **q})
    return shots


def summarise(rows: Sequence[dict[str, Any]], margins: Sequence[float]) -> dict[str, Any]:
    labels = [int(r["y"]) for r in rows]
    correct = [int((m > 0) == (y == 1)) for m, y in zip(margins, labels)]
    by_kind: dict[str, list[int]] = defaultdict(list)
    for r, c in zip(rows, correct):
        by_kind[r["kind"]].append(c)
    k, n = sum(correct), len(correct)
    return {
        "n": n,
        "acc": k / n,
        "acc_wilson95": wilson(k, n),
        "auroc": auroc(margins, labels),
        "yes_rate": sum(m > 0 for m in margins) / n,
        "acc_by_kind": {kind: sum(v) / len(v) for kind, v in sorted(by_kind.items())},
        "auroc_listed_vs_reversed": auroc(
            [m for r, m in zip(rows, margins) if r["kind"] in ("listed", "reversed")],
            [int(r["y"]) for r in rows if r["kind"] in ("listed", "reversed")],
        ),
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Edge-lookup probe for the reference LLMs (MEASURE, inference only).")
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--crossed-data", type=Path, default=DEFAULT_CROSSED)
    p.add_argument("--long-data", type=Path, default=DEFAULT_LONG)
    p.add_argument("--graphs-per-hop", type=int, default=GRAPHS_PER_HOP)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--tokens-per-batch", type=int, default=12_000)
    p.add_argument("--max-memory", default=None)
    p.add_argument("--int8", action="store_true", help="store linear weights in 8 bits (int8_linear)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer
    except ImportError:
        print("FAIL: torch and transformers required", file=sys.stderr)
        return 2
    crossed = load_jsonl(args.crossed_data)
    sets = {
        "crossed_val": probe_rows(sample_graphs([r for r in crossed if r["split"] == "val"], args.graphs_per_hop)),
        "crossed_long": probe_rows(sample_graphs(load_jsonl(args.long_data), args.graphs_per_hop)),
    }
    shots = choose_edge_shots([r for r in crossed if r["split"] == "train"])
    results: dict[str, Any] = {}
    # Everything but the results is assembled before any model runs, so a
    # mistake here fails in seconds instead of after the whole evaluation.
    artifact: dict[str, Any] = {
        "science_open": False,
        "purpose": "edge-lookup probe: can the reference LLMs read the edge list (one-step lookup)?",
        "protocol": {
            "instruction": INSTRUCTION,
            "shots": [{"edge_hash": s["edge_hash"], "u": s["u"], "v": s["v"], "y": s["y"], "kind": s["kind"]} for s in shots],
            "questions_per_graph": {"listed": 2, "reversed": 1, "absent": 1},
            "graphs_per_hop": args.graphs_per_hop,
            "probe_seed": PROBE_SEED,
            "dtype": args.dtype,
            "int8_weights": args.int8,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "sets": {k: [{"edge_hash": r["edge_hash"], "u": r["u"], "v": r["v"], "y": r["y"], "kind": r["kind"]} for r in v]
                 for k, v in sets.items()},
        "models": results,
    }
    t0 = time.perf_counter()
    for name in args.models:
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = load_model(name, args.dtype, args.device, parse_max_memory(args.max_memory), args.int8)
        res: dict[str, Any] = {"sets": {}}
        for set_name, rows in sets.items():
            prompts = [edge_prompt(r["encoding"], r["u"], r["v"], shots) for r in rows]
            margins = score_prompts(model, tokenizer, prompts, args.device, args.tokens_per_batch)
            entry = summarise(rows, margins)
            entry["margins"] = [round(m, 4) for m in margins]
            res["sets"][set_name] = entry
            print(f"[{name}] {set_name}: acc {entry['acc']:.3f} auroc {entry['auroc']:.3f} by kind "
                  f"{ {k: round(v, 3) for k, v in entry['acc_by_kind'].items()} }", file=sys.stderr, flush=True)
        results[name] = res
        del model
        gc.collect()  # a model can stay referenced in cycles until collected
        if args.device == "cuda":
            torch.cuda.empty_cache()
    artifact["elapsed_seconds"] = time.perf_counter() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "out": args.out.as_posix(), "science_open": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Cue-conflict test for the LLM reference points (MEASURE).

On the paired split every graph has one reachable and one unreachable
question. For a feature that reads one endpoint's reach (``reach_cues``), a
graph *agrees* with the cue when the reachable question has the higher value
and *conflicts* when it has the lower one. A model that searches for the path
ranks the reachable question above the unreachable one about equally often in
both groups; a model that follows the cue does so mostly in agreeing graphs
and less than half the time in conflicting ones.

Reads ``artifacts/llm_reference.json`` (its stored margins and sample) and the
paired data; writes ``artifacts/llm_cue_conflict.json``.

``science_open=false`` always.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.encode import parse_instance
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.reach_cues import endpoint_reach_features
from reachability_gen.run_takeoff import fisher_exact, wilson

DEFAULT_REFERENCE = Path("artifacts/llm_reference.json")
DEFAULT_PAIRED = Path("data/id_disjoint_20k.jsonl")
DEFAULT_OUT = Path("artifacts/llm_cue_conflict.json")
FEATURES: tuple[tuple[str, Optional[int]], ...] = (
    ("target_ancestors", 6),
    ("source_reach", None),
    ("target_ancestors", None),
)


def pairwise(rows: Sequence[dict[str, Any]], margins: Sequence[float], feature: str,
             limit: Optional[int]) -> dict[str, Any]:
    """Pairwise ranking accuracy of the margins in cue-agreeing and cue-conflicting graphs."""
    graphs: dict[str, dict[int, tuple[dict[str, Any], float]]] = {}
    for r, m in zip(rows, margins):
        graphs.setdefault(r["edge_hash"], {})[int(r["y"])] = (r, m)
    groups: dict[str, list[float]] = {"agree": [], "conflict": [], "tie": []}
    for pair in graphs.values():
        if 0 not in pair or 1 not in pair:
            continue
        values = {}
        for y, (r, _) in pair.items():
            _, edges, s, t = parse_instance(r["encoding"])
            values[y] = endpoint_reach_features(edges, s, t, limit=limit)[feature]
        diff = pair[1][1] - pair[0][1]
        won = 1.0 if diff > 0 else 0.5 if diff == 0 else 0.0
        key = "agree" if values[1] > values[0] else "conflict" if values[1] < values[0] else "tie"
        groups[key].append(won)
    out: dict[str, Any] = {}
    for key, results in groups.items():
        won = sum(1 for w in results if w == 1.0)
        tied = sum(1 for w in results if w == 0.5)
        lost = len(results) - won - tied
        decided = won + lost
        out[key] = {
            "graphs": len(results),
            "wins": won,
            "ties": tied,
            "losses": lost,
            "ranked_reachable_higher": (sum(results) / len(results)) if results else None,  # a tie counts half
            "decided_win_rate": (won / decided) if decided else None,  # ties left out
            "decided_wilson95": wilson(won, decided) if decided else None,
        }
    return out


def cue_groups(rows: Sequence[dict[str, Any]], feature: str, limit: Optional[int]) -> dict[str, list[str]]:
    """Graphs whose reachable question has the higher (agree) or lower (conflict) feature value."""
    graphs: dict[str, dict[int, dict[str, Any]]] = {}
    for r in rows:
        graphs.setdefault(r["edge_hash"], {})[int(r["y"])] = r
    groups: dict[str, list[str]] = {"agree": [], "conflict": [], "tie": []}
    for eh in sorted(graphs):
        pair = graphs[eh]
        if 0 not in pair or 1 not in pair:
            continue
        value = {}
        for y, r in pair.items():
            _, edges, s, t = parse_instance(r["encoding"])
            value[y] = endpoint_reach_features(edges, s, t, limit=limit)[feature]
        key = "agree" if value[1] > value[0] else "conflict" if value[1] < value[0] else "tie"
        groups[key].append(eh)
    return groups


def enriched(argv_models: Sequence[str], paired: Sequence[dict[str, Any]], crossed: Sequence[dict[str, Any]], *,
             device: str, dtype: str, tokens_per_batch: int, seed: int = 11) -> dict[str, Any]:
    """Every cue-conflicting paired graph plus as many cue-agreeing ones, scored by each model."""
    import random

    import torch
    from transformers import AutoTokenizer

    from reachability_gen.run_llm_reference import build_prompt, choose_shots, load_model, score_prompts

    feature, limit = FEATURES[0]
    groups = cue_groups(paired, feature, limit)
    agree = sorted(random.Random(seed).sample(groups["agree"], min(len(groups["agree"]), len(groups["conflict"]))))
    chosen = {"conflict": groups["conflict"], "agree": agree}
    by_graph: dict[str, list[dict[str, Any]]] = {}
    for r in paired:
        by_graph.setdefault(r["edge_hash"], []).append(r)
    rows = [r for key in ("agree", "conflict") for eh in chosen[key] for r in sorted(by_graph[eh], key=lambda r: r["y"])]
    shots = choose_shots(crossed)
    prompts = [build_prompt(r["encoding"], shots) for r in rows]
    out: dict[str, Any] = {
        "feature": f"{feature}@within_{limit}",
        "available": {k: len(v) for k, v in groups.items()},
        "graphs": {k: len(v) for k, v in chosen.items()},
        "rows": [{"edge_hash": r["edge_hash"], "s": r["s"], "t": r["t"], "y": r["y"]} for r in rows],
        "models": {},
    }
    for name in argv_models:
        tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = load_model(name, dtype, device)
        margins = score_prompts(model, tokenizer, prompts, device, tokens_per_batch)
        result: dict[str, Any] = {}
        members = {key: set(chosen[key]) for key in chosen}
        for key in ("agree", "conflict"):
            keep = [(r, m) for r, m in zip(rows, margins) if r["edge_hash"] in members[key]]
            result[key] = pairwise([r for r, _ in keep], [m for _, m in keep], feature, limit)[key]
        a, c = result["agree"], result["conflict"]
        result["fisher_decided_agree_vs_conflict"] = fisher_exact(
            a["wins"], a["wins"] + a["losses"], c["wins"], c["wins"] + c["losses"])
        result["margins"] = [round(m, 4) for m in margins]
        out["models"][name] = result
        print(f"[{name}] decided win rate agree {a['decided_win_rate']:.3f} | conflict {c['decided_win_rate']:.3f} "
              f"(ties {a['ties']}/{c['ties']}; p={result['fisher_decided_agree_vs_conflict']:.2e})",
              file=sys.stderr, flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return out


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Cue-conflict test for the LLM reference points (MEASURE).")
    p.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    p.add_argument("--paired-data", type=Path, default=DEFAULT_PAIRED)
    p.add_argument("--crossed-data", type=Path, default=Path("data/id_crossed_20k.jsonl"))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--enriched", nargs="*", default=None, metavar="MODEL",
                   help="also score every cue-conflicting paired graph (train and val) and as many agreeing "
                        "ones with these models (default: the reference file's models)")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--tokens-per-batch", type=int, default=12_000)
    args = p.parse_args(argv)
    for path in (args.reference, args.paired_data):
        if not path.exists():
            print(f"FAIL: missing {path}", file=sys.stderr)
            return 1
    ref = json.loads(args.reference.read_text(encoding="utf-8"))
    index = {(r["edge_hash"], int(r["s"]), int(r["t"])): r for r in load_jsonl(args.paired_data)}
    rows = [index[(r["edge_hash"], int(r["s"]), int(r["t"]))] for r in ref["sets"]["paired_val"]["rows"]]
    results: dict[str, Any] = {}
    for name, res in ref["models"].items():
        margins = res["sets"]["paired_val"]["margins"]
        results[name] = {
            f"{feature}@{'unlimited' if limit is None else f'within_{limit}'}": pairwise(rows, margins, feature, limit)
            for feature, limit in FEATURES
        }
    artifact = {
        "science_open": False,
        "purpose": "cue-conflict test: do LLM margins on the paired split follow one-endpoint reach cues?",
        "reference": args.reference.as_posix(),
        "paired_data": args.paired_data.as_posix(),
        "models": results,
    }
    if args.enriched is not None:
        models = args.enriched or list(ref["models"])
        artifact["enriched"] = enriched(models, load_jsonl(args.paired_data), load_jsonl(args.crossed_data),
                                        device=args.device, dtype=args.dtype, tokens_per_batch=args.tokens_per_batch)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    key = "target_ancestors@within_6"
    print(f"=== decided win rate: cue-agreeing vs cue-conflicting graphs ({key}) ===", file=sys.stderr)
    for name, res in results.items():
        g = res[key]
        print(f"{name:38s} agree {g['agree']['decided_win_rate']} (n={g['agree']['graphs']}) | "
              f"conflict {g['conflict']['decided_win_rate']} (n={g['conflict']['graphs']})", file=sys.stderr)
    print(json.dumps({"ok": True, "out": args.out.as_posix(), "science_open": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

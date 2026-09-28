# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Path audit of the step-by-step LLM generations (MEASURE, text only).

The step-by-step runner (``run_llm_cot``) records every generation. This
audit checks the paths the models write against the edges they were shown.
A *written path* is a chain of three or more node numbers joined by arrows
("4->9->3"); the last one in a generation is taken as the path the answer
rests on. Each of its steps is *listed* (an edge of the graph), *reversed*
(only the opposite edge is listed) or *absent* (neither). Single arrows
("4->9") are counted separately as *cited edges*, which shows whether the
edges quoted while reasoning are quoted correctly.

Results are grouped by label and answer (e.g. a Yes on an unreachable pair),
per model and set. The withheld-graph set is skipped: it shows no edges.

``science_open=false`` always.

Usage::

    python -m reachability_gen.llm_cot_paths
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from reachability_gen.encode import parse_instance
from reachability_gen.overfit_ff import load_jsonl

DEFAULT_GENERATIONS = (Path("artifacts/llm_cot_generations.jsonl"), Path("artifacts/llm_cot_7b_generations.jsonl"))
DEFAULT_DATA = (Path("data/id_crossed_20k.jsonl"), Path("data/extended_crossed_2k.jsonl"))
DEFAULT_OUT = Path("artifacts/llm_cot_path_audit.json")
CHAIN = re.compile(r"\d+(?:\s*(?:->|→)\s*\d+)+")
GROUPS = ("yes_on_reachable", "yes_on_unreachable", "no_on_reachable", "no_on_unreachable", "unparsed")


def classify_step(u: int, v: int, edges: set[tuple[int, int]]) -> str:
    if (u, v) in edges:
        return "listed"
    return "reversed" if (v, u) in edges else "absent"


def audit_text(text: str, edges: set[tuple[int, int]], s: int, t: int) -> dict[str, Any]:
    """Cited edges and the final written path of one generation."""
    chains = [[int(x) for x in re.findall(r"\d+", m)] for m in CHAIN.findall(text)]
    cited = [classify_step(c[0], c[1], edges) for c in chains if len(c) == 2]
    paths = [c for c in chains if len(c) >= 3]
    out: dict[str, Any] = {"cited": cited, "path": None}
    if paths:
        path = paths[-1]
        steps = [classify_step(u, v, edges) for u, v in zip(path, path[1:])]
        out["path"] = {"steps": steps, "from_s_to_t": path[0] == s and path[-1] == t}
    return out


def group_of(y: int, answer: Optional[int]) -> str:
    if answer is None:
        return "unparsed"
    return f"{'yes' if answer == 1 else 'no'}_on_{'reachable' if y == 1 else 'unreachable'}"


def audit(generations: Iterable[dict[str, Any]], graphs: dict[tuple[str, int, int], str]) -> dict[str, Any]:
    """Counts per model, set and label/answer group."""
    out: dict[str, Any] = {}
    for g in generations:
        if g["set"] == "no_graph":
            continue
        _, edges, s, t = parse_instance(graphs[(g["edge_hash"], int(g["s"]), int(g["t"]))])
        res = audit_text(g["text"], set(map(tuple, edges)), s, t)
        entry = out.setdefault(g["model"], {}).setdefault(g["set"], {
            "cited_edges": {"listed": 0, "reversed": 0, "absent": 0},
            **{grp: {"generations": 0, "with_written_path": 0, "path_all_steps_listed": 0,
                     "path_with_reversed_step": 0, "path_with_absent_step": 0, "path_from_s_to_t": 0}
               for grp in GROUPS},
        })
        for kind in res["cited"]:
            entry["cited_edges"][kind] += 1
        grp = entry[group_of(int(g["y"]), g["answer"])]
        grp["generations"] += 1
        if res["path"] is not None:
            steps = res["path"]["steps"]
            grp["with_written_path"] += 1
            grp["path_all_steps_listed"] += all(k == "listed" for k in steps)
            grp["path_with_reversed_step"] += "reversed" in steps
            grp["path_with_absent_step"] += "absent" in steps
            grp["path_from_s_to_t"] += res["path"]["from_s_to_t"]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Path audit of step-by-step LLM generations (MEASURE).")
    p.add_argument("--generations", type=Path, nargs="+", default=list(DEFAULT_GENERATIONS))
    p.add_argument("--data", type=Path, nargs="+", default=list(DEFAULT_DATA))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    missing = [str(x) for x in (*args.generations, *args.data) if not x.exists()]
    if missing:
        print(f"FAIL: missing {missing}", file=sys.stderr)
        return 1
    graphs = {(r["edge_hash"], int(r["s"]), int(r["t"])): r["encoding"] for d in args.data for r in load_jsonl(d)}
    gens = [json.loads(line) for path in args.generations for line in path.open(encoding="utf-8") if line.strip()]
    artifact = {
        "science_open": False,
        "purpose": "do the paths written in step-by-step generations use the listed edges?",
        "definitions": {
            "written_path": "the last chain of three or more node numbers joined by arrows in a generation",
            "listed": "the step is an edge of the graph shown",
            "reversed": "only the opposite edge is listed",
            "absent": "neither direction is listed",
            "cited_edges": "single arrows u->v quoted while reasoning",
        },
        "generations_files": [x.as_posix() for x in args.generations],
        "models": audit(gens, graphs),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    for model, sets in artifact["models"].items():
        for set_name, e in sets.items():
            wrong_yes = e["yes_on_unreachable"]
            print(f"{model:38s} {set_name:12s} cited {e['cited_edges']} | Yes on unreachable: "
                  f"{wrong_yes['with_written_path']} written paths, {wrong_yes['path_all_steps_listed']} all listed, "
                  f"{wrong_yes['path_with_reversed_step']} with a reversed step", file=sys.stderr)
    print(json.dumps({"ok": True, "out": args.out.as_posix(), "science_open": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

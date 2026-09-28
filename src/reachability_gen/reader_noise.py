# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""How exact must the reader's graph be? (MEASURE, evaluation only)

Corrupts the true graphs of the crossed validation split and the long-path set
at controlled rates, in one of three ways: deleting each edge with probability
p, inserting pairs that are not edges (r times the number of edges, rounded at
random), or reversing each edge with probability p. The frozen solvers of the
reader study then answer the questions on the corrupted graphs, scored as in
``run_llm_reader`` (reader layer, pipeline accuracy and AUROC, attribution).

Crossed positives have exactly one path, and deleting edges cannot create a
path, so for deletions the expected accuracy is 1/2 + 1/2 * mean (1 - p)^h over
the positives' path lengths h; it is recorded next to the measured curve.

A matched-noise control places the language-model readers of
``artifacts/llm_reader.json`` on these curves: random readings of the same
graphs with each model's recall, extra edges per true edge and share of
reversed extras, scored on the same questions. A model whose pipeline beats its
matched noise makes errors that spare the edges the answers depend on; one that
does worse concentrates them there.

``science_open=false`` always.

Usage::

    python -m reachability_gen.reader_noise --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_llm_reader import SETS, graphs_of, load_solvers, score

DEFAULT_CROSSED = Path("data/id_crossed_20k.jsonl")
DEFAULT_LONG = Path("data/extended_crossed_2k.jsonl")
DEFAULT_STUDY = Path("artifacts/reader_pipeline.json")
DEFAULT_LLM_READER = Path("artifacts/llm_reader.json")
DEFAULT_OUT = Path("artifacts/reader_noise.json")
KINDS: tuple[str, ...] = ("delete", "reverse", "insert")
RATES: dict[str, tuple[float, ...]] = {
    "delete": (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
    "reverse": (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
    "insert": (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0),
}
DRAWS: int = 5
SEED: int = 2026

Edge = tuple[int, int]


def random_count(x: float, rng: random.Random) -> int:
    """``x`` rounded up with probability equal to its fractional part (unbiased)."""
    base = int(x)
    return base + (rng.random() < x - base)


def corrupt(n: int, edges: Sequence[Edge], kind: str, rate: float, rng: random.Random) -> set[Edge]:
    """The graph's edges after one kind of controlled error."""
    true = set(edges)
    if kind == "delete":
        return {e for e in sorted(true) if rng.random() >= rate}
    if kind == "reverse":
        return {(v, u) if rng.random() < rate else (u, v) for u, v in sorted(true)}
    if kind == "insert":
        absent = [(u, v) for u in range(n) for v in range(n) if u != v and (u, v) not in true]
        return true | set(rng.sample(absent, min(len(absent), random_count(rate * len(true), rng))))
    raise ValueError(f"unknown kind {kind!r}")


def matched(n: int, edges: Sequence[Edge], *, recall: float, extra_per_edge: float, reversed_share: float,
            rng: random.Random) -> set[Edge]:
    """A random reading with a reader's recall, extra edges per true edge and reversed share of the extras."""
    true = set(edges)
    kept = {e for e in sorted(true) if rng.random() < recall}
    extra = random_count(extra_per_edge * len(true), rng)
    reversible = [(v, u) for u, v in sorted(true) if (v, u) not in true]
    reversed_ = set(rng.sample(reversible, min(len(reversible), random_count(reversed_share * extra, rng))))
    absent = [(u, v) for u in range(n) for v in range(n)
              if u != v and (u, v) not in true and (u, v) not in reversed_]
    other = set(rng.sample(absent, min(len(absent), max(0, extra - len(reversed_)))))
    return kept | reversed_ | other


def deletion_accuracy(rows: Sequence[dict[str, Any]], rate: float) -> float:
    """Expected accuracy under deletions when every positive has exactly one path of its hop length."""
    return statistics.fmean(1.0 if int(r["y"]) == 0 else (1 - rate) ** int(r["hop_distance"]) for r in rows)


def summarise_draws(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Mean (and range over draws) of the pipeline and reader layers of repeated ``score`` results."""
    def stat(values: list[float]) -> dict[str, float]:
        return {"mean": statistics.fmean(values), "min": min(values), "max": max(values)}

    steps = list(entries[0]["pipeline"])
    return {
        "pipeline": {k: {"accuracy": stat([e["pipeline"][k]["accuracy"]["mean"] for e in entries]),
                         "auroc": stat([e["pipeline"][k]["auroc"]["mean"] for e in entries])} for k in steps},
        "reader": {m: statistics.fmean(e["reader"][m] for e in entries)
                   for m in ("precision", "recall", "f1", "closure_agreement_all_pairs", "closure_exact_graphs",
                             "edges_read_mean")},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="How exact must the reader's graph be? (MEASURE)")
    p.add_argument("--crossed-data", type=Path, default=DEFAULT_CROSSED)
    p.add_argument("--long-data", type=Path, default=DEFAULT_LONG)
    p.add_argument("--study", type=Path, default=DEFAULT_STUDY, help="reader-study result naming the frozen solvers")
    p.add_argument("--llm-reader", type=Path, default=DEFAULT_LLM_READER)
    p.add_argument("--draws", type=int, default=DRAWS)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    missing = [str(x) for x in (args.crossed_data, args.long_data, args.study) if not x.exists()]
    if missing:
        print(f"FAIL: missing {missing}", file=sys.stderr)
        return 1
    import torch

    t0 = time.perf_counter()
    solvers = load_solvers(args.study, args.device)
    crossed = load_jsonl(args.crossed_data)
    long_rows = load_jsonl(args.long_data)
    sets = {"crossed_val": [r for r in crossed if r["split"] == "val"], "crossed_long": long_rows}
    study = json.loads(args.study.read_text(encoding="utf-8"))
    artifact: dict[str, Any] = {
        "science_open": False,
        "purpose": "how exact the reader's graph must be: pipeline accuracy under controlled reading errors",
        "protocol": {
            "kinds": {"delete": "each edge removed with probability p",
                      "reverse": "each edge replaced by its reverse with probability p",
                      "insert": "r times the number of edges of absent ordered pairs added, rounded at random"},
            "rates": {k: list(v) for k, v in RATES.items()},
            "draws": args.draws, "seed": SEED,
            "rows": {k: len(v) for k, v in sets.items()},
            "solver_steps": {name: list(steps) for name, (_, steps) in SETS.items()},
            "solvers": [{"seed": s, "checkpoint_sha256": r["checkpoint_sha256"]}
                        for s, _ in solvers for r in study["runs"] if r["regime"] == "supervised" and r["seed"] == s],
            "deletion_expectation": "1/2 + 1/2 * mean (1 - p)^h over positives (one path each)",
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "torch_version": torch.__version__,
        },
        "true_graph": {}, "curves": {}, "deletion_expectation": {}, "matched_noise": {},
    }
    for set_name, rows in sets.items():
        graphs = graphs_of(rows)
        steps = SETS[set_name][1]
        control = score(rows, {eh: set(e) for eh, (_, e) in graphs.items()}, solvers, steps, args.device)
        artifact["true_graph"][set_name] = {k: v["accuracy"]["mean"] for k, v in control["pipeline"].items()}
        artifact["deletion_expectation"][set_name] = {str(rate): deletion_accuracy(rows, rate)
                                                      for rate in RATES["delete"]}
        curves: dict[str, Any] = {}
        for kind in KINDS:
            curves[kind] = {}
            for rate in RATES[kind]:
                entries = []
                for draw in range(args.draws if rate > 0 else 1):
                    rng = random.Random(f"{SEED}/{set_name}/{kind}/{rate}/{draw}")
                    read = {eh: corrupt(n, e, kind, rate, rng) for eh, (n, e) in graphs.items()}
                    entries.append(score(rows, read, solvers, steps, args.device, with_true_graph=False))
                curves[kind][str(rate)] = summarise_draws(entries)
                first = next(iter(curves[kind][str(rate)]["pipeline"].values()))["accuracy"]["mean"]
                print(f"[{set_name}] {kind:7s} rate {rate:<6} accuracy {first:.3f} "
                      f"edge F1 {curves[kind][str(rate)]['reader']['f1']:.3f}", file=sys.stderr, flush=True)
        artifact["curves"][set_name] = curves
    if args.llm_reader.exists():
        llm = json.loads(args.llm_reader.read_text(encoding="utf-8"))
        lookup = {(r["edge_hash"], int(r["s"]), int(r["t"])): r for rows in sets.values() for r in rows}
        for model, res in llm["models"].items():
            artifact["matched_noise"][model] = {}
            for set_name, entry in res["sets"].items():
                rows = [lookup[(r["edge_hash"], int(r["s"]), int(r["t"]))] for r in llm["sets"][set_name]]
                rd, n_rows = entry["reader"], len(rows)
                true_edges = rd["edges_true_mean"] * n_rows
                tp = rd["recall"] * true_edges
                fp = max(0.0, rd["edges_read_mean"] * n_rows - tp)
                target = {"recall": rd["recall"], "extra_per_edge": fp / true_edges,
                          "reversed_share": min(1.0, rd["reversed_errors"] / fp) if fp else 0.0}
                graphs = graphs_of(rows)
                entries = []
                for draw in range(args.draws):
                    rng = random.Random(f"{SEED}/matched/{model}/{set_name}/{draw}")
                    read = {eh: matched(n, e, rng=rng, **target) for eh, (n, e) in graphs.items()}
                    entries.append(score(rows, read, solvers, SETS[set_name][1], args.device, with_true_graph=False))
                noise = summarise_draws(entries)
                artifact["matched_noise"][model][set_name] = {
                    "target": target,
                    "llm": {"reader_f1": rd["f1"],
                            "accuracy": {k: v["accuracy"]["mean"] for k, v in entry["pipeline"].items()}},
                    "matched": noise,
                }
                k0 = str(SETS[set_name][1][0])
                print(f"[matched] {model:34s} {set_name:12s} LLM acc {entry['pipeline'][k0]['accuracy']['mean']:.3f} "
                      f"(F1 {rd['f1']:.3f}) | matched noise acc {noise['pipeline'][k0]['accuracy']['mean']:.3f} "
                      f"(F1 {noise['reader']['f1']:.3f})", file=sys.stderr, flush=True)
    artifact["elapsed_seconds"] = time.perf_counter() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "out": args.out.as_posix(), "science_open": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

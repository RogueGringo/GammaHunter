# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Construction coverage: how reading held-out constructions scales with the frames trained on (MEASURE).

Fixed before any run. Under the PI's standing delegation (the PI asked "what
comes next?" after "lets make a small llm from the ground up ... you and i can
parse out some ideas that will convey some optimizations", having earlier
handed over the open choices with "infer the best path all context eval for
value based on prime intent"), the author chose this leg: the first
measurement of a data-design optimization with the small byte LM of
``tinylm`` (no pretrained weights). Items 19–20 found that readers read new
words in constructions they had seen almost completely, and unseen
constructions only partly and unevenly; in item 22, tuning a pretrained encoder
did not help. A pretrained encoder blurs what "new" means; here a held-out
frame is exactly unseen.

The model learns to read one sentence into its edge ("u>v", or "none" for a
distractor), trained on k frames of the 12-frame pool of ``tinylm.grammar`` (6
source-first, 6 target-first) and scored on every frame, the 4 held-out frames
included.

What k varies: frame coverage, i.e. constructions together with their words.
Only at k = 12 has every word and byte of every held-out frame been trained in
every run (tested), so only the k = 12 level is new in structure only for
certain; below it, a held-out frame may also contain words a run never saw
(some k = 8 runs may cover them all), and each run records how many of the
held-out frames' words its frames cover. Role order is
balanced at every level: an ordering alternates source-first and target-first
frames, so every prefix k in {2, 4, 8, 12} holds k/2 of each, and no run can
score by a fixed role order.

Design: k in {2, 4, 8, 12}; 6 orderings (seeded permutations of each order's
frames, interleaved, starting source-first in even orderings and target-first in
odd ones); for each ordering the first k frames (nested sets); 24 runs. In
ordering o every run starts from the same weights (seed o), so within an
ordering only the data differ across k.

Measures (exact match, greedy decoding, 400 fixed sentences per frame and 400
distractors; wrong answers sorted into reversed, "none", malformed and other):
``in_distribution`` (mean over the run's own frames); ``heldout_T`` (mean over
the three held-out target-first frames), the primary measure; ``heldout_S``
(the participial frame); ``heldout_word_coverage`` (the share of the held-out
frames' words that the run's frames use).

Controls, which read no structure: the order heuristic (the first number is the
source: 1 on source-first, 0 on target-first frames), the mirror heuristic (the
reverse), and two marker heuristics that read roles from familiar local markers
("from N", "to N", "into N", and "N" before a source verb). With prepositions
first, the marker heuristic reads every frame of this grammar, held-out frames
included, so a held-out frame read correctly is read at least as well as
familiar local markers allow, and this study cannot show reading beyond them;
with the subject position first, it misreads locative inversion, so reading that
frame takes more than trusting the subject position. Also: the untrained model on the
held-out frames, and the in-distribution check (a run reading less than 0.99 of
its own frames is undertrained).

Predictions and decision rules, fixed before any run (``--decide``):

* H1 (frame coverage helps): the mean ``heldout_T`` of the six k = 12 runs
  exceeds that of the six k = 2 runs by at least 0.30 with an exact one-sided
  permutation p < 0.05 (all 924 splits) — "supported". Before that:
  "uninterpretable (undertrained)" when two or more runs at k = 2 or at k = 12
  are undertrained. After it: "excluded" when a gain of 0.30 is rejected (the
  same test on the k = 12 values lowered by 0.30, p < 0.05), else
  "inconclusive", which supports neither side.
* The k = 12 level, classified by its mean ``heldout_T``: "uninterpretable
  (undertrained)" under the k = 12 half of that guard; otherwise "generalizes"
  at 0.9 or more,
  "partial" from 0.5, "fails" below 0.5 (a description fixed in advance, not a
  test).
* Reported, not decided: the held-out frames' accuracy and error kinds by k
  (every frame's per run), ``heldout_S``, word coverage, the four heuristics per
  frame, losses, timing,
  and, as a description only, the rank correlation between k and ``heldout_T``
  over all 24 runs (it pools runs that share starting weights within an
  ordering).

Power: with 6 runs per level the smallest attainable one-sided p is 1/924, and
p < 0.05 needs the k = 12 and k = 2 values to separate almost completely (at
most 46 of the 924 splits as extreme as the one observed); the observed gain
must itself reach 0.30. This model has no earlier runs, so the run-to-run
spread, and with it the power, is unknown before the study runs. Only H1 is
tested; no correction is applied.

Fixed choices (set by judgement, not tuned; no run informed them): the model
config ``CONFIG`` (4.9M parameters), 2,000 steps of 128 sentences, AdamW
(betas 0.9 and 0.95, weight decay 0.1 on every parameter) at 1e-3 with 100
warm-up steps and cosine decay, clipping at 1.0, bf16 autocast on CUDA, a
distractor rate of 0.2, the margins and thresholds above. "Before any run"
means before any run on the study's data: the test suite runs miniature
versions with its own evaluation and data seeds (orderings are part of the
design and shared); an early version of that smoke test wrote a stray run file
into the repository's artifacts, because the output path was bound when the
function was defined; it was deleted unread and the binding fixed.

``science_open=false`` always.

Usage::

    python -m reachability_gen.run_coverage --device cuda
    python -m reachability_gen.run_coverage --decide
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.run_selection_reader import permutation_p
from reachability_gen.tinylm import grammar
from reachability_gen.tinylm.grammar import FRAMES, HELD_OUT, POOL, eval_sets
from reachability_gen.tinylm.model import TinyConfig

KS: tuple[int, ...] = (2, 4, 8, 12)
ORDERINGS: int = 6
CONFIG = TinyConfig(d=256, layers=6, heads=8, mlp_hidden=704, max_len=128)
STEPS: int = 2000
BATCH: int = 128
LR: float = 1e-3
BETAS: tuple[float, float] = (0.9, 0.95)
WARMUP: int = 100
WEIGHT_DECAY: float = 0.1
CLIP: float = 1.0
DISTRACTOR_RATE: float = 0.2
EVAL_PER_FRAME: int = 400
EVAL_SEED: int = 2029
DATA_PREFIX: str = "data"
H1_MARGIN: float = 0.30
ALPHA_TEST: float = 0.05
INDIST_BAR: float = 0.99
UNDERTRAINED_LIMIT: int = 2
GENERALIZES: float = 0.9
PARTIAL: float = 0.5
HELD_OUT_T: tuple[str, ...] = tuple(n for n in HELD_OUT if FRAMES[n].order == "T")
HELD_OUT_S: tuple[str, ...] = tuple(n for n in HELD_OUT if FRAMES[n].order == "S")
HEURISTICS = {"order": grammar.order_heuristic, "mirror": grammar.mirror_heuristic,
              "marker": grammar.marker_heuristic, "subject_marker": grammar.subject_marker_heuristic}
RUNS_OUT = Path("artifacts/coverage_runs.json")
DECIDE_OUT = Path("artifacts/coverage.json")
# protocol entries that are the code's own constants (checked against a run file at --decide)
_CODE_KEYS = ("config", "steps", "batch", "lr", "betas", "warmup", "weight_decay", "clip", "distractor_rate",
              "eval_per_frame", "eval_seed", "data_prefix", "ks", "orderings", "pool", "held_out", "frames",
              "ordering_frames", "lexicon", "max_answer")


def ordering(o: int) -> list[str]:
    """Source-first and target-first frames, each order shuffled, interleaved (starting S when o is even)."""
    rng = random.Random(f"ordering/{o}")
    s = [n for n in POOL if FRAMES[n].order == "S"]
    t = [n for n in POOL if FRAMES[n].order == "T"]
    rng.shuffle(s)
    rng.shuffle(t)
    first, second = (s, t) if o % 2 == 0 else (t, s)
    return [x for pair in zip(first, second) for x in pair]


def _revision() -> dict[str, Any]:
    """The checked-out commit and whether tracked files differ from it (untracked run outputs do not count)."""
    here = Path(__file__).resolve().parent
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                             cwd=here).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                    text=True, check=True, cwd=here).stdout.strip())
        return {"code_revision": rev, "code_dirty": dirty}
    except Exception:  # noqa: BLE001 - not a git checkout
        return {"code_revision": None, "code_dirty": None}


def code_protocol() -> dict[str, Any]:
    """The protocol entries that are the code's own constants (what --decide checks a run file against)."""
    from reachability_gen.tinylm.extract import MAX_ANSWER

    return {"config": CONFIG.as_dict(), "steps": STEPS, "batch": BATCH, "lr": LR, "betas": list(BETAS),
            "warmup": WARMUP, "weight_decay": WEIGHT_DECAY, "weight_decay_on": "every parameter", "clip": CLIP,
            "distractor_rate": DISTRACTOR_RATE, "eval_per_frame": EVAL_PER_FRAME,
            "eval_seed": EVAL_SEED, "data_prefix": DATA_PREFIX, "ks": list(KS), "orderings": ORDERINGS,
            "pool": list(POOL), "held_out": list(HELD_OUT),
            "frames": {n: {"order": f.order, "pattern": f.pattern} for n, f in FRAMES.items()},
            "ordering_frames": [ordering(o) for o in range(ORDERINGS)],
            "lexicon": {"sv": list(grammar.SV), "pv": list(grammar.PV), "iv": list(grammar.IV),
                        "nouns": list(grammar.NOUNS), "to_into": list(grammar.TO_INTO),
                        "distractors": list(grammar.DISTRACTORS)},
            "max_answer": MAX_ANSWER}


def protocol(device: str) -> dict[str, Any]:
    import torch

    return {**code_protocol(), "bf16_autocast": device == "cuda", "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None, "torch_version": torch.__version__,
            "python_version": platform.python_version(), **_revision()}


def heuristic_scores(evals: dict[str, list[tuple[str, str]]]) -> dict[str, dict[str, float]]:
    return {h: {name: sum(fn(s) == a for s, a in rows) / len(rows) for name, rows in evals.items()}
            for h, fn in HEURISTICS.items()}


def heldout_word_coverage(frames: Sequence[str]) -> dict[str, float]:
    trained = set().union(*(grammar.frame_words(n) for n in frames))
    return {h: len(grammar.frame_words(h) & trained) / len(grammar.frame_words(h)) for h in HELD_OUT}


def run_one(o: int, k: int, evals: dict[str, list[tuple[str, str]]], device: str) -> dict[str, Any]:
    import torch

    from reachability_gen.tinylm.extract import score, train
    from reachability_gen.tinylm.model import TinyLM

    frames = tuple(ordering(o)[:k])
    torch.manual_seed(o)  # every run of an ordering starts from the same weights
    model = TinyLM(CONFIG).to(device)
    untrained = {name: score(model, evals[name], device) for name in HELD_OUT}
    log = train(model, frames, steps=STEPS, batch=BATCH, lr=LR, warmup=WARMUP, weight_decay=WEIGHT_DECAY,
                clip=CLIP, distractor_rate=DISTRACTOR_RATE, data_seed=f"{DATA_PREFIX}/{o}/{k}", device=device,
                betas=BETAS)
    scored = {name: score(model, rows, device) for name, rows in evals.items()}
    acc = {name: s["accuracy"] for name, s in scored.items()}
    coverage = heldout_word_coverage(frames)
    return {"ordering": o, "k": k, "frames": list(frames), "accuracy": acc,
            "errors": {name: s["errors"] for name, s in scored.items()},
            "in_distribution": statistics.fmean(acc[n] for n in frames),
            "heldout_T": statistics.fmean(acc[n] for n in HELD_OUT_T),
            "heldout_S": statistics.fmean(acc[n] for n in HELD_OUT_S),
            "distractor": acc["distractor"], "heldout_word_coverage": coverage,
            "untrained_heldout": {name: s["accuracy"] for name, s in untrained.items()},
            "training": log, "science_open": False}


def run_all(device: str, out: Optional[Path] = None) -> int:
    out = out or RUNS_OUT  # read at call time
    evals = eval_sets(EVAL_PER_FRAME, EVAL_SEED)
    artifact = {"science_open": False, "purpose": "frame coverage of a from-scratch byte LM reading edges",
                "protocol": protocol(device), "heuristics": heuristic_scores(evals), "runs": [], "complete": False}
    t0 = time.perf_counter()
    for o in range(ORDERINGS):
        for k in KS:
            run = run_one(o, k, evals, device)
            artifact["runs"].append(run)
            print(f"[ordering {o} k={k}] in-dist {run['in_distribution']:.3f} held-out T {run['heldout_T']:.3f} "
                  f"S {run['heldout_S']:.3f} distractor {run['distractor']:.3f} "
                  f"({run['training']['seconds']:.0f} s)", file=sys.stderr, flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")  # kept current after each run
    artifact["complete"] = True
    artifact["elapsed_seconds"] = time.perf_counter() - t0
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "out": out.as_posix(), "science_open": False}, sort_keys=True))
    return 0


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for t in range(i, j + 1):
                r[order[t]] = (i + j) / 2
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (sx * sy) if sx and sy else 0.0


def decide(art: dict[str, Any]) -> dict[str, Any]:
    """Apply the fixed rules (per-run values kept, ordered by ordering)."""
    runs = art["runs"]
    by_k = {k: sorted((r for r in runs if r["k"] == k), key=lambda r: r["ordering"]) for k in KS}
    t = {k: [r["heldout_T"] for r in by_k[k]] for k in KS}
    under = {k: sum(r["in_distribution"] < INDIST_BAR for r in by_k[k]) for k in KS}
    lo, hi = KS[0], KS[-1]
    gain = statistics.fmean(t[hi]) - statistics.fmean(t[lo])
    p = permutation_p(t[hi], t[lo])
    p_excluded = permutation_p(t[lo], [v - H1_MARGIN for v in t[hi]])
    if under[lo] >= UNDERTRAINED_LIMIT or under[hi] >= UNDERTRAINED_LIMIT:
        h1 = "uninterpretable (undertrained)"
    elif gain >= H1_MARGIN and p < ALPHA_TEST:
        h1 = "supported"
    elif p_excluded < ALPHA_TEST:
        h1 = "excluded"
    else:
        h1 = "inconclusive"
    level = statistics.fmean(t[hi])
    if under[hi] >= UNDERTRAINED_LIMIT:
        classification = "uninterpretable (undertrained)"
    else:
        classification = "generalizes" if level >= GENERALIZES else "partial" if level >= PARTIAL else "fails"

    def mean_errors(rows: list[dict[str, Any]], name: str) -> dict[str, float]:
        return {kind: statistics.fmean(r["errors"][name][kind] for r in rows)
                for kind in ("reversed", "none", "malformed", "other")}

    return {
        "H1": {"verdict": h1, "gain": gain, "p": p, "p_gain_below_margin": p_excluded,
               "heldout_T_by_k": {str(k): v for k, v in t.items()},
               "undertrained_by_k": {str(k): v for k, v in under.items()}},
        "k12_level": {"classification": classification, "mean_heldout_T": level},
        "reported": {
            "mean_by_k": {str(k): {
                "heldout_T": statistics.fmean(t[k]),
                "heldout_S": statistics.fmean(r["heldout_S"] for r in by_k[k]),
                "in_distribution": statistics.fmean(r["in_distribution"] for r in by_k[k]),
                "distractor": statistics.fmean(r["distractor"] for r in by_k[k]),
                "heldout_word_coverage": {h: statistics.fmean(r["heldout_word_coverage"][h] for r in by_k[k])
                                          for h in HELD_OUT},
                **{name: statistics.fmean(r["accuracy"][name] for r in by_k[k]) for name in HELD_OUT},
                "errors": {name: mean_errors(by_k[k], name) for name in HELD_OUT}} for k in KS},
            "spearman_k_heldout_T_descriptive": spearman([r["k"] for r in runs], [r["heldout_T"] for r in runs]),
            "runs": len(runs)},
    }


def check(art: dict[str, Any]) -> list[str]:
    issues = []
    if not art.get("complete"):
        issues.append("the run file is not complete")
    found = sorted((r["ordering"], r["k"]) for r in art["runs"])
    if found != sorted((o, k) for o in range(ORDERINGS) for k in KS):
        issues.append(f"runs present {found} differ from the design")
    expected = code_protocol()  # no device needed: only the code's constants are compared
    for key in _CODE_KEYS:
        if art["protocol"].get(key) != expected[key]:
            issues.append(f"protocol {key} differs from the code's")
    return issues


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Frame coverage of a from-scratch byte LM (MEASURE).")
    p.add_argument("--decide", action="store_true", help="apply the fixed rules to the run file")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args(argv)
    if args.decide:
        if not RUNS_OUT.exists():
            print(f"FAIL: missing {RUNS_OUT}", file=sys.stderr)
            return 1
        art = json.loads(RUNS_OUT.read_text(encoding="utf-8"))
        issues = check(art)
        if issues:
            print(f"FAIL: {issues}", file=sys.stderr)
            return 1
        result = {"science_open": False, "verdicts": decide(art), "heuristics": art["heuristics"]}
        DECIDE_OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"ok": True, "out": DECIDE_OUT.as_posix(), "science_open": False}, sort_keys=True))
        return 0
    return run_all(args.device)


if __name__ == "__main__":
    raise SystemExit(main())

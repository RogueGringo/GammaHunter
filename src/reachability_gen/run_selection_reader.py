# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Selection normalizers in the word reader (MEASURE; stage B of the selection leg, fixed before any run).

The word reader of item 19 (limitations), trained exactly as there (five
training templates, distractor rate 0.25, reader rate 1e-2, 5 epochs, 10 seeds),
with its attention normalized by one of ``softmax``, ``ssmax``, ``entmax15`` or
``sparsemax`` (``reachability_gen.selection``). Each run is scored on:

* ``val_in_0.25``, ``val_in_1``, ``val_in_4``: the validation graphs in the
  training wording with 0.25, 1 and 4 distractor sentences per edge (pure
  dilution: the wording is the one trained on);
* ``long_in``: the long-path graphs in the training wording (pure length);
* ``val_out``, ``long_out``: the held-out wording of item 19.

Predictions and decision rules, fixed before any run (``--decide`` applies them
to the four arms' result files):

* H1: the sparse arms (entmax15, sparsemax) lose less edge F1 than softmax from
  ``val_in_0.25`` to ``val_in_4``. Moot if softmax's mean drop is below 0.01.
* H2: the same comparison from ``val_in_0.25`` to ``long_in`` (at the first
  long-path step count; edge F1 does not depend on the solver's steps), with
  H1's moot rule.
* H3: no arm fixes the held-out wording. Measured by ``val_out`` edge F1 (not
  recall, which a reader that over-predicts edges can raise), for every
  non-softmax arm including SSMax: "fails" when the arm's mean exceeds
  softmax's by at least 0.05 with an exact one-sided permutation p < 0.05;
  "holds" when a gain of 0.05 or more is rejected (the same test on the arm's
  values shifted down by 0.05, p < 0.05); otherwise "inconclusive".
* H4: the sparse arms start slower: a higher median per-batch training loss
  over the first epoch than softmax. The median, not the epoch's mean loss:
  item 19's recorded epoch-1 mean losses (1.1–18.8 against untrained losses of
  0.35–1.2 per pair) are dominated by loss spikes, which a mean would measure
  instead of the start. Every per-batch loss is recorded.
* SSMax: exploratory, no directional prediction except its place in H3.

"Less", "higher": the arm's mean over seeds differs from softmax's in the
predicted direction and an exact one-sided permutation test on the difference
of means over the per-seed values (all 184,756 splits of 10 + 10) gives
p < 0.05. No correction is made for the several tests; every p is reported.

Controls that could produce a verdict without the normalizer:

* H1, H2: a smaller drop could come from a lower starting point, so an arm
  whose mean ``val_in_0.25`` edge F1 is more than 0.01 below softmax's is
  reported as confounded instead of supported.
* H4: a higher epoch-1 loss could come from a higher loss before training, so an
  arm whose mean untrained topology loss on ``val_in_0.25`` (the same per-pair
  loss, in the training wording) is more than 0.01 above softmax's is reported
  as confounded instead of supported. It could also come from a loss that
  stays higher throughout (a worse fit rather than a slower start), so an arm
  is also reported as confounded when its gap to softmax in the median
  per-batch loss of the last epoch is at least half its gap in the first
  (the gap has not closed by half). Loss spikes are handled by the median; the
  number of epoch-1 batches with a loss above 1 is reported per seed.
* Recorded with every run: the untrained reader on ``val_in_0.25``,
  ``val_in_4``, ``long_in`` and ``val_out``; the solver on the true graphs
  (``oracle``); and, for every arm with a normalizer module (all but softmax,
  which keeps the original code path), the attention support actually used
  (mean number and fraction of unmasked key positions with weight above 0 per
  layer, one rendering per graph of the same four sets, before and after
  training), so that a dilution or length effect can be set against how sparse
  the attention was.

Power: item 19's per-seed held-out edge F1 has a standard deviation of 0.22, so
with 10 seeds per arm the difference of two means has a standard error of about
0.10. If an arm truly equals softmax, H3 "holds" with probability of only about
0.1 and "fails" with about 0.05; "inconclusive" is the expected outcome then,
and is support for neither side.

Replication of item 19 (fixed before any run): the softmax arm is compared seed
by seed with item 19's recorded word reader on ``val_in`` (here
``val_in_0.25``) and ``val_out`` (edge F1, precision, recall, exact graphs) and
on item 19's pass criteria. "exact" when every value matches; otherwise
"in distribution" when, for edge F1 on both sets, the mean difference is below
0.05 and an exact two-sided permutation test gives p ≥ 0.05; otherwise "not
replicated". Item 19's numbers enter no verdict: H1–H4 compare arms run in this
code, so a failed replication is reported beside them, not used to change them.

``--decide`` refuses to decide unless the four result files have the same seeds,
the same protocol, clean self-audits and verified datasets.

Departures from the queued specification, fixed here before any run: H2 takes
H1's moot rule; the confound guards for H1, H2 and H4; H3's metric, its
three-way outcome and SSMax's place in it (the specification fixed only its
0.05 margin); H4's statistic, the median per-batch loss of the first epoch
rather than the epoch's mean loss; the replication criterion. SSMax's scale starts at 1, without the
optional bias of the paper that defines it, and like every reader parameter it
is under AdamW weight decay (0.01), which pulls it towards 0 wherever its
gradient does not; the learned scales are recorded. "Before any run" means
before any run on the study's data: the test suite runs miniature versions
(two seeds, forty rows, one epoch) into temporary directories.

``science_open=false`` always.

Usage::

    python -m reachability_gen.run_selection_reader --arm softmax --device cuda
    python -m reachability_gen.run_selection_reader --decide
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.gen_crossed import CROSSED_EXTENDED_SPEC, CROSSED_ID_SPEC, verify_crossed
from reachability_gen.nl_render import DISTRACTORS, TEMPLATES, build_vocab, render, word_tokens
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_llm_reader import graphs_of
from reachability_gen.run_mp_calibration import BATCH, GRAD_CLIP, LR, WEIGHT_DECAY
from reachability_gen.run_nl_reader import (
    DEFAULT_EXTENDED,
    DEFAULT_TRAIN,
    MAX_OFFSET,
    PASS,
    READER_EPOCHS,
    NLData,
    train_reader,
)
from reachability_gen.run_reader import (
    LONG_STEPS,
    READER_LR,
    SOLVER_EPOCHS,
    TRAIN_STEPS,
    build_solver,
    evaluate,
    train_solver,
)

ARMS: tuple[str, ...] = ("softmax", "ssmax", "entmax15", "sparsemax")
SPARSE: tuple[str, ...] = ("entmax15", "sparsemax")
DILUTION_RATES: tuple[float, ...] = (0.25, 1.0, 4.0)
TRAIN_RATE: float = 0.25
MOOT_DROP: float = 0.01
START_GAP: float = 0.01  # H1/H2: starting F1 more than this below softmax's; H4: untrained loss more than this above
END_SHRINK: float = 0.5  # H4: the last epoch's loss gap must be below this fraction of the first epoch's
H3_MARGIN: float = 0.05
REPLICATION_MARGIN: float = 0.05
ALPHA_TEST: float = 0.05
ITEM19 = Path("artifacts/nl_reader_words.json")
SUPPORT_SETS: tuple[str, ...] = ("val_in_0.25", "val_in_4", "long_in", "val_out")
# Protocol entries that must agree across the four result files before --decide
SHARED_PROTOCOL: tuple[str, ...] = (
    "train_wording", "train_distractor_rate", "dilution_rates", "sets", "max_offset", "train_steps", "long_steps",
    "solver_epochs", "reader_epochs", "reader_lr", "solver_lr", "batch_size", "weight_decay", "grad_clip", "seeds",
    "device", "torch_version", "dataset_verified")
HYPOTHESES: dict[str, str] = {
    "H1": "sparse arms lose less edge F1 than softmax from val_in_0.25 to val_in_4; moot if softmax's mean drop "
          "< 0.01; confounded if the arm's mean val_in_0.25 F1 is > 0.01 below softmax's",
    "H2": "sparse arms lose less edge F1 than softmax from val_in_0.25 to long_in; moot if softmax's mean drop "
          "< 0.01; confounded if the arm's mean val_in_0.25 F1 is > 0.01 below softmax's",
    "H3": "no arm fixes the held-out wording: val_out edge F1 gain over softmax of 0.05 or more rejected (holds), "
          "shown (fails), or neither (inconclusive); every non-softmax arm",
    "H4": "sparse arms start slower: higher median per-batch epoch-1 training loss than softmax; confounded if the "
          "arm's mean untrained val_in_0.25 topology loss is > 0.01 above softmax's, or if its last-epoch median loss "
          "gap is at least half its first-epoch gap",
    "SSMax": "exploratory, no directional prediction except its place in H3",
}


def out_path(arm: str) -> Path:
    return Path(f"artifacts/selection_reader_{arm}.json")


def permutation_p(a: Sequence[float], b: Sequence[float]) -> float:
    """Exact one-sided p for mean(a) - mean(b) being at least as large as observed, over all equal splits."""
    pooled, k = list(a) + list(b), len(a)
    observed = statistics.fmean(a) - statistics.fmean(b)
    total = sum(pooled)
    hits = count = 0
    for idx in itertools.combinations(range(len(pooled)), k):
        sa = sum(pooled[i] for i in idx)
        diff = sa / k - (total - sa) / (len(pooled) - k)
        hits += diff >= observed - 1e-12
        count += 1
    return hits / count


def permutation_p_two_sided(a: Sequence[float], b: Sequence[float]) -> float:
    return min(1.0, 2 * min(permutation_p(a, b), permutation_p(b, a)))


def sets_for(rows: list[dict[str, Any]], ext_rows: list[dict[str, Any]]) -> dict[str, tuple[list, str, float]]:
    """Name -> (rows, template split, distractor rate) of every set a run reads."""
    val = [r for r in rows if r["split"] == "val"]
    out = {f"val_in_{rate:g}": (val, "train", rate) for rate in DILUTION_RATES}
    out.update({"long_in": (ext_rows, "train", TRAIN_RATE), "val_out": (val, "heldout", TRAIN_RATE),
                "long_out": (ext_rows, "heldout", TRAIN_RATE)})
    return out


def attention_support(reader, data) -> Optional[dict[str, Any]]:
    """Mean attention support per layer: unmasked key positions with weight > 0, over valid query positions.

    One rendering per graph (the first row of each). None when the reader keeps
    the original softmax code path, which has no normalizer module to observe.
    """
    import torch

    mods = [layer.normalizer for layer in reader.layers]
    if any(m is None for m in mods):
        return None
    sums = [[0.0, 0.0, 0] for _ in mods]  # support size, support fraction, query positions

    def hook(i):
        def fn(_module, inputs, out):
            keys = inputs[1][:, 0, 0, :]  # [B, L] unmasked keys (= valid positions)
            size = (out > 0).sum(-1).float()  # [B, heads, L]
            frac = size / keys.sum(-1).float()[:, None, None]
            valid = keys[:, None, :].expand_as(size)
            sums[i][0] += size[valid].sum().item()
            sums[i][1] += frac[valid].sum().item()
            sums[i][2] += int(valid.sum())
        return fn

    first: dict[str, int] = {}
    for i, r in enumerate(data.rows):
        first.setdefault(r["edge_hash"], i)
    idx = list(first.values())
    handles = [m.register_forward_hook(hook(i)) for i, m in enumerate(mods)]
    reader.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(idx), data.eval_batch):
                _, tokens = data.batch(idx[start:start + data.eval_batch])
                reader.edge_log_evidence(tokens)
    finally:
        for h in handles:
            h.remove()
    return {"graphs": len(idx),
            "layers": [{"mean_support": s / q, "mean_support_fraction": f / q, "query_positions": q}
                       for s, f, q in sums]}


def passes_item19(final: dict[str, Any]) -> bool:
    """Item 19's pass criteria on the held-out wording, unchanged."""
    first = LONG_STEPS[0]
    return (final["val_out"]["reader"]["exact_graphs"] >= PASS["exact_graphs_heldout_val"]
            and final[f"long_out_{first}"]["reader"]["closure_agreement_all_pairs"]
            >= PASS["closure_agreement_heldout_long"]
            and all(final[f"long_out_{s}"]["accuracy"] >= PASS["long_path_accuracy_heldout"] for s in LONG_STEPS))


def run_arm(args) -> int:
    import torch

    from reachability_gen.models.reader import GraphReader, collate_tokens

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
    train_rows = [r for r in rows if r["split"] == "train"]
    vocab = build_vocab(TEMPLATES["train"] + DISTRACTORS)

    def tokens(rs, split, rate):
        return {eh: word_tokens(render(n, e, split, eh, rate), n, vocab) for eh, (n, e) in graphs_of(rs).items()}

    train = NLData(train_rows, args.device, tokens(train_rows, "train", TRAIN_RATE), collate_tokens)
    data = {name: NLData(rs, args.device, tokens(rs, split, rate), collate_tokens)
            for name, (rs, split, rate) in sets_for(rows, ext_rows).items()}

    def make_reader():
        return GraphReader(vocab_size=len(vocab) + 2, max_offset=MAX_OFFSET, normalizer=args.arm)

    runs, t0 = [], time.perf_counter()
    for seed in args.seeds:
        t1 = time.perf_counter()
        torch.manual_seed(seed)
        reader, solver = make_reader().to(args.device), build_solver().to(args.device)
        train_solver(solver, train, epochs=args.solver_epochs, device=args.device)
        untrained = {name: evaluate(reader, solver, data[name], TRAIN_STEPS) for name in SUPPORT_SETS}
        untrained_support = {name: attention_support(reader, data[name]) for name in SUPPORT_SETS}
        batch_losses: list[list[float]] = []
        history = train_reader(reader, solver, train, data["val_out"], epochs=args.reader_epochs,
                               reader_lr=args.reader_lr, label=f"{args.arm}/seed{seed}", device=args.device,
                               batch_losses=batch_losses)
        path = args.ckpt_dir / out_path(args.arm).stem / f"{args.arm}_seed{seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"reader": reader.state_dict(), "solver": solver.state_dict(), "seed": seed, "arm": args.arm,
                    "science_open": False}, path)
        fresh, fresh_solver = make_reader().to(args.device), build_solver().to(args.device)
        state = torch.load(path, map_location="cpu", weights_only=True)
        fresh.load_state_dict(state["reader"])
        fresh_solver.load_state_dict(state["solver"])
        final = {name: evaluate(fresh, fresh_solver, d, TRAIN_STEPS) for name, d in data.items()
                 if name.startswith("val")}
        for name in ("long_in", "long_out"):
            final.update({f"{name}_{s}": evaluate(fresh, fresh_solver, data[name], s) for s in LONG_STEPS})
        run = {"seed": seed, "arm": args.arm, "science_open": False, "history": history, "final": final,
               "batch_losses": batch_losses,
               "untrained_reader": untrained, "passes_item19_criteria": passes_item19(final),
               "attention_support": {"trained": {name: attention_support(fresh, data[name]) for name in SUPPORT_SETS},
                                     "untrained": untrained_support},
               "oracle": {"val_out": evaluate(fresh, fresh_solver, data["val_out"], TRAIN_STEPS, oracle=True)},
               "checkpoint_path": path.as_posix(), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "rescore_matches_record": final["val_out"]["accuracy"] == history[-1]["val_accuracy"],
               "seconds": time.perf_counter() - t1}
        if args.arm == "ssmax":
            run["ssmax_scales"] = [layer.normalizer.scale.detach().cpu().tolist() for layer in fresh.layers]
        runs.append(run)
        print(f"[{args.arm}/seed{seed}] F1 in@0.25 {final['val_in_0.25']['reader']['f1']:.3f} "
              f"in@4 {final['val_in_4']['reader']['f1']:.3f} "
              f"long_in {final[f'long_in_{LONG_STEPS[0]}']['reader']['f1']:.3f} "
              f"out {final['val_out']['reader']['f1']:.3f}", file=sys.stderr, flush=True)
    mismatches = [f"seed{r['seed']}" for r in runs if not r["rescore_matches_record"]]
    artifact = {
        "science_open": False,
        "purpose": "selection normalizers in the item-19 word reader: dilution, length and held-out wording",
        "arm": args.arm,
        "hypotheses": HYPOTHESES,
        "protocol": {
            "train_wording": "train", "train_distractor_rate": TRAIN_RATE, "dilution_rates": list(DILUTION_RATES),
            "sets": {name: {"template_split": split, "distractor_rate": rate}
                     for name, (_, split, rate) in sets_for(rows, ext_rows).items()},
            "reader_params": sum(p.numel() for p in make_reader().parameters()), "max_offset": MAX_OFFSET,
            "train_steps": TRAIN_STEPS, "long_steps": list(LONG_STEPS), "solver_epochs": args.solver_epochs,
            "reader_epochs": args.reader_epochs, "reader_lr": args.reader_lr, "solver_lr": LR, "batch_size": BATCH,
            "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "seeds": args.seeds, "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "torch_version": torch.__version__, "checkpoints_versioned": False, "dataset_verified": not args.no_verify,
            "eval_batch_rows": {k: d.eval_batch for k, d in data.items()},
        },
        "runs": runs,
        "self_audit_mismatches": mismatches,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    path = args.out or out_path(args.arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": not mismatches, "out": path.as_posix(), "science_open": False}, sort_keys=True))
    return 1 if mismatches else 0


def check_arts(arts: dict[str, dict[str, Any]]) -> list[str]:
    """Reasons the four result files cannot be decided together (empty when they can)."""
    issues = []
    ref = arts["softmax"]
    seeds = sorted(r["seed"] for r in ref["runs"])
    for arm, art in arts.items():
        if art.get("arm") != arm:
            issues.append(f"{arm}: file records arm {art.get('arm')!r}")
        if sorted(r["seed"] for r in art["runs"]) != seeds:
            issues.append(f"{arm}: seeds differ from softmax's")
        if art.get("self_audit_mismatches"):
            issues.append(f"{arm}: self-audit mismatches {art['self_audit_mismatches']}")
        for key in SHARED_PROTOCOL:
            if art["protocol"].get(key) != ref["protocol"].get(key):
                issues.append(f"{arm}: protocol {key} differs from softmax's")
        if art["protocol"].get("dataset_verified") is not True:
            issues.append(f"{arm}: datasets not verified (--no-verify is for tests only)")
    return issues


def per_seed(art: dict[str, Any], fn) -> list[float]:
    return [fn(r) for r in sorted(art["runs"], key=lambda r: r["seed"])]


def decide(arts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the fixed decision rules to the four arms' results (per-seed values kept, ordered by seed)."""
    first = LONG_STEPS[0]
    seeds = sorted(r["seed"] for r in arts["softmax"]["runs"])

    def drop(set_b):
        return lambda r: r["final"]["val_in_0.25"]["reader"]["f1"] - r["final"][set_b]["reader"]["f1"]

    def start(r):
        return r["final"]["val_in_0.25"]["reader"]["f1"]

    out: dict[str, Any] = {"seeds": seeds, "splits_per_test": math.comb(2 * len(seeds), len(seeds))}
    for h, set_b in (("H1", "val_in_4"), ("H2", f"long_in_{first}")):
        soft = per_seed(arts["softmax"], drop(set_b))
        soft_start = per_seed(arts["softmax"], start)
        verdicts = {}
        for arm in SPARSE:
            vals = per_seed(arts[arm], drop(set_b))
            arm_start = per_seed(arts[arm], start)
            if statistics.fmean(soft) < MOOT_DROP:
                verdict, p = "moot (softmax's mean drop is below 0.01)", None
            else:
                p = permutation_p(soft, vals)  # softmax's drop larger than the arm's
                supported = statistics.fmean(vals) < statistics.fmean(soft) and p < ALPHA_TEST
                verdict = "supported" if supported else "not supported"
                if supported and statistics.fmean(arm_start) < statistics.fmean(soft_start) - START_GAP:
                    verdict = "confounded (lower val_in_0.25 F1 than softmax)"
            verdicts[arm] = {"verdict": verdict, "p": p, "arm_mean_drop": statistics.fmean(vals),
                             "softmax_mean_drop": statistics.fmean(soft), "arm_drops": vals, "softmax_drops": soft,
                             "arm_start_f1": arm_start, "softmax_start_f1": soft_start}
        out[h] = verdicts

    def f1_out(r):
        return r["final"]["val_out"]["reader"]["f1"]

    soft_f1 = per_seed(arts["softmax"], f1_out)
    out["H3"] = {}
    for arm in (a for a in ARMS if a != "softmax"):
        vals = per_seed(arts[arm], f1_out)
        diff = statistics.fmean(vals) - statistics.fmean(soft_f1)
        p_gain = permutation_p(vals, soft_f1)  # the arm reads the held-out wording better
        p_below = permutation_p(soft_f1, [v - H3_MARGIN for v in vals])  # a gain of H3_MARGIN or more rejected
        if diff >= H3_MARGIN and p_gain < ALPHA_TEST:
            verdict = "fails"
        elif p_below < ALPHA_TEST:
            verdict = "holds"
        else:
            verdict = "inconclusive"
        out["H3"][arm] = {"verdict": verdict, "difference": diff, "p_gain": p_gain, "p_gain_below_margin": p_below,
                          "arm_f1": vals, "softmax_f1": soft_f1}

    def loss1(r):
        return statistics.median(r["batch_losses"][0])

    def loss_last(r):
        return statistics.median(r["batch_losses"][-1])

    def spikes1(r):
        return sum(1 for v in r["batch_losses"][0] if v > 1.0)

    def untrained_loss(r):
        return r["untrained_reader"]["val_in_0.25"]["reader"]["topology_bce"]

    soft_loss, soft_init = per_seed(arts["softmax"], loss1), per_seed(arts["softmax"], untrained_loss)
    soft_last = per_seed(arts["softmax"], loss_last)
    out["H4"] = {}
    for arm in SPARSE:
        vals, init = per_seed(arts[arm], loss1), per_seed(arts[arm], untrained_loss)
        last = per_seed(arts[arm], loss_last)
        gap_first = statistics.fmean(vals) - statistics.fmean(soft_loss)
        gap_last = statistics.fmean(last) - statistics.fmean(soft_last)
        p = permutation_p(vals, soft_loss)
        supported = gap_first > 0 and p < ALPHA_TEST
        verdict = "supported" if supported else "not supported"
        if supported and statistics.fmean(init) > statistics.fmean(soft_init) + START_GAP:
            verdict = "confounded (higher untrained loss than softmax)"
        elif supported and gap_last >= END_SHRINK * gap_first:
            verdict = "confounded (the loss gap persists to the last epoch)"
        out["H4"][arm] = {"verdict": verdict, "p": p, "arm_mean": statistics.fmean(vals),
                          "softmax_mean": statistics.fmean(soft_loss), "arm_epoch1_median_loss": vals,
                          "softmax_epoch1_median_loss": soft_loss, "arm_untrained_loss": init,
                          "softmax_untrained_loss": soft_init, "gap_first_epoch": gap_first,
                          "gap_last_epoch": gap_last, "arm_last_epoch_median_loss": last,
                          "softmax_last_epoch_median_loss": soft_last,
                          "arm_epoch1_batches_above_1": per_seed(arts[arm], spikes1),
                          "softmax_epoch1_batches_above_1": per_seed(arts["softmax"], spikes1)}
    return out


def replication(soft_art: dict[str, Any], item19: dict[str, Any]) -> dict[str, Any]:
    """The softmax arm against item 19's recorded word reader, seed by seed, with the fixed criterion."""
    ref = {r["seed"]: r for r in item19["runs"]}
    mine = {r["seed"]: r for r in soft_art["runs"]}
    seeds = sorted(set(ref) & set(mine))
    out: dict[str, Any] = {"seeds": seeds}
    exact = bool(seeds) and set(ref) == set(mine)
    in_dist = bool(seeds)
    for ours, theirs in (("val_in_0.25", "val_in"), ("val_out", "val_out")):
        block = {}
        for key in ("f1", "precision", "recall", "exact_graphs"):
            a = [mine[s]["final"][ours]["reader"][key] for s in seeds]
            b = [ref[s]["final"][theirs]["reader"][key] for s in seeds]
            block[key] = {"softmax_arm": dict(zip(map(str, seeds), a)), "item19": dict(zip(map(str, seeds), b))}
            exact = exact and a == b
            if key == "f1" and seeds:
                diff = statistics.fmean(a) - statistics.fmean(b)
                p = permutation_p_two_sided(a, b)
                block[key].update({"mean_difference": diff, "p_two_sided": p})
                in_dist = in_dist and abs(diff) < REPLICATION_MARGIN and p >= ALPHA_TEST
        out[ours] = block
    passes = {"softmax_arm": {str(s): mine[s]["passes_item19_criteria"] for s in seeds},
              "item19": {str(s): ref[s]["passes"] for s in seeds}}
    out["passes"] = passes
    exact = exact and passes["softmax_arm"] == passes["item19"]
    out["verdict"] = "exact" if exact else "in distribution" if in_dist else "not replicated"
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Selection normalizers in the word reader (MEASURE).")
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--decide", action="store_true", help="apply the fixed decision rules to the four result files")
    p.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--extended-data", type=Path, default=DEFAULT_EXTENDED)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--reader-epochs", type=int, default=READER_EPOCHS)
    p.add_argument("--solver-epochs", type=int, default=SOLVER_EPOCHS)
    p.add_argument("--reader-lr", type=float, default=READER_LR)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/selection_reader"))
    p.add_argument("--no-verify", action="store_true", help="skip dataset verification (tests only)")
    args = p.parse_args(argv)
    if args.decide:
        missing = [out_path(a).as_posix() for a in ARMS if not out_path(a).exists()]
        if missing:
            print(f"FAIL: missing {missing}", file=sys.stderr)
            return 1
        arts = {a: json.loads(out_path(a).read_text(encoding="utf-8")) for a in ARMS}
        issues = check_arts(arts)
        if issues:
            print(f"FAIL: the result files cannot be decided together: {issues}", file=sys.stderr)
            return 1
        result = {"science_open": False, "hypotheses": HYPOTHESES,
                  "decision_rule": f"arm mean differs from softmax's in the predicted direction and an exact one-sided "
                                   f"permutation test on per-seed values gives p < {ALPHA_TEST}; H1/H2 moot when "
                                   f"softmax's mean drop < {MOOT_DROP}, confounded when the arm's mean val_in_0.25 "
                                   f"F1 is more than {START_GAP} below softmax's; H3 on val_out edge F1: fails when "
                                   f"the gain is >= {H3_MARGIN} with p < {ALPHA_TEST}, holds when a gain of "
                                   f"{H3_MARGIN} or more is rejected at p < {ALPHA_TEST}, else inconclusive; H4 on "
                                   f"the median per-batch epoch-1 loss, confounded when the last-epoch gap is at "
                                   f"least {END_SHRINK} of the first-epoch gap or when the arm's mean untrained val_in_0.25 loss is more than "
                                   f"{START_GAP} above softmax's",
                  "verdicts": decide(arts)}
        if ITEM19.exists():
            result["replication_of_item19"] = replication(arts["softmax"],
                                                          json.loads(ITEM19.read_text(encoding="utf-8")))
        out = args.out or Path("artifacts/selection_reader.json")
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"ok": True, "out": out.as_posix(), "science_open": False}, sort_keys=True))
        return 0
    if not args.arm:
        p.error("choose --arm or --decide")
    return run_arm(args)


if __name__ == "__main__":
    raise SystemExit(main())

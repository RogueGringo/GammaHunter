# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Embedding and representation tuning for reading unseen constructions (MEASURE; fixed before any run).

The PI's hypothesis, recorded verbatim: "Prolly a combo of embedding and
representation tuning across all subsets in modern convention." The PI asked
for LoReFT-style interventions with tuned embeddings, ablations, optionally
entmax-gated intervention sites, and a pass defined as item 19's criteria plus
recall ≥ 0.99 on "Starting at {u}, one step takes you to {v}." plus a novel-set
score, and then delegated the open choices ("infer the best path all context
eval for value based on prime intent", verbatim). Choices made under that
delegation, with their reasons:

1. "All subsets" is read as every block below the read layer and every
   position, not every wording family: training on the held-out or novel
   wordings would leave nothing unseen to test.
2. The primary measure is recall on the construction, not the PI's pass: item
   20's reader read at most 95% of the validation graphs exactly even in its own
   training wording (0.73–0.95 per seed), below item 19's 0.99, so a pass count
   would be about 0 in every arm whatever tuning does. The pass and the novel-set
   score (mean recall over the four novel templates; bar 0.90) are reported per
   seed, not decided.
3. The entmax-gated arm is dropped: it was conditional on item 21's dilution
   hypothesis (H1) being supported, and H1 came out moot.
4. Two parameter-matched controls are added (``capacity``, ``interventions_matched``).
5. Learning rates are fixed without tuning (below); every seed enters every
   verdict.

Reading of the hypothesis: the language-model reader of item 20 (limitations)
fails on the one held-out construction absent from its training wordings (recall
0.37–0.50 per seed) because its encoder is frozen; tuning the encoder's input
embeddings together with low-rank representation interventions at every block
below the read layer and every position (``tuned_encoder``), in the
parameter-efficient way now conventional, lets it read that construction.

Arms (10 seeds each; item 20's protocol otherwise: the 51 diverse training
wordings, distractor rate 0.25, reader rate 1e-2, 5 epochs, the frozen solvers of
item 15):

* ``frozen``: item 20's reader on cached frozen features (a replication);
* ``embeddings``: the input embeddings of the tokens that occur in training
  tuned (every other token keeps its embedding);
* ``interventions``: LoReFT interventions of rank 4 at the output of each of the
  12 blocks, every position, tied across positions;
* ``interventions_matched``: the same with the rank whose parameter count is
  closest to what ``both`` adds (rank 7: 150,612 against 159,536), so that
  "a combo" is not credited for having more parameters than one component;
* ``both``: the hypothesis (embeddings and rank-4 interventions);
* ``capacity``: the frozen encoder with a wider reader, whose extra trainable
  parameters match what ``both`` adds (could more parameters alone do it?).

Scored on the validation graphs in the training wording (``val_in``), in item
19's four held-out templates (``val_out``; ``long_out`` at 16, 48 and 192
steps) and in item 20's four novel templates (``val_novel``, ``long_novel``),
with recall per template. The primary measure is recall on the construction
(held-out template 2) on the validation graphs.

Predictions and decision rules, fixed before any run (``--decide``). "Beats X by
m" means: the mean over seeds is higher by at least m and an exact one-sided
permutation test over all equal splits of the per-seed values gives p < 0.05;
"a gain of m over X is rejected" means the same test on the values lowered by m
(p < 0.05). A run whose ``val_in`` edge F1 ends below 0.5 counts as collapsed.

* P1 (the hypothesis): ``both`` beats ``frozen`` and ``capacity`` by 0.30 —
  "supported", unless a guard applies: "confounded (in-distribution)" when its
  mean ``val_in`` edge F1 is more than 0.01 below ``frozen``'s (a gain bought
  with the reading of the training wording), "confounded (over-prediction)"
  when its mean ``val_out`` precision is more than 0.05 below ``frozen``'s
  (recall bought by listing more edges), "confounded (degraded control)" when
  the mean ``val_in`` edge F1 of ``frozen`` or ``capacity`` is more than 0.01
  below ``both``'s (a control whose collapsed or degraded seeds pull its mean
  down). When it beats ``frozen`` but not ``capacity`` by 0.30: "not separated
  from capacity" (its gain over the parameter-matched reader did not reach the
  margin, so the specific claim over added reader parameters is not shown; how
  much of the gain the capacity arm reaches is in the reported comparisons).
  Otherwise, "uninterpretable (training collapsed)" when 3 or more of its 10
  seeds collapsed; else "excluded at these rates" when a gain of 0.30 over
  ``frozen`` is rejected; else "inconclusive", which supports neither side.
* P2 ("a combo"): ``both`` against ``embeddings``, ``interventions`` and
  ``interventions_matched`` — "shown" when it beats each by 0.10, unless a guard
  applies: "confounded (collapsed component)" when 3 or more seeds of a
  component arm collapsed, "confounded (degraded component)" when a
  component's mean ``val_in`` edge F1 is more than 0.01 below ``both``'s,
  "confounded (in-distribution)" or "confounded (over-prediction)" when
  ``both``'s mean ``val_in`` edge F1 or ``val_out`` precision is more than 0.01
  or 0.05 below a component's (each label names the arms concerned). Otherwise
  "uninterpretable (training collapsed)" under the collapse rule for ``both``;
  else "excluded" when a gain of 0.10 over one of the components is rejected at
  p < 0.05/3 (any one of three tests can fire, so each is Bonferroni-corrected);
  else "inconclusive". The embeddings arm cannot be
  parameter-matched (only the tokens that occur in training have a delta), so
  beating it does not separate the combination from its larger parameter count;
  beating ``interventions_matched`` does, for the interventions.
* Reported, not decided: the PI's pass per seed, the novel-set score per seed,
  every arm against ``frozen`` on the construction, collapses, per-batch
  losses, and how far training moved each adapter.
* Replication (as in item 21): ``frozen`` against item 20's recorded reader,
  seed by seed, on held-out edge F1 and recall on the construction: "exact" when
  every per-seed value matches; else "in distribution" when both mean
  differences are below 0.05 and an exact two-sided permutation test gives
  p ≥ 0.05; else "not replicated". Item 20's numbers enter no verdict.

Power and multiplicity: item 20's per-seed recall on the construction has a
standard deviation of 0.05, so with 10 seeds per arm the standard error of a
difference is about 0.02 at that spread. Because "supported" needs the observed
gains themselves to reach 0.30, a true gain of exactly 0.30 over both ``frozen``
and ``capacity`` would be detected only about a third of the time (the two
comparisons share ``both``'s values), a true gain of 0.35 about 97% of the time
(normal approximation); seeds made more variable by tuning (item 21 found the
reader rate unstable) would lower the latter. "Supported", "shown" and P1's "excluded" need every one of
their tests, so no correction is applied to them; P2's "excluded" needs any one
of three and is corrected as above; reported comparisons are not decided.

Fixed choices, not tuned (no run informed them; the constants below, like the
rank of 4, the margins of 0.30 and 0.10, the guard sizes of 0.01 and 0.05 and
the collapse rule of F1 below 0.5 in 3 of 10 seeds, were set by judgement, not
derived): interventions at rate 1e-3; embedding deltas at 0.01 × the RMS of the
encoder's embedding matrix (the rate
is set on the embeddings' scale; how far the deltas moved is recorded per run);
no weight decay on either; each component's gradients clipped on its own at the
reader's norm; the reader's rate, weight decay and clipping as in item 20. The
interventions depart from the paper in form (``tuned_encoder``: W = R + D with D
and b starting at 0, which changes the training dynamics, and R obtained by QR),
and act at every position (the paper intervenes on chosen prefix and suffix
positions), because every sentence of a rendering carries edges. Every tuned arm
starts bit for bit as ``frozen``: each run audits its fresh adapter on 16
held-out renderings in evaluation and on 32 training renderings through the
training path (train mode, checkpointing, one batch), against ``lm_features``.
The capacity arm's wider reader draws its initialisation from the same stream
as its solver, so its solver and batch order differ from ``frozen``'s for the
same seed (the comparisons are between distributions, not paired).

"Before any run" means before any run on the study's data: the test suite runs
miniature versions with a tiny random model, and engineering checks on the real
encoder (forward and backward passes, no training) timed one training batch
(about 0.6 s at the longest renderings) and confirmed the exact starts above
(which a free W copied from R did not give, prompting the form used).

``science_open=false`` always.

Usage::

    python -m reachability_gen.run_tuned_reader --arm frozen --device cuda
    python -m reachability_gen.run_tuned_reader --decide
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.gen_crossed import CROSSED_EXTENDED_SPEC, CROSSED_ID_SPEC, verify_crossed
from reachability_gen.nl_render import TEMPLATES, render
from reachability_gen.nl_template_audit import recall_by_template, template_of_edges
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_llm_reader import graphs_of
from reachability_gen.run_mp_calibration import BATCH, GRAD_CLIP, LR, WEIGHT_DECAY
from reachability_gen.run_nl_reader import (
    DEFAULT_ENCODER,
    DEFAULT_EXTENDED,
    DEFAULT_TRAIN,
    ENCODER_LAYER,
    MAX_OFFSET,
    PASS,
    READER_EPOCHS,
    NLData,
    lm_features,
    train_reader,
)
from reachability_gen.run_reader import (
    LONG_STEPS,
    READER_LR,
    SOLVER_EPOCHS,
    TRAIN_STEPS,
    build_solver,
    evaluate,
    topology_bce,
    train_solver,
)
from reachability_gen.run_selection_reader import permutation_p, permutation_p_two_sided

HYPOTHESIS_VERBATIM = "Prolly a combo of embedding and representation tuning across all subsets in modern convention."
DELEGATION_VERBATIM = "infer the best path all context eval for value based on prime intent"
ARMS: tuple[str, ...] = ("frozen", "embeddings", "interventions", "interventions_matched", "both", "capacity")
TUNED: dict[str, tuple[bool, bool]] = {"embeddings": (True, False), "interventions": (False, True),
                                       "interventions_matched": (False, True), "both": (True, True)}
COMPONENTS: tuple[str, ...] = ("embeddings", "interventions", "interventions_matched")
WORDING = "diverse"
RANK: int = 4
INTERVENTION_LR: float = 1e-3
EMBEDDING_LR_OF_RMS: float = 0.01
STARTING_AT: int = 2  # index of "Starting at {u}, one step takes you to {v}." among the held-out templates
P1_MARGIN: float = 0.30
P2_MARGIN: float = 0.10
INDIST_GAP: float = 0.01
PRECISION_GAP: float = 0.05
NOVEL_BAR: float = 0.90
CONSTRUCTION_BAR: float = 0.99
REPLICATION_MARGIN: float = 0.05
COLLAPSE_F1: float = 0.5
COLLAPSE_LIMIT: int = 3
ALPHA_TEST: float = 0.05
AUDIT_EVAL: int = 16  # held-out renderings audited in evaluation
AUDIT_TRAIN: int = 32  # training renderings audited through the training path (one batch)
ITEM20 = Path("artifacts/nl_reader_lm_diverse.json")
ITEM20_TEMPLATES = Path("artifacts/nl_templates_lm_diverse.json")
SHARED_PROTOCOL: tuple[str, ...] = (
    "train_wording", "encoder_name", "encoder_layer", "reader_epochs", "reader_lr", "solver_epochs", "solver_lr",
    "batch_size", "weight_decay", "grad_clip", "seeds", "device", "torch_version", "dataset_verified",
    "max_offset", "train_steps", "long_steps", "sizes")


def out_path(arm: str) -> Path:
    return Path(f"artifacts/tuned_reader_{arm}.json")


def sets_for(rows, ext_rows) -> dict[str, tuple[list, str]]:
    """Name -> (rows, template split) of every set scored."""
    val = [r for r in rows if r["split"] == "val"]
    return {"val_in": (val, WORDING), "val_out": (val, "heldout"), "long_out": (ext_rows, "heldout"),
            "val_novel": (val, "novel"), "long_novel": (ext_rows, "novel")}


def intervention_params(layers: int, d: int, rank: int) -> int:
    """Per block: the r × d basis of R, D (r × d) and b (r)."""
    return layers * (2 * d * rank + rank)


def adapter_sizes(name: str, layer: int, train_texts: Sequence[str]) -> dict[str, Any]:
    """The tokens that occur in training, what ``both`` adds, and the rank that matches it with interventions only."""
    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    d = int(AutoConfig.from_pretrained(name, local_files_only=True).hidden_size)
    ids = sorted({int(i) for t in train_texts for i in tok(t, add_special_tokens=False)["input_ids"]})
    both = len(ids) * d + intervention_params(layer, d, RANK)
    matched = min(range(1, 65), key=lambda r: abs(intervention_params(layer, d, r) - both))
    return {"train_ids": ids, "d": d, "both_params": both, "matched_rank": matched,
            "matched_params": intervention_params(layer, d, matched)}


def reader_params(d: int, feature_dim: int) -> int:
    from reachability_gen.models.reader import FeatureReader

    return sum(p.numel() for p in FeatureReader(feature_dim, d=d, max_offset=MAX_OFFSET).parameters())


def capacity_width(extra: int, feature_dim: int, base_d: int = 64) -> int:
    """The reader width (a multiple of its 4 heads) whose parameter count is closest to the base reader's + ``extra``."""
    target = reader_params(base_d, feature_dim) + extra
    return min(range(base_d, 4 * base_d + 1, 4), key=lambda d: abs(reader_params(d, feature_dim) - target))


def template_recall(reader, items: Sequence[Any], which: Sequence[dict], ns: Sequence[int], n_templates: int,
                    device: str, batch: int = 16) -> dict[str, Any]:
    """Recall per template on one rendering per graph (``nl_template_audit``'s grouping)."""
    import torch

    from reachability_gen.models.reader import collate_features

    hits, totals, extra = [0] * n_templates, [0] * n_templates, 0
    reader.eval()
    with torch.no_grad():
        for i in range(0, len(items), batch):
            adj = reader(collate_features(items[i : i + batch], device), hard=True).cpu()
            h, t, x = recall_by_template(adj, which[i : i + batch], ns[i : i + batch], n_templates)
            hits, totals, extra = [a + b for a, b in zip(hits, h)], [a + b for a, b in zip(totals, t)], extra + x
    return {"recall": [a / b if b else None for a, b in zip(hits, totals)], "edges": totals, "extra_edges": extra,
            "recall_all_templates": sum(hits) / sum(totals)}


def passes_pi(final: dict[str, Any], construction: float) -> bool:
    """The PI's pass: item 19's criteria on the held-out templates and recall ≥ 0.99 on the construction."""
    first = LONG_STEPS[0]
    return (final["val_out"]["reader"]["exact_graphs"] >= PASS["exact_graphs_heldout_val"]
            and final[f"long_out_{first}"]["reader"]["closure_agreement_all_pairs"]
            >= PASS["closure_agreement_heldout_long"]
            and all(final[f"long_out_{s}"]["accuracy"] >= PASS["long_path_accuracy_heldout"] for s in LONG_STEPS)
            and construction >= CONSTRUCTION_BAR)


def train_tuned(reader, solver, encoder, train: NLData, val_items: dict[str, Any], val_rows, *, epochs: int,
                reader_lr: float, embedding_lr: float, label: str, device: str,
                batch_losses: list[list[float]]) -> list[dict[str, Any]]:
    """``run_nl_reader.train_reader`` with live encoder features and the adapter trained alongside the reader."""
    import torch
    from torch.nn.utils import clip_grad_norm_

    from reachability_gen.models.reader import collate_features

    opt = torch.optim.AdamW(reader.parameters(), lr=reader_lr, weight_decay=WEIGHT_DECAY)
    groups = encoder.adapter.groups(intervention_lr=INTERVENTION_LR, embedding_lr=embedding_lr)
    enc_opt = torch.optim.AdamW(groups, weight_decay=0.0)
    history = []
    for epoch in range(1, epochs + 1):
        reader.train()
        encoder.model.train()
        order = torch.randperm(len(train.rows)).tolist()
        loss_sum, seen = 0.0, 0
        batch_losses.append([])
        for i in range(0, len(order), BATCH):
            idx = order[i : i + BATCH]
            graph, texts = train.batch(idx)
            bce, pairs = topology_bce(reader.edge_log_evidence(encoder.reader_batch(texts)), graph)
            loss = bce / pairs
            opt.zero_grad(set_to_none=True)
            enc_opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(reader.parameters(), GRAD_CLIP)
            for group in groups:  # each component on its own, so both's embedding updates are clipped as alone
                clip_grad_norm_(group["params"], GRAD_CLIP)
            opt.step()
            enc_opt.step()
            value = float(loss.item())
            loss_sum += value * len(idx)
            seen += len(idx)
            batch_losses[-1].append(value)
        val = NLData(val_rows, device, dict(zip(val_items["keys"], encoder.feature_tokens(val_items["encoded"]))),
                     collate_features)
        ev = evaluate(reader, solver, val, TRAIN_STEPS)
        history.append({"epoch": epoch, "train_loss": loss_sum / seen, "val_accuracy": ev["accuracy"],
                        "val_exact_graphs": ev["reader"]["exact_graphs"], "val_edge_f1": ev["reader"]["f1"],
                        "val_closure_agreement": ev["reader"]["closure_agreement_all_pairs"]})
        print(f"[{label}] epoch {epoch}/{epochs}: loss {loss_sum / seen:.4f} held-out val acc {ev['accuracy']:.4f} "
              f"exact graphs {ev['reader']['exact_graphs']:.4f} edge F1 {ev['reader']['f1']:.4f}",
              file=sys.stderr, flush=True)
    return history


def run_arm(args) -> int:
    import torch

    from reachability_gen.models.reader import FeatureReader, collate_features
    from reachability_gen.tuned_encoder import EncoderAdapter, TunedEncoder

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
    val_rows = [r for r in rows if r["split"] == "val"]
    sets = sets_for(rows, ext_rows)
    train_graphs = graphs_of(train_rows)
    train_texts = {eh: render(n, e, WORDING, eh) for eh, (n, e) in train_graphs.items()}
    texts = {name: {eh: render(n, e, split, eh) for eh, (n, e) in graphs_of(rs).items()}
             for name, (rs, split) in sets.items()}
    sizes = {**train_graphs, **graphs_of(val_rows), **graphs_of(ext_rows)}
    val_graphs = graphs_of(val_rows)
    audit_keys = list(val_graphs)
    audit = {}  # split -> (template map per graph, template count)
    for name, split in (("val_in", WORDING), ("val_out", "heldout"), ("val_novel", "novel")):
        which = [template_of_edges(texts[name][k], TEMPLATES[split]) for k in audit_keys]
        if any(set(w) != set(map(tuple, val_graphs[k][1])) for k, w in zip(audit_keys, which)):
            print(f"FAIL: a {name} rendering does not parse back to its edges", file=sys.stderr)
            return 1
        audit[name] = (which, len(TEMPLATES[split]))
    audit_ns = [val_graphs[k][0] for k in audit_keys]
    adapter_plan = adapter_sizes(args.encoder, args.layer, list(train_texts.values()))
    size_record = {k: v for k, v in adapter_plan.items() if k != "train_ids"}
    size_record["train_token_ids"] = len(adapter_plan["train_ids"])

    t0 = time.perf_counter()
    tuned = args.arm in TUNED
    embedding_lr = rank = None
    if tuned:
        emb, itv = TUNED[args.arm]
        rank = (adapter_plan["matched_rank"] if args.arm == "interventions_matched" else RANK) if itv else None
        eval_keys = list(texts["val_out"])[: AUDIT_EVAL]
        train_keys = list(train_texts)[: AUDIT_TRAIN]
        ref_eval = lm_features([texts["val_out"][k] for k in eval_keys], [sizes[k][0] for k in eval_keys],
                               args.encoder, args.layer, args.device)
        ref_train = lm_features([train_texts[k] for k in train_keys], [sizes[k][0] for k in train_keys],
                                args.encoder, args.layer, args.device)
        encoder = TunedEncoder(args.encoder, args.layer, args.device, checkpointing=args.device == "cuda")
        feature_dim = encoder.d
        encoded = {name: dict(zip(d, encoder.encode_texts(list(d.values()), [sizes[k][0] for k in d])))
                   for name, d in texts.items()}
        train_encoded = dict(zip(train_texts, encoder.encode_texts(list(train_texts.values()),
                                                                   [sizes[k][0] for k in train_texts])))
        if sorted({int(i) for x in train_encoded.values() for i in x.ids.tolist()}) != adapter_plan["train_ids"]:
            print("FAIL: the training token ids differ between the tokenizer and the encoder", file=sys.stderr)
            return 1
        embedding_rms = encoder.embedding_rms()
        embedding_lr = EMBEDDING_LR_OF_RMS * embedding_rms if emb else None
        train = NLData(train_rows, args.device, train_encoded, lambda items, device: list(items))
        audit_eval_items = [encoded["val_out"][k] for k in eval_keys]
        audit_train_items = [train_encoded[k] for k in train_keys]

        def new_adapter(seed: int) -> Any:
            return EncoderAdapter(d=encoder.d, layers=args.layer, vocab_size=encoder.vocab_size,
                                  train_ids=adapter_plan["train_ids"], embeddings=emb, interventions=itv,
                                  rank=rank or RANK, seed=seed).to(args.device)
    else:
        feats = {}
        for name, d in {"train": train_texts, **texts}.items():
            keys = list(d)
            feats[name] = dict(zip(keys, lm_features([d[k] for k in keys], [sizes[k][0] for k in keys],
                                                     args.encoder, args.layer, args.device)))
        feature_dim = int(next(iter(feats["train"].values())).features.shape[-1])
        train = NLData(train_rows, args.device, feats["train"], collate_features)
    feature_seconds = time.perf_counter() - t0
    extra = adapter_plan["both_params"] if args.arm == "capacity" else None
    width = capacity_width(extra, feature_dim) if args.arm == "capacity" else 64

    def make_reader():
        return FeatureReader(feature_dim, d=width, max_offset=MAX_OFFSET)

    def eval_data(items_by_set: dict[str, dict[str, Any]]) -> dict[str, NLData]:
        return {name: NLData(sets[name][0], args.device, items_by_set[name], collate_features) for name in sets}

    runs, t1, adapter_params = [], time.perf_counter(), None
    for seed in args.seeds:
        ts = time.perf_counter()
        torch.manual_seed(seed)
        reader, solver = make_reader().to(args.device), build_solver().to(args.device)
        train_solver(solver, train, epochs=args.solver_epochs, device=args.device)
        batch_losses: list[list[float]] = []
        run: dict[str, Any] = {"seed": seed, "arm": args.arm, "science_open": False}
        if tuned:
            encoder.adapter = new_adapter(seed)
            adapter_params = sum(p.numel() for p in encoder.adapter.parameters())
            live_eval = encoder.feature_tokens(audit_eval_items)
            encoder.model.train()
            live_train = encoder.reader_batch(audit_train_items)["features"].detach()  # the training path, one batch
            run["start_audit"] = {
                "eval_identical": all(torch.equal(a.features, b.features) for a, b in zip(live_eval, ref_eval)),
                "train_path_identical": all(
                    torch.equal(live_train[i, : x.ids.numel()].cpu(), ref.features)
                    for i, (x, ref) in enumerate(zip(audit_train_items, ref_train)))}
            del live_eval, live_train
            val_items = {"keys": list(encoded["val_out"]), "encoded": list(encoded["val_out"].values())}
            init_val = NLData(val_rows, args.device,
                              dict(zip(val_items["keys"], encoder.feature_tokens(val_items["encoded"]))),
                              collate_features)
            run["untrained_reader"] = {"val_out": evaluate(reader, solver, init_val, TRAIN_STEPS)}
            del init_val
            history = train_tuned(reader, solver, encoder, train, val_items, val_rows, epochs=args.reader_epochs,
                                  reader_lr=args.reader_lr, embedding_lr=embedding_lr,
                                  label=f"{args.arm}/seed{seed}", device=args.device, batch_losses=batch_losses)
            adapter_state = {"state": {k: v.detach().cpu() for k, v in encoder.adapter.state_dict().items()},
                             "train_ids": list(encoder.adapter.train_ids), "embeddings": emb,
                             "interventions": itv, "rank": rank}
        else:
            data0 = eval_data(feats)
            run["untrained_reader"] = {"val_out": evaluate(reader, solver, data0["val_out"], TRAIN_STEPS)}
            history = train_reader(reader, solver, train, data0["val_out"], epochs=args.reader_epochs,
                                   reader_lr=args.reader_lr, label=f"{args.arm}/seed{seed}", device=args.device,
                                   batch_losses=batch_losses)
            adapter_state = None
        path = args.ckpt_dir / out_path(args.arm).stem / f"{args.arm}_seed{seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"reader": reader.state_dict(), "solver": solver.state_dict(), "adapter": adapter_state,
                    "seed": seed, "arm": args.arm, "science_open": False}, path)
        fresh, fresh_solver = make_reader().to(args.device), build_solver().to(args.device)
        state = torch.load(path, map_location="cpu", weights_only=True)
        fresh.load_state_dict(state["reader"])
        fresh_solver.load_state_dict(state["solver"])
        if tuned:  # re-score through a fresh adapter rebuilt from the checkpoint alone
            saved = state["adapter"]
            restored = EncoderAdapter(d=encoder.d, layers=args.layer, vocab_size=encoder.vocab_size,
                                      train_ids=saved["train_ids"], embeddings=saved["embeddings"],
                                      interventions=saved["interventions"], rank=saved["rank"] or RANK, seed=seed)
            restored.load_state_dict(saved["state"])
            encoder.adapter = restored.to(args.device)
            run["adapter_moved"] = encoder.adapter.summary(embedding_rms)
            items_by_set = {name: dict(zip(encoded[name], encoder.feature_tokens(list(encoded[name].values()))))
                            for name in sets}
        else:
            items_by_set = feats
        data = eval_data(items_by_set)
        final = {name: evaluate(fresh, fresh_solver, data[name], TRAIN_STEPS) for name in ("val_in", "val_out",
                                                                                           "val_novel")}
        for name in ("long_out", "long_novel"):
            final.update({f"{name}_{s}": evaluate(fresh, fresh_solver, data[name], s) for s in LONG_STEPS})
        templates = {name: template_recall(fresh, [items_by_set[name][k] for k in audit_keys], audit[name][0],
                                           audit_ns, audit[name][1], args.device)
                     for name in ("val_in", "val_out", "val_novel")}
        construction = templates["val_out"]["recall"][STARTING_AT]
        novel = [r for r in templates["val_novel"]["recall"] if r is not None]
        run.update({
            "history": history, "batch_losses": batch_losses, "final": final, "templates": templates,
            "construction_recall": construction, "novel_mean_recall": statistics.fmean(novel),
            "passes_pi": passes_pi(final, construction),
            "novel_bar_met": statistics.fmean(novel) >= NOVEL_BAR,
            "collapsed": final["val_in"]["reader"]["f1"] < COLLAPSE_F1,
            "oracle": {"val_out": evaluate(fresh, fresh_solver, data["val_out"], TRAIN_STEPS, oracle=True)},
            "checkpoint_path": path.as_posix(), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rescore_matches_record": final["val_out"]["accuracy"] == history[-1]["val_accuracy"],
            "seconds": time.perf_counter() - ts})
        runs.append(run)
        print(f"[{args.arm}/seed{seed}] F1 in {final['val_in']['reader']['f1']:.3f} out "
              f"{final['val_out']['reader']['f1']:.3f} construction recall {construction:.3f} "
              f"novel {statistics.fmean(novel):.3f}", file=sys.stderr, flush=True)
        del data, items_by_set, reader, solver, fresh, fresh_solver, state
        if tuned:
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
    mismatches = [f"seed{r['seed']}" for r in runs if not r["rescore_matches_record"]]
    if tuned:
        mismatches += [f"seed{r['seed']}:start" for r in runs
                       if not (r["start_audit"]["eval_identical"] and r["start_audit"]["train_path_identical"])]
        expected = (len(adapter_plan["train_ids"]) * adapter_plan["d"] if TUNED[args.arm][0] else 0) + (
            intervention_params(args.layer, adapter_plan["d"], rank) if TUNED[args.arm][1] else 0)
        if adapter_params != expected:
            mismatches.append(f"adapter_params {adapter_params} != {expected}")
    artifact = {
        "science_open": False,
        "hypothesis_verbatim": HYPOTHESIS_VERBATIM,
        "delegation_verbatim": DELEGATION_VERBATIM,
        "purpose": "embedding and representation tuning for reading a construction absent from the training wordings",
        "arm": args.arm,
        "protocol": {
            "train_wording": WORDING, "encoder_name": args.encoder, "encoder_layer": args.layer,
            "tuned": {"embeddings": TUNED.get(args.arm, (False, False))[0],
                      "interventions": TUNED.get(args.arm, (False, False))[1]},
            "rank": rank, "adapter_params": adapter_params,
            "intervention_lr": INTERVENTION_LR if tuned and TUNED[args.arm][1] else None,
            "embedding_lr": embedding_lr, "embedding_lr_rule": f"{EMBEDDING_LR_OF_RMS} x RMS of the embedding matrix",
            "tuned_token_ids": adapter_plan["train_ids"] if tuned and TUNED[args.arm][0] else None,
            "sizes": size_record, "reader_width": width,
            "reader_params": sum(p.numel() for p in make_reader().parameters()),
            "capacity_extra_params": extra, "max_offset": MAX_OFFSET,
            "reader_epochs": args.reader_epochs, "reader_lr": args.reader_lr, "solver_epochs": args.solver_epochs,
            "solver_lr": LR, "batch_size": BATCH, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "seeds": args.seeds, "device": args.device, "train_steps": TRAIN_STEPS, "long_steps": list(LONG_STEPS),
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "torch_version": torch.__version__, "checkpoints_versioned": False,
            "dataset_verified": not args.no_verify, "feature_seconds": feature_seconds,
            "templates": {"in": list(TEMPLATES[WORDING]), "heldout": list(TEMPLATES["heldout"]),
                          "novel": list(TEMPLATES["novel"])},
            "construction": TEMPLATES["heldout"][STARTING_AT],
        },
        "runs": runs,
        "self_audit_mismatches": mismatches,
        "elapsed_seconds": time.perf_counter() - t1,
    }
    path = args.out or out_path(args.arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": not mismatches, "out": path.as_posix(), "science_open": False}, sort_keys=True))
    return 1 if mismatches else 0


def check_arts(arts: dict[str, dict[str, Any]]) -> list[str]:
    """Reasons the result files cannot be decided together (empty when they can)."""
    issues = []
    ref = arts["frozen"]
    seeds = sorted(r["seed"] for r in ref["runs"])
    for arm, art in arts.items():
        if art.get("arm") != arm:
            issues.append(f"{arm}: file records arm {art.get('arm')!r}")
        if sorted(r["seed"] for r in art["runs"]) != seeds:
            issues.append(f"{arm}: seeds differ from frozen's")
        if art.get("self_audit_mismatches"):
            issues.append(f"{arm}: self-audit mismatches {art['self_audit_mismatches']}")
        for key in SHARED_PROTOCOL:
            if art["protocol"].get(key) != ref["protocol"].get(key):
                issues.append(f"{arm}: protocol {key} differs from frozen's")
        if art["protocol"].get("dataset_verified") is not True:
            issues.append(f"{arm}: datasets not verified (--no-verify is for tests only)")
    sizes = ref["protocol"].get("sizes") or {}
    expected = {"interventions": RANK, "both": RANK, "interventions_matched": sizes.get("matched_rank")}
    for arm, rank in expected.items():
        if arm in arts and arts[arm]["protocol"].get("rank") != rank:
            issues.append(f"{arm}: rank {arts[arm]['protocol'].get('rank')} is not {rank}")
    if "both" in arts and arts["both"]["protocol"].get("adapter_params") != sizes.get("both_params"):
        issues.append("both: adapter size differs from the recorded plan")
    if "capacity" in arts and arts["capacity"]["protocol"].get("capacity_extra_params") != sizes.get("both_params"):
        issues.append("capacity: extra parameters differ from what both adds")
    for key, group in (("intervention_lr", ("interventions", "interventions_matched", "both")),
                       ("embedding_lr", ("embeddings", "both")), ("tuned_token_ids", ("embeddings", "both"))):
        values = {json.dumps(arts[a]["protocol"].get(key)) for a in group if a in arts}
        if len(values) > 1:
            issues.append(f"{key} differs between {group}")
    return issues


def per_seed(art: dict[str, Any], fn) -> list[Any]:
    return [fn(r) for r in sorted(art["runs"], key=lambda r: r["seed"])]


def beats(a: Sequence[float], b: Sequence[float], margin: float) -> tuple[bool, float, float]:
    """(a beats b by ``margin``, the one-sided p, the p for rejecting a gain of ``margin``)."""
    gain = statistics.fmean(a) - statistics.fmean(b)
    p = permutation_p(a, b)
    return gain >= margin and p < ALPHA_TEST, p, permutation_p(b, [v - margin for v in a])


def decide(arts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the fixed decision rules (per-seed values kept, ordered by seed)."""
    seeds = sorted(r["seed"] for r in arts["frozen"]["runs"])
    rec = {arm: per_seed(art, lambda r: r["construction_recall"]) for arm, art in arts.items()}
    mean = {arm: statistics.fmean(v) for arm, v in rec.items()}
    collapsed = {arm: per_seed(art, lambda r: r["collapsed"]) for arm, art in arts.items()}
    out: dict[str, Any] = {"seeds": seeds, "splits_per_test": math.comb(2 * len(seeds), len(seeds)),
                           "construction_recall": rec, "mean_construction_recall": mean}
    over_frozen, p_frozen, p_excluded = beats(rec["both"], rec["frozen"], P1_MARGIN)
    over_capacity, p_capacity, _ = beats(rec["both"], rec["capacity"], P1_MARGIN)
    in_f1_all = {arm: per_seed(art, lambda r: r["final"]["val_in"]["reader"]["f1"]) for arm, art in arts.items()}
    prec_all = {arm: per_seed(art, lambda r: r["final"]["val_out"]["reader"]["precision"])
                for arm, art in arts.items()}
    f1_mean = {arm: statistics.fmean(v) for arm, v in in_f1_all.items()}
    prec_mean = {arm: statistics.fmean(v) for arm, v in prec_all.items()}

    def degraded(arms: Sequence[str]) -> list[str]:
        """Control arms reading their own training wording worse than ``both`` (collapsed or degraded seeds)."""
        return [a for a in arms if f1_mean[a] < f1_mean["both"] - INDIST_GAP]

    n_collapsed = sum(collapsed["both"])
    if over_frozen and over_capacity:
        if f1_mean["both"] < f1_mean["frozen"] - INDIST_GAP:
            verdict = "confounded (in-distribution)"
        elif prec_mean["both"] < prec_mean["frozen"] - PRECISION_GAP:
            verdict = "confounded (over-prediction)"
        elif degraded(("frozen", "capacity")):
            verdict = f"confounded (degraded control: {', '.join(degraded(('frozen', 'capacity')))})"
        else:
            verdict = "supported"
    elif over_frozen:
        verdict = "not separated from capacity"
    elif n_collapsed >= COLLAPSE_LIMIT:
        verdict = "uninterpretable (training collapsed)"
    elif p_excluded < ALPHA_TEST:
        verdict = "excluded at these rates"
    else:
        verdict = "inconclusive"
    out["P1"] = {"verdict": verdict, "gain_over_frozen": mean["both"] - mean["frozen"], "p_over_frozen": p_frozen,
                 "gain_over_capacity": mean["both"] - mean["capacity"], "p_over_capacity": p_capacity,
                 "p_gain_below_margin": p_excluded, "both_collapsed_seeds": n_collapsed,
                 "mean_val_in_f1": f1_mean, "mean_val_out_precision": prec_mean}
    comps = {c: beats(rec["both"], rec[c], P2_MARGIN) for c in COMPONENTS}
    if all(v[0] for v in comps.values()):
        dead = [c for c in COMPONENTS if sum(collapsed[c]) >= COLLAPSE_LIMIT]
        worse_in = [c for c in COMPONENTS if f1_mean["both"] < f1_mean[c] - INDIST_GAP]
        worse_prec = [c for c in COMPONENTS if prec_mean["both"] < prec_mean[c] - PRECISION_GAP]
        if dead:
            p2 = f"confounded (collapsed component: {', '.join(dead)})"
        elif degraded(COMPONENTS):
            p2 = f"confounded (degraded component: {', '.join(degraded(COMPONENTS))})"
        elif worse_in:
            p2 = f"confounded (in-distribution vs {', '.join(worse_in)})"
        elif worse_prec:
            p2 = f"confounded (over-prediction vs {', '.join(worse_prec)})"
        else:
            p2 = "shown"
    elif n_collapsed >= COLLAPSE_LIMIT:
        p2 = "uninterpretable (training collapsed)"
    else:
        within = [c for c, v in comps.items() if v[2] < ALPHA_TEST / len(COMPONENTS)]  # a union: Bonferroni
        p2 = f"excluded ({', '.join(within)} within {P2_MARGIN})" if within else "inconclusive"
    out["P2"] = {"verdict": p2, "excluded_alpha": ALPHA_TEST / len(COMPONENTS),
                 **{c: {"gain": mean["both"] - mean[c], "p": v[1], "p_gain_below_margin": v[2]}
                    for c, v in comps.items()}}
    out["reported"] = {
        arm: {"construction_gain_over_frozen": mean[arm] - mean["frozen"],
              "p_gain_over_frozen": permutation_p(rec[arm], rec["frozen"]) if arm != "frozen" else None,
              "passes_pi": per_seed(art, lambda r: r["passes_pi"]),
              "novel_mean_recall": per_seed(art, lambda r: r["novel_mean_recall"]),
              "novel_bar_met": per_seed(art, lambda r: r["novel_bar_met"]),
              "collapsed": collapsed[arm], "val_in_f1": in_f1_all[arm], "val_out_precision": prec_all[arm]}
        for arm, art in arts.items()}
    return out


def replication(frozen: dict[str, Any], item20: dict[str, Any], item20_templates: dict[str, Any]) -> dict[str, Any]:
    """The frozen arm against item 20's recorded reader, seed by seed, with the fixed criterion."""
    ref_f1 = {r["seed"]: r["final"]["val_out"]["reader"]["f1"] for r in item20["runs"]}
    ref_rec = {s["seed"]: s["heldout"]["recall"][STARTING_AT] for s in item20_templates["seeds"]}
    mine = {r["seed"]: r for r in frozen["runs"]}
    seeds = sorted(set(ref_f1) & set(ref_rec) & set(mine))
    out: dict[str, Any] = {"seeds": seeds}
    exact = bool(seeds) and set(ref_f1) == set(mine)
    in_dist = bool(seeds)
    for key, ref, fn in (("val_out_f1", ref_f1, lambda r: r["final"]["val_out"]["reader"]["f1"]),
                         ("construction_recall", ref_rec, lambda r: r["construction_recall"])):
        a, b = [fn(mine[s]) for s in seeds], [ref[s] for s in seeds]
        diff = statistics.fmean(a) - statistics.fmean(b) if seeds else None
        p = permutation_p_two_sided(a, b) if seeds else None
        out[key] = {"frozen_arm": dict(zip(map(str, seeds), a)), "item20": dict(zip(map(str, seeds), b)),
                    "mean_difference": diff, "p_two_sided": p}
        exact = exact and a == b
        in_dist = in_dist and abs(diff) < REPLICATION_MARGIN and p >= ALPHA_TEST
    out["verdict"] = "exact" if exact else "in distribution" if in_dist else "not replicated"
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Embedding and representation tuning for reading (MEASURE).")
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--decide", action="store_true", help="apply the fixed decision rules to the six result files")
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--layer", type=int, default=ENCODER_LAYER)
    p.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--extended-data", type=Path, default=DEFAULT_EXTENDED)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--reader-epochs", type=int, default=READER_EPOCHS)
    p.add_argument("--solver-epochs", type=int, default=SOLVER_EPOCHS)
    p.add_argument("--reader-lr", type=float, default=READER_LR)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/tuned_reader"))
    p.add_argument("--no-verify", action="store_true", help="skip dataset verification (tests only)")
    p.add_argument("--no-replication", action="store_true",
                   help="decide without item 20's files (tests only); otherwise they are required")
    args = p.parse_args(argv)
    if args.decide:
        missing = [out_path(a).as_posix() for a in ARMS if not out_path(a).exists()]
        if not args.no_replication:
            missing += [f.as_posix() for f in (ITEM20, ITEM20_TEMPLATES) if not f.exists()]
        if missing:
            print(f"FAIL: missing {missing}", file=sys.stderr)
            return 1
        arts = {a: json.loads(out_path(a).read_text(encoding="utf-8")) for a in ARMS}
        issues = check_arts(arts)
        if issues:
            print(f"FAIL: the result files cannot be decided together: {issues}", file=sys.stderr)
            return 1
        result = {"science_open": False, "hypothesis_verbatim": HYPOTHESIS_VERBATIM,
                  "delegation_verbatim": DELEGATION_VERBATIM,
                  "construction": TEMPLATES["heldout"][STARTING_AT], "verdicts": decide(arts)}
        if not args.no_replication:
            result["replication_of_item20"] = replication(
                arts["frozen"], json.loads(ITEM20.read_text(encoding="utf-8")),
                json.loads(ITEM20_TEMPLATES.read_text(encoding="utf-8")))
        out = args.out or Path("artifacts/tuned_reader.json")
        out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"ok": True, "out": out.as_posix(), "science_open": False}, sort_keys=True))
        return 0
    if not args.arm:
        p.error("choose --arm or --decide")
    return run_arm(args)


if __name__ == "__main__":
    raise SystemExit(main())

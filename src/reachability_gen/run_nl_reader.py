# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Reading graphs from natural language, including phrasings never seen in training (MEASURE).

Each crossed graph is rendered as shuffled sentences (``nl_render``): the
training graphs with the training templates, and the evaluation graphs with
the training templates (in-distribution) and with held-out templates whose
wording never appears in training. A question-blind reader is trained on the
true edges of the training renderings and supplies the frozen anchored solver
(trained first on true graphs) with an explicit 0/1 graph, scored in the three
layers of ``run_reader``. Two readers:

* ``words``: ``GraphReader`` trained from scratch on word tokens (node numbers
  share one embedding; unseen words map to UNK); no language knowledge;
* ``lm``: ``FeatureReader`` on the frozen hidden states of a small language
  model from the local cache (one layer, fixed in advance); language knowledge
  plus a trained structural head.

Pass criteria (held-out phrasings), fixed before any run: exact graphs ≥ 0.99 on
the validation graphs, closure agreement ≥ 0.99 on the long-path graphs, and
long-path accuracy ≥ 0.99 at 16, 48 and 192 steps. The questions of the
language-model reader samples (``artifacts/llm_reader.json``) are scored too,
for a comparison on identical questions.

``science_open=false`` always.

Usage::

    python -m reachability_gen.run_nl_reader --reader words --device cuda
    python -m reachability_gen.run_nl_reader --reader lm --device cuda
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from reachability_gen.gen_crossed import CROSSED_EXTENDED_SPEC, CROSSED_ID_SPEC, verify_crossed
from reachability_gen.nl_render import HELDOUT_TEMPLATES, TRAIN_TEMPLATES, build_vocab, render, word_tokens
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_llm_reader import graphs_of
from reachability_gen.run_mp_calibration import BATCH, GRAD_CLIP, LR, WEIGHT_DECAY
from reachability_gen.run_reader import (
    EVAL_BATCH,
    LONG_STEPS,
    READER_LR,
    SOLVER_EPOCHS,
    TRAIN_STEPS,
    Data,
    build_solver,
    evaluate,
    topology_bce,
    train_solver,
)

DEFAULT_TRAIN = Path("data/id_crossed_20k.jsonl")
DEFAULT_EXTENDED = Path("data/extended_crossed_2k.jsonl")
DEFAULT_LLM_READER = Path("artifacts/llm_reader.json")
DEFAULT_ENCODER = "Qwen/Qwen2.5-0.5B"
ENCODER_LAYER: int = 12  # of 24: a middle layer, fixed in advance
READER_EPOCHS: int = 5
MAX_OFFSET: int = 8
PASS: dict[str, float] = {"exact_graphs_heldout_val": 0.99, "closure_agreement_heldout_long": 0.99,
                          "long_path_accuracy_heldout": 0.99}
NUMBER = re.compile(r"\d+")
# Rows x tokens^2 per scoring batch: bounds the reader's attention memory on long renderings (a
# long-path graph runs to about a thousand words). The graphs read, accuracy and edge metrics do not
# depend on it; float32 rounding moves margins by ~1e-8, which can shift AUROC only where margins
# tie. Set after the first words run ran out of GPU memory while scoring the long-path set (log kept).
EVAL_PAIR_BUDGET: int = 2**24


class NLData(Data):
    """Rows with the solver's view of each graph and the reader's view of its rendering."""

    def __init__(self, rows: Sequence[dict[str, Any]], device: str, tokens_by_graph: dict[str, Any],
                 collate_fn: Callable[..., dict[str, Any]]) -> None:
        from reachability_gen.models.message_passing import collate, parse_rows

        self.rows = list(rows)
        self.device = device
        self.graphs = parse_rows(self.rows)
        self.tokens = [tokens_by_graph[r["edge_hash"]] for r in self.rows]
        self._collate, self._collate_tokens = collate, collate_fn
        hop = {r["edge_hash"]: int(r["hop_distance"]) for r in self.rows if int(r["y"]) == 1}
        self.graph_hop = [hop.get(r["edge_hash"], -1) for r in self.rows]
        longest = max((len(t.kinds) for t in self.tokens), default=1)
        self.eval_batch = max(1, min(EVAL_BATCH, EVAL_PAIR_BUDGET // longest**2))

    def batch(self, idx: Sequence[int]) -> tuple[dict[str, Any], dict[str, Any]]:
        return (self._collate([self.graphs[i] for i in idx], self.device),
                self._collate_tokens([self.tokens[i] for i in idx], self.device))


def node_token_marks(text: str, offsets: Sequence[tuple[int, int]]) -> tuple[list[int], list[int]]:
    """Kinds (NODE on the token that ends each number, 1 elsewhere) and node numbers, from token offsets."""
    from reachability_gen.models.reader import NODE

    ends = {m.end(): int(m.group()) for m in NUMBER.finditer(text)}
    kinds, symbol = [], []
    for start, end in offsets:
        if end > start and end in ends and text[end - 1].isdigit():
            kinds.append(NODE)
            symbol.append(ends.pop(end))
        else:
            kinds.append(1)
            symbol.append(-1)
    if ends:
        raise ValueError(f"numbers without a token ending them: {sorted(ends.values())}")
    return kinds, symbol


def lm_features(texts: Sequence[str], ns: Sequence[int], name: str, layer: int, device: str,
                batch: int = 16) -> list[Any]:
    """Frozen hidden states (layer ``layer``) of each rendering, with its node tokens marked."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from reachability_gen.models.reader import FeatureTokens

    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(name, local_files_only=True, dtype=torch.bfloat16).to(device).eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            chunk = list(texts[i : i + batch])
            enc = tokenizer(chunk, return_offsets_mapping=True, padding=True, return_tensors="pt",
                            add_special_tokens=False)
            offsets = enc.pop("offset_mapping")
            states = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True).hidden_states[layer]
            for b, text in enumerate(chunk):
                length = int(enc["attention_mask"][b].sum())
                kinds, symbol = node_token_marks(text, [tuple(map(int, o)) for o in offsets[b, :length]])
                out.append(FeatureTokens(states[b, :length].clone(), torch.tensor(kinds), torch.tensor(symbol),
                                         ns[i + b]))
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def train_reader(reader, solver, train: NLData, val: NLData, *, epochs: int, reader_lr: float, label: str,
                 device: str) -> list[dict[str, Any]]:
    """Supervised on the true edges (the ``supervised`` regime of ``run_reader``); history on ``val``."""
    import torch
    from torch.nn.utils import clip_grad_norm_

    opt = torch.optim.AdamW(reader.parameters(), lr=reader_lr, weight_decay=WEIGHT_DECAY)
    history = []
    for epoch in range(1, epochs + 1):
        reader.train()
        order = torch.randperm(len(train.rows)).tolist()
        loss_sum, seen = 0.0, 0
        for i in range(0, len(order), BATCH):
            idx = order[i : i + BATCH]
            graph, tokens = train.batch(idx)
            bce, pairs = topology_bce(reader.edge_log_evidence(tokens), graph)
            loss = bce / pairs
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(reader.parameters(), GRAD_CLIP)
            opt.step()
            loss_sum += float(loss.item()) * len(idx)
            seen += len(idx)
        ev = evaluate(reader, solver, val, TRAIN_STEPS)
        history.append({"epoch": epoch, "train_loss": loss_sum / seen, "val_accuracy": ev["accuracy"],
                        "val_exact_graphs": ev["reader"]["exact_graphs"], "val_edge_f1": ev["reader"]["f1"],
                        "val_closure_agreement": ev["reader"]["closure_agreement_all_pairs"]})
        print(f"[{label}] epoch {epoch}/{epochs}: loss {loss_sum / seen:.4f} held-out val acc {ev['accuracy']:.4f} "
              f"exact graphs {ev['reader']['exact_graphs']:.4f} edge F1 {ev['reader']['f1']:.4f}",
              file=sys.stderr, flush=True)
    return history


def summarise(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def stat(fn):
        vals = [fn(r) for r in runs]
        return {"mean": statistics.fmean(vals), "min": min(vals), "max": max(vals)}

    out: dict[str, Any] = {"seeds": [r["seed"] for r in runs], "passes": sum(r["passes"] for r in runs)}
    for name in ("val_in", "val_out", "sample_val"):
        out[name] = {"accuracy": stat(lambda r, n=name: r["final"][n]["accuracy"]),
                     "auroc": stat(lambda r, n=name: r["final"][n]["auroc"]),
                     "exact_graphs": stat(lambda r, n=name: r["final"][n]["reader"]["exact_graphs"]),
                     "edge_f1": stat(lambda r, n=name: r["final"][n]["reader"]["f1"])}
    for name in ("long_out", "sample_long"):
        out[name] = {**{f"accuracy_{s}": stat(lambda r, n=name, s=s: r["final"][f"{n}_{s}"]["accuracy"])
                        for s in LONG_STEPS},
                     "closure_agreement": stat(lambda r, n=name: r["final"][f"{n}_{LONG_STEPS[0]}"]["reader"][
                         "closure_agreement_all_pairs"]),
                     "edge_f1": stat(lambda r, n=name: r["final"][f"{n}_{LONG_STEPS[0]}"]["reader"]["f1"])}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Reading graphs from natural language (MEASURE).")
    p.add_argument("--reader", choices=("words", "lm"), required=True)
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--layer", type=int, default=ENCODER_LAYER)
    p.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--extended-data", type=Path, default=DEFAULT_EXTENDED)
    p.add_argument("--llm-reader", type=Path, default=DEFAULT_LLM_READER, help="its question samples are scored too")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--reader-epochs", type=int, default=READER_EPOCHS)
    p.add_argument("--solver-epochs", type=int, default=SOLVER_EPOCHS)
    p.add_argument("--reader-lr", type=float, default=READER_LR)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--ckpt-dir", type=Path, default=Path("artifacts/nl_reader"))
    p.add_argument("--no-verify", action="store_true", help="skip dataset verification (tests only)")
    args = p.parse_args(argv)
    import torch

    from reachability_gen.models.reader import FeatureReader, GraphReader, collate_features, collate_tokens

    out_path = args.out or Path(f"artifacts/nl_reader_{args.reader}.json")
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
    texts = {
        "train": {eh: render(n, e, "train", eh) for eh, (n, e) in graphs_of(train_rows).items()},
        "val_in": {eh: render(n, e, "train", eh) for eh, (n, e) in graphs_of(val_rows).items()},
        "val_out": {eh: render(n, e, "heldout", eh) for eh, (n, e) in graphs_of(val_rows).items()},
        "long_out": {eh: render(n, e, "heldout", eh) for eh, (n, e) in graphs_of(ext_rows).items()},
    }
    sizes = {**graphs_of(train_rows), **graphs_of(val_rows), **graphs_of(ext_rows)}
    t0 = time.perf_counter()
    if args.reader == "words":
        vocab = build_vocab()
        tokens = {name: {eh: word_tokens(t, sizes[eh][0], vocab) for eh, t in d.items()} for name, d in texts.items()}
        collate_fn = collate_tokens

        def make_reader():
            return GraphReader(vocab_size=len(vocab) + 2, max_offset=MAX_OFFSET)
        encoder = None
    else:
        tokens = {}
        for name, d in texts.items():
            keys = list(d)
            feats = lm_features([d[k] for k in keys], [sizes[k][0] for k in keys], args.encoder, args.layer,
                                args.device)
            tokens[name] = dict(zip(keys, feats))
        feature_dim = int(next(iter(tokens["train"].values())).features.shape[-1])
        collate_fn = collate_features

        def make_reader():
            return FeatureReader(feature_dim, max_offset=MAX_OFFSET)
        encoder = {"name": args.encoder, "layer": args.layer, "feature_dim": feature_dim, "frozen": True}
    feature_seconds = time.perf_counter() - t0
    data = {
        "train": NLData(train_rows, args.device, tokens["train"], collate_fn),
        "val_in": NLData(val_rows, args.device, tokens["val_in"], collate_fn),
        "val_out": NLData(val_rows, args.device, tokens["val_out"], collate_fn),
        "long_out": NLData(ext_rows, args.device, tokens["long_out"], collate_fn),
    }
    if args.llm_reader.exists():
        llm = json.loads(args.llm_reader.read_text(encoding="utf-8"))
        lookup = {(r["edge_hash"], int(r["s"]), int(r["t"])): r for r in val_rows + ext_rows}
        pick = lambda name: [lookup[(r["edge_hash"], int(r["s"]), int(r["t"]))] for r in llm["sets"][name]]
        data["sample_val"] = NLData(pick("crossed_val"), args.device, tokens["val_out"], collate_fn)
        data["sample_long"] = NLData(pick("crossed_long"), args.device, tokens["long_out"], collate_fn)
    runs = []
    for seed in args.seeds:
        t1 = time.perf_counter()
        torch.manual_seed(seed)
        reader, solver = make_reader().to(args.device), build_solver().to(args.device)
        train_solver(solver, data["train"], epochs=args.solver_epochs, device=args.device)
        untrained = {"val_out": evaluate(reader, solver, data["val_out"], TRAIN_STEPS)}
        history = train_reader(reader, solver, data["train"], data["val_out"], epochs=args.reader_epochs,
                               reader_lr=args.reader_lr, label=f"{args.reader}/seed{seed}", device=args.device)
        path = args.ckpt_dir / out_path.stem / f"{args.reader}_seed{seed}.pt"  # one directory per result file
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"reader": reader.state_dict(), "solver": solver.state_dict(), "seed": seed,
                    "science_open": False}, path)
        fresh_reader, fresh_solver = make_reader().to(args.device), build_solver().to(args.device)
        state = torch.load(path, map_location="cpu", weights_only=True)
        fresh_reader.load_state_dict(state["reader"])
        fresh_solver.load_state_dict(state["solver"])
        final = {name: evaluate(fresh_reader, fresh_solver, d, TRAIN_STEPS)
                 for name, d in data.items() if name in ("val_in", "val_out", "sample_val")}
        for name in ("long_out", "sample_long"):
            if name in data:
                final.update({f"{name}_{s}": evaluate(fresh_reader, fresh_solver, data[name], s) for s in LONG_STEPS})
        run = {
            "seed": seed, "science_open": False, "untrained_reader": untrained, "history": history,
            "final": final,
            "oracle": {"val_out": evaluate(fresh_reader, fresh_solver, data["val_out"], TRAIN_STEPS, oracle=True)},
            "checkpoint_path": path.as_posix(),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rescore_matches_record": final["val_out"]["accuracy"] == history[-1]["val_accuracy"],
            "seconds": time.perf_counter() - t1,
        }
        run["passes"] = (final["val_out"]["reader"]["exact_graphs"] >= PASS["exact_graphs_heldout_val"]
                         and final[f"long_out_{LONG_STEPS[0]}"]["reader"]["closure_agreement_all_pairs"]
                         >= PASS["closure_agreement_heldout_long"]
                         and all(final[f"long_out_{s}"]["accuracy"] >= PASS["long_path_accuracy_heldout"]
                                 for s in LONG_STEPS))
        runs.append(run)
    mismatches = [f"seed{r['seed']}" for r in runs if not r["rescore_matches_record"]]
    artifact = {
        "science_open": False,
        "purpose": "reading graphs from natural language, including phrasings never seen in training",
        "reader": args.reader,
        "protocol": {
            "templates": {"train": list(TRAIN_TEMPLATES), "heldout": list(HELDOUT_TEMPLATES)},
            "encoder": encoder,
            "reader_params": sum(p.numel() for p in make_reader().parameters()),
            "max_offset": MAX_OFFSET, "train_steps": TRAIN_STEPS, "long_steps": list(LONG_STEPS),
            "solver_epochs": args.solver_epochs, "reader_epochs": args.reader_epochs, "reader_lr": args.reader_lr,
            "solver_lr": LR, "batch_size": BATCH, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "pass_criteria": PASS, "seeds": args.seeds, "checkpoints_versioned": False,
            "eval_pair_budget": EVAL_PAIR_BUDGET, "eval_batch_rows": {k: d.eval_batch for k, d in data.items()},
            "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "torch_version": torch.__version__,
        },
        "feature_seconds": feature_seconds,
        "runs": runs,
        "summary": summarise(runs) if all(k in runs[0]["final"] for k in ("sample_val", "sample_long_16")) else None,
        "self_audit_mismatches": mismatches,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": not mismatches, "out": out_path.as_posix(), "self_audit_mismatches": mismatches,
                      "science_open": False}, sort_keys=True))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())

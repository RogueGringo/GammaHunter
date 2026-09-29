# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Which phrasings does a natural-language reader read? (MEASURE, post-hoc analysis)

Added after the natural-language reader runs; not fixed before them. For each
reader saved by ``run_nl_reader`` (checked against the SHA-256 its study
recorded), the edges of the validation graphs are grouped by the template that
rendered them, and the reader's recall is taken per template, together with the
edges it read that the graph does not have. Both template sets are scored: the
training templates (in distribution) and the held-out ones. Each reader's recall
over all templates is checked against the recall its study recorded on the same
graphs (``val_in`` and ``val_out``), so the audit reads the graphs the study scored.

For the language models of ``run_llm_reader --rendering nl`` (``--reader llm``),
the recorded replies are grouped the same way, per model and question set, with
the share of each template's edges that a model listed backwards (the target's
successor given as the source); each model's recall is checked against its study.

``science_open=false`` always.

Usage::

    python -m reachability_gen.nl_template_audit --reader words
    python -m reachability_gen.nl_template_audit --reader lm --device cuda
    python -m reachability_gen.nl_template_audit --reader llm
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.nl_render import TEMPLATES, build_vocab, render, word_tokens
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_llm_reader import graphs_of
from reachability_gen.run_nl_reader import DEFAULT_EXTENDED, DEFAULT_TRAIN, MAX_OFFSET, lm_features

SENTENCE = re.compile(r"(?<=\.)\s+")
GRAPHS_PER_BATCH: int = 16
STUDY_SET: dict[str, str] = {"train": "val_in", "heldout": "val_out"}  # the study's scores on the same graphs


def template_pattern(template: str) -> re.Pattern:
    return re.compile(re.escape(template).replace(r"\{u\}", r"(?P<u>\d+)").replace(r"\{v\}", r"(?P<v>\d+)") + "$")


def template_of_edges(text: str, templates: Sequence[str]) -> dict[tuple[int, int], int]:
    """The template index that rendered each edge of ``text`` (distractor sentences match none)."""
    patterns = [template_pattern(t) for t in templates]
    found: dict[tuple[int, int], int] = {}
    for sentence in SENTENCE.split(text.strip()):
        for k, pattern in enumerate(patterns):
            m = pattern.match(sentence)
            if m:
                found[(int(m["u"]), int(m["v"]))] = k
                break
    return found


def recall_by_template(adj, which: Sequence[dict[tuple[int, int], int]], ns: Sequence[int],
                       n_templates: int) -> tuple[list[int], list[int], int]:
    """Edges read and edges rendered per template, and edges read that the graph does not have."""
    hits, totals, extra = [0] * n_templates, [0] * n_templates, 0
    for b, (tmpl, n) in enumerate(zip(which, ns)):
        read = adj[b, :n, :n] > 0.5
        for (u, v), k in tmpl.items():
            totals[k] += 1
            hits[k] += int(read[u, v])
        extra += int(read.sum()) - sum(int(read[u, v]) for u, v in tmpl)
    return hits, totals, extra


def audit_replies(study: dict[str, Any], rows: Sequence[dict[str, Any]], generations: Path) -> dict[str, Any]:
    """Per model and question set: recall per held-out template, and the share of its edges listed backwards."""
    by_key = {(r["edge_hash"], int(r["s"]), int(r["t"])): r for r in rows}
    graphs = {name: graphs_of([by_key[(r["edge_hash"], int(r["s"]), int(r["t"]))] for r in sample])
              for name, sample in study["sets"].items()}
    listed: dict[tuple[str, str, str], set[tuple[int, int]]] = {}
    for line in generations.read_text(encoding="utf-8").splitlines():
        if line.strip():
            g = json.loads(line)
            listed.setdefault((g["model"], g["set"], g["edge_hash"]), set()).update((g["node"], v) for v in g["successors"])
    templates = TEMPLATES["heldout"]
    out: dict[str, Any] = {}
    for model, res in study["models"].items():
        out[model] = {}
        for name, gs in graphs.items():
            hits, totals, backwards = [0] * len(templates), [0] * len(templates), [0] * len(templates)
            for eh, (n, edges) in gs.items():
                which = template_of_edges(render(n, edges, "heldout", eh), templates)
                read = listed.get((model, name, eh), set())
                for (u, v), k in which.items():
                    totals[k] += 1
                    hits[k] += (u, v) in read
                    backwards[k] += (v, u) in read and (v, u) not in which
            overall = sum(hits) / sum(totals)
            recorded = res["sets"][name]["reader"]["recall"]
            out[model][name] = {"recall": [h / t for h, t in zip(hits, totals)],
                                "read_backwards": [b / t for b, t in zip(backwards, totals)], "edges": totals,
                                "recall_all_templates": overall, "study_recall": recorded,
                                "matches_study": abs(overall - recorded) < 1e-9}
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Recall of a natural-language reader per template (MEASURE, post-hoc).")
    p.add_argument("--reader", choices=("words", "lm", "llm"), required=True)
    p.add_argument("--study", type=Path, default=None,
                   help="the run_nl_reader artifact whose checkpoints are audited (llm: the run_llm_reader artifact)")
    p.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--extended-data", type=Path, default=DEFAULT_EXTENDED, help="long-path graphs (llm only)")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    out_path = args.out or Path(f"artifacts/nl_templates_{args.reader}.json")
    if args.reader == "llm":
        study_path = args.study or Path("artifacts/llm_reader_nl.json")
        study = json.loads(study_path.read_text(encoding="utf-8"))
        if study["protocol"].get("rendering") != "nl" or not study.get("complete", True):
            print(f"FAIL: {study_path} is not a complete natural-language run", file=sys.stderr)
            return 1
        generations = Path(study["protocol"]["generations_file"])
        models = audit_replies(study, load_jsonl(args.train_data) + load_jsonl(args.extended_data), generations)
        for model, sets in models.items():
            for name, s in sets.items():
                print(f"[{model}] {name}: recall by held-out template {[round(r, 3) for r in s['recall']]}, "
                      f"read backwards {[round(r, 3) for r in s['read_backwards']]}", file=sys.stderr, flush=True)
        artifact = {
            "science_open": False,
            "post_hoc": True,
            "purpose": "which phrasings language models read when listing successors: recall per held-out template",
            "reader": "llm",
            "study": study_path.as_posix(),
            "generations": generations.as_posix(),
            "templates": {"heldout": list(TEMPLATES["heldout"])},
            "models": models,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"ok": True, "out": out_path.as_posix(), "science_open": False}, sort_keys=True))
        return 0
    import torch

    from reachability_gen.models.reader import FeatureReader, GraphReader, collate_features, collate_tokens

    study_path = args.study or Path(f"artifacts/nl_reader_{args.reader}.json")
    study = json.loads(study_path.read_text(encoding="utf-8"))
    for run in study["runs"]:
        digest = hashlib.sha256(Path(run["checkpoint_path"]).read_bytes()).hexdigest()
        if digest != run["checkpoint_sha256"]:
            print(f"FAIL: {run['checkpoint_path']} does not match the SHA-256 recorded in {study_path}",
                  file=sys.stderr)
            return 1
    graphs = graphs_of([r for r in load_jsonl(args.train_data) if r["split"] == "val"])
    keys = list(graphs)
    ns = [graphs[k][0] for k in keys]
    splits: dict[str, Any] = {}
    for split in ("train", "heldout"):
        texts = [render(n, e, split, k) for k, (n, e) in ((k, graphs[k]) for k in keys)]
        which = [template_of_edges(t, TEMPLATES[split]) for t in texts]
        for k, tmpl in zip(keys, which):
            if set(tmpl) != set(map(tuple, graphs[k][1])):
                print(f"FAIL: the rendering of {k} does not parse back to its edges", file=sys.stderr)
                return 1
        if args.reader == "words":
            vocab = build_vocab()
            items = [word_tokens(t, n, vocab) for t, n in zip(texts, ns)]
        else:
            enc = study["protocol"]["encoder"]
            items = lm_features(texts, ns, enc["name"], enc["layer"], args.device)
        splits[split] = (items, which)
    collate = collate_tokens if args.reader == "words" else collate_features
    seeds = []
    for run in study["runs"]:
        if args.reader == "words":
            reader = GraphReader(vocab_size=len(build_vocab()) + 2, max_offset=MAX_OFFSET)
        else:
            reader = FeatureReader(study["protocol"]["encoder"]["feature_dim"], max_offset=MAX_OFFSET)
        state = torch.load(run["checkpoint_path"], map_location="cpu", weights_only=True)
        reader.load_state_dict(state["reader"])
        reader.to(args.device).eval()
        entry: dict[str, Any] = {"seed": run["seed"], "checkpoint_sha256": run["checkpoint_sha256"]}
        for split, (items, which) in splits.items():
            n_t = len(TEMPLATES[split])
            hits, totals, extra = [0] * n_t, [0] * n_t, 0
            with torch.no_grad():
                for i in range(0, len(items), GRAPHS_PER_BATCH):
                    adj = reader(collate(items[i : i + GRAPHS_PER_BATCH], args.device), hard=True).cpu()
                    h, t, x = recall_by_template(adj, which[i : i + GRAPHS_PER_BATCH], ns[i : i + GRAPHS_PER_BATCH], n_t)
                    hits, totals, extra = [a + b for a, b in zip(hits, h)], [a + b for a, b in zip(totals, t)], extra + x
            recorded = run["final"][STUDY_SET[split]]["reader"]["recall"]
            entry[split] = {"recall": [a / b for a, b in zip(hits, totals)], "edges": totals, "extra_edges": extra,
                            "recall_all_templates": sum(hits) / sum(totals), "study_recall": recorded,
                            "matches_study": abs(sum(hits) / sum(totals) - recorded) < 1e-9}
        seeds.append(entry)
        print(f"[{args.reader}/seed{run['seed']}] recall by held-out template "
              f"{[round(r, 3) for r in entry['heldout']['recall']]}, extra edges {entry['heldout']['extra_edges']}",
              file=sys.stderr, flush=True)
    summary = {split: {
        "recall_mean": [statistics.fmean(s[split]["recall"][k] for s in seeds) for k in range(len(TEMPLATES[split]))],
        "recall_min": [min(s[split]["recall"][k] for s in seeds) for k in range(len(TEMPLATES[split]))],
        "recall_max": [max(s[split]["recall"][k] for s in seeds) for k in range(len(TEMPLATES[split]))],
        "seeds_reading_template_at_0_99": [sum(s[split]["recall"][k] >= 0.99 for s in seeds)
                                           for k in range(len(TEMPLATES[split]))],
        "extra_edges_total": sum(s[split]["extra_edges"] for s in seeds),
        "seeds_matching_study_recall": sum(s[split]["matches_study"] for s in seeds),
    } for split in splits}
    artifact = {
        "science_open": False,
        "post_hoc": True,
        "purpose": "which phrasings a natural-language reader reads: recall per template on the validation graphs",
        "reader": args.reader,
        "study": study_path.as_posix(),
        "templates": {split: list(TEMPLATES[split]) for split in splits},
        "graphs": len(keys),
        "seeds": seeds,
        "summary": summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "out": out_path.as_posix(), "science_open": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

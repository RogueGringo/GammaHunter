# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for natural-language renderings and the natural-language readers (no language model needed)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.gen_crossed import CROSSED_EXTENDED_SPEC, generate_crossed  # noqa: E402
from reachability_gen.models.reader import (  # noqa: E402
    NODE,
    FeatureReader,
    FeatureTokens,
    collate_features,
)
from reachability_gen.nl_render import (  # noqa: E402
    DISTRACTORS,
    HELDOUT_TEMPLATES,
    TEMPLATES,
    UNK,
    build_vocab,
    render,
    word_tokens,
)
from reachability_gen.run_llm_reader import successor_prompt_nl  # noqa: E402
from reachability_gen.run_nl_reader import node_token_marks  # noqa: E402

EDGES = [(0, 1), (1, 2), (3, 4), (4, 0), (2, 3), (1, 4), (0, 2), (3, 1)]


def pattern(template: str) -> re.Pattern:
    return re.compile(re.escape(template).replace(r"\{u\}", r"(?P<u>\d+)").replace(r"\{v\}", r"(?P<v>\d+)")
                      .replace(r"\{w\}", r"(?P<w>\d+)") + "$")


def parse(text: str, split: str) -> tuple[list[tuple[int, int]], int]:
    edges, distractors = [], 0
    for sentence in re.split(r"(?<=\.)\s+", text):
        for t in TEMPLATES[split]:
            m = pattern(t).match(sentence)
            if m:
                edges.append((int(m["u"]), int(m["v"])))
                break
        else:
            assert any(pattern(t).match(sentence) for t in DISTRACTORS), sentence
            distractors += 1
    return edges, distractors


@pytest.mark.parametrize("split", ["train", "heldout", "diverse", "novel"])
def test_rendering_keeps_every_edge_and_its_direction(split):
    text = render(5, EDGES, split, "g")
    assert text == render(5, EDGES, split, "g") and text != render(5, EDGES, split, "h")
    edges, distractors = parse(text, split)
    assert sorted(edges) == sorted(EDGES) and distractors == round(0.25 * len(EDGES))


def test_diverse_wordings_never_use_the_held_out_or_novel_wording():
    from reachability_gen.nl_render import (
        DIVERSE_TEMPLATES,
        FORBIDDEN_WORDS,
        NOVEL_TEMPLATES,
        TRAIN_TEMPLATES,
        WORD,
        skeleton,
    )

    def words(t):
        return {w.lower() for w in WORD.findall(t.replace("{u}", "0").replace("{v}", "0")) if w.isalpha()}

    assert set(TRAIN_TEMPLATES) <= set(DIVERSE_TEMPLATES) and len(set(DIVERSE_TEMPLATES)) == len(DIVERSE_TEMPLATES) >= 40
    target_first = [t for t in DIVERSE_TEMPLATES if t.index("{v}") < t.index("{u}")]
    assert 0.2 <= len(target_first) / len(DIVERSE_TEMPLATES) <= 0.3
    assert not set(DIVERSE_TEMPLATES) & set(HELDOUT_TEMPLATES + NOVEL_TEMPLATES)
    assert all(not words(t) & FORBIDDEN_WORDS for t in DIVERSE_TEMPLATES)
    assert all(words(t) & FORBIDDEN_WORDS for t in HELDOUT_TEMPLATES + NOVEL_TEMPLATES)  # each has unseen wording
    for t in DIVERSE_TEMPLATES:  # one source, one target, and nothing the word tokenizer would drop
        assert t.count("{u}") == 1 and t.count("{v}") == 1
        assert not re.sub(r"\{u\}|\{v\}|->|[A-Za-z ,.]", "", t)
    # no diverse wording is a novel construction with only its unseen words swapped
    assert not any(skeleton(nv).fullmatch(t) for nv in NOVEL_TEMPLATES for t in DIVERSE_TEMPLATES)
    # "by" introduces the source in some templates and the target in another
    by = [t for t in DIVERSE_TEMPLATES if " by " in t]
    assert any(t.index("{u}") > t.index(" by ") for t in by) and any(t.index("{v}") > t.index(" by ") for t in by)
    vocab = build_vocab(DIVERSE_TEMPLATES + DISTRACTORS)
    assert all(w not in vocab for t in HELDOUT_TEMPLATES + NOVEL_TEMPLATES for w in words(t) & FORBIDDEN_WORDS)


def test_word_tokens_group_numbers_and_map_unseen_words_to_unk():
    vocab = build_vocab()
    assert "->" in vocab and "leads" not in vocab and min(vocab.values()) == 2
    tok = word_tokens("3 leads to 7. There is an edge from 7 to 3.", 8, vocab)
    kinds, symbol = tok.kinds.tolist(), tok.symbol.tolist()
    assert kinds[0] == NODE and symbol[0] == 3 and kinds[1] == UNK and kinds[2] == vocab["to"]
    assert [s for k, s in zip(kinds, symbol) if k == NODE] == [3, 7, 7, 3]


def test_node_token_marks_use_the_token_that_ends_each_number():
    text = "From 17 to 2."
    pieces = ["From", " ", "1", "7", " to", " ", "2", "."]
    offsets, pos = [], 0
    for piece in pieces:
        offsets.append((pos, pos + len(piece)))
        pos += len(piece)
    kinds, symbol = node_token_marks(text, offsets)
    assert [(i, s) for i, (k, s) in enumerate(zip(kinds, symbol)) if k == NODE] == [(3, 17), (6, 2)]
    with pytest.raises(ValueError):
        node_token_marks("From 17 to 2.", offsets[:5])  # the token ending 2 is missing


def test_feature_reader_reads_padded_feature_batches():
    torch.manual_seed(0)
    items = [FeatureTokens(torch.randn(6, 12), torch.tensor([1, 0, 1, 1, 0, 1]), torch.tensor([-1, 2, -1, -1, 0, -1]), 3),
             FeatureTokens(torch.randn(4, 12), torch.tensor([0, 1, 0, 1]), torch.tensor([1, -1, 0, -1]), 2)]
    batch = collate_features(items)
    reader = FeatureReader(12, d=16, layers=1, heads=2)
    adj = reader(batch, hard=True, through="logit")
    assert adj.shape == (2, 3, 3) and set(adj.detach().unique().tolist()) <= {0.0, 1.0}
    adj.sum().backward()
    assert reader.proj.weight.grad is not None and reader.proj.weight.grad.abs().sum() > 0


def test_nl_prompt_carries_no_question():
    q = successor_prompt_nl(render(4, [(0, 1), (2, 3)], "heldout", "g"), 2)
    assert "node 2 has a direct one-way connection" in q and "path" not in q.lower()
    fragments = ("leads to", "A link runs from", "one step takes you to", "can be entered straight from")
    assert any(f in q for f in fragments)  # the held-out wording is what the model reads


def test_words_reader_smoke(tmp_path, monkeypatch):
    from reachability_gen import run_nl_reader as rn

    monkeypatch.setattr(rn, "LONG_STEPS", (16,))
    rows = [e.to_dict() for e in generate_crossed(seed=21, n_total=40, n_val=20)[0]]
    ext = [e.to_dict() for e in generate_crossed(seed=22, n_total=20, n_val=20, spec=CROSSED_EXTENDED_SPEC)[0]]
    train_path, ext_path = tmp_path / "train.jsonl", tmp_path / "ext.jsonl"
    train_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ext_path.write_text("".join(json.dumps(r) + "\n" for r in ext))
    out = tmp_path / "nl.json"
    assert rn.main(["--reader", "words", "--train-data", str(train_path), "--extended-data", str(ext_path),
                    "--llm-reader", str(tmp_path / "absent.json"), "--seeds", "0", "--reader-epochs", "1",
                    "--solver-epochs", "1", "--out", str(out), "--ckpt-dir", str(tmp_path / "ckpt"),
                    "--no-verify"]) == 0
    art = json.loads(out.read_text())
    run = art["runs"][0]
    assert art["self_audit_mismatches"] == [] and art["protocol"]["templates"]["heldout"] == list(HELDOUT_TEMPLATES)
    assert set(run["final"]) == {"val_in", "val_out", "long_out_16", "val_novel", "long_novel_16"}
    assert isinstance(run["passes"], bool) and isinstance(run["passes_novel"], bool)
    assert art["protocol"]["train_wording"] == "train" and art["protocol"]["templates"]["train"] == list(TEMPLATES["train"])
    assert run["oracle"]["val_out"]["accuracy"] >= 0.0


def test_words_reader_smoke_with_diverse_wording(tmp_path, monkeypatch):
    from reachability_gen import run_nl_reader as rn
    from reachability_gen.nl_render import DIVERSE_TEMPLATES, FORBIDDEN_WORDS, NOVEL_TEMPLATES, skeleton

    monkeypatch.setattr(rn, "LONG_STEPS", (16,))
    rows = [e.to_dict() for e in generate_crossed(seed=21, n_total=40, n_val=20)[0]]
    ext = [e.to_dict() for e in generate_crossed(seed=22, n_total=20, n_val=20, spec=CROSSED_EXTENDED_SPEC)[0]]
    train_path, ext_path = tmp_path / "train.jsonl", tmp_path / "ext.jsonl"
    train_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ext_path.write_text("".join(json.dumps(r) + "\n" for r in ext))
    common = ["--reader", "words", "--train-data", str(train_path), "--extended-data", str(ext_path),
              "--llm-reader", str(tmp_path / "absent.json"), "--seeds", "0", "--reader-epochs", "1",
              "--solver-epochs", "1", "--ckpt-dir", str(tmp_path / "ckpt"), "--no-verify"]
    monkeypatch.chdir(tmp_path)  # the default result paths are relative to the working directory
    assert rn.main(common) == 0 and rn.main(common + ["--train-wording", "diverse"]) == 0
    five, diverse = (json.loads((tmp_path / "artifacts" / f"nl_reader_{n}.json").read_text())
                     for n in ("words", "words_diverse"))
    pr = diverse["protocol"]
    assert five["protocol"]["train_wording"] == "train" and pr["train_wording"] == "diverse"
    assert pr["templates"]["train"] == list(DIVERSE_TEMPLATES) and pr["templates"]["novel"] == list(NOVEL_TEMPLATES)
    assert pr["forbidden_words"] == sorted(FORBIDDEN_WORDS) and pr["distractors"] == list(DISTRACTORS)
    # construction coverage is recorded as computed; only the novel set's disjointness is required
    assert pr["skeleton_in_training"] == {split: [any(skeleton(t).fullmatch(g) for g in DIVERSE_TEMPLATES)
                                                  for t in TEMPLATES[split]] for split in ("heldout", "novel")}
    assert pr["skeleton_in_training"]["novel"] == [False] * len(NOVEL_TEMPLATES)
    assert pr["reader_params"] > five["protocol"]["reader_params"]  # the larger vocabulary
    assert Path(diverse["runs"][0]["checkpoint_path"]).parent.name == "nl_reader_words_diverse"


def test_only_the_training_wording_changes_with_train_wording():
    from reachability_gen.run_nl_reader import renderings

    rows = [e.to_dict() for e in generate_crossed(seed=21, n_total=40, n_val=20)[0]]
    ext = [e.to_dict() for e in generate_crossed(seed=22, n_total=20, n_val=20, spec=CROSSED_EXTENDED_SPEC)[0]]
    train_rows, val_rows = [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "val"]
    five, diverse = (renderings(w, train_rows, val_rows, ext) for w in ("train", "diverse"))
    for name in ("val_out", "long_out", "val_novel", "long_novel"):
        assert five[name] == diverse[name] and len(five[name]) > 0
    assert five["train"] != diverse["train"] and five["val_in"] != diverse["val_in"]


def test_scoring_batches_shrink_with_rendering_length_and_leave_scores_unchanged():
    from reachability_gen import run_nl_reader as rn
    from reachability_gen.models.reader import GraphReader, collate_tokens
    from reachability_gen.run_llm_reader import graphs_of
    from reachability_gen.run_reader import EVAL_BATCH, build_solver, evaluate

    rows = [e.to_dict() for e in generate_crossed(seed=21, n_total=40, n_val=20)[0]]
    vocab = build_vocab()
    tokens = {eh: word_tokens(render(n, e, "heldout", eh), n, vocab) for eh, (n, e) in graphs_of(rows).items()}
    data = rn.NLData(rows, "cpu", tokens, collate_tokens)
    longest = max(len(t.kinds) for t in tokens.values())
    assert data.eval_batch == max(1, min(EVAL_BATCH, rn.EVAL_PAIR_BUDGET // longest**2))
    torch.manual_seed(0)
    reader, solver = GraphReader(vocab_size=len(vocab) + 2, max_offset=rn.MAX_OFFSET), build_solver()
    whole = evaluate(reader, solver, data, 6)
    data.eval_batch = 3
    split = evaluate(reader, solver, data, 6)
    assert split["accuracy"] == whole["accuracy"]  # AUROC may move where rounding breaks tied margins
    for key in ("exact_graphs", "f1", "closure_agreement_all_pairs", "reversed_errors"):
        assert split["reader"][key] == whole["reader"][key]


def test_recording_batch_losses_leaves_training_unchanged():
    from reachability_gen import run_nl_reader as rn
    from reachability_gen.models.reader import GraphReader, collate_tokens
    from reachability_gen.run_llm_reader import graphs_of
    from reachability_gen.run_reader import build_solver

    rows = [e.to_dict() for e in generate_crossed(seed=21, n_total=40, n_val=20)[0]]
    vocab = build_vocab()
    tokens = {eh: word_tokens(render(n, e, "train", eh), n, vocab) for eh, (n, e) in graphs_of(rows).items()}
    train = rn.NLData([r for r in rows if r["split"] == "train"], "cpu", tokens, collate_tokens)
    val = rn.NLData([r for r in rows if r["split"] == "val"], "cpu", tokens, collate_tokens)

    def trained(record):
        torch.manual_seed(3)
        reader, solver = GraphReader(vocab_size=len(vocab) + 2, max_offset=rn.MAX_OFFSET), build_solver()
        losses: list = []
        history = rn.train_reader(reader, solver, train, val, epochs=2, reader_lr=1e-2, label="t", device="cpu",
                                  batch_losses=losses if record else None)
        return reader.state_dict(), history, losses

    plain, plain_history, _ = trained(False)
    rec, rec_history, losses = trained(True)
    assert plain_history == rec_history and all(torch.equal(plain[k], rec[k]) for k in plain)
    assert len(losses) == 2 and all(losses)
    batches = -(-len(train.rows) // rn.BATCH)
    assert len(losses[0]) == batches


def _tiny_words_study(tmp_path, monkeypatch, wording="train"):
    from reachability_gen import run_nl_reader as rn

    monkeypatch.setattr(rn, "LONG_STEPS", (16,))
    rows = [e.to_dict() for e in generate_crossed(seed=21, n_total=40, n_val=20)[0]]
    ext = [e.to_dict() for e in generate_crossed(seed=22, n_total=20, n_val=20, spec=CROSSED_EXTENDED_SPEC)[0]]
    train_path, ext_path = tmp_path / "train.jsonl", tmp_path / "ext.jsonl"
    train_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ext_path.write_text("".join(json.dumps(r) + "\n" for r in ext))
    study = tmp_path / f"nl_{wording}.json"
    assert rn.main(["--reader", "words", "--train-data", str(train_path), "--extended-data", str(ext_path),
                    "--llm-reader", str(tmp_path / "absent.json"), "--seeds", "0", "--reader-epochs", "1",
                    "--solver-epochs", "1", "--out", str(study), "--ckpt-dir", str(tmp_path / "ckpt"),
                    "--train-wording", wording, "--no-verify"]) == 0
    return study, train_path, rows


def test_template_audit_of_a_diverse_study(tmp_path, monkeypatch):
    from reachability_gen import nl_template_audit as audit

    study, train_path, _ = _tiny_words_study(tmp_path, monkeypatch, wording="diverse")
    out = tmp_path / "templates_diverse.json"
    assert audit.main(["--reader", "words", "--study", str(study), "--train-data", str(train_path),
                       "--out", str(out)]) == 0
    art = json.loads(out.read_text())
    assert art["train_wording"] == "diverse" and set(art["templates"]) == {"diverse", "heldout", "novel"}
    for split in ("diverse", "heldout", "novel"):
        s = art["seeds"][0][split]
        assert len(s["recall"]) == len(TEMPLATES[split]) and s["matches_study"] is True


def test_template_audit_parses_every_edge_and_checks_checkpoints(tmp_path, monkeypatch):
    from reachability_gen import nl_template_audit as audit
    from reachability_gen.run_llm_reader import graphs_of

    for split in ("train", "heldout", "diverse", "novel"):
        found = audit.template_of_edges(render(5, EDGES, split, "g"), TEMPLATES[split])
        assert set(found) == set(EDGES) and set(found.values()) <= set(range(len(TEMPLATES[split])))
    study, train_path, rows = _tiny_words_study(tmp_path, monkeypatch)
    out = tmp_path / "templates.json"
    argv = ["--reader", "words", "--study", str(study), "--train-data", str(train_path), "--out", str(out)]
    assert audit.main(argv) == 0
    art = json.loads(out.read_text())
    edges = sum(len(e) for _, e in graphs_of([r for r in rows if r["split"] == "val"]).values())
    assert art["post_hoc"] is True and art["science_open"] is False and len(art["seeds"]) == 1
    assert art["train_wording"] == "train" and set(art["templates"]) == {"train", "heldout", "novel"}
    for split in ("train", "heldout", "novel"):
        s = art["seeds"][0][split]
        assert len(s["recall"]) == len(TEMPLATES[split]) and sum(s["edges"]) == edges
        assert all(0.0 <= r <= 1.0 for r in s["recall"]) and s["extra_edges"] >= 0
        assert s["matches_study"]  # the audit reads the graphs its study scored
    ckpt = Path(json.loads(study.read_text())["runs"][0]["checkpoint_path"])
    ckpt.write_bytes(ckpt.read_bytes() + b"tampered")
    assert audit.main(argv) == 1


def test_template_audit_of_language_model_replies(tmp_path):
    """Replies that list the source-first templates' edges and read the target-first one backwards."""
    from reachability_gen import nl_template_audit as audit
    from reachability_gen.run_llm_reader import graphs_of

    rows = [e.to_dict() for e in generate_crossed(seed=23, n_total=20, n_val=20)[0]]
    lines, read_total, true_total = [], 0, 0
    for eh, (n, edges) in graphs_of(rows).items():
        which = audit.template_of_edges(render(n, edges, "heldout", eh), HELDOUT_TEMPLATES)
        read = {(u, v) for (u, v), k in which.items() if k < 3} | {(v, u) for (u, v), k in which.items() if k == 3}
        lines += [json.dumps({"model": "m", "set": "crossed_val", "edge_hash": eh, "node": u,
                              "successors": sorted(v for a, v in read if a == u)}) for u in range(n)]
        read_total += len(read & set(which))
        true_total += len(which)
    (tmp_path / "gen.jsonl").write_text("\n".join(lines) + "\n")
    data = tmp_path / "rows.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    study = {"protocol": {"rendering": "nl", "generations_file": str(tmp_path / "gen.jsonl")}, "complete": True,
             "sets": {"crossed_val": rows},
             "models": {"m": {"sets": {"crossed_val": {"reader": {"recall": read_total / true_total}}}}}}
    (tmp_path / "study.json").write_text(json.dumps(study))
    out = tmp_path / "llm.json"
    assert audit.main(["--reader", "llm", "--study", str(tmp_path / "study.json"), "--train-data", str(data),
                       "--extended-data", str(data), "--out", str(out)]) == 0
    res = json.loads(out.read_text())["models"]["m"]["crossed_val"]
    assert res["recall"][:3] == [1.0, 1.0, 1.0] and res["recall"][3] < 0.1 and res["read_backwards"][3] > 0.9
    assert res["read_backwards"][:3] == [0.0, 0.0, 0.0] and res["matches_study"]
    study["protocol"]["rendering"] = "edges"
    (tmp_path / "study.json").write_text(json.dumps(study))
    assert audit.main(["--reader", "llm", "--study", str(tmp_path / "study.json"), "--train-data", str(data),
                       "--extended-data", str(data), "--out", str(out)]) == 1


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["words", "lm", "words_diverse", "lm_diverse"])
def test_nl_reader_artifact_contract(name):
    from reachability_gen.run_nl_reader import EVAL_PAIR_BUDGET, PASS
    from reachability_gen.run_reader import LONG_STEPS

    path = ROOT / "artifacts" / f"nl_reader_{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    art = json.loads(path.read_text(encoding="utf-8"))
    pr = art["protocol"]
    reader, wording = name.split("_")[0], pr.get("train_wording", "train")  # item 19 predates the field
    assert art["science_open"] is False and art["self_audit_mismatches"] == [] and art["reader"] == reader
    assert pr["pass_criteria"] == PASS and pr["eval_pair_budget"] == EVAL_PAIR_BUDGET
    assert wording == ("diverse" if name.endswith("_diverse") else "train")
    assert pr["templates"]["train"] == list(TEMPLATES[wording]) and pr["templates"]["heldout"] == list(HELDOUT_TEMPLATES)
    assert (pr["encoder"] is None) == (reader == "words") and len(art["runs"]) == 10
    assert art["summary"]["passes"] == sum(r["passes"] for r in art["runs"])

    def meets(f, held):
        return (f[f"val_{held}"]["reader"]["exact_graphs"] >= PASS["exact_graphs_heldout_val"]
                and f[f"long_{held}_16"]["reader"]["closure_agreement_all_pairs"] >= PASS["closure_agreement_heldout_long"]
                and all(f[f"long_{held}_{s}"]["accuracy"] >= PASS["long_path_accuracy_heldout"] for s in LONG_STEPS))

    for run in art["runs"]:
        assert run["passes"] == meets(run["final"], "out") and run["rescore_matches_record"]
        assert run["oracle"]["val_out"]["accuracy"] == 1.0
        if "passes_novel" in run:  # scored from this leg on, reported apart from the pass criteria
            assert run["passes_novel"] == meets(run["final"], "novel")


@pytest.mark.parametrize("name", ["words", "lm", "words_diverse", "lm_diverse"])
def test_template_audit_artifact_contract(name):
    path = ROOT / "artifacts" / f"nl_templates_{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    art = json.loads(path.read_text(encoding="utf-8"))
    study = json.loads((ROOT / art["study"]).read_text(encoding="utf-8"))
    wording = study["protocol"].get("train_wording", "train")
    assert art["science_open"] is False and art["post_hoc"] is True and art["reader"] == name.split("_")[0]
    assert art["train_wording"] == wording and set(art["templates"]) == {wording, "heldout", "novel"}
    assert all(art["templates"][split] == list(TEMPLATES[split]) for split in art["templates"])
    assert [s["checkpoint_sha256"] for s in art["seeds"]] == [r["checkpoint_sha256"] for r in study["runs"]]
    scored_novel = "val_novel" in study["runs"][0]["final"]  # studies before the novel set did not score it
    for split, s in art["summary"].items():
        assert s["seeds_matching_study_recall"] == s["seeds_with_study_record"]  # every recorded recall reproduced
        assert s["seeds_with_study_record"] == (len(art["seeds"]) if split != "novel" or scored_novel else 0)
        assert all(r is not None and 0.0 <= r <= 1.0 for r in s["recall_mean"])  # every template rendered some edge


def test_template_audit_of_language_models_artifact_contract():
    path = ROOT / "artifacts" / "nl_templates_llm.json"
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    art = json.loads(path.read_text(encoding="utf-8"))
    study = json.loads((ROOT / art["study"]).read_text(encoding="utf-8"))
    assert art["science_open"] is False and art["post_hoc"] is True and art["reader"] == "llm"
    assert art["templates"] == {"heldout": list(HELDOUT_TEMPLATES)} and set(art["models"]) == set(study["models"])
    for sets in art["models"].values():
        assert set(sets) == set(study["sets"]) and all(s["matches_study"] for s in sets.values())

# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for natural-language renderings and the natural-language readers (no language model needed)."""

from __future__ import annotations

import json
import re

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


@pytest.mark.parametrize("split", ["train", "heldout"])
def test_rendering_keeps_every_edge_and_its_direction(split):
    text = render(5, EDGES, split, "g")
    assert text == render(5, EDGES, split, "g") and text != render(5, EDGES, split, "h")
    edges, distractors = parse(text, split)
    assert sorted(edges) == sorted(EDGES) and distractors == round(0.25 * len(EDGES))


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
    assert set(run["final"]) == {"val_in", "val_out", "long_out_16"} and isinstance(run["passes"], bool)
    assert run["oracle"]["val_out"]["accuracy"] >= 0.0


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

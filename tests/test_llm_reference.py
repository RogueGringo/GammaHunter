# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the LLM reference runner (no model download; heavy parts skip without a local cache)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reachability_gen.encode import encode_instance
from reachability_gen.gen_crossed import generate_crossed
from reachability_gen.run_llm_reference import (
    WITHHELD,
    answer_tokens,
    auroc,
    build_prompt,
    choose_shots,
    format_item,
    metrics,
    sample_graphs,
)

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "llm_reference.json"


def _rows():
    return [e.to_dict() for e in generate_crossed(seed=5, n_total=200, n_val=40)[0]]


def test_item_and_prompt_format():
    enc = encode_instance(4, [(0, 1), (1, 3)], 0, 3)
    assert format_item(enc) == "Edges: 0->1, 1->3\nQuestion: Is there a directed path from node 0 to node 3?\nAnswer:"
    assert WITHHELD in format_item(enc, graph=False) and "0->1" not in format_item(enc, graph=False)
    shots = [{"encoding": enc, "y": 1}, {"encoding": encode_instance(4, [(1, 0)], 0, 1), "y": 0}]
    prompt = build_prompt(enc, shots)
    assert prompt.count("Answer: Yes") == 1 and prompt.count("Answer: No") == 1
    assert prompt.endswith("Answer:")


def test_shots_are_balanced_distinct_training_graphs():
    rows = _rows()
    shots = choose_shots(rows, k=4)
    assert [s["y"] for s in shots] == [1, 0, 1, 0]
    assert len({s["edge_hash"] for s in shots}) == 4 and all(s["split"] == "train" for s in shots)
    assert choose_shots(rows, k=4) == shots  # seeded


def test_samples_keep_whole_graphs_per_hop():
    rows = [r for r in _rows() if r["split"] == "val"]
    chosen = sample_graphs(rows, per_hop=1)
    per_graph = {}
    for r in chosen:
        per_graph.setdefault(r["edge_hash"], []).append(r)
    assert all(len(v) == 4 for v in per_graph.values())  # crossed: all four questions
    hops = sorted(next(r["hop_distance"] for r in v if r["y"] == 1) for v in per_graph.values())
    assert hops == [2, 3, 4, 5, 6]
    assert sample_graphs(rows, per_hop=1) == chosen


def test_auroc_and_metrics():
    assert auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert auroc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5
    rows = [r for r in _rows() if r["split"] == "val"][:8]
    m = metrics(rows, [1.0 if r["y"] == 1 else -1.0 for r in rows])
    assert m["acc"] == 1.0 and m["auroc"] == 1.0 and m["yes_rate"] == 0.5


class _Tok:
    """Word-level stand-in: 'Yes' and 'No' diverge after a shared space token."""

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=True):
        ids = []
        for word in text.replace(" ", " ▁").split(" "):
            ids.append(self.vocab.setdefault(word, len(self.vocab)))
        return {"input_ids": ids}


def test_answer_tokens_find_the_first_diverging_token():
    tok = _Tok()
    prefix, yes, no = answer_tokens(tok, "Question: path?\nAnswer:")
    assert yes != no and prefix == tok("Question: path?\nAnswer:")["input_ids"]


def test_batched_scores_match_direct_forward_passes():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from reachability_gen.run_llm_reference import score_prompts

    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", local_files_only=True)
    except Exception:
        pytest.skip("no locally cached tokenizer")
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2)
    model = transformers.Qwen2ForCausalLM(cfg).eval()
    rows = [r for r in _rows() if r["split"] == "val"][:6]
    prompts = [build_prompt(r["encoding"], choose_shots(_rows(), k=2)) for r in rows]
    batched = score_prompts(model, tok, prompts, "cpu", tokens_per_batch=900)
    with torch.inference_mode():
        for p, m in zip(prompts, batched):
            pre, y, n = answer_tokens(tok, p)
            logits = model(torch.tensor([pre])).logits[0, -1]
            assert abs(float(logits[y] - logits[n]) - m) < 1e-4


@pytest.mark.skipif(not ARTIFACT.exists(), reason="LLM reference artifact not present")
def test_llm_reference_artifact_contract():
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert art["science_open"] is False
    for res in art["models"].values():
        assert set(res["sets"]) == {"crossed_val", "crossed_long", "paired_val", "no_graph"}
        for name, entry in res["sets"].items():
            assert entry["n"] == len(art["sets"][name]["rows"]) == len(entry["margins"])
            assert 0.0 <= entry["acc"] <= 1.0

# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the step-by-step LLM reference runner (parsing and scoring; no model needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reachability_gen.encode import encode_instance
from reachability_gen.run_llm_cot import WITHHELD, parse_answer, question, summarise

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "llm_cot.json"


def test_question_lists_edges_or_withholds_them():
    enc = encode_instance(4, [(0, 1), (1, 3)], 0, 3)
    q = question(enc)
    assert "0->1, 1->3" in q and "from node 0 to node 3" in q and "'Answer: Yes'" in q
    assert WITHHELD in question(enc, graph=False) and "0->1" not in question(enc, graph=False)


@pytest.mark.parametrize("text, want", [
    ("0 reaches 1, 1 reaches 3.\nAnswer: Yes", 1),
    ("... so no path exists.\n**Answer: No**", 0),
    ("Answer: Yes? Let me recheck... Answer: No", 0),  # the last one counts
    ("ANSWER - yes", 1),
    ("I cannot tell.", None),
    ("The answer is yesterday's", None),
])
def test_parse_answer(text, want):
    assert parse_answer(text) == want


def test_summarise_counts_unparsed_as_wrong():
    rows = [{"edge_hash": "g", "y": 1, "hop_distance": 2}, {"edge_hash": "g", "y": 0, "hop_distance": -1},
            {"edge_hash": "h", "y": 1, "hop_distance": 3}, {"edge_hash": "h", "y": 0, "hop_distance": -1}]
    out = summarise(rows, [1, 0, None, 1], [10, 20, 768, 30], limit=768)
    assert out["acc"] == 0.5 and out["parsed_rate"] == 0.75 and out["hit_token_limit"] == 1
    assert out["acc_by_graph_hop"] == {"2": 1.0, "3": 0.0}


@pytest.mark.skipif(not ARTIFACT.exists(), reason="step-by-step LLM artifact not present")
def test_llm_cot_artifact_contract():
    art = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert art["science_open"] is False and art["protocol"]["decoding"] == "greedy"
    for res in art["models"].values():
        for name, entry in res["sets"].items():
            assert entry["n"] == len(art["sets"][name])
            assert 0.0 <= entry["acc"] <= entry["parsed_rate"] <= 1.0


def test_generate_halves_a_batch_that_runs_out_of_memory():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from reachability_gen.run_llm_cot import generate

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = 0
        padding_side = "right"

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return messages[0]["content"]

        def __call__(self, texts, return_tensors=None, padding=False, add_special_tokens=True):
            if isinstance(texts, str):
                return {"input_ids": [int(x) for x in texts.split()]}
            rows = [[int(x) for x in t.split()] for t in texts]
            width = max(map(len, rows))
            return Batch(input_ids=torch.tensor([[0] * (width - len(r)) + r for r in rows]))

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(str(int(i)) for i in ids)

    class Model(torch.nn.Module):
        config = SimpleNamespace(num_attention_heads=1, num_key_value_heads=1, head_dim=1, hidden_size=1,
                                 num_hidden_layers=1)

        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(10, 1)
            self.calls = []

        def get_input_embeddings(self):
            return self.emb

        def generate(self, input_ids, do_sample, max_new_tokens, pad_token_id):
            self.calls.append(len(input_ids))
            if len(input_ids) > 2:
                raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
            return torch.cat([input_ids, input_ids[:, -1:] * 10], dim=1)  # "answer": last token times ten

    model, stats = Model(), {"oom_splits": 0}
    texts, lengths = generate(model, Tokenizer(), ["1", "2", "3", "4", "5"], "cpu", batch=8, max_new_tokens=1,
                              stats=stats)
    assert texts == ["10", "20", "30", "40", "50"] and lengths == [1] * 5
    assert stats["oom_splits"] == 2 and model.calls == [5, 2, 3, 1, 2]

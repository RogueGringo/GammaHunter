# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the edge-sentence grammar, the extraction task and the frame-coverage study."""

from __future__ import annotations

import json
import random
import re

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.tinylm import grammar as g  # noqa: E402
from reachability_gen.tinylm.extract import (  # noqa: E402
    MALFORMED,
    accuracy,
    classify,
    collate,
    greedy_answers,
    train,
)
from reachability_gen.tinylm.model import TinyConfig, TinyLM  # noqa: E402
from reachability_gen.tinylm.tokenizer import EOS  # noqa: E402

TINY = TinyConfig(d=64, layers=2, heads=4, mlp_hidden=192, max_len=128)


def test_every_frame_states_its_edge_in_its_recorded_order_in_lower_case():
    rng = random.Random(0)
    for name, frame in g.FRAMES.items():
        for _ in range(200):
            u, v = rng.sample(range(g.N_NODES), 2)
            s = g.realize(frame, u, v, rng)
            assert s == s.lower(), s
            nums = [int(x) for x in re.findall(r"\d+", s)]
            assert nums == ([u, v] if frame.order == "S" else [v, u]), (name, nums)


def test_pool_is_balanced_and_held_out_frames_are_new_in_structure_only():
    assert len(g.POOL) == 12 and len(g.HELD_OUT) == 4 and not set(g.POOL) & set(g.HELD_OUT)
    assert sum(g.FRAMES[n].order == "T" for n in g.POOL) == 6
    assert [g.FRAMES[n].order for n in g.HELD_OUT] == ["S", "T", "T", "T"]
    pool_words = set().union(*(g.frame_words(n) for n in g.POOL))
    pool_chars = set().union(*(g.frame_chars(n) for n in g.POOL))
    for name in g.HELD_OUT:
        assert g.frame_words(name) <= pool_words, (name, g.frame_words(name) - pool_words)
        assert g.frame_chars(name) <= pool_chars, (name, g.frame_chars(name) - pool_chars)  # bytes too
    rng = random.Random(1)
    seen = set()
    for name in g.POOL:
        for _ in range(300):
            seen |= set(g.realize(g.FRAMES[name], *rng.sample(range(g.N_NODES), 2), rng))
    for name in g.HELD_OUT:
        for _ in range(300):
            assert set(g.realize(g.FRAMES[name], *rng.sample(range(g.N_NODES), 2), rng)) <= seen
    skeletons = [g.skeleton(n) for n in g.FRAMES]
    assert len(set(skeletons)) == len(skeletons)  # no two frames share a structure


def test_the_controls_score_as_documented():
    evals = g.eval_sets(50, 1)
    assert evals == g.eval_sets(50, 1)  # fixed
    for name, rows in evals.items():
        order = sum(g.order_heuristic(s) == a for s, a in rows) / len(rows)
        mirror = sum(g.mirror_heuristic(s) == a for s, a in rows) / len(rows)
        marker = sum(g.marker_heuristic(s) == a for s, a in rows) / len(rows)
        subject = sum(g.subject_marker_heuristic(s) == a for s, a in rows) / len(rows)
        if name == "distractor":
            assert order == mirror == marker == subject == 1.0
            continue
        source_first = g.FRAMES[name].order == "S"
        assert (order, mirror) == ((1.0, 0.0) if source_first else (0.0, 1.0)), name
        assert marker == 1.0, name  # familiar local markers, prepositions first, read every frame
        assert subject == (0.0 if name == "locative_inversion" else 1.0), name  # the target stands before the verb


def test_orderings_alternate_role_order_so_every_even_prefix_is_balanced():
    from reachability_gen.run_coverage import KS, ORDERINGS, ordering

    for o in range(ORDERINGS):
        frames = ordering(o)
        assert sorted(frames) == sorted(g.POOL)
        for k in KS:
            assert sum(g.FRAMES[n].order == "T" for n in frames[:k]) == k // 2
        assert g.FRAMES[frames[0]].order == ("S" if o % 2 == 0 else "T")


def test_loss_mask_covers_the_answer_and_eos_only():
    ids, mask = collate([("3 leads to 7.", "3>7"), ("node 5 is marked.", "none")], "cpu")
    for b, ans in enumerate(("3>7", "none")):
        assert ids[b, 1:][mask[b]].tolist() == list(ans.encode()) + [EOS]


def test_scoring_is_exact_and_sorts_errors():
    assert classify("3>7", "3>7") == "correct" and classify("7>3", "3>7") == "reversed"
    assert classify("none", "3>7") == "none" and classify("3>9", "3>7") == "other"
    assert classify(MALFORMED, "3>7") == "malformed" and classify("3>", "3>7") == "malformed"
    torch.manual_seed(0)
    out = greedy_answers(TinyLM(TINY), ["3 leads to 7.", "there is a link to 40 from 12.", "3 goes to 9."], "cpu")
    assert len(out) == 3 and all(isinstance(a, str) for a in out)


def test_a_reply_without_eos_or_with_a_special_token_is_malformed():
    from reachability_gen.tinylm.tokenizer import SEP, VOCAB_SIZE

    class Stub(torch.nn.Module):  # always predicts one token
        def __init__(self, token):
            super().__init__()
            self.token = token

        def forward(self, ids):
            logits = torch.zeros(ids.shape[0], ids.shape[1], VOCAB_SIZE)
            logits[..., self.token] = 1.0
            return logits

    assert greedy_answers(Stub(SEP), ["3 leads to 7."], "cpu") == [MALFORMED]  # a special token, never the end
    assert greedy_answers(Stub(ord("3")), ["3 leads to 7."], "cpu") == [MALFORMED]  # no end within the budget
    assert greedy_answers(Stub(EOS), ["3 leads to 7."], "cpu") == [""]  # an empty answer, scored wrong


def test_a_tiny_model_learns_to_read_one_frame():
    torch.manual_seed(0)
    model = TinyLM(TINY)
    log = train(model, ("svo",), steps=1500, batch=32, lr=3e-3, warmup=50, weight_decay=0.0, clip=1.0,
                distractor_rate=0.2, data_seed="t", device="cpu", log_every=100)
    assert log["losses"][-1] < log["losses"][0]
    rows = g.eval_sets(40, 3)
    assert accuracy(model, rows["svo"], "cpu") >= 0.8 and accuracy(model, rows["distractor"], "cpu") >= 0.9


def test_coverage_study_smoke_and_decision(tmp_path, monkeypatch):
    from reachability_gen import run_coverage as rc

    for name, value in (("KS", (2, 4)), ("ORDERINGS", 2), ("STEPS", 20), ("BATCH", 8), ("EVAL_PER_FRAME", 10),
                        ("CONFIG", TINY), ("WARMUP", 5), ("EVAL_SEED", 7), ("DATA_PREFIX", "test-data")):
        monkeypatch.setattr(rc, name, value)  # the smoke run never touches the study's evaluation or data seeds
    monkeypatch.setattr(rc, "RUNS_OUT", tmp_path / "runs.json")
    monkeypatch.setattr(rc, "DECIDE_OUT", tmp_path / "decide.json")
    assert rc.main(["--device", "cpu"]) == 0
    art = json.loads((tmp_path / "runs.json").read_text())
    assert art["complete"] and len(art["runs"]) == 4 and rc.check(art) == []
    assert "code_dirty" in art["protocol"] and "code_revision" in art["protocol"]
    run = art["runs"][0]
    assert set(run["accuracy"]) == set(g.FRAMES) | {"distractor"} and set(run["untrained_heldout"]) == set(g.HELD_OUT)
    assert set(run["errors"]["cleft_target"]) == {"reversed", "none", "malformed", "other"}
    assert set(run["heldout_word_coverage"]) == set(g.HELD_OUT)
    assert art["heuristics"]["order"]["cleft_target"] == 0.0 and art["heuristics"]["mirror"]["cleft_target"] == 1.0
    assert art["protocol"]["eval_seed"] == 7 and art["protocol"]["betas"] == [0.9, 0.95]
    first = [r for r in art["runs"] if r["ordering"] == 0]
    assert first[0]["frames"] == first[1]["frames"][:2]  # nested within an ordering
    assert rc.main(["--decide"]) == 0
    verdicts = json.loads((tmp_path / "decide.json").read_text())["verdicts"]
    assert set(verdicts) == {"H1", "k12_level", "reported"}
    art["protocol"]["steps"] = 999
    assert any("steps" in i for i in rc.check(art))  # the run file must match the code


def _runs(t2, t12, *, under=0, under_k=2):
    from reachability_gen import run_coverage as rc

    runs = []
    for o in range(6):
        for k in rc.KS:
            ht = t2[o] if k == 2 else t12[o] if k == 12 else 0.5
            runs.append({"ordering": o, "k": k, "heldout_T": ht, "heldout_S": 1.0, "distractor": 1.0,
                         "in_distribution": 0.5 if (k == under_k and o < under) else 1.0,
                         "accuracy": {n: ht for n in g.HELD_OUT},
                         "errors": {n: {"reversed": 0, "none": 0, "malformed": 0, "other": 0} for n in g.HELD_OUT},
                         "heldout_word_coverage": {n: 1.0 for n in g.HELD_OUT}})
    return {"runs": runs}


def test_decision_rules_cover_every_branch():
    from reachability_gen.run_coverage import decide, spearman

    low, high = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05], [0.80, 0.85, 0.90, 0.92, 0.95, 0.97]
    v = decide(_runs(low, high))
    assert v["H1"]["verdict"] == "supported" and v["k12_level"]["classification"] == "partial"
    assert decide(_runs(low, [x + 0.05 for x in high]))["k12_level"]["classification"] == "generalizes"
    assert decide(_runs(low, low))["H1"]["verdict"] == "excluded"
    assert decide(_runs(low, low))["k12_level"]["classification"] == "fails"
    wide = [0.0, 0.9, 0.0, 0.8, 0.0, 0.2]  # a gain short of the margin that cannot be ruled out either
    assert decide(_runs(low, wide))["H1"]["verdict"] == "inconclusive"
    assert decide(_runs(low, high, under=2))["H1"]["verdict"] == "uninterpretable (undertrained)"
    guarded = decide(_runs(low, high, under=2, under_k=12))
    assert guarded["H1"]["verdict"] == guarded["k12_level"]["classification"] == "uninterpretable (undertrained)"
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0) and spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)

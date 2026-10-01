# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the edge-sentence grammar, the extraction task and the frame-coverage study."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

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
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts" / "coverage_runs.json"
DECISION = ROOT / "artifacts" / "coverage.json"


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


@pytest.mark.skipif(not (RUNS.exists() and DECISION.exists()), reason="frame-coverage artifacts not present")
def test_coverage_artifacts_contract():
    """The recorded study: the verdicts re-derive from the run file, and every value limitations item 23
    reports is pinned to the run file, the grammar or the runner."""
    from reachability_gen import run_coverage as rc

    art = json.loads(RUNS.read_text(encoding="utf-8"))
    assert rc.check(art) == [] and art["science_open"] is False and len(art["runs"]) == 24
    pr = art["protocol"]
    assert pr["code_revision"].startswith("bdae5a1") and pr["code_dirty"] is False  # the design commit, clean
    assert "RTX 5070" in pr["gpu"] and pr["steps"] == 2000 and pr["eval_per_frame"] == 400
    assert round(TinyLM(rc.CONFIG).num_params() / 1e6, 1) == 4.9
    assert g.N_NODES == 48 and len(g.FRAMES) == 16 and (rc.H1_MARGIN, rc.PARTIAL) == (0.30, 0.5)
    assert len(g.eval_sets(rc.EVAL_PER_FRAME, rc.EVAL_SEED)["distractor"]) == 400
    for words in ("infer the best path all context eval for value based on prime intent", "what comes next?",
                  "lets make a small llm from the ground up ... you and i can parse out some ideas that will "
                  "convey some optimizations"):
        assert words in " ".join(rc.__doc__.split())  # the PI's words as the runner records them
    decided = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decided["science_open"] is False and decided["heuristics"] == art["heuristics"]
    assert decided["verdicts"] == json.loads(json.dumps(rc.decide(art)))  # the verdicts re-derive

    runs = {(r["ordering"], r["k"]): r for r in art["runs"]}
    orderings = range(rc.ORDERINGS)
    seconds = [r["training"]["seconds"] for r in art["runs"]]
    assert (round(min(seconds)), round(max(seconds)), round(art["elapsed_seconds"] / 60)) == (73, 81, 32)

    # controls
    assert all(r["accuracy"][n] == 1.0 for r in art["runs"] for n in r["frames"])
    assert all(r["distractor"] == 1.0 and r["training"]["losses"][-1] < 1.6e-6 for r in art["runs"])
    assert all(v == 0.0 for r in art["runs"] for v in r["untrained_heldout"].values())
    heur = art["heuristics"]
    for name in g.FRAMES:
        source_first = g.FRAMES[name].order == "S"
        assert (heur["order"][name], heur["mirror"][name]) == ((1.0, 0.0) if source_first else (0.0, 1.0))
        assert heur["marker"][name] == 1.0
        assert heur["subject_marker"][name] == (0.0 if name == "locative_inversion" else 1.0)

    # verdicts
    v = decided["verdicts"]
    h1 = v["H1"]
    assert h1["verdict"] == "inconclusive" and round(h1["gain"], 3) == 0.265
    assert round(h1["p"], 4) == 0.0032 and round(h1["p_gain_below_margin"], 2) == 0.33
    assert all(n == 0 for n in h1["undertrained_by_k"].values())
    t = h1["heldout_T_by_k"]
    assert all(a > b for a, b in zip(t["12"], t["2"]))  # higher at k = 12 in every ordering
    assert v["k12_level"]["classification"] == "fails" and round(v["k12_level"]["mean_heldout_T"], 3) == 0.383

    # per held-out construction at k = 12
    def at(k, name):
        return [runs[o, k]["accuracy"][name] for o in orderings]

    assert at(12, "locative_inversion") == [1.0] * 6
    assert (round(min(at(12, "participial")), 4), max(at(12, "participial"))) == (0.9475, 1.0)
    assert (round(min(at(12, "cleft_target")), 4), round(max(at(12, "cleft_target")), 4)) == (0.0075, 0.0725)
    assert (round(min(at(12, "relative_target")), 4), round(max(at(12, "relative_target")), 4)) == (0.0125, 0.32)

    def pooled(k, name):
        return {kind: sum(runs[o, k]["errors"][name][kind] for o in orderings) for kind in
                ("reversed", "none", "malformed", "other")}

    cleft, relative = pooled(12, "cleft_target"), pooled(12, "relative_target")
    assert (cleft["other"], sum(cleft.values()), cleft["reversed"]) == (2100, 2343, 239)
    assert (relative["other"], sum(relative.values()), relative["reversed"]) == (2011, 2100, 83)
    assert (cleft["malformed"], cleft["none"], relative["malformed"], relative["none"]) == (4, 0, 6, 0)
    assert all(r["accuracy"]["relative_source"] == 1.0 for r in art["runs"] if "relative_source" in r["frames"])

    # which numbers a preposition stands directly before, and the marker rule's fallback on the subject position
    prepositions = {"at", "from", "to", "into", "{to}"}

    def tokens(name):
        return re.findall(r"\{[a-z_]+\}|[a-z]+", g.FRAMES[name].pattern)

    def marked(name):
        toks = tokens(name)
        return [i > 0 and toks[i - 1] in prepositions for i, tok in enumerate(toks) if tok in ("{u}", "{v}")]

    assert [n for n in g.HELD_OUT if all(marked(n))] == ["participial", "locative_inversion"]
    assert [n for n in g.HELD_OUT if not any(marked(n))] == ["cleft_target", "relative_target"]
    assert [n for n in g.POOL if not any(marked(n))] == ["relative_source"]
    for name in ("cleft_target", "relative_target"):  # no preposition marks a number; the source stands before a verb
        assert tokens(name)[tokens(name).index("{u}") + 1] == "{sv}"

    # not monotone in k (reported, not decided)
    means = v["reported"]["mean_by_k"]
    assert [round(means[str(k)]["heldout_T"], 3) for k in rc.KS] == [0.118, 0.239, 0.576, 0.383]
    assert [round(means[str(k)]["heldout_T"], 2) for k in (2, 12)] == [0.12, 0.38]  # related work
    assert round(v["reported"]["spearman_k_heldout_T_descriptive"], 2) == 0.70
    above = [a - b for a, b in zip(t["8"], t["12"])]
    assert sum(d > 0 for d in above) == 5 and round(min(d for d in above if d > 0), 3) == 0.004
    coverage8 = [sum(runs[o, 8]["heldout_word_coverage"].values()) / 4 for o in orderings]
    assert (round(min(coverage8), 3), max(coverage8), sum(c < 1.0 for c in coverage8)) == (0.959, 1.0, 5)
    assert all(c == 1.0 for o in orderings for c in runs[o, 12]["heldout_word_coverage"].values())
    full = [(r["ordering"], r["k"]) for r in art["runs"]
            if r["k"] < 12 and all(c == 1.0 for c in r["heldout_word_coverage"].values())]
    assert full == [(1, 8)]  # the one run below k = 12 that covered every held-out word
    rng = random.Random(0)  # at fixed steps, each construction is drawn two-thirds as often among 12 as among 8
    drawn = {k: sum(g.sample(tuple(g.POOL[:k]), rng, rc.DISTRACTOR_RATE)[2] == g.POOL[0] for _ in range(60000))
             for k in (8, 12)}
    assert drawn[12] / drawn[8] == pytest.approx(2 / 3, rel=0.05)

    # post hoc
    with_svo = [o for o in orderings if "svo" in runs[o, 8]["frames"]]
    without_conditional = [o for o in orderings if "conditional_source" not in runs[o, 8]["frames"]]
    assert with_svo == without_conditional == [0, 1]
    rel8 = {o: runs[o, 8]["accuracy"]["relative_target"] for o in orderings}
    assert (round(min(rel8[o] for o in (2, 3, 4, 5)), 4), max(rel8[o] for o in (2, 3, 4, 5))) == (0.6025, 1.0)
    assert (round(min(rel8[o] for o in (0, 1)), 4), round(max(rel8[o] for o in (0, 1)), 4)) == (0.0125, 0.0475)
    rel12 = [runs[o, 12]["accuracy"]["relative_target"] for o in (2, 3, 4, 5)]
    assert all(b < rel8[o] for o, b in zip((2, 3, 4, 5), rel12))
    added = set.intersection(*(set(runs[o, 12]["frames"][8:]) for o in (2, 3, 4, 5)))
    assert added == {"svo", "noun_target"}  # the only constructions added in all four of those orderings
    assert (round(min(rel12), 4), round(max(rel12), 4)) == (0.0125, 0.32)
    read = [(r["ordering"], r["k"]) for r in art["runs"] if r["accuracy"]["cleft_target"] > 0.5]
    assert read == [(5, 8)] and round(runs[5, 8]["accuracy"]["cleft_target"], 3) == 0.855
    assert [o for o in orderings if "cleft_source" not in runs[o, 8]["frames"]] == [5]
    assert round(max(r["accuracy"]["cleft_target"] for r in art["runs"]
                     if (r["ordering"], r["k"]) != (5, 8)), 4) == 0.0725
    assert round(max(r["accuracy"]["cleft_target"] for r in art["runs"]
                     if r["k"] in (2, 4) and "cleft_source" not in r["frames"]), 4) == 0.0625

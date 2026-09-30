# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the tuned encoder (LoReFT, embedding deltas) and the embedding-and-representation tuning study."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from reachability_gen.tuned_encoder import EncoderAdapter, LoReFT, TunedEncoder  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
QWEN = "Qwen/Qwen2.5-0.5B"


def _tokenizer():
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    except Exception:  # noqa: BLE001 - the tokenizer is not in the local cache
        pytest.skip("the Qwen2.5 tokenizer is not in the local model cache")


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A tiny random Qwen2 model with the real tokenizer, saved locally (nothing is downloaded)."""
    from transformers import Qwen2Config, Qwen2Model

    tok = _tokenizer()
    torch.manual_seed(0)
    config = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    path = tmp_path_factory.mktemp("tiny_qwen")
    Qwen2Model(config).to(torch.bfloat16).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def test_loreft_starts_as_the_identity_with_orthonormal_rows():
    gen = torch.Generator().manual_seed(1)
    ft = LoReFT(16, 4, gen)
    r = ft.projection().detach()
    assert torch.allclose(r @ r.T, torch.eye(4), atol=1e-5) and torch.allclose(r, ft.basis.detach(), atol=1e-5)
    h = torch.randn(3, 5, 16) * 100  # large coordinates, as in the encoder's hidden states
    assert torch.equal(ft(h), h)
    ft(h).pow(2).sum().backward()  # the identity, but D and b receive a gradient
    assert ft.learned.weight.grad.abs().sum() > 0 and ft.learned.bias.grad.abs().sum() > 0
    with torch.no_grad():
        ft.learned.bias.fill_(0.5)  # once the edit is non-zero, R receives a gradient too
    ft.zero_grad()
    ft(h).pow(2).sum().backward()
    assert ft.basis.grad.abs().sum() > 0
    # the same interventions as a free W = R + D: h + Rᵀ(W h + b − R h)
    r, dm, b = ft.projection().detach(), ft.learned.weight.detach(), ft.learned.bias.detach()
    w = r + dm
    assert torch.allclose(ft(h).detach(), h + (h @ w.T + b - h @ r.T) @ r, atol=1e-3)
    with torch.no_grad():
        ft.basis.mul_(3.0)  # R stays orthonormal whatever the basis's scale
    r = ft.projection().detach()
    assert torch.allclose(r @ r.T, torch.eye(4), atol=1e-5)


def test_adapter_leaves_the_global_random_stream_untouched():
    torch.manual_seed(5)
    before = torch.get_rng_state()
    EncoderAdapter(d=16, layers=3, vocab_size=50, train_ids=[1, 4, 9], embeddings=True, interventions=True,
                   rank=4, seed=7)
    assert torch.equal(torch.get_rng_state(), before)
    a = EncoderAdapter(d=16, layers=2, vocab_size=50, train_ids=[1], embeddings=False, interventions=True, rank=4,
                       seed=7)
    b = EncoderAdapter(d=16, layers=2, vocab_size=50, train_ids=[1], embeddings=False, interventions=True, rank=4,
                       seed=7)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    summary = EncoderAdapter(d=16, layers=2, vocab_size=50, train_ids=[1, 2], embeddings=True, interventions=True,
                             rank=4, seed=7).summary(0.02)
    assert summary["embedding_delta"]["rms_over_embedding_rms"] == 0 and len(summary["interventions"]) == 2


def test_tuned_encoder_reproduces_the_frozen_features_and_trains(tiny_model):
    from transformers import AutoModel

    from reachability_gen.run_nl_reader import lm_features

    texts = ["0 -> 3. There is an edge from 3 to 1. Node 2 is marked.", "Starting at 2, one step takes you to 0."]
    frozen = lm_features(texts, [4, 4], tiny_model, 2, "cpu")
    enc = TunedEncoder(tiny_model, 2, "cpu")
    items = enc.encode_texts(texts, [4, 4])
    assert all(torch.equal(a.features, b.features) for a, b in zip(enc.feature_tokens(items), frozen))
    assert [x.kinds.tolist() for x in items] == [f.kinds.tolist() for f in frozen]
    full = AutoModel.from_pretrained(tiny_model, local_files_only=True, dtype=torch.bfloat16)
    ids = items[1].ids[None]
    with torch.no_grad():
        assert torch.equal(full(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
                           .hidden_states[2][0], enc.hidden([items[1]])[0][0])
    train_ids = sorted({int(i) for i in items[0].ids.tolist()})
    enc.adapter = EncoderAdapter(d=enc.d, layers=2, vocab_size=enc.vocab_size, train_ids=train_ids,
                                 embeddings=True, interventions=True, rank=4, seed=0)
    assert all(torch.equal(a.features, b.features) for a, b in zip(enc.feature_tokens(items), frozen))  # identity
    enc.model.train()
    live = enc.reader_batch(items)["features"]
    assert all(torch.equal(live[i, : x.ids.numel()].detach(), f.features) for i, (x, f) in enumerate(zip(items, frozen)))
    live.float().pow(2).mean().backward()
    assert enc.adapter.delta.grad.abs().sum() > 0 and all(ft.learned.weight.grad.abs().sum() > 0
                                                          for ft in enc.adapter.refts)
    unseen = [i for i in items[1].ids.tolist() if i not in train_ids]
    assert unseen and all(enc.adapter.slot[i] == -1 for i in unseen)  # tokens absent from training have no delta
    with torch.no_grad():
        enc.adapter.delta.add_(0.1)
    assert not torch.equal(enc.feature_tokens(items)[0].features, frozen[0].features)


def test_interventions_alone_receive_gradients_through_checkpointed_blocks(tiny_model):
    enc = TunedEncoder(tiny_model, 2, "cpu", checkpointing=True)
    items = enc.encode_texts(["0 -> 3. 3 heads to 1.", "Node 2 is marked. 2 feeds 0."], [4, 4])
    enc.adapter = EncoderAdapter(d=enc.d, layers=2, vocab_size=enc.vocab_size, train_ids=[], embeddings=False,
                                 interventions=True, rank=4, seed=0)
    assert enc.model.is_gradient_checkpointing
    enc.model.train()
    out = enc.reader_batch(items)["features"]
    assert out.requires_grad  # no input requires a gradient; only the interventions inside the blocks do
    out.float().pow(2).mean().backward()
    assert all(ft.learned.weight.grad is not None and ft.learned.weight.grad.abs().sum() > 0 for ft in enc.adapter.refts)


def test_sizes_and_capacity_width():
    from reachability_gen.run_tuned_reader import capacity_width, intervention_params, reader_params

    base = reader_params(64, 32)
    d = capacity_width(50_000, 32)
    assert d % 4 == 0 and abs(reader_params(d, 32) - (base + 50_000)) <= abs(reader_params(d + 4, 32) - (base + 50_000))
    assert capacity_width(0, 32) == 64
    assert intervention_params(12, 896, 4) == 86_064 and intervention_params(12, 896, 7) == 150_612


def test_tuned_study_smoke_and_decision(tiny_model, tmp_path, monkeypatch):
    from reachability_gen import run_tuned_reader as rt
    from reachability_gen.gen_crossed import CROSSED_EXTENDED_SPEC, generate_crossed

    monkeypatch.setattr(rt, "LONG_STEPS", (16,))
    rows = [e.to_dict() for e in generate_crossed(seed=21, n_total=40, n_val=20)[0]]
    ext = [e.to_dict() for e in generate_crossed(seed=22, n_total=20, n_val=20, spec=CROSSED_EXTENDED_SPEC)[0]]
    train_path, ext_path = tmp_path / "train.jsonl", tmp_path / "ext.jsonl"
    train_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ext_path.write_text("".join(json.dumps(r) + "\n" for r in ext))
    monkeypatch.chdir(tmp_path)
    for arm in rt.ARMS:
        assert rt.main(["--arm", arm, "--encoder", tiny_model, "--layer", "2", "--train-data", str(train_path),
                        "--extended-data", str(ext_path), "--seeds", "0", "1", "--reader-epochs", "1",
                        "--solver-epochs", "1", "--ckpt-dir", str(tmp_path / "ckpt")]) == 0
    load = lambda arm: json.loads((tmp_path / "artifacts" / f"tuned_reader_{arm}.json").read_text())  # noqa: E731
    both = load("both")
    run = both["runs"][0]
    assert run["start_audit"] == {"eval_identical": True, "train_path_identical": True}
    assert both["self_audit_mismatches"] == [] and both["protocol"]["adapter_params"] == both["protocol"]["sizes"]["both_params"]
    assert set(run["templates"]) == {"val_in", "val_out", "val_novel"} and len(run["templates"]["val_out"]["recall"]) == 4
    assert 0 <= run["construction_recall"] <= 1 and len(run["batch_losses"]) == 1 and "adapter_moved" in run
    assert both["protocol"]["tuned"] == {"embeddings": True, "interventions": True} and both["protocol"]["embedding_lr"] > 0
    ckpt = torch.load(run["checkpoint_path"], map_location="cpu", weights_only=True)
    assert ckpt["adapter"]["train_ids"] == both["protocol"]["tuned_token_ids"]  # the checkpoint is self-contained
    matched = load("interventions_matched")
    assert matched["protocol"]["rank"] == both["protocol"]["sizes"]["matched_rank"]
    cap = load("capacity")
    extra = cap["protocol"]["capacity_extra_params"]
    assert extra == both["protocol"]["sizes"]["both_params"] and cap["protocol"]["reader_width"] == rt.capacity_width(extra, 32)
    assert load("frozen")["protocol"]["reader_width"] == 64
    assert rt.main(["--decide"]) == 1  # item 20's files are required unless explicitly waived
    assert rt.main(["--decide", "--no-replication"]) == 0
    verdicts = json.loads((tmp_path / "artifacts" / "tuned_reader.json").read_text())["verdicts"]
    assert verdicts["splits_per_test"] == math.comb(4, 2) and set(verdicts["reported"]) == set(rt.ARMS)
    assert set(verdicts["P2"]) == {"verdict", "excluded_alpha", *rt.COMPONENTS}


def _art(arm, recall, *, in_f1=0.99, precision=0.95, collapsed=0):
    runs = [{"seed": s, "construction_recall": r, "passes_pi": False, "novel_mean_recall": 0.4, "novel_bar_met": False,
             "collapsed": s < collapsed, "final": {"val_in": {"reader": {"f1": in_f1}},
                                                   "val_out": {"reader": {"precision": precision, "f1": 0.9}}}}
            for s, r in enumerate(recall)]
    rank = 4 if arm in ("interventions", "both") else None
    return {"arm": arm, "runs": runs, "self_audit_mismatches": [],
            "protocol": {"seeds": list(range(len(recall))), "dataset_verified": True, "rank": rank}}


BASE = [0.40, 0.42, 0.44, 0.46, 0.48]
HIGH = [0.95, 0.96, 0.97, 0.98, 0.99]


def _arts(both, *, capacity=BASE, components=BASE, **kw):
    return {"frozen": _art("frozen", BASE), "embeddings": _art("embeddings", components),
            "interventions": _art("interventions", components),
            "interventions_matched": _art("interventions_matched", components),
            "capacity": _art("capacity", capacity), "both": _art("both", both, **kw)}


def test_decision_rules_cover_every_branch():
    from reachability_gen.run_tuned_reader import check_arts, decide

    assert decide(_arts(HIGH))["P1"]["verdict"] == "supported"
    assert decide(_arts(HIGH))["P2"]["verdict"] == "shown"
    assert decide(_arts(HIGH, in_f1=0.90))["P1"]["verdict"] == "confounded (in-distribution)"
    assert decide(_arts(HIGH, precision=0.80))["P1"]["verdict"] == "confounded (over-prediction)"
    assert decide(_arts(HIGH, capacity=HIGH))["P1"]["verdict"] == "not separated from capacity"
    assert decide(_arts(BASE))["P1"]["verdict"] == "excluded at these rates"  # a gain of 0.30 is rejected
    wide = [0.10, 0.95, 0.20, 0.90, 0.60]
    assert decide(_arts(wide))["P1"]["verdict"] == "inconclusive"
    assert decide(_arts(BASE, collapsed=3))["P1"]["verdict"] == "uninterpretable (training collapsed)"
    p2 = decide(_arts(HIGH, components=HIGH))["P2"]["verdict"]
    assert p2.startswith("excluded") and "interventions_matched" in p2  # one component alone comes within 0.10
    assert decide(_arts(HIGH, components=HIGH))["P2"]["excluded_alpha"] == pytest.approx(0.05 / 3)
    assert decide(_arts(wide, components=wide))["P2"]["verdict"] == "inconclusive"
    assert decide(_arts(BASE, collapsed=3))["P2"]["verdict"] == "uninterpretable (training collapsed)"
    assert decide(_arts(HIGH, precision=0.80))["P2"]["verdict"].startswith("confounded (over-prediction vs")
    assert decide(_arts(HIGH, in_f1=0.90))["P2"]["verdict"].startswith("confounded (in-distribution vs")
    arts = _arts(HIGH)
    for r in arts["embeddings"]["runs"][:3]:
        r["collapsed"] = True  # a component that collapsed cannot show the combination is needed
    assert decide(arts)["P2"]["verdict"] == "confounded (collapsed component: embeddings)"
    assert len(decide(arts)["reported"]["embeddings"]["val_out_precision"]) == 5
    arts = _arts(HIGH)  # a control whose training-wording reading degraded cannot carry a verdict
    for arm in ("capacity", "interventions_matched"):
        for r in arts[arm]["runs"][:2]:
            r["final"]["val_in"]["reader"]["f1"] = 0.0  # two collapsed seeds: below the collapse rule's 3
    v = decide(arts)
    assert v["P1"]["verdict"] == "confounded (degraded control: capacity)"
    assert v["P2"]["verdict"] == "confounded (degraded component: interventions_matched)"
    arts = _arts(HIGH)
    assert check_arts(arts) == []
    arts["both"]["runs"] = arts["both"]["runs"][:4]
    arts["capacity"]["protocol"]["dataset_verified"] = False
    arts["interventions"]["protocol"]["rank"] = 8
    issues = check_arts(arts)
    assert any("both: seeds" in i for i in issues) and any("capacity: datasets" in i for i in issues)
    assert any("interventions: rank 8" in i for i in issues)


def test_replication_of_item20_criterion():
    from reachability_gen.run_tuned_reader import replication

    f1s, recalls = [0.92, 0.93, 0.90, 0.91, 0.94], [0.40, 0.45, 0.37, 0.50, 0.42]
    item20 = {"runs": [{"seed": s, "final": {"val_out": {"reader": {"f1": f}}}} for s, f in enumerate(f1s)]}
    templates = {"seeds": [{"seed": s, "heldout": {"recall": [1.0, 1.0, r, 0.9]}} for s, r in enumerate(recalls)]}

    def frozen(fs, rs):
        return {"runs": [{"seed": s, "final": {"val_out": {"reader": {"f1": f}}}, "construction_recall": r}
                         for s, (f, r) in enumerate(zip(fs, rs))]}

    assert replication(frozen(f1s, recalls), item20, templates)["verdict"] == "exact"
    near = frozen([f + 0.005 for f in f1s], [0.41, 0.44, 0.38, 0.49, 0.43])
    assert replication(near, item20, templates)["verdict"] == "in distribution"
    far = frozen(f1s, [r + 0.3 for r in recalls])
    assert replication(far, item20, templates)["verdict"] == "not replicated"

# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Torch-required tests for FeedForward model / trainer / overfit plumbing.

Skipped entirely when torch is not installed so pure schema CI stays green.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.arms import FeedForwardArm  # noqa: E402
from reachability_gen.harness.runner import (  # noqa: E402
    ExperimentHarness,
    default_stub_arms,
    eval_jsonl,
)
from reachability_gen.models.feedforward import FeedForward  # noqa: E402
from reachability_gen.overfit_ff import (  # noqa: E402
    ensure_balanced_batch,
    ensure_id_hop_batch,
    filter_id_hop,
    run_overfit,
)
from reachability_gen.tokenize import (  # noqa: E402
    build_vocab,
    encode_to_ids,
    split_encoding_tokens,
)
from reachability_gen.train.ff_trainer import FeedForwardTrainer, examples_to_batch  # noqa: E402


def test_tokenize_splits_edges():
    toks = split_encoding_tokens("N 3 EDGES 0,2 1,0 QUERY 0 2")
    assert toks == ["N", "3", "EDGES", "0", ",", "2", "1", ",", "0", "QUERY", "0", "2"]
    vocab = build_vocab(max_node_id=16)
    ids = encode_to_ids("N 3 EDGES 0,2 QUERY 0 2", vocab)
    assert vocab.pad_id == 0
    assert all(isinstance(i, int) for i in ids)


def test_feedforward_forward_shape_and_empty_traj():
    vocab = build_vocab()
    model = FeedForward(vocab_size=len(vocab), d=32, L=2, n_heads=4, max_len=64)
    ids = torch.tensor([[1, 2, 3, 4, 0, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
    logits, traj = model(ids, mask, return_trajectory=False)
    assert tuple(logits.shape) == (1, 2)
    assert traj is None
    logits2, traj2 = model(ids, mask, return_trajectory=True)
    assert tuple(logits2.shape) == (1, 2)
    assert traj2 is None


def test_trainer_train_step_returns_loss_acc():
    vocab = build_vocab()
    model = FeedForward(vocab_size=len(vocab), d=32, L=2, n_heads=4, max_len=64)
    trainer = FeedForwardTrainer(model, lr=1e-2, grad_clip=1.0)
    examples = [
        {"encoding": "N 3 EDGES 0,1 QUERY 0 1", "y": 1},
        {"encoding": "N 3 EDGES QUERY 1 0", "y": 0},
    ]
    token_ids, mask, labels, _ = examples_to_batch(examples, vocab)
    loss, acc = trainer.train_step(token_ids, labels, mask)
    assert isinstance(loss, float) and loss >= 0.0
    assert 0.0 <= acc <= 1.0


def test_ff_arm_real_forward_empty_trajectory():
    arm = FeedForwardArm(L=2, d=32)
    arm.attach_default_model(seed=0)
    assert arm.has_real_model
    logits, traj = arm.forward(
        {"encoding": ["N 4 EDGES 0,1 1,2 QUERY 0 2"], "y": [1]},
        return_trajectory=True,
    )
    assert isinstance(logits, list) and len(logits) == 1
    assert len(logits[0]) == 2
    assert traj == []


def test_harness_ff_real_loss_and_empty_drift(tmp_path: Path):
    from reachability_gen.generate import generate_split

    examples, _ = generate_split("train", n_per_cell=2, max_rejects=2000)
    ex = examples[0].to_dict()
    arms = default_stub_arms(d=32, T=4, L=1, attach_ff_model=True, ff_seed=0)
    ff = next(a for a in arms if a.name.startswith("ff-"))
    assert ff.has_real_model
    metrics = tmp_path / "m.jsonl"
    with ExperimentHarness(arms[:3], metrics_path=metrics, run_id="ff-real") as h:
        rec = h.eval_example(ex, ff)
    assert rec["arm"].startswith("ff-")
    # Real CE — not the stub 0.0 placeholder (untrained ≈ ln2, but any >0 or finite).
    assert isinstance(rec["loss"], float)
    assert rec["loss"] != 0.0 or rec["accuracy"] in (0.0, 1.0)
    assert rec["loss"] >= 0.0
    assert rec["drift_trajectory"] == []
    assert rec["terminal_drift"] is None
    assert rec["perturbation_delta"] is None
    assert rec["tokens_decoded"] is None


def test_eval_jsonl_ff_metrics_nonzero_loss(tmp_path: Path):
    from reachability_gen.generate import write_jsonl, generate_split

    examples, _ = generate_split("train", n_per_cell=2, max_rejects=2000)
    ex_path = tmp_path / "train.jsonl"
    write_jsonl(ex_path, examples)
    metrics = tmp_path / "metrics.jsonl"
    arms = default_stub_arms(d=32, attach_ff_model=True, ff_seed=1)
    n = eval_jsonl(ex_path, metrics, arms, limit=2, run_id="ff-eval")
    assert n == 8
    rows = [json.loads(line) for line in metrics.read_text().splitlines()]
    ff_rows = [r for r in rows if r["arm"].startswith("ff-")]
    assert ff_rows
    for r in ff_rows:
        assert r["drift_trajectory"] == []
        assert r["terminal_drift"] is None
        assert r["perturbation_delta"] is None
        assert r["tokens_decoded"] is None
        assert isinstance(r["loss"], float)
        assert isinstance(r["accuracy"], float)


@pytest.mark.slow
def test_overfit_gate_small_batch(tmp_path: Path):
    """Overfit a tiny synthetic batch (may regenerate); allow up to 100 steps."""
    # Build a tiny all-positive ID-hop-like set via generator, then filter.
    examples_path = tmp_path / "train.jsonl"
    batch, note = ensure_id_hop_batch(examples_path, target_n=16, regenerate=True)
    assert len(batch) >= 4, note
    # Keep the run short-ish but within the gate budget.
    result = run_overfit(batch[:16], steps=100, d=64, L=2, lr=5e-3, seed=0)
    assert result["ok"], (
        f"overfit failed: final_loss={result['final_loss']:.4f} "
        f"final_acc={result['final_acc']:.4f} note={note}"
    )
    assert result["final_loss"] < 0.05
    assert result["final_acc"] == pytest.approx(1.0)


@pytest.mark.slow
def test_balanced_overfit_gate(tmp_path: Path):
    """Balanced 16+16 hard-neg overfit; CE < 1e-3 and per-class acc=1.0."""
    examples_path = tmp_path / "train.jsonl"
    batch, note, meta = ensure_balanced_batch(
        examples_path, n_pos=8, n_neg=8, regenerate=True
    )
    assert len(batch) == 16, note
    assert sum(1 for e in batch if int(e["y"]) == 1) == 8
    assert sum(1 for e in batch if int(e["y"]) == 0) == 8
    assert meta.get("reject_reasons") is not None
    result = run_overfit(
        batch,
        steps=100,
        d=64,
        L=2,
        lr=5e-3,
        seed=0,
        loss_threshold=1e-3,
        require_per_class=True,
    )
    assert result["ok"], (
        f"balanced overfit failed: loss={result['final_loss']:.6f} "
        f"acc={result['final_acc']:.4f} per_class={result.get('per_class')} note={note}"
    )
    assert result["final_loss"] < 1e-3
    assert result["per_class"]["y0"] == pytest.approx(1.0)
    assert result["per_class"]["y1"] == pytest.approx(1.0)

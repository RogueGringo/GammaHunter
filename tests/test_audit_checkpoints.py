# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the truncation guard, collapse diagnostics and checkpoint audit."""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import pytest

from reachability_gen.diagnostics import (
    TOKEN_COLLAPSE_COS,
    collapse_flags,
    largest_train_acc_drop,
    median_abs_deviation,
)
from reachability_gen.tokenize import (
    build_vocab,
    encode_to_ids,
    max_token_len,
    pad_batch,
    required_max_len,
)

ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "artifacts" / "id_2k_checkpoint_audit.json"
BOUND30_PATH = ROOT / "artifacts" / "id_2k_rematch_bound30.json"
ID_2K_PATH = ROOT / "data" / "id_2k.jsonl"


# --- truncation guard ------------------------------------------------------


def test_pad_batch_overflow_policies():
    seqs = [[5, 6, 7, 8], [5, 6]]
    with pytest.warns(UserWarning, match="QUERY tokens are dropped"):
        ids, mask = pad_batch(seqs, pad_id=0, max_len=3)
    assert ids == [[5, 6, 7], [5, 6, 0]]
    assert mask == [[1, 1, 1], [1, 1, 0]]
    with pytest.raises(ValueError, match="longer than max_len=3"):
        pad_batch(seqs, pad_id=0, max_len=3, on_overflow="error")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert pad_batch(seqs, pad_id=0, max_len=3, on_overflow="allow")[0] == ids
        # No overflow → no warning under any policy.
        pad_batch(seqs, pad_id=0, max_len=4, on_overflow="error")
    with pytest.raises(ValueError, match="on_overflow"):
        pad_batch(seqs, pad_id=0, max_len=3, on_overflow="truncate")


def test_required_max_len_fits_every_encoding():
    vocab = build_vocab()
    encs = ["N 3 EDGES 0,1 1,2 QUERY 0 2", "N 2 EDGES QUERY 0 1"]
    longest = max(len(encode_to_ids(e, vocab)) for e in encs)
    assert max_token_len(encs, vocab) == longest == 12
    assert required_max_len(encs, vocab) == 64  # floor
    assert required_max_len(encs, vocab, headroom=0, floor=1) == longest
    assert max_token_len([], vocab) == 0


# --- diagnostics -----------------------------------------------------------


def test_token_coherence_extremes_and_mask():
    torch = pytest.importorskip("torch")
    from reachability_gen.diagnostics import token_coherence

    same = torch.ones(1, 4, 8)
    cos, ratio = token_coherence(same)
    assert math.isclose(cos.item(), 1.0, abs_tol=1e-6)
    assert math.isclose(ratio.item(), 1.0, abs_tol=1e-6)

    ortho = torch.eye(4).unsqueeze(0)  # 4 orthonormal tokens
    cos, ratio = token_coherence(ortho)
    assert math.isclose(cos.item(), 0.0, abs_tol=1e-6)
    assert math.isclose(ratio.item(), 0.5, abs_tol=1e-6)  # 1/sqrt(4)

    # Padded tokens are ignored: two real identical tokens + junk padding.
    z = torch.cat([torch.ones(1, 2, 4), torch.randn(1, 3, 4)], dim=1)
    mask = torch.tensor([[1, 1, 0, 0, 0]])
    cos, ratio = token_coherence(z, mask)
    assert math.isclose(cos.item(), 1.0, abs_tol=1e-6)
    cos, _ = token_coherence(z, torch.tensor([[1, 0, 0, 0, 0]]))
    assert math.isnan(cos.item())  # fewer than two real tokens


def test_perturbation_gain_tracks_scaling():
    torch = pytest.importorskip("torch")
    from reachability_gen.diagnostics import perturbation_gain

    clean = [torch.ones(2, 3, 4) * (t + 1) for t in range(3)]
    delta = torch.full((2, 3, 4), 0.0625)  # exact in float32
    perturbed = [c + delta * k for c, k in zip(clean, (1.0, 2.0, 0.5))]
    gain = perturbation_gain(clean, perturbed)
    assert [round(g, 4) for g in gain] == [1.0, 2.0, 0.5]
    # Relative gain divides by ||z_t|| (which grows 1×, 2×, 3× here).
    rel = perturbation_gain(clean, perturbed, relative=True)
    assert [round(g, 4) for g in rel] == [1.0, 1.0, round(0.5 / 3, 4)]


def test_collapse_flags():
    quiet = collapse_flags(
        final_token_cos=0.35,
        margins=[-4.0, -2.0, 1.0, 3.0, 5.0],
        train_history=[
            {"epoch": 1, "train_acc": 0.6},
            {"epoch": 2, "train_acc": 0.8},
            {"epoch": 3, "train_acc": 0.79},
        ],
        best_val_acc=0.90,
        last_epoch_val_acc=0.88,
    )
    assert quiet["any"] is False and quiet["science_open"] is False
    loud = collapse_flags(
        final_token_cos=TOKEN_COLLAPSE_COS + 0.05,
        margins=[0.5] * 9 + [3.0],
        train_history=[
            {"epoch": 1, "train_acc": 0.85},
            {"epoch": 2, "train_acc": 0.70},
        ],
        best_val_acc=0.85,
        last_epoch_val_acc=0.60,
    )
    for key in (
        "token_collapse",
        "output_concentration",
        "train_acc_drop",
        "last_epoch_below_best",
    ):
        assert loud[key]["flag"] is True, key
    assert loud["train_acc_drop"]["epoch"] == 2
    # Only flags whose inputs are given are computed.
    assert set(collapse_flags(final_token_cos=0.2)) == {"token_collapse", "any", "science_open"}
    assert median_abs_deviation([1.0, 2.0, 3.0, 4.0, 100.0]) == 1.0


def test_largest_train_acc_drop_finds_bound30_geo_breakdown():
    if not BOUND30_PATH.exists():
        pytest.skip("bound30 artifact missing")
    hist = json.loads(BOUND30_PATH.read_text())["geo"]["train_history"]
    worst = largest_train_acc_drop(hist)
    assert worst["epoch"] == 19
    assert worst["drop"] > 0.1


# --- model token_states ----------------------------------------------------


def test_token_states_reproduce_forward():
    torch = pytest.importorskip("torch")
    from reachability_gen.models.euclidean_loop import EuclideanLoop
    from reachability_gen.models.feedforward import FeedForward
    from reachability_gen.models.geometric import GeometricRecurrent

    torch.manual_seed(0)
    ids = torch.tensor([[2, 7, 3, 8, 9, 0], [2, 7, 4, 10, 0, 0]])
    mask = (ids != 0).long()
    for model in (
        GeometricRecurrent(71, d=32, T=4, n_heads=4, max_len=16, apply_cycle_rmsnorm=True),
        EuclideanLoop(71, d=32, T=4, n_heads=4, max_len=16),
    ):
        model.eval()
        with torch.no_grad():
            _, traj = model(ids, mask, return_trajectory=True)
            states = model.token_states(ids, mask)
            noisy = model.token_states(ids, mask, context_noise=torch.zeros(2, 6, 32))
        assert len(states) == 5
        for z, pooled in zip(states, traj):
            assert torch.equal(model._pool(z, mask), pooled)
        assert all(torch.equal(a, b) for a, b in zip(states, noisy))
    ff = FeedForward(71, d=32, L=3, n_heads=4, max_len=16).eval()
    with torch.no_grad():
        states = ff.token_states(ids, mask)
        logits, _ = ff(ids, mask)
        x = ff.ln_f(states[-1])
        m = mask.unsqueeze(-1).float()
        again = ff.head((x * m).sum(1) / m.sum(1))
    assert len(states) == 4
    assert torch.allclose(logits, again)


def test_build_model_from_state_round_trip():
    torch = pytest.importorskip("torch")
    from reachability_gen.audit_checkpoints import build_model_from_state
    from reachability_gen.models.geometric import GeometricRecurrent

    torch.manual_seed(0)
    src = GeometricRecurrent(
        71, d=32, T=3, n_heads=4, max_len=40, mlp_expansion=6, use_tau=True,
        residual_alpha=0.5, apply_cycle_rmsnorm=True,
    ).eval()
    rebuilt = build_model_from_state(
        "geo", src.state_dict(), T=3, residual_alpha=0.5, pad_id=0
    )
    assert rebuilt.max_len == 40 and rebuilt.mlp_expansion == 6
    assert rebuilt.use_tau and rebuilt.apply_cycle_rmsnorm
    ids = torch.tensor([[2, 7, 3, 8, 9]])
    with torch.no_grad():
        assert torch.equal(src(ids)[0], rebuilt(ids)[0])


# --- dataset / audit -------------------------------------------------------


def test_dataset_report_query_blind_baseline():
    pytest.importorskip("torch")
    from reachability_gen.audit_checkpoints import dataset_report

    def row(split, g, s, t, y):
        return {"split": split, "edge_hash": g, "s": s, "t": t, "y": y,
                "hop_distance": 2 if y else -1}

    rows = [
        row("train", "A", 0, 1, 1), row("train", "A", 0, 2, 1), row("train", "A", 1, 0, 0),
        row("train", "B", 0, 1, 0),
        row("val", "A", 2, 0, 0),  # A's majority is y=1 → blind baseline wrong
        row("val", "B", 1, 0, 0),  # B's majority is y=0 → right
        row("val", "C", 0, 1, 1),  # unseen graph → predicts 1 → right
    ]
    rep = dataset_report(rows)
    assert rep["distinct_graphs"] == 3
    assert rep["val_graphs"] == 3 and rep["val_graphs_seen_in_train"] == 2
    assert rep["single_label_graphs"] == 2  # B, C
    assert rep["val_queries_duplicated_in_train"] == 0
    assert math.isclose(rep["query_blind_baseline"]["val_acc"], 2 / 3)


def test_eval_set_review_flags_unseen_tokens_and_size_split():
    pytest.importorskip("torch")
    from reachability_gen.audit_checkpoints import eval_set_review

    def row(n, enc, y, g):
        return {"n": n, "encoding": enc, "y": y, "edge_hash": g}

    train = [row(3, "N 3 EDGES 0,1 1,2 QUERY 0 2", 1, "A")]
    evals = [
        row(3, "N 3 EDGES 0,1 QUERY 1 0", 0, "B"),  # all tokens seen
        row(5, "N 5 EDGES 3,4 QUERY 3 4", 1, "C"),  # 3, 4, 5 unseen
    ]
    rep = eval_set_review(train, evals)
    assert rep["train_n_values"] == [3]
    assert rep["unseen_tokens"] == ["4", "5"]  # "3" appears in training
    assert rep["by_label"]["y1"]["rows_with_unseen_tokens"] == 1
    assert rep["by_label"]["y0"]["rows_with_unseen_tokens"] == 0
    assert rep["by_label"]["y1"]["n_values"] == {"5": 1}
    assert rep["by_label"]["y0"]["distinct_graphs"] == 1


def test_audit_artifact_if_present():
    if not AUDIT_PATH.exists():
        pytest.skip("artifacts/id_2k_checkpoint_audit.json not written yet")
    data = json.loads(AUDIT_PATH.read_text())

    def _walk(obj):
        if isinstance(obj, dict):
            if obj.get("science_open") is True:
                return False
            return all(_walk(v) for v in obj.values())
        if isinstance(obj, list):
            return all(_walk(v) for v in obj)
        return True

    assert data["science_open"] is False and _walk(data)
    trunc = data["truncation"]
    assert trunc["n_truncated"] > 0
    assert trunc["required_max_len"] > trunc["max_len_used_by_runs"]
    ds = data["dataset"]
    assert 0.0 <= ds["query_blind_baseline"]["val_acc"] <= 1.0
    assert 0.0 <= ds["endpoint_rule_baseline"]["val_acc"] <= 1.0
    for name, review in data["eval_set_reviews"].items():
        if review is None:  # set not generated when the audit ran
            continue
        for label in ("y0", "y1"):
            block = review["by_label"][label]
            assert 0 <= block["rows_with_unseen_tokens"] <= block["n_rows"], name
    for run in ("fixed30", "bound30"):
        for arm in ("ff", "geo", "loop"):
            a = data["runs"][run]["arms"][arm]
            # Every table must describe the saved checkpoint it came from.
            assert a["rescore_matches_record"] is True, f"{run}/{arm}"
            assert math.isclose(a["rescored_val_acc"], a["recorded_best_val_acc"])
            assert a["checkpoint_epoch"] == a["recorded_best_epoch"]
            for k in ["-1", "2", "3", "4", "5", "6"]:
                assert "acc_answerable" in a["val_by_hop"][k]
    assert data["runs"]["bound30"]["arms"]["geo"]["run_flags"]["train_acc_drop"]["flag"]
    assert data["runs"]["fixed30"]["arms"]["geo"]["checkpoint_flags"]["token_collapse"]["flag"]
    for run in ("fixed30", "bound30"):
        ff = data["runs"][run]["arms"]["ff"]
        assert ff["checkpoint_flags"]["any"] is False
        assert ff["run_flags"]["any"] is False


@pytest.mark.slow
def test_rescore_reproduces_recorded_best_val_acc():
    pytest.importorskip("torch")
    if not (ID_2K_PATH.exists() and BOUND30_PATH.exists()):
        pytest.skip("id_2k.jsonl or bound30 artifact missing")
    from reachability_gen.audit_checkpoints import audit_arm
    from reachability_gen.overfit_ff import load_jsonl

    rows = load_jsonl(ID_2K_PATH)
    val = [r for r in rows if r["split"] == "val"]
    block = json.loads(BOUND30_PATH.read_text())["loop"]
    block = dict(block, checkpoint_path=str(ROOT / block["checkpoint_path"]))
    a = audit_arm("loop", block, val, build_vocab(), residual_alpha=0.5)
    assert a["rescore_matches_record"] is True
    assert a["summary"]["n_answerable"] < len(val)

# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the graph-disjoint ID set and its multi-seed rematch runner."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from reachability_gen.gen_id_disjoint import (
    endpoint_rule_accuracy,
    generate_id_disjoint,
    verify_id_disjoint,
)
from reachability_gen.hard_negatives import has_endpoint_cue

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = [
    ROOT / "artifacts" / "id_disjoint_rematch.json",
    ROOT / "artifacts" / "id_disjoint_20k_rematch.json",
]


@pytest.fixture(scope="module")
def generated():
    examples, report = generate_id_disjoint()
    return [e.to_dict() for e in examples], report


def test_has_endpoint_cue():
    edges = [(0, 1), (1, 2)]
    assert has_endpoint_cue(3, edges, 2, 0)  # source 2 has no outgoing edge
    assert has_endpoint_cue(3, edges, 1, 0)  # target 0 has no incoming edge
    assert not has_endpoint_cue(3, edges, 0, 2)  # reachable: both edges present
    # Strong hard negative: 2 has an outgoing edge, 0 an incoming one, yet 2 ↛ 0.
    strong = [(0, 1), (1, 2), (2, 1), (3, 0)]
    assert not has_endpoint_cue(4, strong, 2, 0)


def test_generated_set_meets_its_contract(generated):
    rows, report = generated
    ok, issues = verify_id_disjoint(rows)
    assert ok, issues
    assert report["graphs_shared_by_train_and_val"] == 0
    assert report["negatives_with_endpoint_cue"]["count"] == 0
    assert report["endpoint_rule_accuracy"] == {"train": 0.5, "val": 0.5}
    assert report["n_values_by_label"]["y0"] == report["n_values_by_label"]["y1"]
    assert report["by_split"]["train"]["graphs"] == 800
    assert report["by_split"]["val"]["graphs"] == 200
    assert endpoint_rule_accuracy(rows) == 0.5


def test_generation_is_deterministic(generated):
    rows, _ = generated
    again = [e.to_dict() for e in generate_id_disjoint()[0]]
    assert [r["encoding"] for r in again] == [r["encoding"] for r in rows]


def test_custom_sizes_scale_quotas():
    examples, report = generate_id_disjoint(seed=7, n_total=200, n_val=40)
    rows = [e.to_dict() for e in examples]
    ok, issues = verify_id_disjoint(rows, n_total=200, n_val=40)
    assert ok, issues
    assert report["by_split"]["val"]["graphs"] == 20
    assert report["by_split"]["train"]["hop_counts"]["6"] == 16  # (100-20)/5 graphs
    assert not verify_id_disjoint(rows)[0]  # default 2k sizes must not match
    with pytest.raises(ValueError, match="multiples"):
        generate_id_disjoint(n_total=205, n_val=40)


def test_verify_catches_split_leak_and_label_break(generated):
    rows, _ = generated
    leaked = copy.deepcopy(rows)
    first_val = next(r for r in leaked if r["split"] == "val")
    first_val["split"] = "train"
    ok, issues = verify_id_disjoint(leaked)
    assert not ok and any("spans splits" in i or "sizes" in i for i in issues)
    broken = copy.deepcopy(rows)
    broken[0]["y"] = 1 - int(broken[0]["y"])
    ok, issues = verify_id_disjoint(broken)
    assert not ok


def test_runner_smoke_self_audit(generated, tmp_path):
    pytest.importorskip("torch")
    from reachability_gen.run_disjoint_rematch import parity_section, train_arm
    from reachability_gen.tokenize import build_vocab, required_max_len

    rows, _ = generated
    train = [r for r in rows if r["split"] == "train"][:48]
    val = [r for r in rows if r["split"] == "val"][:24]
    vocab = build_vocab()
    max_len = required_max_len((r["encoding"] for r in rows), vocab)
    assert parity_section(vocab, max_len)["within_5pct"] is True
    out = train_arm(
        "loop", train, val, vocab,
        seed=0, epochs=1, max_len=max_len, ckpt_dir=tmp_path, telemetry=False,
    )
    assert len(out["history"]) == 1
    for which in ("best", "final"):
        assert out[which]["rescore_matches_record"] is True
        assert Path(out[which]["checkpoint_path"]).exists()
    assert out["run_flags"]["science_open"] is False


def test_query_features_pick_s_and_t():
    torch = pytest.importorskip("torch")
    from reachability_gen.models.feedforward import query_features

    x = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])  # row 1 has 3 real tokens
    out = query_features(x, mask)
    assert out.shape == (2, 6)
    assert torch.equal(out[0], torch.cat([x[0, 3], x[0, 4]]))
    assert torch.equal(out[1], torch.cat([x[1, 1], x[1, 2]]))


def test_query_readout_models():
    torch = pytest.importorskip("torch")
    from reachability_gen.models.euclidean_loop import EuclideanLoop
    from reachability_gen.models.feedforward import FeedForward
    from reachability_gen.models.geometric import GeometricRecurrent

    ids = torch.tensor([[2, 7, 3, 8, 9, 0], [2, 7, 4, 10, 0, 0]])
    mask = (ids != 0).long()
    for model in (
        FeedForward(71, d=32, L=2, n_heads=4, max_len=16, readout="query"),
        GeometricRecurrent(71, d=32, T=3, n_heads=4, max_len=16, readout="query"),
        EuclideanLoop(71, d=32, T=3, n_heads=4, max_len=16, readout="query"),
    ):
        assert model.head.in_features == 64
        assert model(ids, mask)[0].shape == (2, 2)
    with pytest.raises(ValueError, match="readout"):
        FeedForward(71, d=32, L=1, n_heads=4, max_len=16, readout="cls")


def test_curriculum_schedule():
    from reachability_gen.run_disjoint_rematch import curriculum_hop_cap

    assert [curriculum_hop_cap(e, 30) for e in range(1, 31)] == (
        [2] * 6 + [3] * 6 + [4] * 6 + [5] * 6 + [6] * 6
    )
    assert [curriculum_hop_cap(e, 3) for e in (1, 2, 3)] == [2, 3, 4]


def test_runner_smoke_query_curriculum(generated, tmp_path):
    pytest.importorskip("torch")
    from reachability_gen.run_disjoint_rematch import train_arm
    from reachability_gen.tokenize import build_vocab, required_max_len

    rows, _ = generated
    train = [r for r in rows if r["split"] == "train"][:200]
    val = [r for r in rows if r["split"] == "val"][:40]
    vocab = build_vocab()
    max_len = required_max_len((r["encoding"] for r in rows), vocab)
    out = train_arm(
        "ff", train, val, vocab, seed=0, epochs=5, max_len=max_len,
        ckpt_dir=tmp_path, telemetry=False, readout="query", curriculum=True,
    )
    caps = [h["hop_cap"] for h in out["history"]]
    sizes = [h["train_rows"] for h in out["history"]]
    assert caps == [2, 3, 4, 5, 6]
    assert sizes == sorted(sizes) and sizes[-1] == len(train)
    assert set(out["history"][0]["val_acc_by_graph_hop"]) <= {"2", "3", "4", "5", "6"}
    for which in ("best", "final"):
        assert out[which]["rescore_matches_record"] is True


@pytest.mark.parametrize("artifact", ARTIFACTS, ids=lambda p: p.stem)
def test_disjoint_artifact_if_present(artifact):
    if not artifact.exists():
        pytest.skip(f"{artifact.name} not written yet")
    data = json.loads(artifact.read_text())

    def _walk(obj):
        if isinstance(obj, dict):
            if obj.get("science_open") is True:
                return False
            return all(_walk(v) for v in obj.values())
        if isinstance(obj, list):
            return all(_walk(v) for v in obj)
        return True

    assert data["science_open"] is False and _walk(data)
    assert data["self_audit_mismatches"] == []
    assert data["param_match"]["within_5pct"] is True
    assert data["protocol"]["early_stopping"] is False
    for seed, per_seed in data["runs"].items():
        for arm, block in per_seed.items():
            assert len(block["history"]) == data["protocol"]["epochs"], (seed, arm)
            for which in ("best", "final"):
                assert block[which]["rescore_matches_record"] is True, (seed, arm, which)
    for arm, entry in data["aggregate"].items():
        assert len(entry["seeds"]) == len(data["protocol"]["seeds"]), arm

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
ARTIFACT = ROOT / "artifacts" / "id_disjoint_rematch.json"


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


def test_disjoint_artifact_if_present():
    if not ARTIFACT.exists():
        pytest.skip("artifacts/id_disjoint_rematch.json not written yet")
    data = json.loads(ARTIFACT.read_text())

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

# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the message-passing arms and the calibration runner."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.encode import encode_instance  # noqa: E402
from reachability_gen.models.message_passing import (  # noqa: E402
    MessagePassing,
    collate,
    match_width,
    param_formula,
    parse_rows,
)


def _row(n, edges, s, t, y, hop=-1, eh="g"):
    return {"encoding": encode_instance(n, edges, s, t), "y": y, "hop_distance": hop, "edge_hash": eh}


def test_param_formula_matches_model():
    for d, steps, looped in ((16, 6, False), (37, 6, True), (64, 3, False)):
        assert MessagePassing(d, steps, looped=looped).param_count() == param_formula(
            d, 1 if looped else steps
        )


def test_looped_width_matched_within_tolerance():
    target = param_formula(64, 6)
    d = match_width(target, 6, looped=True)
    assert abs(param_formula(d, 1) / target - 1.0) <= 0.05


def test_collate_adjacency_and_flags():
    graphs = parse_rows([_row(3, [(0, 1), (1, 2)], 0, 2, 1), _row(2, [(1, 0)], 1, 0, 1)])
    b = collate(graphs)
    assert b["adj"].shape == (2, 3, 3)
    assert b["adj"][0, 0, 1] and b["adj"][0, 1, 2] and not b["adj"][0, 1, 0]
    assert b["node_mask"][1].tolist() == [True, True, False]
    assert b["feats"][0, 0].tolist() == [1.0, 0.0] and b["feats"][0, 2].tolist() == [0.0, 1.0]


def test_unlooped_depth_is_fixed():
    model = MessagePassing(8, 3, looped=False)
    batch = collate(parse_rows([_row(3, [(0, 1), (1, 2)], 0, 2, 1)]))
    assert model(batch).shape == (1, 2)
    with pytest.raises(ValueError, match="fixed depth"):
        model(batch, steps=5)


@pytest.mark.parametrize("looped", [False, True])
def test_information_travels_one_hop_per_step(looped):
    torch.manual_seed(0)
    model = MessagePassing(16, 6, looped=looped).eval()
    path = [(i, i + 1) for i in range(8)]

    def target_state(s):
        batch = collate(parse_rows([_row(9, path, s, 8, 1)]))
        with torch.no_grad():
            return model.node_states(batch)[-1][0, 8]

    # Sources 8 and 7 hops upstream are out of reach in 6 steps: identical state.
    assert torch.equal(target_state(0), target_state(1))
    # A source 5 hops upstream is in reach: the target's state must change.
    assert not torch.equal(target_state(0), target_state(3))


CALIBRATION = Path(__file__).resolve().parents[1] / "artifacts" / "mp_calibration.json"


@pytest.mark.skipif(not CALIBRATION.exists(), reason="calibration artifact not present")
def test_calibration_artifact_contract():
    import json

    from reachability_gen.adr_invariants import PARAM_TOL

    art = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    root = CALIBRATION.parents[1]
    assert art["science_open"] is False
    assert art["self_audit_mismatches"] == []
    assert abs(art["protocol"]["looped_over_unlooped_params"] - 1.0) <= PARAM_TOL
    assert set(art["aggregate"]) == {"unlooped", "looped"}
    for per_seed in art["runs"].values():
        for kind, block in per_seed.items():
            assert block["science_open"] is False
            for which in ("best", "final"):
                assert block[which]["rescore_matches_record"] is True
                assert (root / block[which]["checkpoint_path"]).is_file()
                steps = set(block[which]["extended_by_steps"])
                if kind == "unlooped":
                    assert steps == {"6"}
                else:
                    assert "6" in steps and len(steps) > 1


def test_calibration_smoke(tmp_path: Path):
    from reachability_gen.gen_id_disjoint import EXTENDED_SPEC, generate_id_disjoint
    from reachability_gen.run_mp_calibration import train_arm

    id_rows = [e.to_dict() for e in generate_id_disjoint(seed=3, n_total=100, n_val=50)[0]]
    ext_rows = [
        e.to_dict()
        for e in generate_id_disjoint(seed=4, n_total=20, n_val=20, spec=EXTENDED_SPEC)[0]
    ]
    train_rows = [r for r in id_rows if r["split"] == "train"]
    val_rows = [r for r in id_rows if r["split"] == "val"]
    for kind in ("unlooped", "looped"):
        out = train_arm(
            kind, parse_rows(train_rows), parse_rows(val_rows), val_rows,
            parse_rows(ext_rows), ext_rows,
            seed=0, epochs=1, d=16, device="cpu", ckpt_dir=tmp_path, test_steps=[10],
        )
        for which in ("best", "final"):
            assert out[which]["rescore_matches_record"] is True
            expected = {"6"} if kind == "unlooped" else {"6", "10"}
            assert set(out[which]["extended_by_steps"]) == expected

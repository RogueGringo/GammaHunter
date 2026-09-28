# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the geometric step, the ported SheafInferCore and the stability ring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.encode import encode_instance  # noqa: E402
from reachability_gen.gen_crossed import (  # noqa: E402
    CROSSED_EXTENDED_SPEC,
    CROSSED_ID_SPEC,
    generate_crossed,
)
from reachability_gen.models.message_passing import (  # noqa: E402
    MessagePassing,
    collate,
    parse_rows,
    take,
)

RING = Path(__file__).resolve().parents[1] / "artifacts" / "stability_ring.json"


def _row(n, edges, s, t, y=1):
    return {"encoding": encode_instance(n, edges, s, t), "y": y, "hop_distance": -1, "edge_hash": "g", "n": n}


def _crossed(seed=5, n_total=40, n_val=20, spec=CROSSED_ID_SPEC):
    return [e.to_dict() for e in generate_crossed(seed=seed, n_total=n_total, n_val=n_val, spec=spec)[0]]


@pytest.mark.parametrize("update, tau", [("residual", False), ("geo", False), ("geo", True)])
def test_looped_arms_move_information_one_hop_per_step(update, tau):
    torch.manual_seed(0)
    model = MessagePassing(16, 6, looped=True, update=update, tau=tau).eval()
    path = [(i, i + 1) for i in range(8)]

    def target_state(s):
        batch = collate(parse_rows([_row(9, path, s, 8)]))
        with torch.no_grad():
            return model.node_states(batch)[-1][0, 8]

    assert torch.equal(target_state(0), target_state(1))  # 8 and 7 hops: out of reach
    assert not torch.equal(target_state(0), target_state(3))  # 5 hops: in reach


def test_tau_needs_the_looped_geo_update():
    with pytest.raises(ValueError):
        MessagePassing(8, 6, update="residual", tau=True)
    with pytest.raises(ValueError):
        MessagePassing(8, 6, looped=False, update="geo", tau=True)
    model = MessagePassing(8, 6, update="geo", tau=True)
    batch = collate(parse_rows([_row(3, [(0, 1), (1, 2)], 0, 2)]))
    assert model(batch, steps=40).shape == (1, 2)  # steps past 6 reuse the last vector


def test_whole_set_batches_match_per_batch_padding():
    rows = _crossed()
    graphs = parse_rows(rows)
    packed = collate(graphs)
    idx = torch.tensor([3, 17, 25])
    for update in ("residual", "geo"):
        torch.manual_seed(1)
        model = MessagePassing(16, 6, update=update).eval()
        with torch.no_grad():
            a = model(take(packed, idx), steps=9)
            b = model(collate([graphs[i] for i in idx.tolist()]), steps=9)
        assert torch.allclose(a, b, atol=1e-5)


def test_neutral_init_resets_every_hand_set_value():
    from reachability_gen.models.sheaf_adapter import neutral_init
    from reachability_gen.models.sheaf_infer_core import SheafInferCore

    torch.manual_seed(0)
    model = neutral_init(SheafInferCore(d=16, T=6, mlp_expansion=2, max_nodes=64, residual_alpha=1.0))
    eye = torch.eye(16)
    assert not torch.equal(model.phi.W_msg.weight, eye) and not torch.equal(model.phi.W_out.weight, eye)
    assert not torch.equal(model.stalk_proj.weight, eye)
    assert model.phi.mlp_up.weight.abs().sum() > 0 and model.phi.mlp_down.weight.abs().sum() > 0
    assert float(model.edge_encoder[-1].bias) != 4.0 and model.edge_encoder[-1].weight.abs().sum() > 0
    assert model.head.bias.tolist() != [5.0, -5.0]
    assert float(model.absent_bias) == 0.0


def test_sealed_sheaf_is_correct_before_training():
    """The ported arm's hand-set weights already compute T-step reachability."""
    from reachability_gen.models.sheaf_adapter import pack_sheaf, sheaf_logits
    from reachability_gen.models.sheaf_infer_core import SheafInferCore

    rows = _crossed()
    torch.manual_seed(0)
    # d as sealed: the head's threshold (5) sits below the arriving state's norm (~sqrt(d)).
    model = SheafInferCore(d=64, T=6, mlp_expansion=2, max_nodes=64, residual_alpha=1.0).eval()
    batch = pack_sheaf(rows)
    with torch.no_grad():
        logits, aux = sheaf_logits(model, batch, steps=6)
        logits_aux, loss = sheaf_logits(model, batch, steps=6, with_aux=True)
    assert aux is None and loss is not None and loss.dim() == 0
    assert torch.equal(logits, logits_aux)
    assert (logits.argmax(-1) == batch["labels"]).all()  # hops 2-6: no training needed


def test_ring_arms_are_parameter_matched():
    from reachability_gen.adr_invariants import PARAM_TOL
    from reachability_gen.run_stability import ARMS, build, param_count, target_params

    for kind in ARMS:
        assert abs(param_count(build(kind)) / target_params() - 1.0) <= PARAM_TOL, kind


def test_ring_smoke_two_seeds_and_merge(tmp_path, monkeypatch):
    from reachability_gen import run_stability as rs

    for name, steps in (("TEST_STEPS", (6, 8)), ("STABLE_STEPS", (8,)), ("ID_STEPS", (6, 8)),
                        ("DRIFT_AT", (6, 8)), ("UNTRAINED_STEPS", (6, 8)), ("PROBE_STEPS", (16, 48))):
        monkeypatch.setattr(rs, name, steps)
    train = tmp_path / "train.jsonl"
    ext = tmp_path / "ext.jsonl"
    train.write_text("".join(json.dumps(r) + "\n" for r in _crossed(n_total=80, n_val=40)))
    ext.write_text("".join(json.dumps(r) + "\n" for r in _crossed(seed=6, n_total=20, n_val=20,
                                                                    spec=CROSSED_EXTENDED_SPEC)))
    parts = []
    for seed in (0, 1):
        out = tmp_path / f"seed{seed}.json"
        code = rs.main(["--train-data", str(train), "--extended-data", str(ext), "--seeds", str(seed),
                        "--epochs", "1", "--arms", "loop", "geo_tau", "sheaf", "--out", str(out),
                        "--ckpt-dir", str(tmp_path / "ckpt"), "--no-verify"])
        assert code == 0
        parts.append(out)
    merged = tmp_path / "merged.json"
    assert rs.main(["--merge", *map(str, parts), "--out", str(merged)]) == 0
    art = json.loads(merged.read_text())
    assert art["protocol"]["seeds"] == [0, 1] and art["self_audit_mismatches"] == []
    for kind in ("loop", "geo_tau", "sheaf"):
        assert art["aggregate"][kind]["final"]["of_seeds"] == 2
        for seed in ("0", "1"):
            block = art["runs"][seed][kind]
            assert set(block["final"]["extended_by_steps"]) == {"6", "8"}
            assert all(0.0 <= v <= 2.0 for v in block["final"]["target_drift"].values())
    assert rs.main(["--merge", str(parts[0]), str(parts[0]), "--out", str(merged)]) == 1  # overlapping seeds


@pytest.mark.skipif(not RING.exists(), reason="stability ring artifact not present")
def test_stability_ring_artifact_contract():
    from reachability_gen.adr_invariants import PARAM_TOL

    art = json.loads(RING.read_text(encoding="utf-8"))
    root = RING.parents[1]
    assert art["science_open"] is False and art["self_audit_mismatches"] == []
    assert all(abs(r - 1.0) <= PARAM_TOL for r in art["protocol"]["param_ratios"].values())
    for per_seed in art["runs"].values():
        for block in per_seed.values():
            for which in ("best", "final"):
                assert block[which]["rescore_matches_record"] is True
                assert (root / block[which]["checkpoint_path"]).is_file()

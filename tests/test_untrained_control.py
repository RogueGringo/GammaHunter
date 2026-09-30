# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the untrained-model control."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reachability_gen import run_mp_calibration  # noqa: E402
from reachability_gen.models.message_passing import parse_rows  # noqa: E402
from reachability_gen.untrained_control import control_arm, learned_margin  # noqa: E402

CONTROL = Path(__file__).resolve().parents[1] / "artifacts" / "mp_calibration_untrained_control.json"


def _sets():
    from reachability_gen.gen_id_disjoint import EXTENDED_SPEC, generate_id_disjoint

    id_rows = [e.to_dict() for e in generate_id_disjoint(seed=3, n_total=100, n_val=50)[0]]
    ext_rows = [
        e.to_dict()
        for e in generate_id_disjoint(seed=4, n_total=20, n_val=20, spec=EXTENDED_SPEC)[0]
    ]
    train = [r for r in id_rows if r["split"] == "train"]
    val = [r for r in id_rows if r["split"] == "val"]
    return train, val, ext_rows


def test_training_starts_from_init_model(tmp_path: Path, monkeypatch):
    train_rows, val_rows, ext_rows = _sets()
    original = run_mp_calibration.init_model
    seen: dict[str, dict[str, torch.Tensor]] = {}

    def recording(kind, d, seed, device="cpu"):
        model = original(kind, d, seed, device)
        seen[kind] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return model

    monkeypatch.setattr(run_mp_calibration, "init_model", recording)
    for kind in ("unlooped", "looped"):
        run_mp_calibration.train_arm(
            kind, parse_rows(train_rows), parse_rows(val_rows), val_rows,
            parse_rows(ext_rows), ext_rows,
            seed=5, epochs=1, d=16, device="cpu", ckpt_dir=tmp_path, test_steps=[10],
        )
        fresh = original(kind, 16, 5).state_dict()
        assert seen[kind].keys() == fresh.keys()
        assert all(torch.equal(seen[kind][k], fresh[k]) for k in fresh)


def test_control_arm_reports_every_step_count():
    _, val_rows, ext_rows = _sets()
    for kind, expected in (("unlooped", {"6"}), ("looped", {"6", "10"})):
        out = control_arm(
            kind, parse_rows(val_rows), val_rows, parse_rows(ext_rows), ext_rows,
            seed=0, d=16, device="cpu", test_steps=[10],
        )
        assert set(out["extended_by_steps"]) == expected
        assert 0.0 <= out["id_val_acc"] <= 1.0


def test_learned_margin_subtracts_untrained_from_trained():
    control = {"0": {"looped": {"id_val_acc": 0.5,
                                "extended_by_steps": {"6": {"acc": 0.5}, "16": {"acc": 0.5}}}}}
    trained = {"id_val_acc": 1.0, "extended_by_steps": {"6": {"acc": 0.5}, "16": {"acc": 0.97}}}
    calibration = {"runs": {"0": {"looped": {"best": trained, "final": trained}}}}
    margin = learned_margin(control, calibration)
    assert set(margin) == {"seed0/looped/best", "seed0/looped/final"}
    assert margin["seed0/looped/best"]["id_val"] == pytest.approx(0.5)
    assert margin["seed0/looped/final"]["extended_by_steps"] == pytest.approx({"6": 0.0, "16": 0.47})


@pytest.mark.skipif(not CONTROL.exists(), reason="untrained-control artifact not present")
def test_untrained_control_artifact_contract():
    art = json.loads(CONTROL.read_text(encoding="utf-8"))
    assert art["science_open"] is False
    assert set(art["aggregate"]) == {"unlooped", "looped"}
    for per_seed in art["control"].values():
        assert set(per_seed) == {"unlooped", "looped"}
        assert set(per_seed["unlooped"]["extended_by_steps"]) == {"6"}
    if "learned_margin" in art:
        assert len(art["learned_margin"]) == 2 * 2 * len(art["control"])


ANCHORED = Path(__file__).resolve().parents[1] / "artifacts" / "anchored_untrained_control.json"


def test_anchored_zero_test_needs_no_training():
    """A random anchored core read by the fixed zero test answers every question within its step budget."""
    from reachability_gen.gen_crossed import CROSSED_EXTENDED_SPEC, generate_crossed
    from reachability_gen.run_stability import SetData
    from reachability_gen.untrained_control import anchored_control

    rows = [e.to_dict() for e in generate_crossed(seed=21, n_total=40, n_val=20)[0]]
    ext = [e.to_dict() for e in generate_crossed(seed=22, n_total=20, n_val=20, spec=CROSSED_EXTENDED_SPEC)[0]]
    crossed = SetData([r for r in rows if r["split"] == "train"], "cpu")
    val, long = SetData([r for r in rows if r["split"] == "val"], "cpu"), SetData(ext, "cpu")
    out = anchored_control(0, crossed, val, long, device="cpu", epochs=1)
    zt = out["zero_test"]
    assert zt["val_by_steps"]["6"]["acc"] == 1.0  # every crossed path is at most 6 hops
    assert all(v["acc"] == 1.0 for v in zt["long_by_steps"].values())
    # with fewer steps only reachable pairs beyond the budget are missed
    for steps in ("4", "5"):
        missed = [h for h, a in zt["val_by_steps"][steps]["by_graph_hop"].items() if a < 1.0]
        assert all(int(h) > int(steps) for h in missed)
    assert out["trained_head"]["core_unchanged"] is True


@pytest.mark.skipif(not ANCHORED.exists(), reason="anchored-control artifact not present")
def test_anchored_control_artifact_contract():
    art = json.loads(ANCHORED.read_text(encoding="utf-8"))
    assert art["science_open"] is False and len(art["runs"]) == len(art["seeds"])
    within_budget = [("val_by_steps", "6"), ("long_by_steps", "16"), ("long_by_steps", "48"),
                     ("long_by_steps", "192")]
    for run in art["runs"]:
        assert run["trained_head"]["core_unchanged"] is True
        assert set(run["zero_test"]["val_by_steps"]) == {"4", "5", "6"}
        assert set(run["zero_test"]["long_by_steps"]) == {"16", "48", "192"}
        for group, steps in within_budget:  # the element rule answers everything within the step budget
            assert run["zero_test"][group][steps]["acc"] == 1.0
        for by_steps in run["zero_test"].values():
            for zt in by_steps.values():
                assert 0.0 <= zt["norm_rule_acc"] <= 1.0
                for stats in zt["reachable_target_state_by_graph_hop"].values():
                    assert 0 <= stats["norms_exactly_zero"] <= stats["targets"]
    trained = art["trained_takeoff_checkpoints"]
    assert trained is not None and set(trained) == {str(s) for s in art["seeds"]}
    for ckpt in trained.values():
        assert len(ckpt["checkpoint_sha256"]) == 64
        for group, steps in within_budget:
            zt = ckpt[group][steps]
            assert zt["acc"] == 1.0 and zt["norm_rule_acc"] == 1.0
            for stats in zt["reachable_target_state_by_graph_hop"].values():
                assert stats["norms_exactly_zero"] == 0 and stats["norm_min"] > 1.0  # reached states stay large

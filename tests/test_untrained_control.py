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

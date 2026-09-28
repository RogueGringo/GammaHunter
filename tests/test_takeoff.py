# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the anchored arm and the take-off study."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.encode import encode_instance  # noqa: E402
from reachability_gen.gen_crossed import CROSSED_EXTENDED_SPEC, generate_crossed  # noqa: E402
from reachability_gen.models.message_passing import AnchoredMP, collate, parse_rows, rms_cap  # noqa: E402

STUDY = Path(__file__).resolve().parents[1] / "artifacts" / "takeoff_study.json"


def _row(n, edges, s, t, y=1):
    return {"encoding": encode_instance(n, edges, s, t), "y": y, "hop_distance": -1, "edge_hash": "g", "n": n}


def test_rms_cap_keeps_zero_and_bounds():
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.1, 0.1]])
    y = rms_cap(x)
    assert torch.equal(y[0], x[0]) and torch.equal(y[2], x[2])
    assert torch.isclose(y[1].pow(2).mean().sqrt(), torch.tensor(1.0))


def test_rms_cap_gradient_is_finite_at_zero():
    """A square root at zero gives 0 * inf = NaN gradients; exactly-zero states are the design."""
    h = torch.zeros(2, 4, requires_grad=True)
    with torch.no_grad():
        h[1] = torch.tensor([3.0, 4.0, 0.0, 0.0])
    rms_cap(h).sum().backward()
    assert torch.isfinite(h.grad).all()
    assert torch.equal(h.grad[0], torch.ones(4))  # identity below RMS 1


def test_anchored_state_is_zero_exactly_where_the_source_has_not_reached():
    torch.manual_seed(0)
    model = AnchoredMP(16).eval()
    edges = [(i, i + 1) for i in range(8)] + [(9, 3)]  # node 9 feeds the path, unreachable from 0
    batch = collate(parse_rows([_row(10, edges, 0, 8)]))
    with torch.no_grad():
        for steps in (1, 4, 8, 60):
            h = model.node_states(batch, steps)[-1][0]
            nonzero = {v for v in range(10) if h[v].abs().sum() > 0}
            assert nonzero == set(range(min(steps, 8) + 1))


def test_anchored_readout_ignores_the_target_flag_and_node_ids():
    torch.manual_seed(1)
    model = AnchoredMP(16).eval()
    a = collate(parse_rows([_row(4, [(0, 1), (1, 2)], 0, 2)]))
    b = collate(parse_rows([_row(4, [(3, 1), (1, 2)], 3, 2)]))  # same shape, relabelled source
    with torch.no_grad():
        assert torch.allclose(model(a, steps=5), model(b, steps=5), atol=1e-6)


def test_wilson_and_fisher():
    from reachability_gen.run_takeoff import fisher_exact, wilson

    lo, hi = wilson(10, 20)
    assert lo < 0.5 < hi and 0.27 < lo < 0.31 and 0.69 < hi < 0.73
    assert fisher_exact(10, 20, 10, 20) == pytest.approx(1.0)
    assert fisher_exact(20, 20, 0, 20) < 1e-9
    # References computed with exact fractions over all tables with these margins.
    assert math.isclose(fisher_exact(18, 20, 8, 20), 0.0021996412430228, rel_tol=1e-9)
    assert math.isclose(fisher_exact(3, 10, 8, 10), 0.0697785186949274, rel_tol=1e-9)


def test_starts_select_the_intended_rows():
    from reachability_gen.run_stability import SetData
    from reachability_gen.run_takeoff import epoch_indices

    crossed = SetData([e.to_dict() for e in generate_crossed(seed=5, n_total=200, n_val=40)[0]
                       if e.split == "train"], "cpu")
    paired = SetData([_row(3, [(0, 1)], 0, 1)], "cpu")
    data, idx = epoch_indices("paired_warm", 1, crossed, paired)
    assert data is paired and len(idx) == 1
    data, idx = epoch_indices("paired_warm", 2, crossed, paired)
    assert data is crossed and len(idx) == len(crossed.rows)
    for epoch, cap in ((1, 2), (2, 3), (5, 6), (9, 6)):
        data, idx = epoch_indices("curriculum", epoch, crossed, paired)
        hops = {crossed.graph_hop[i] for i in idx.tolist()}
        assert max(hops) == cap and data is crossed


def test_takeoff_smoke_two_parts_and_merge(tmp_path, monkeypatch):
    from reachability_gen import run_takeoff as rt

    monkeypatch.setattr(rt, "FINAL_STEPS", (8, 10))
    rows = [e.to_dict() for e in generate_crossed(seed=5, n_total=80, n_val=40)[0]]
    ext = [e.to_dict() for e in generate_crossed(seed=6, n_total=20, n_val=20, spec=CROSSED_EXTENDED_SPEC)[0]]
    paired = [dict(r, split="train") for r in rows[:20]]
    files = {}
    for name, data in (("train", rows), ("ext", ext), ("paired", paired)):
        files[name] = tmp_path / f"{name}.jsonl"
        files[name].write_text("".join(json.dumps(r) + "\n" for r in data))
    parts = []
    for seeds in (["0"], ["1"]):
        out = tmp_path / f"part{seeds[0]}.json"
        code = rt.main(["--train-data", str(files["train"]), "--extended-data", str(files["ext"]),
                        "--paired-data", str(files["paired"]), "--seeds", *seeds, "--epochs", "2",
                        "--out", str(out), "--ckpt-dir", str(tmp_path / "ckpt"), "--no-verify"])
        assert code == 0
        parts.append(out)
    merged = tmp_path / "merged.json"
    assert rt.main(["--merge", *map(str, parts), "--out", str(merged)]) == 0
    art = json.loads(merged.read_text())
    assert art["self_audit_mismatches"] == [] and art["protocol"]["seeds"] == [0, 1]
    assert len(art["runs"]) == 2 * len(rt.ARMS) * len(rt.STARTS)
    assert set(art["summary"]["cells"]) == {f"{a}/{s}" for a in rt.ARMS for s in rt.STARTS}
    assert all(len(r["final"]["checkpoint_sha256"]) == 64 for r in art["runs"])
    assert rt.main(["--merge", str(parts[0]), str(parts[0]), "--out", str(merged)]) == 1


@pytest.mark.skipif(not STUDY.exists(), reason="take-off study artifact not present")
def test_takeoff_study_artifact_contract():
    from reachability_gen.adr_invariants import PARAM_TOL

    art = json.loads(STUDY.read_text(encoding="utf-8"))
    assert art["science_open"] is False and art["self_audit_mismatches"] == []
    assert all(abs(r - 1.0) <= PARAM_TOL for r in art["protocol"]["param_ratios"].values())
    for cell in art["summary"]["cells"].values():
        assert 0 <= cell["took_off"] <= cell["of"]

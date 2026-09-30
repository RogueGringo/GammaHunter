# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the selection normalizers, their certificate, the battery, stage A and the stage-B runner."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.selection.certificate import certify_entmax15, certify_sparsemax  # noqa: E402
from reachability_gen.selection.conformance import run_battery  # noqa: E402
from reachability_gen.selection.reference import (  # noqa: E402
    BACKENDS,
    Normalizer,
    broken_sparsemax,
    entmax15,
    softmax,
    sparsemax,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kind", sorted(BACKENDS))
def test_reference_backends_pass_the_battery(kind):
    report = run_battery(BACKENDS[kind], kind, rows=24, length=9)
    assert report["passed"], report["checks"]


def test_battery_rejects_the_broken_backend():
    report = run_battery(broken_sparsemax, "sparsemax", rows=24, length=9)
    assert not report["passed"]
    assert not report["checks"]["simplex"] and not report["checks"]["certificate"]


def test_sparsemax_and_entmax_known_values():
    z = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float64)
    assert torch.allclose(sparsemax(z), torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64))
    z = torch.tensor([[0.3, 0.1, -2.0]], dtype=torch.float64)  # τ = (0.3 + 0.1 - 1) / 2 = -0.3
    assert torch.allclose(sparsemax(z), torch.tensor([[0.6, 0.4, 0.0]], dtype=torch.float64))
    p = entmax15(torch.tensor([[0.0, 0.0]], dtype=torch.float64))
    assert torch.allclose(p, torch.full((1, 2), 0.5, dtype=torch.float64))
    masked = sparsemax(torch.tensor([[5.0, 0.2, 0.1]]), torch.tensor([[False, True, True]]))
    assert masked[0, 0] == 0 and abs(float(masked.sum()) - 1) < 1e-6


def test_certificates_grade_outputs_exactly():
    z = [0.3, 0.1, -2.0]
    assert certify_sparsemax(z, [0.6, 0.4, 0.0]).passed
    assert not certify_sparsemax(z, [0.55, 0.45, 0.0]).passed  # values off the exact solution
    assert not certify_sparsemax(z, [0.6, 0.0, 0.4]).passed  # wrong support
    assert certify_sparsemax(z, [0.6, 0.4, 0.0], mask=[True, True, False]).passed  # masked position at exactly 0
    cert = certify_sparsemax(z, [0.6, 0.3, 0.1], mask=[True, True, False])  # weight on a masked position
    assert not cert.passed and not cert.checks["masked_exactly_zero"]
    assert not certify_sparsemax(z, [0.3, 0.2, 0.0]).checks["kernel_sums_to_one"]
    zs = torch.randn(1, 7, dtype=torch.float64)
    assert certify_entmax15(zs[0].tolist(), entmax15(zs)[0].tolist()).passed
    assert not certify_entmax15(zs[0].tolist(), softmax(zs)[0].tolist()).passed


def test_analytic_backward_formulas():
    z = torch.randn(3, 6, dtype=torch.float64, requires_grad=True)
    g = torch.randn(3, 6, dtype=torch.float64)
    p = sparsemax(z)
    (p * g).sum().backward()
    s = (p.detach() > 0).double()
    expected = s * (g - (g * s).sum(-1, keepdim=True) / s.sum(-1, keepdim=True))
    assert torch.allclose(z.grad, expected)
    z.grad = None
    p = entmax15(z)
    (p * g).sum().backward()
    root = p.detach().sqrt()
    expected = root * g - root * (root * g).sum(-1, keepdim=True) / root.sum(-1, keepdim=True)
    assert torch.allclose(z.grad, expected)


def test_stage_a_predictions_hold_for_small_n():
    from reachability_gen.selection.stage_a import gaussian_excess, run, sparsemax_gaussian_prediction

    out = run(ns=(1, 10, 100), draws=1)
    pred = out["predictions"]
    for kind, per_n in pred["bounded"].items():
        assert all(e["holds"] for e in per_n.values()), kind
    assert abs(pred["bounded"]["softmax"]["100"]["measured"] - math.log(100)) < 1e-9
    assert all(e["measured"] < 0.5 for e in pred["bounded"]["sparsemax"].values())
    assert all(e["measured"] < math.sqrt(2) for e in pred["bounded"]["entmax15"].values())
    assert all(pred["appending_changes_nothing_for_alpha_above_1"].values()) and pred["appending_dilutes_softmax"]
    assert all("holds" not in e for per_n in pred["gaussian"].values() for e in per_n.values())  # n < 1,000
    assert len(out["gaussian"]["sparsemax"]["10"]["per_draw"]) == 1 and "claimed" not in out
    from reachability_gen.selection.stage_a import claim_reproduced

    assert claim_reproduced(4.5, [4.2, 4.4, 4.6, 4.8, 4.9]) and claim_reproduced(12.0, [12.01, 12.02])
    assert not claim_reproduced(4.5, [4.7, 4.8, 4.9])
    t = sparsemax_gaussian_prediction(100_000) - 0.5  # half the weight left to the distractors
    assert abs(100_000 * gaussian_excess(t) - 0.5) < 1e-9 and 4.5 < t + 0.5 < 4.7


def test_normalizer_in_the_reader():
    from reachability_gen.models.reader import GraphReader

    attn = torch.randn(2, 4, 5, 5)
    mask = torch.tensor([[True] * 5, [True, True, True, False, False]])[:, None, None, :]
    original = attn.masked_fill(~mask, float("-inf")).softmax(dim=-1)
    assert torch.equal(Normalizer("softmax")(attn, mask), original)
    ssm = Normalizer("ssmax", heads=4)
    assert torch.equal(ssm.scale.detach(), torch.ones(4))  # the SSMax paper's from-scratch start
    for kind in ("softmax", "ssmax", "entmax15", "sparsemax"):
        reader = GraphReader(d=16, layers=1, heads=2, normalizer=kind)
        assert (reader.layers[0].normalizer is None) == (kind == "softmax")  # default: the original code path
    with pytest.raises(ValueError):
        Normalizer("sparsest")


def test_selection_runner_smoke_and_decision(tmp_path, monkeypatch):
    from reachability_gen import run_selection_reader as rs
    from reachability_gen.gen_crossed import CROSSED_EXTENDED_SPEC, generate_crossed

    monkeypatch.setattr(rs, "LONG_STEPS", (16,))
    rows = [e.to_dict() for e in generate_crossed(seed=21, n_total=40, n_val=20)[0]]
    ext = [e.to_dict() for e in generate_crossed(seed=22, n_total=20, n_val=20, spec=CROSSED_EXTENDED_SPEC)[0]]
    train_path, ext_path = tmp_path / "train.jsonl", tmp_path / "ext.jsonl"
    train_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ext_path.write_text("".join(json.dumps(r) + "\n" for r in ext))
    monkeypatch.chdir(tmp_path)
    for arm in rs.ARMS:
        assert rs.main(["--arm", arm, "--train-data", str(train_path), "--extended-data", str(ext_path),
                        "--seeds", "0", "1", "--reader-epochs", "1", "--solver-epochs", "1",
                        "--ckpt-dir", str(tmp_path / "ckpt")]) == 0
    art = json.loads((tmp_path / "artifacts" / "selection_reader_sparsemax.json").read_text())
    run = art["runs"][0]
    assert set(run["final"]) == {"val_in_0.25", "val_in_1", "val_in_4", "val_out", "long_in_16", "long_out_16"}
    assert art["self_audit_mismatches"] == [] and art["protocol"]["sets"]["val_in_4"]["distractor_rate"] == 4.0
    assert set(run["untrained_reader"]) == set(rs.SUPPORT_SETS) and isinstance(run["passes_item19_criteria"], bool)
    support = run["attention_support"]["trained"]["val_in_4"]["layers"]
    assert support and all(0 < layer["mean_support_fraction"] <= 1 for layer in support)
    soft = json.loads((tmp_path / "artifacts" / "selection_reader_softmax.json").read_text())["runs"][0]
    assert soft["attention_support"]["trained"]["val_in_4"] is None  # original code path: nothing to observe
    assert "ssmax_scales" in json.loads((tmp_path / "artifacts" / "selection_reader_ssmax.json").read_text())["runs"][0]
    assert rs.main(["--decide"]) == 0
    verdicts = json.loads((tmp_path / "artifacts" / "selection_reader.json").read_text())["verdicts"]
    assert set(verdicts) == {"H1", "H2", "H3", "H4", "seeds", "splits_per_test"} and set(verdicts["H1"]) == set(rs.SPARSE)
    assert set(verdicts["H3"]) == {"ssmax", "entmax15", "sparsemax"} and verdicts["splits_per_test"] == 6
    assert len(verdicts["H1"]["sparsemax"]["arm_start_f1"]) == 2  # per seed
    assert len(run["batch_losses"]) == 1 and len(run["batch_losses"][0]) >= 1
    assert art["protocol"]["dataset_verified"] is True


def test_permutation_p_is_exact():
    from reachability_gen.run_selection_reader import permutation_p

    assert permutation_p([1.0, 1.0], [0.0, 0.0]) == pytest.approx(1 / 6)  # 1 of the 6 splits is as extreme
    assert permutation_p([0.0, 0.0], [1.0, 1.0]) == 1.0


def _synthetic(arm, seeds, *, start, at4, long_, out, loss1, init, last=None):
    last = last or [0.01] * len(seeds)  # every arm ends at the same loss unless stated
    from reachability_gen.run_reader import LONG_STEPS

    runs = []
    for i, s in enumerate(seeds):
        runs.append({"seed": s,
                     "final": {"val_in_0.25": {"reader": {"f1": start[i]}}, "val_in_4": {"reader": {"f1": at4[i]}},
                               f"long_in_{LONG_STEPS[0]}": {"reader": {"f1": long_[i]}},
                               "val_out": {"reader": {"f1": out[i]}}},
                     "history": [{"train_loss": loss1[i] + 10}],  # a spiky mean; H4 reads the median
                     "batch_losses": [[loss1[i], loss1[i], 50.0 * (i == 0) + loss1[i]], [last[i]] * 3],
                     "untrained_reader": {"val_in_0.25": {"reader": {"topology_bce": init[i]}}}})
    return {"arm": arm, "runs": runs, "self_audit_mismatches": [],
            "protocol": {"seeds": list(seeds), "dataset_verified": True}}


def _arts():
    seeds = [0, 1, 2, 3, 4]
    base = [0.90, 0.91, 0.92, 0.93, 0.94]
    low = [b - 0.3 for b in base]
    out = [0.30, 0.31, 0.32, 0.33, 0.34]
    loss = [0.10, 0.11, 0.12, 0.13, 0.14]
    init = [0.5] * 5
    return {
        "softmax": _synthetic("softmax", seeds, start=base, at4=low, long_=low, out=out, loss1=loss, init=init),
        # no dilution loss, same length loss, same held-out F1, slower start from the same untrained loss
        "sparsemax": _synthetic("sparsemax", seeds, start=base, at4=base, long_=low, out=out,
                                loss1=[v + 0.5 for v in loss], init=init),
        # no dilution loss but a lower start; better held-out F1; slower start from a higher untrained loss
        "entmax15": _synthetic("entmax15", seeds, start=[b - 0.2 for b in base], at4=[b - 0.2 for b in base],
                               long_=[b - 0.5 for b in base], out=[v + 0.3 for v in out],
                               loss1=[v + 0.5 for v in loss], init=[0.6] * 5),
        # same mean held-out F1 with a wide spread: neither a gain nor its absence is shown
        "ssmax": _synthetic("ssmax", seeds, start=base, at4=low, long_=low, out=[0.10, 0.50, 0.20, 0.45, 0.35],
                            loss1=loss, init=init),
    }


def test_decide_applies_every_branch():
    from reachability_gen import run_selection_reader as rs

    v = rs.decide(_arts())
    assert v["seeds"] == [0, 1, 2, 3, 4] and v["splits_per_test"] == 252
    assert v["H1"]["sparsemax"]["verdict"] == "supported"
    assert v["H1"]["entmax15"]["verdict"].startswith("confounded")
    assert v["H2"]["sparsemax"]["verdict"] == "not supported"
    assert v["H3"]["sparsemax"]["verdict"] == "holds"
    assert v["H3"]["entmax15"]["verdict"] == "fails"
    assert v["H3"]["ssmax"]["verdict"] == "inconclusive"
    assert v["H4"]["sparsemax"]["verdict"] == "supported"
    assert v["H4"]["entmax15"]["verdict"].startswith("confounded")
    assert v["H1"]["sparsemax"]["arm_drops"] == [0.0] * 5  # per-seed values kept, ordered by seed
    arts = _arts()
    for r in arts["softmax"]["runs"]:
        r["final"]["val_in_4"] = r["final"]["val_in_0.25"]  # softmax loses nothing
    assert rs.decide(arts)["H1"]["sparsemax"]["verdict"].startswith("moot")
    arts = _arts()
    for r in arts["sparsemax"]["runs"]:
        r["batch_losses"][-1] = [0.5] * 3  # the gap to softmax does not close: a worse fit, not a slower start
    h4 = rs.decide(arts)["H4"]["sparsemax"]
    assert h4["verdict"] == "confounded (the loss gap persists to the last epoch)"
    assert h4["gap_last_epoch"] >= 0.5 * h4["gap_first_epoch"]


def test_decide_refuses_mismatched_result_files():
    from reachability_gen import run_selection_reader as rs

    arts = _arts()
    assert rs.check_arts(arts) == []
    arts["ssmax"]["runs"] = arts["ssmax"]["runs"][:4]
    arts["entmax15"]["self_audit_mismatches"] = ["seed3"]
    arts["sparsemax"]["protocol"]["reader_lr"] = 1e-3
    arts["softmax"]["protocol"]["dataset_verified"] = False
    issues = rs.check_arts(arts)
    assert any("ssmax: seeds" in i for i in issues) and any("self-audit" in i for i in issues)
    assert any("reader_lr" in i for i in issues) and any("softmax: datasets not verified" in i for i in issues)


def test_replication_criterion():
    from reachability_gen.run_selection_reader import replication

    def art(key_in, key_out, f1s, passes_key):
        runs = []
        for s, f in enumerate(f1s):
            reader = {"f1": f, "precision": 1.0, "recall": f, "exact_graphs": 0.0}
            runs.append({"seed": s, "final": {key_in: {"reader": reader}, key_out: {"reader": reader}},
                         passes_key: False})
        return {"runs": runs}

    f1s = [0.10, 0.50, 0.20, 0.45, 0.35]
    item19 = art("val_in", "val_out", f1s, "passes")
    assert replication(art("val_in_0.25", "val_out", f1s, "passes_item19_criteria"), item19)["verdict"] == "exact"
    near = [0.12, 0.48, 0.22, 0.44, 0.33]
    assert replication(art("val_in_0.25", "val_out", near, "passes_item19_criteria"), item19)["verdict"] == \
        "in distribution"
    far = [f + 0.4 for f in f1s]
    rep = replication(art("val_in_0.25", "val_out", far, "passes_item19_criteria"), item19)
    assert rep["verdict"] == "not replicated" and rep["val_out"]["f1"]["item19"]["0"] == 0.10


def test_float32_agreement_of_reference_backends():
    from reachability_gen.selection.conformance import float32_agreement

    for kind, fn in BACKENDS.items():
        report = float32_agreement(fn)
        assert report["passed"], (kind, report)

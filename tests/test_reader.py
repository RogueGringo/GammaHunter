# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the graph reader and the reader → anchored pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.encode import encode_instance  # noqa: E402
from reachability_gen.gen_crossed import CROSSED_EXTENDED_SPEC, generate_crossed  # noqa: E402
from reachability_gen.models.reader import (  # noqa: E402
    EdgeListTokens,
    GraphReader,
    closure,
    collate_tokens,
    edge_list_tokens,
    edge_metrics,
)


def test_edge_list_tokens_drop_the_query():
    a = edge_list_tokens(encode_instance(4, [(0, 1), (2, 3)], 0, 3))
    b = edge_list_tokens(encode_instance(4, [(0, 1), (2, 3)], 2, 1))  # same graph, other question
    assert a.kinds.tolist() == [0, 1, 0, 0, 1, 0] and a.slots.tolist() == [0, 1, 2, 0, 1, 2]
    assert a.symbol.tolist() == [0, -1, 1, 2, -1, 3]
    for field in ("kinds", "slots", "symbol"):
        assert torch.equal(getattr(a, field), getattr(b, field))


def test_reader_output_is_a_hard_adjacency_with_gradients():
    torch.manual_seed(0)
    reader = GraphReader(d=16, layers=1, heads=2)
    batch = collate_tokens([edge_list_tokens(encode_instance(5, [(0, 1), (1, 2), (3, 4)], 0, 2)),
                            edge_list_tokens(encode_instance(3, [(2, 0)], 2, 0))])
    adj = reader(batch)
    assert adj.shape == (2, 5, 5) and set(adj.detach().unique().tolist()) <= {0.0, 1.0}
    assert adj.diagonal(dim1=1, dim2=2).abs().sum() == 0
    adj.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in reader.parameters())


def test_node_symbols_only_decide_grouping():
    """Renaming the nodes (same token order) permutes the reader's graph exactly."""
    torch.manual_seed(1)
    reader = GraphReader(d=16, layers=2, heads=2).eval()
    base = edge_list_tokens(encode_instance(6, [(0, 1), (1, 2), (2, 5), (3, 4)], 0, 5))
    perm = torch.tensor([4, 0, 5, 1, 3, 2])
    renamed = EdgeListTokens(base.kinds, base.slots,
                             torch.where(base.symbol >= 0, perm[base.symbol.clamp(min=0)], base.symbol), base.n)
    with torch.no_grad():
        a = reader(collate_tokens([base]), hard=False)[0]
        b = reader(collate_tokens([renamed]), hard=False)[0]
    assert torch.allclose(b[perm][:, perm], a, atol=1e-6)


def test_edge_metrics_and_closure():
    gold = torch.zeros(1, 4, 4)
    gold[0, 0, 1] = gold[0, 1, 2] = 1
    pred = torch.zeros(1, 4, 4)
    pred[0, 0, 1] = pred[0, 2, 1] = 1  # one right, one reversed
    mask = torch.ones(1, 4, dtype=torch.bool)
    m = edge_metrics(pred, gold, mask)
    assert (m["tp"], m["fp"], m["fn"], m["reversed_errors"], m["exact_graphs"]) == (1, 1, 1, 1, 0)
    reach = closure(gold, mask)[0]
    assert reach[0, 2] and reach[0, 1] and not reach[2, 0] and not reach[0, 3]


def test_pipeline_smoke_all_regimes(tmp_path, monkeypatch):
    from reachability_gen import run_reader as rr

    monkeypatch.setattr(rr, "LONG_STEPS", (16,))

    rows = [e.to_dict() for e in generate_crossed(seed=5, n_total=80, n_val=40)[0]]
    ext = [e.to_dict() for e in generate_crossed(seed=6, n_total=20, n_val=20, spec=CROSSED_EXTENDED_SPEC)[0]]
    train_path, ext_path = tmp_path / "train.jsonl", tmp_path / "ext.jsonl"
    train_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ext_path.write_text("".join(json.dumps(r) + "\n" for r in ext))
    out = tmp_path / "reader.json"
    code = rr.main(["--train-data", str(train_path), "--extended-data", str(ext_path), "--seeds", "0",
                    "--reader-epochs", "1", "--solver-epochs", "1", "--out", str(out),
                    "--ckpt-dir", str(tmp_path / "ckpt"), "--no-verify"])
    assert code == 0
    art = json.loads(out.read_text())
    assert art["self_audit_mismatches"] == [] and set(art["summary"]) == set(rr.REGIMES)
    for run in art["runs"]:
        r = run["final"]["val"]["reader"]
        assert 0.0 <= r["closure_agreement_all_pairs"] <= 1.0 and 0.0 <= r["exact_graphs"] <= 1.0
        errors = run["final"]["long_16"]["errors"]
        wrong = round((1 - run["final"]["long_16"]["accuracy"]) * len(ext))
        assert errors["reader_caused"] + errors["solver_caused"] == wrong
        assert run["oracle"]["val"]["reader"]["exact_graphs"] == 1.0  # the ceiling uses the true graph
        assert r["topology_bce"] >= 0.0 and run["oracle"]["val"]["reader"]["topology_bce"] is None
        probe = run["untrained_reader"]["gradient_probe"]
        shares = probe["share_query_pair"] + probe["share_true_edges"] + probe["share_other"]
        assert abs(shares - 1.0) < 1e-6 or shares == 0.0
        assert -1.0 <= probe["net_push_to_add"] <= 1.0 and "gradient_probe" in run["history"][-1]


def test_merge_combines_parallel_parts(tmp_path, monkeypatch):
    from reachability_gen import run_reader as rr

    monkeypatch.setattr(rr, "LONG_STEPS", (16,))
    rows = [e.to_dict() for e in generate_crossed(seed=7, n_total=40, n_val=20)[0]]
    ext = [e.to_dict() for e in generate_crossed(seed=8, n_total=20, n_val=20, spec=CROSSED_EXTENDED_SPEC)[0]]
    train_path, ext_path = tmp_path / "train.jsonl", tmp_path / "ext.jsonl"
    train_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ext_path.write_text("".join(json.dumps(r) + "\n" for r in ext))
    common = ["--train-data", str(train_path), "--extended-data", str(ext_path), "--seeds", "0",
              "--reader-epochs", "1", "--solver-epochs", "1", "--ckpt-dir", str(tmp_path / "ckpt"), "--no-verify"]
    parts = []
    for regime in ("supervised", "answers_joint"):
        parts.append(tmp_path / f"{regime}.json")
        assert rr.main(common + ["--regimes", regime, "--out", str(parts[-1])]) == 0
    merged = tmp_path / "merged.json"
    assert rr.main(["--merge", *map(str, parts), "--out", str(merged)]) == 0
    art = json.loads(merged.read_text())
    assert set(art["summary"]) == {"supervised", "answers_joint"} and len(art["runs"]) == 2
    assert art["protocol"]["seeds"] == [0] and art["science_open"] is False
    assert rr.main(["--merge", str(parts[0]), str(parts[0]), "--out", str(tmp_path / "dup.json")]) == 1


def _reference_evidence(reader, batch):
    """The noisy-OR evidence summed directly: z = sum of softplus(score) over occurrence pairs."""
    scores = reader.occurrence_scores(batch)
    bsz, width = scores.shape[0], int(batch["width"])
    slot = batch["node_slot"]
    pair = slot[:, :, None] * width + slot[:, None, :]
    pair = pair.masked_fill((slot[:, :, None] < 0) | (slot[:, None, :] < 0), width * width)
    flat = torch.zeros(bsz, width * width + 1).scatter_add(1, pair.flatten(1),
                                                           torch.nn.functional.softplus(scores).flatten(1))
    z = flat[:, : width * width].view(bsz, width, width)
    return z.masked_fill(torch.eye(width, dtype=torch.bool)[None], 0.0)


def test_log_evidence_matches_the_direct_sum():
    torch.manual_seed(2)
    reader = GraphReader(d=16, layers=2, heads=2).eval()
    batch = collate_tokens([edge_list_tokens(encode_instance(6, [(0, 1), (1, 2), (2, 5), (3, 4), (0, 1)], 0, 5)),
                            edge_list_tokens(encode_instance(4, [(2, 0), (3, 1)], 2, 0))])
    with torch.no_grad():
        z = reader.edge_evidence(batch)
        ref = _reference_evidence(reader, batch)
    assert torch.allclose(z, ref, rtol=1e-5, atol=1e-7)
    assert z[1, 4, 0] == 0 and z[0, 5, 5] == 0  # a node absent from the list / the diagonal


def test_no_dead_zone_when_every_score_is_very_negative():
    """Scores far below -104 make z round to 0; the loss and both carriers still get finite gradients."""
    from reachability_gen.run_reader import topology_bce

    torch.manual_seed(4)
    reader = GraphReader(d=16, layers=1, heads=2)
    torch.nn.init.constant_(reader.pair_offset.weight, -300.0)
    enc = encode_instance(5, [(0, 1), (1, 2), (3, 4)], 0, 2)
    batch = collate_tokens([edge_list_tokens(enc)])
    assert reader.edge_evidence(batch).max() == 0  # the direct evidence has underflowed
    gold = torch.zeros(1, 5, 5)
    for u, v in [(0, 1), (1, 2), (3, 4)]:
        gold[0, u, v] = 1
    graph = {"node_mask": torch.ones(1, 5, dtype=torch.bool), "adj": gold}
    bce, pairs = topology_bce(reader.edge_log_evidence(batch), graph)
    bce.backward()
    g = reader.pair_offset.weight.grad
    assert torch.isfinite(bce) and torch.isfinite(g).all() and g.abs().sum() > 0.1
    reader.zero_grad()
    adj = reader(batch, hard=True, through="logit")
    assert adj.detach().sum() == 0
    (adj * gold).sum().backward()
    g = reader.pair_offset.weight.grad
    assert torch.isfinite(g).all() and g.abs().sum() > 0.1


STUDY = Path(__file__).resolve().parents[1] / "artifacts" / "reader_pipeline.json"


@pytest.mark.skipif(not STUDY.exists(), reason="reader-pipeline artifact not present")
def test_reader_pipeline_artifact_contract():
    from reachability_gen.run_reader import PASS

    art = json.loads(STUDY.read_text(encoding="utf-8"))
    assert art["science_open"] is False and art["self_audit_mismatches"] == []
    assert art["protocol"]["pass_criteria"] == PASS
    for run in art["runs"]:
        f = run["final"]
        passes = (f["val"]["reader"]["exact_graphs"] >= PASS["exact_graphs"]
                  and f["long_16"]["reader"]["closure_agreement_all_pairs"] >= PASS["closure_agreement_long"]
                  and all(f[f"long_{s}"]["accuracy"] >= PASS["long_path_accuracy"] for s in art["protocol"]["long_steps"]))
        assert run["passes"] == passes and run["rescore_matches_record"]
        assert len(run["checkpoint_sha256"]) == 64 and len(run["history"]) == art["protocol"]["reader_epochs"]
    for regime, s in art["summary"].items():
        assert s["passes"] == sum(r["passes"] for r in art["runs"] if r["regime"] == regime)


def test_forward_is_adjacency_of_log_evidence():
    torch.manual_seed(6)
    reader = GraphReader(d=16, layers=1, heads=2).eval()
    batch = collate_tokens([edge_list_tokens(encode_instance(5, [(0, 1), (1, 2), (3, 4)], 0, 2))])
    with torch.no_grad():
        log_z = reader.edge_log_evidence(batch)
        for through in ("probability", "logit"):
            assert torch.equal(reader(batch, hard=True, through=through), reader.adjacency(log_z, hard=True, through=through))
        assert torch.equal(reader(batch, hard=False), reader.adjacency(log_z, hard=False))


def test_density_prior_pulls_towards_the_target_density():
    from reachability_gen.run_reader import density_kl

    graph = {"node_mask": torch.ones(1, 4, dtype=torch.bool)}
    off = 1 - torch.eye(4)
    for fill, sign in ((0.9, 1.0), (0.01, -1.0)):  # too dense: push down; too sparse: push up
        adj = (fill * off)[None].clone().requires_grad_(True)
        kl = density_kl(adj, graph, 0.25)
        kl.backward()
        assert kl.item() > 0 and (adj.grad[0][off.bool()] * sign > 0).all()
    assert abs(density_kl((0.25 * off)[None], graph, 0.25).item()) < 1e-6


def test_answers_only_variants_smoke(tmp_path, monkeypatch):
    from reachability_gen import run_reader as rr

    monkeypatch.setattr(rr, "LONG_STEPS", (16,))
    rows = [e.to_dict() for e in generate_crossed(seed=9, n_total=40, n_val=20)[0]]
    ext = [e.to_dict() for e in generate_crossed(seed=10, n_total=20, n_val=20, spec=CROSSED_EXTENDED_SPEC)[0]]
    train_path, ext_path = tmp_path / "train.jsonl", tmp_path / "ext.jsonl"
    train_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ext_path.write_text("".join(json.dumps(r) + "\n" for r in ext))
    out = tmp_path / "variants.json"
    assert rr.main(["--train-data", str(train_path), "--extended-data", str(ext_path), "--seeds", "0",
                    "--reader-epochs", "1", "--solver-epochs", "1", "--out", str(out),
                    "--ckpt-dir", str(tmp_path / "ckpt"), "--no-verify", "--regimes", *rr.VARIANTS]) == 0
    art = json.loads(out.read_text())
    assert set(art["summary"]) == set(rr.VARIANTS) and art["self_audit_mismatches"] == []
    assert 0 < art["protocol"]["edge_density_prior"] < 1 and art["protocol"]["prior_weight"] == rr.PRIOR_WEIGHT


def test_density_prior_acts_on_an_empty_graph():
    """The prior must push an empty graph towards more edges (a clamp would give no gradient here)."""
    from reachability_gen.run_reader import density_kl

    graph = {"node_mask": torch.ones(1, 4, dtype=torch.bool)}
    adj = torch.zeros(1, 4, 4, requires_grad=True)
    density_kl(adj, graph, 0.25).backward()
    off = ~torch.eye(4, dtype=torch.bool)
    assert (adj.grad[0][off] < 0).all()  # descent raises every off-diagonal entry


def test_merge_accepts_prior_settings_absent_from_parts_without_a_prior(tmp_path, monkeypatch):
    from reachability_gen import run_reader as rr

    monkeypatch.setattr(rr, "LONG_STEPS", (16,))
    rows = [e.to_dict() for e in generate_crossed(seed=11, n_total=40, n_val=20)[0]]
    ext = [e.to_dict() for e in generate_crossed(seed=12, n_total=20, n_val=20, spec=CROSSED_EXTENDED_SPEC)[0]]
    train_path, ext_path = tmp_path / "train.jsonl", tmp_path / "ext.jsonl"
    train_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ext_path.write_text("".join(json.dumps(r) + "\n" for r in ext))
    common = ["--train-data", str(train_path), "--extended-data", str(ext_path), "--seeds", "0",
              "--reader-epochs", "1", "--solver-epochs", "1", "--ckpt-dir", str(tmp_path / "ckpt"), "--no-verify"]
    soft, prior = tmp_path / "soft.json", tmp_path / "prior.json"
    assert rr.main(common + ["--regimes", "answers_frozen_soft", "--out", str(soft)]) == 0
    assert rr.main(common + ["--regimes", "answers_frozen_prior", "--out", str(prior)]) == 0
    assert "edge_density_prior" not in json.loads(soft.read_text())["protocol"]
    merged = tmp_path / "merged.json"
    assert rr.main(["--merge", str(soft), str(prior), "--out", str(merged)]) == 0
    p = json.loads(merged.read_text())["protocol"]
    assert p["prior_weight"] == rr.PRIOR_WEIGHT and 0 < p["edge_density_prior"] < 1
    other = json.loads(prior.read_text())
    other["protocol"]["prior_weight"] = 2.0
    other["runs"][0]["seed"] = 1
    conflict = tmp_path / "conflict.json"
    conflict.write_text(json.dumps(other))
    assert rr.main(["--merge", str(prior), str(conflict), "--out", str(tmp_path / "bad.json")]) == 1


VARIANTS_STUDY = Path(__file__).resolve().parents[1] / "artifacts" / "reader_variants.json"


@pytest.mark.skipif(not VARIANTS_STUDY.exists(), reason="reader-variants artifact not present")
def test_reader_variants_artifact_contract():
    from reachability_gen.run_reader import PRIOR_WEIGHT, VARIANTS

    art = json.loads(VARIANTS_STUDY.read_text(encoding="utf-8"))
    assert art["science_open"] is False and art["self_audit_mismatches"] == []
    assert set(art["summary"]) == set(VARIANTS) and art["protocol"]["prior_weight"] == PRIOR_WEIGHT
    assert all(len(s["seeds"]) == 10 for s in art["summary"].values())

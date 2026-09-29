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
    # checkpoints sit in a directory named after the result file, so concurrent runs cannot overwrite each other's
    assert {Path(run["checkpoint_path"]).parent for run in art["runs"]} == {tmp_path / "ckpt" / out.stem}
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


def test_relu_slope_one_at_zero_changes_only_the_gradient():
    from reachability_gen.models.message_passing import AnchoredMP, ReLUSlopeOneAtZero, relu_slope_one_at_zero

    x = torch.tensor([-1.0, 0.0, 2.0], requires_grad=True)
    y = ReLUSlopeOneAtZero()(x)
    y.sum().backward()
    assert torch.equal(y, torch.relu(x.detach())) and x.grad.tolist() == [0.0, 1.0, 1.0]
    torch.manual_seed(0)
    solver = AnchoredMP(8, 3).eval()
    graph = {"feats": torch.zeros(1, 4, 2), "adj": torch.eye(4)[None].roll(1, dims=2), "node_mask": torch.ones(1, 4, dtype=torch.bool),
             "s": torch.tensor([0]), "t": torch.tensor([2])}
    before = solver(graph, 3)
    assert relu_slope_one_at_zero(solver) == 2 and torch.equal(solver(graph, 3), before)


def test_kink_free_gradient_reaches_an_edge_into_an_unreached_node():
    """On an empty graph the default gradient on the edge s->t is exactly 0; with slope 1 at zero it is not."""
    import torch.nn.functional as F

    from reachability_gen.models.message_passing import AnchoredMP, relu_slope_one_at_zero

    torch.manual_seed(3)
    solver = AnchoredMP(8, 3).eval()
    grads = []
    for kink_free in (False, True):
        if kink_free:
            relu_slope_one_at_zero(solver)
        adj = torch.zeros(1, 4, 4, requires_grad=True)
        graph = {"feats": torch.zeros(1, 4, 2), "adj": adj, "node_mask": torch.ones(1, 4, dtype=torch.bool),
                 "s": torch.tensor([0]), "t": torch.tensor([2])}
        F.cross_entropy(solver(graph, 3), torch.tensor([1])).backward()
        grads.append(adj.grad[0, 0, 2].item())
    assert grads[0] == 0.0 and grads[1] != 0.0


def test_dense_answers_labels_and_masks():
    from reachability_gen.run_reader import Data, DenseAnswers, shortest_hops

    enc = encode_instance(4, [(0, 1), (1, 2)], 0, 2)
    rows = [{"edge_hash": "g", "encoding": enc, "s": 0, "t": 2, "y": 1, "hop_distance": 2},
            {"edge_hash": "g", "encoding": enc, "s": 2, "t": 0, "y": 0, "hop_distance": -1}]
    data = Data(rows, "cpu")
    assert shortest_hops(data.graphs[0])[0] == [0, 1, 2, -1]
    dense = DenseAnswers(data, steps=1)  # 0->2 needs two steps, so it is masked out
    assert len(dense) == 1
    graph, tokens, row_graph, labels, mask = dense.batch([0])
    assert graph["s"].tolist() == [0, 1, 2, 3] and row_graph.tolist() == [0, 0, 0, 0]
    assert labels[0].tolist() == [0, 1, 1, 0] and mask[0].tolist() == [False, True, False, True]
    assert mask[1].tolist() == [True, False, True, True] and labels[1, 2].item() == 1


def test_density_study_smoke_with_closure_criteria(tmp_path, monkeypatch):
    from reachability_gen import run_reader as rr

    monkeypatch.setattr(rr, "LONG_STEPS", (16,))
    rows = [e.to_dict() for e in generate_crossed(seed=13, n_total=40, n_val=20)[0]]
    ext = [e.to_dict() for e in generate_crossed(seed=14, n_total=20, n_val=20, spec=CROSSED_EXTENDED_SPEC)[0]]
    train_path, ext_path = tmp_path / "train.jsonl", tmp_path / "ext.jsonl"
    train_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ext_path.write_text("".join(json.dumps(r) + "\n" for r in ext))
    out = tmp_path / "density.json"
    regimes = ["answers_frozen_prior", *rr.DENSITY_STUDY]
    assert rr.main(["--train-data", str(train_path), "--extended-data", str(ext_path), "--seeds", "0",
                    "--reader-epochs", "1", "--solver-epochs", "1", "--out", str(out), "--criteria", "closure",
                    "--ckpt-dir", str(tmp_path / "ckpt"), "--no-verify", "--regimes", *regimes]) == 0
    art = json.loads(out.read_text())
    p = art["protocol"]
    assert p["criteria"] == "closure" and p["pass_criteria"] == rr.PASS_CLOSURE and p["dense_graphs_per_batch"] == 8
    assert set(art["summary"]) == set(regimes) and art["self_audit_mismatches"] == []
    for run in art["runs"]:
        assert run["oracle"]["val"]["reader"]["closure_exact_graphs"] == 1.0
        assert 0.0 <= run["final"]["val"]["reader"]["closure_exact_graphs"] <= 1.0


def test_merge_accepts_dense_setting_absent_from_sparse_parts(tmp_path, monkeypatch):
    from reachability_gen import run_reader as rr

    monkeypatch.setattr(rr, "LONG_STEPS", (16,))
    rows = [e.to_dict() for e in generate_crossed(seed=15, n_total=40, n_val=20)[0]]
    ext = [e.to_dict() for e in generate_crossed(seed=16, n_total=20, n_val=20, spec=CROSSED_EXTENDED_SPEC)[0]]
    train_path, ext_path = tmp_path / "train.jsonl", tmp_path / "ext.jsonl"
    train_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ext_path.write_text("".join(json.dumps(r) + "\n" for r in ext))
    common = ["--train-data", str(train_path), "--extended-data", str(ext_path), "--seeds", "0", "--criteria", "closure",
              "--reader-epochs", "1", "--solver-epochs", "1", "--ckpt-dir", str(tmp_path / "ckpt"), "--no-verify"]
    sparse, dense = tmp_path / "sparse.json", tmp_path / "dense.json"
    assert rr.main(common + ["--regimes", "answers_frozen_prior", "--out", str(sparse)]) == 0
    assert rr.main(common + ["--regimes", "answers_dense_prior", "--out", str(dense)]) == 0
    assert "dense_graphs_per_batch" not in json.loads(sparse.read_text())["protocol"]
    merged = tmp_path / "merged.json"
    assert rr.main(["--merge", str(sparse), str(dense), "--out", str(merged)]) == 0
    p = json.loads(merged.read_text())["protocol"]
    assert p["dense_graphs_per_batch"] == rr.DENSE_GRAPHS_PER_BATCH and p["criteria"] == "closure"


DENSITY_STUDY_ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "reader_answers_density.json"


@pytest.mark.skipif(not DENSITY_STUDY_ARTIFACT.exists(), reason="answer-density artifact not present")
def test_answer_density_artifact_contract():
    from reachability_gen.run_reader import DENSITY_STUDY, LONG_STEPS, PASS_CLOSURE

    art = json.loads(DENSITY_STUDY_ARTIFACT.read_text(encoding="utf-8"))
    assert art["science_open"] is False and art["self_audit_mismatches"] == []
    assert art["protocol"]["criteria"] == "closure" and art["protocol"]["pass_criteria"] == PASS_CLOSURE
    assert set(art["summary"]) == {"answers_frozen_prior", *DENSITY_STUDY}
    for run in art["runs"]:
        f = run["final"]
        passes = (f["val"]["reader"]["closure_agreement_all_pairs"] >= PASS_CLOSURE["closure_agreement_val"]
                  and f["long_16"]["reader"]["closure_agreement_all_pairs"] >= PASS_CLOSURE["closure_agreement_long"]
                  and all(f[f"long_{s}"]["accuracy"] >= PASS_CLOSURE["long_path_accuracy"] for s in LONG_STEPS))
        assert run["passes"] == passes


def test_rloo_advantages_have_zero_mean_per_reading():
    from reachability_gen.run_reader import rloo_advantages

    adv = rloo_advantages(torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]]))
    assert torch.allclose(adv[0], torch.tensor([-2.0, -2 / 3, 2 / 3, 2.0])) and torch.all(adv[1] == 0)
    assert torch.allclose(adv.sum(dim=1), torch.zeros(2))


def test_sample_log_prob_matches_bernoulli():
    from reachability_gen.run_reader import sample_log_prob

    torch.manual_seed(5)
    log_z = torch.randn(2, 4, 4)
    node_mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    prob = -torch.expm1(-log_z.exp())
    samples = torch.bernoulli(prob[:, None].expand(-1, 3, -1, -1))
    valid = (node_mask[:, :, None] & node_mask[:, None, :] & ~torch.eye(4, dtype=torch.bool)[None]).float()
    direct = (torch.distributions.Bernoulli(probs=prob[:, None]).log_prob(samples) * valid[:, None]).flatten(2).sum(-1)
    assert torch.allclose(sample_log_prob(log_z, samples, node_mask), direct, atol=1e-4)


def test_score_function_gradient_favours_rewarded_edges():
    """Rewarding samples that contain edge 0->1 pushes its log-evidence up, and only its own."""
    from reachability_gen.run_reader import rloo_advantages, sample_log_prob

    torch.manual_seed(6)
    log_z = torch.zeros(1, 3, 3, requires_grad=True)
    node_mask = torch.ones(1, 3, dtype=torch.bool)
    prob = -torch.expm1(-log_z.exp())
    samples = torch.bernoulli(prob.detach()[:, None].expand(-1, 256, -1, -1))
    adv = rloo_advantages(samples[:, :, 0, 1])  # reward: the sample contains 0 -> 1
    (-(adv * sample_log_prob(log_z, samples, node_mask)).mean()).backward()
    g = log_z.grad[0]
    assert g[0, 1] < 0 and g[0, 1].abs() > 5 * g[1, 2].abs()  # descent raises 0 -> 1 far more than any other pair


def test_reinforce_regimes_smoke_and_merge(tmp_path, monkeypatch):
    from reachability_gen import run_reader as rr

    monkeypatch.setattr(rr, "LONG_STEPS", (16,))
    rows = [e.to_dict() for e in generate_crossed(seed=17, n_total=40, n_val=20)[0]]
    ext = [e.to_dict() for e in generate_crossed(seed=18, n_total=20, n_val=20, spec=CROSSED_EXTENDED_SPEC)[0]]
    train_path, ext_path = tmp_path / "train.jsonl", tmp_path / "ext.jsonl"
    train_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ext_path.write_text("".join(json.dumps(r) + "\n" for r in ext))
    common = ["--train-data", str(train_path), "--extended-data", str(ext_path), "--seeds", "0", "--criteria", "closure",
              "--reader-epochs", "1", "--solver-epochs", "1", "--ckpt-dir", str(tmp_path / "ckpt"), "--no-verify"]
    est, plain = tmp_path / "estimator.json", tmp_path / "plain.json"
    assert rr.main(common + ["--regimes", *rr.ESTIMATOR_STUDY, "--out", str(est)]) == 0
    assert rr.main(common + ["--regimes", "answers_frozen_prior", "--out", str(plain)]) == 0
    art = json.loads(est.read_text())
    assert art["protocol"]["reinforce_samples"] == rr.REINFORCE_SAMPLES and art["self_audit_mismatches"] == []
    assert set(art["summary"]) == set(rr.ESTIMATOR_STUDY)
    assert rr.main(["--merge", str(est), str(plain), "--out", str(tmp_path / "merged.json")]) == 0


def test_reinforce_stays_finite_when_the_evidence_runs_away():
    """Very large scores give finite losses and gradients (log z grows only logarithmically with them)."""
    from reachability_gen.run_reader import reinforce_rows

    torch.manual_seed(8)
    reader = GraphReader(d=16, layers=1, heads=2)
    torch.nn.init.constant_(reader.pair_offset.weight, 300.0)
    from reachability_gen.models.message_passing import AnchoredMP, collate, parse_rows

    enc = encode_instance(5, [(0, 1), (1, 2), (3, 4)], 0, 2)
    rows = [{"encoding": enc, "y": 1}]
    graph = collate(parse_rows(rows))
    tokens = collate_tokens([edge_list_tokens(enc)])
    loss = reinforce_rows(reader, AnchoredMP(8, 3).eval(), graph, tokens, density=0.1)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in reader.parameters() if p.grad is not None)

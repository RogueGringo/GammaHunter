# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the graph reader and the reader → anchored pipeline."""

from __future__ import annotations

import json

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

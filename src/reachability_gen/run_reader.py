# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Reader → anchored solver: can the graph come from the text? (MEASURE)

The anchored solver (``AnchoredMP``) learns reachability reliably when it is
given each graph. Here a question-blind, identity-free reader
(``models.reader``) must supply that graph from the edge-list tokens as an
explicit 0/1 adjacency. Three regimes:

* ``supervised``: the reader is trained on the true edges; the solver, trained
  first on true graphs, stays frozen (pipeline check);
* ``answers_frozen``: the reader learns from the reachability answers only,
  through the hard adjacency, with the same frozen solver;
* ``answers_joint``: reader and solver both start untrained and learn from the
  answers only.

Scoring, in three layers per run: the reader (edge precision/recall/F1,
exact-graph rate, reversed-direction errors, agreement of the reachability
closure with the true graph's over all node pairs), the pipeline (accuracy and
AUROC on the crossed validation split at 6 steps and on the long-path set at
16, 48 and 192 steps), and attribution of every wrong answer to the reader
(its graph's closure disagrees with the true one on that pair) or the solver.
Controls: the untrained pipeline, and the solver on the true graph (ceiling).
A gradient probe (measurement only) records, before training and after every
epoch, where the answers loss pushes the reader's graph on a fixed batch of
training rows: on each question's own endpoint pair, on the true edges, or
elsewhere, and whether it pushes towards adding or removing edges.

Pass criteria, fixed in advance: exact-graph rate ≥ 0.99 on the validation
graphs, closure agreement ≥ 0.99 on the long-path graphs, and long-path
accuracy ≥ 0.99 at 16, 48 and 192 steps.

``science_open=false`` always.

Usage::

    python -m reachability_gen.run_reader --device cuda
    # or one process per regime (--regimes ... --out PART.json), then
    python -m reachability_gen.run_reader --merge PART.json [PART.json ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.adr_invariants import PARAM_TOL
from reachability_gen.gen_crossed import CROSSED_EXTENDED_SPEC, CROSSED_ID_SPEC, verify_crossed
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_mp_calibration import BATCH, GRAD_CLIP, LR, WEIGHT_DECAY
from reachability_gen.run_takeoff import wilson

DEFAULT_TRAIN = Path("data/id_crossed_20k.jsonl")
DEFAULT_EXTENDED = Path("data/extended_crossed_2k.jsonl")
DEFAULT_OUT = Path("artifacts/reader_pipeline.json")
DEFAULT_CKPT_DIR = Path("artifacts/reader_pipeline")
REGIMES: dict[str, str] = {
    "supervised": "reader trained on the true edges; solver trained on true graphs first, then frozen",
    "answers_frozen": "reader trained on reachability answers only; solver trained on true graphs first, then frozen",
    "answers_joint": "reader and solver both untrained at the start, trained on reachability answers only",
}
# Answers-only variants, fixed after the main study (limitations item 15) and run as a
# separate study: during training the frozen solver receives the reader's soft graph
# P(edge) instead of the hard one, and/or a prior pulls the density of the graph it
# receives towards the training graphs' edge density. Evaluation always uses the hard graph.
VARIANTS: dict[str, str] = {
    "answers_frozen_soft": "as answers_frozen, but the solver receives the soft graph P(edge) during training",
    "answers_frozen_prior": "as answers_frozen, plus a prior on the density of the graph the solver receives",
    "answers_frozen_soft_prior": "soft graph during training and the density prior",
}
# Answer density x subgradient at zero (limitations item 17), fixed before its runs. With
# answers_frozen_prior these form a 2 x 2, scored with --criteria closure: a reachability
# answer cannot reveal an edge implied by another path, so the attainable target is the
# true closure, not the exact edge set.
DENSITY_STUDY: dict[str, str] = {
    "answers_dense_prior": "as answers_frozen_prior, but every source of each graph with every target answered",
    "answers_frozen_prior_kinkfree": "as answers_frozen_prior, with slope 1 at zero in the solver's message and "
                                     "update ReLUs during reader training (the forward pass is unchanged)",
    "answers_dense_prior_kinkfree": "dense answers and slope 1 at zero",
}
# Score-function (REINFORCE) estimator for answers-only readers (limitations item 18), fixed before
# its runs: graphs are sampled edge by edge from the reader's P(edge) and scored by the frozen
# solver's answers; nothing is differentiated through the solver, so its ReLU kink plays no part.
ESTIMATOR_STUDY: dict[str, str] = {
    "answers_frozen_prior_reinforce": "as answers_frozen_prior, trained with a score-function (REINFORCE) "
                                      "estimator on sampled graphs",
    "answers_dense_prior_reinforce": "as answers_dense_prior, trained with a score-function (REINFORCE) "
                                     "estimator on sampled graphs",
}
ALL_REGIMES: dict[str, str] = {**REGIMES, **VARIANTS, **DENSITY_STUDY, **ESTIMATOR_STUDY}
REINFORCE_SAMPLES: int = 4  # graphs sampled per reading; each sample's baseline is the others' mean (RLOO)
PRIOR_WEIGHT: float = 1.0  # weight of the density prior, fixed before the variant runs
TRAIN_STEPS: int = 6
SOLVER_EPOCHS: int = 2
READER_EPOCHS: int = 5
THROUGH: str = "logit"  # straight-through carrier for the answers-only regimes (pilot: "probability" saturated)
READER_LR: float = 1e-2  # the reader's own rate (solver: shared 1e-3); 1e-3 had not converged in a 4-epoch pilot
LONG_STEPS: tuple[int, ...] = (16, 48, 192)
EVAL_BATCH: int = 250
PASS: dict[str, float] = {"exact_graphs": 0.99, "closure_agreement_long": 0.99, "long_path_accuracy": 0.99}
PROBE_ROWS: int = 256  # fixed training rows on which the gradient probe is taken (measurement only)
PASS_CLOSURE: dict[str, float] = {"closure_agreement_val": 0.99, "closure_agreement_long": 0.99,
                                  "long_path_accuracy": 0.99}
CRITERIA: dict[str, dict[str, float]] = {"exact": PASS, "closure": PASS_CLOSURE}
DENSE_GRAPHS_PER_BATCH: int = BATCH // 4  # four question rows per graph: dense epochs take as many steps


def anchored_width() -> int:
    from reachability_gen.run_takeoff import anchored_width as width

    return width()


class Data:
    """Rows plus both views of each graph: the solver batch and the reader tokens."""

    def __init__(self, rows: Sequence[dict[str, Any]], device: str) -> None:
        from reachability_gen.models.message_passing import collate, parse_rows
        from reachability_gen.models.reader import edge_list_tokens

        self.rows = list(rows)
        self.device = device
        self.graphs = parse_rows(self.rows)
        self.tokens = [edge_list_tokens(r["encoding"]) for r in self.rows]
        self._collate = collate
        hop = {r["edge_hash"]: int(r["hop_distance"]) for r in self.rows if int(r["y"]) == 1}
        self.graph_hop = [hop.get(r["edge_hash"], -1) for r in self.rows]

    def batch(self, idx: Sequence[int]) -> tuple[dict[str, Any], dict[str, Any]]:
        from reachability_gen.models.reader import collate_tokens

        return (self._collate([self.graphs[i] for i in idx], self.device),
                collate_tokens([self.tokens[i] for i in idx], self.device))


def shortest_hops(graph) -> list[list[int]]:
    """``[n][n]`` shortest path lengths in edges (0 on the diagonal, -1 where unreachable)."""
    from collections import deque

    out_edges: list[list[int]] = [[] for _ in range(graph.n)]
    for u, v in zip(graph.src.tolist(), graph.dst.tolist()):
        out_edges[u].append(v)
    table = []
    for s in range(graph.n):
        dist = [-1] * graph.n
        dist[s] = 0
        queue = deque([s])
        while queue:
            u = queue.popleft()
            for v in out_edges[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        table.append(dist)
    return table


class DenseAnswers:
    """Every (source, target) answer of each graph in ``data``, for the dense regimes.

    One entry per distinct graph; a batch of graphs expands to one solver row per
    source, and every other node is a target. Labels are true reachability; a pair
    reachable only in more than ``steps`` steps is masked out, since the solver
    answers within ``steps``.
    """

    def __init__(self, data: Data, steps: int) -> None:
        import torch

        first: dict[str, int] = {}
        for i, r in enumerate(data.rows):
            first.setdefault(r["edge_hash"], i)
        self.data, self.index = data, list(first.values())
        self.labels, self.masks = [], []
        for i in self.index:
            dist = torch.tensor(shortest_hops(data.graphs[i]))
            off_diagonal = ~torch.eye(len(dist), dtype=torch.bool)
            self.labels.append((dist > 0).long())
            self.masks.append(((dist < 0) | ((dist > 0) & (dist <= steps))) & off_diagonal)

    def __len__(self) -> int:
        return len(self.index)

    def batch(self, graph_ids: Sequence[int]):
        """Solver batch (one row per source), reader tokens (one per graph), row -> graph, labels, mask."""
        import torch

        from reachability_gen.models.message_passing import ParsedGraph
        from reachability_gen.models.reader import collate_tokens

        parsed, row_graph = [], []
        for k, gi in enumerate(graph_ids):
            g = self.data.graphs[self.index[gi]]
            parsed += [ParsedGraph(g.n, g.src, g.dst, s, s, 0) for s in range(g.n)]
            row_graph += [k] * g.n
        graph = self.data._collate(parsed, self.data.device)
        tokens = collate_tokens([self.data.tokens[self.index[gi]] for gi in graph_ids], self.data.device)
        width = graph["adj"].shape[1]
        labels = torch.zeros(len(parsed), width, dtype=torch.long)
        mask = torch.zeros(len(parsed), width, dtype=torch.bool)
        row = 0
        for gi in graph_ids:
            n = len(self.labels[gi])
            labels[row : row + n, :n], mask[row : row + n, :n] = self.labels[gi], self.masks[gi]
            row += n
        device = self.data.device
        return graph, tokens, torch.tensor(row_graph, device=device), labels.to(device), mask.to(device)


def all_target_logits(solver, graph: dict[str, Any], steps: int):
    """``[B, N, 2]``: the solver's answer for every node as the target (the target is never marked)."""
    import torch

    for h in solver.iter_states(graph, steps):
        pass
    return solver.head(torch.cat([h, h.norm(dim=-1, keepdim=True)], dim=-1))


def build_solver():
    from reachability_gen.models.message_passing import AnchoredMP

    return AnchoredMP(anchored_width(), TRAIN_STEPS)


def build_reader():
    from reachability_gen.models.reader import GraphReader

    return GraphReader()


def train_solver(solver, data: Data, *, epochs: int, device: str) -> None:
    """Train the solver on the true graphs (regimes with a frozen solver)."""
    import torch
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_

    opt = torch.optim.AdamW(solver.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for _ in range(epochs):
        solver.train()
        order = torch.randperm(len(data.rows)).tolist()
        for i in range(0, len(order), BATCH):
            graph, _ = data.batch(order[i : i + BATCH])
            loss = F.cross_entropy(solver(graph, TRAIN_STEPS), graph["y"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(solver.parameters(), GRAD_CLIP)
            opt.step()
    for p in solver.parameters():
        p.requires_grad_(False)
    solver.eval()


def topology_bce(log_z, graph: dict[str, Any]):
    """Sum and count of the per-pair BCE of P(edge) = 1 - exp(-z) against the true edges.

    Over ordered pairs of distinct nodes of each graph, from log z
    (``GraphReader.edge_log_evidence``): -log P(edge) = -log z + z/2 for small z,
    so neither the loss nor its gradient underflows as P(edge) approaches 0.
    """
    import torch

    mask = graph["node_mask"]
    valid = (mask[:, :, None] & mask[:, None, :]).float()
    valid = valid * (1 - torch.eye(log_z.shape[1], device=log_z.device))[None]
    gold = graph["adj"].float()
    z = log_z.exp()
    nll = torch.where(log_z < -9, -log_z + 0.5 * z, -torch.log(-torch.expm1(-z.clamp(min=1e-4))))
    per_pair = gold * nll + (1 - gold) * z
    return (per_pair * valid).sum(), valid.sum()


def pipeline_logits(reader, solver, graph: dict[str, Any], tokens: dict[str, Any], steps: int, *,
                    oracle: bool = False, through: str = "probability"):
    """Solver logits on the reader's adjacency (or on the true one when ``oracle``)."""
    if oracle:
        return solver(graph, steps), graph["adj"].float()
    adj = reader(tokens, hard=True, through=through)
    return solver(dict(graph, adj=adj), steps), adj


def edge_density(data: Data) -> float:
    """True edges over ordered pairs of distinct nodes, pooled over the rows' graphs."""
    edges = sum(int(g.src.numel()) for g in data.graphs)
    pairs = sum(g.n * (g.n - 1) for g in data.graphs)
    return edges / pairs


def density_kl(adj, graph: dict[str, Any], density: float):
    """KL(Bernoulli(density) || Bernoulli(mean of adj over ordered pairs of distinct nodes)).

    ``adj`` is the graph the solver receives in training (hard with a straight-through
    gradient, or soft), so the prior acts on that graph's density. The mean is squeezed
    into [1e-6, 1 - 1e-6], not clamped: a clamp has no gradient at an empty graph, the
    state the prior must act on.
    """
    import torch

    mask = graph["node_mask"]
    valid = (mask[:, :, None] & mask[:, None, :]).to(adj.dtype)
    valid = valid * (1 - torch.eye(adj.shape[1], device=adj.device, dtype=adj.dtype))[None]
    mean = (adj * valid).sum() / valid.sum() * (1 - 2e-6) + 1e-6
    return density * torch.log(density / mean) + (1 - density) * torch.log((1 - density) / (1 - mean))


def gradient_probe(reader, solver, data: Data, idx: Sequence[int], *, through: str) -> dict[str, float]:
    """Where the answers loss pushes the reader's graph (measurement only; nothing is updated).

    The total |dL/dA| (the answers loss's gradient on the hard adjacency, mean
    loss over the rows), the shares of it that fall on each question's own
    (source, target) pair, on the other true edges and elsewhere, and the net
    push towards adding edges, -sum(dL/dA) / sum|dL/dA|. All are 0 when the
    gradient is exactly zero.
    """
    import torch
    import torch.nn.functional as F

    graph, tokens = data.batch(idx)
    with torch.enable_grad():
        adj = reader(tokens, hard=True, through=through)
        loss = F.cross_entropy(solver(dict(graph, adj=adj), TRAIN_STEPS), graph["y"])
        (grad,) = torch.autograd.grad(loss, adj)
    mass = grad.abs()
    total = mass.sum().item()
    if not total:
        return {"abs_total": 0.0, "share_query_pair": 0.0, "share_true_edges": 0.0, "share_other": 0.0,
                "net_push_to_add": 0.0}
    query = torch.zeros_like(mass, dtype=torch.bool)
    query[torch.arange(len(idx), device=mass.device), graph["s"], graph["t"]] = True
    q = mass[query].sum().item() / total
    t = mass[(graph["adj"] > 0.5) & ~query].sum().item() / total
    return {"abs_total": total, "share_query_pair": q, "share_true_edges": t, "share_other": 1.0 - q - t,
            "net_push_to_add": -grad.sum().item() / total}


def evaluate(reader, solver, data: Data, steps: int, *, oracle: bool = False) -> dict[str, Any]:
    """Pipeline accuracy/AUROC by graph hop, reader metrics and error attribution."""
    import torch

    from reachability_gen.models.reader import closure, edge_metrics
    from reachability_gen.run_llm_reference import auroc

    reader.eval()
    solver.eval()
    margins: list[float] = []
    correct: list[int] = []
    reader_caused = solver_caused = 0
    pair_agree = pair_total = query_agree = closure_exact = 0
    bce_sum = bce_pairs = 0.0
    totals: dict[str, int] = {"edges_true": 0, "tp": 0, "fp": 0, "fn": 0, "reversed_errors": 0,
                              "exact_graphs": 0, "graphs": 0}
    with torch.no_grad():
        for start in range(0, len(data.rows), EVAL_BATCH):
            idx = list(range(start, min(start + EVAL_BATCH, len(data.rows))))
            graph, tokens = data.batch(idx)
            logits, adj = pipeline_logits(reader, solver, graph, tokens, steps, oracle=oracle)
            if not oracle:
                s, c = topology_bce(reader.edge_log_evidence(tokens), graph)
                bce_sum, bce_pairs = bce_sum + s.item(), bce_pairs + c.item()
            margin = (logits[:, 1] - logits[:, 0]).float()
            pred = margin > 0
            y = graph["y"].bool()
            margins += margin.tolist()
            correct += (pred == y).long().tolist()
            mask = graph["node_mask"]
            m = edge_metrics(adj, graph["adj"].float(), mask)
            for k in totals:
                totals[k] += m[k]
            true_reach, read_reach = closure(graph["adj"].float(), mask), closure(adj, mask)
            valid = mask[:, :, None] & mask[:, None, :]
            pair_agree += ((true_reach == read_reach) & valid).sum().item()
            pair_total += valid.sum().item()
            closure_exact += ((true_reach == read_reach) | ~valid).flatten(1).all(dim=1).sum().item()
            rows = torch.arange(len(idx), device=adj.device)
            q_agree = true_reach[rows, graph["s"], graph["t"]] == read_reach[rows, graph["s"], graph["t"]]
            query_agree += q_agree.sum().item()
            wrong = pred != y
            reader_caused += (wrong & ~q_agree).sum().item()
            solver_caused += (wrong & q_agree).sum().item()
    labels = [int(r["y"]) for r in data.rows]
    k, n = sum(correct), len(correct)
    by_hop: dict[str, list[int]] = {}
    for c, h in zip(correct, data.graph_hop):
        by_hop.setdefault(str(h), []).append(c)
    tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {
        "steps": steps,
        "accuracy": k / n,
        "accuracy_wilson95": wilson(k, n),
        "auroc": auroc(margins, labels),
        "accuracy_by_graph_hop": {h: sum(v) / len(v) for h, v in sorted(by_hop.items(), key=lambda kv: int(kv[0]))},
        "reader": {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "reversed_errors": totals["reversed_errors"],
            "topology_bce": bce_sum / bce_pairs if bce_pairs else None,  # None for the true graph (oracle)
            "edges_true_mean": totals["edges_true"] / totals["graphs"],  # per question row (graphs repeat)
            "edges_predicted_mean": (tp + fp) / totals["graphs"],
            "exact_graphs": totals["exact_graphs"] / totals["graphs"],
            "closure_agreement_all_pairs": pair_agree / pair_total,
            "closure_agreement_queried_pairs": query_agree / n,
            "closure_exact_graphs": closure_exact / n,  # rows whose read graph has the true closure
        },
        "errors": {"reader_caused": reader_caused, "solver_caused": solver_caused},
    }


def sample_log_prob(log_z, samples, node_mask):
    """``[B, K]`` log-probability of sampled 0/1 graphs ``[B, K, N, N]`` under P(edge) = 1 - exp(-z).

    Summed over ordered pairs of distinct nodes, from log z with the terms of
    ``topology_bce``: log P = log z - z/2 for small z, and log(1 - P) = -z.
    """
    import torch

    eye = torch.eye(log_z.shape[1], dtype=torch.bool, device=log_z.device)
    valid = (node_mask[:, :, None] & node_mask[:, None, :] & ~eye[None]).to(log_z.dtype)
    z = log_z.exp()
    log_p = -torch.where(log_z < -9, -log_z + 0.5 * z, -torch.log(-torch.expm1(-z.clamp(min=1e-4))))
    terms = samples * log_p[:, None] + (1 - samples) * (-z)[:, None]
    return (terms * valid[:, None]).flatten(2).sum(-1)


def rloo_advantages(rewards):
    """Each sample's reward minus the mean reward of the other samples of the same reading (RLOO)."""
    k = rewards.shape[1]
    return rewards - (rewards.sum(dim=1, keepdim=True) - rewards) / (k - 1)


def pair_counts(node_mask):
    """``[B]`` ordered pairs of distinct nodes per graph."""
    n = node_mask.sum(dim=1).float()
    return n * (n - 1)


def reinforce_rows(reader, solver, graph: dict[str, Any], tokens: dict[str, Any], *, density: Optional[float]):
    """Score-function loss for question rows: each row's graph sampled K times, rewarded by the answer.

    The reward is the log-likelihood the frozen solver gives the correct answer on a
    sample; advantages are leave-one-out (RLOO) and normalised over the batch; the
    log-probability of a sample is divided by the graph's number of node pairs.
    """
    import torch
    import torch.nn.functional as F

    log_z = reader.edge_log_evidence(tokens)
    prob = -torch.expm1(-log_z.exp())
    rows, k = prob.shape[0], REINFORCE_SAMPLES
    with torch.no_grad():
        samples = torch.bernoulli(prob.detach()[:, None].expand(-1, k, -1, -1).contiguous())
        repeated = {key: value.repeat_interleave(k, dim=0) for key, value in graph.items()}
        logits = solver(dict(repeated, adj=samples.flatten(0, 1)), TRAIN_STEPS)
        rewards = F.log_softmax(logits.float(), dim=-1).gather(1, repeated["y"][:, None]).view(rows, k)
        adv = rloo_advantages(rewards)
        adv = adv / (adv.std() + 1e-8)
    log_prob = sample_log_prob(log_z, samples, graph["node_mask"]) / pair_counts(graph["node_mask"])[:, None]
    loss = -(adv * log_prob).mean()
    if density is not None:
        loss = loss + PRIOR_WEIGHT * density_kl(prob, graph, density)
    return loss


def dense_epoch(reader, solver, dense: DenseAnswers, opt, params, *, through: str,
                density: Optional[float], reinforce: bool = False) -> tuple[float, int, int]:
    """One epoch of a dense regime: each graph once, with all its (source, target) answers.

    Returns the loss summed over answers, the correct answers and the answers seen.
    The reader reads each graph once; its graph is shared by that graph's source rows.
    """
    import torch
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_

    loss_sum, hits, seen = 0.0, 0, 0
    order = torch.randperm(len(dense)).tolist()
    for i in range(0, len(order), DENSE_GRAPHS_PER_BATCH):
        gids = order[i : i + DENSE_GRAPHS_PER_BATCH]
        graph, tokens, row_graph, labels, mask = dense.batch(gids)
        first_rows = torch.searchsorted(row_graph, torch.arange(len(gids), device=row_graph.device))
        node_mask = graph["node_mask"][first_rows]
        if reinforce:
            loss, logits = dense_reinforce_loss(reader, solver, graph, tokens, row_graph, labels, mask, node_mask,
                                                density=density)
        else:
            adj = reader.adjacency(reader.edge_log_evidence(tokens), hard=True, through=through)  # one per graph
            logits = all_target_logits(solver, dict(graph, adj=adj[row_graph]), TRAIN_STEPS)
            loss = F.cross_entropy(logits[mask], labels[mask])
            if density is not None:
                loss = loss + PRIOR_WEIGHT * density_kl(adj, {"node_mask": node_mask}, density)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
        answered = int(mask.sum().item())
        loss_sum += float(loss.item()) * answered
        hits += int(((logits.argmax(dim=-1) == labels) & mask).sum().item())
        seen += answered
    return loss_sum, hits, seen


def dense_reinforce_loss(reader, solver, graph: dict[str, Any], tokens: dict[str, Any], row_graph, labels, mask,
                         node_mask, *, density: Optional[float]):
    """Score-function loss for dense answers: each graph sampled K times, rewarded by all its answers.

    A sample's reward is the mean log-likelihood of the graph's (source, target)
    answers; also returns the logits of the reader's own hard graph (for logging).
    """
    import torch
    import torch.nn.functional as F

    log_z = reader.edge_log_evidence(tokens)
    prob = -torch.expm1(-log_z.exp())
    graphs, k = prob.shape[0], REINFORCE_SAMPLES
    answers = torch.zeros(graphs, device=prob.device).index_add(0, row_graph, mask.sum(dim=1).float()).clamp(min=1)
    with torch.no_grad():
        samples = torch.bernoulli(prob.detach()[:, None].expand(-1, k, -1, -1).contiguous())
        rewards = torch.zeros(graphs, k, device=prob.device)
        for j in range(k):
            logits = all_target_logits(solver, dict(graph, adj=samples[row_graph, j]), TRAIN_STEPS)
            ll = F.log_softmax(logits.float(), dim=-1).gather(2, labels[..., None]).squeeze(2)
            rewards[:, j] = torch.zeros(graphs, device=prob.device).index_add(0, row_graph, (ll * mask).sum(dim=1))
        rewards = rewards / answers[:, None]
        adv = rloo_advantages(rewards)
        adv = adv / (adv.std() + 1e-8)
        own = all_target_logits(solver, dict(graph, adj=(prob > 0.5).to(prob.dtype)[row_graph]), TRAIN_STEPS)
    log_prob = sample_log_prob(log_z, samples, node_mask) / pair_counts(node_mask)[:, None]
    loss = -(adv * log_prob).mean()
    if density is not None:
        loss = loss + PRIOR_WEIGHT * density_kl(prob, {"node_mask": node_mask}, density)
    return loss, own


def run(regime: str, seed: int, train: Data, val: Data, ext: Data, *, device: str, ckpt_dir: Path,
        reader_epochs: int = READER_EPOCHS, solver_epochs: int = SOLVER_EPOCHS,
        reader_lr: float = READER_LR, through: str = THROUGH, density: Optional[float] = None,
        criteria: str = "exact") -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from torch.nn.utils import clip_grad_norm_

    torch.manual_seed(seed)
    reader = build_reader().to(device)
    solver = build_solver().to(device)
    out: dict[str, Any] = {"regime": regime, "seed": seed, "science_open": False}
    t0 = time.perf_counter()
    if regime != "answers_joint":
        train_solver(solver, train, epochs=solver_epochs, device=device)
    if regime.endswith("_kinkfree"):
        from reachability_gen.models.message_passing import relu_slope_one_at_zero

        relu_slope_one_at_zero(solver)  # the gradient only: the forward pass, and so evaluation, is unchanged
    dense = DenseAnswers(train, TRAIN_STEPS) if regime.startswith("answers_dense") else None
    probe_idx = list(range(min(PROBE_ROWS, len(train.rows))))
    out["untrained_reader"] = {
        "val": evaluate(reader, solver, val, TRAIN_STEPS),
        "long_16": evaluate(reader, solver, ext, 16),
        "gradient_probe": gradient_probe(reader, solver, train, probe_idx, through=through),
    }
    params = list(reader.parameters()) + ([] if regime != "answers_joint" else list(solver.parameters()))
    groups = [{"params": list(reader.parameters()), "lr": reader_lr}]
    if regime == "answers_joint":
        groups.append({"params": list(solver.parameters()), "lr": LR})
    opt = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    history = []
    for epoch in range(1, reader_epochs + 1):
        reader.train()
        solver.train(regime == "answers_joint")
        if dense is not None:
            loss_sum, hits, seen = dense_epoch(reader, solver, dense, opt, params, through=through,
                                               density=density if "_prior" in regime else None,
                                               reinforce=regime.endswith("_reinforce"))
        else:
            loss_sum, hits, seen = 0.0, 0, 0
            order = torch.randperm(len(train.rows)).tolist()
            for i in range(0, len(order), BATCH):
                idx = order[i : i + BATCH]
                graph, tokens = train.batch(idx)
                if regime == "supervised":
                    bce, pairs = topology_bce(reader.edge_log_evidence(tokens), graph)
                    loss = bce / pairs
                    with torch.no_grad():
                        logits, _ = pipeline_logits(reader, solver, graph, tokens, TRAIN_STEPS)
                elif regime.endswith("_reinforce"):
                    loss = reinforce_rows(reader, solver, graph, tokens,
                                          density=density if "_prior" in regime else None)
                    with torch.no_grad():
                        logits, _ = pipeline_logits(reader, solver, graph, tokens, TRAIN_STEPS)
                else:
                    adj = reader.adjacency(reader.edge_log_evidence(tokens), hard="_soft" not in regime,
                                           through=through)
                    logits = solver(dict(graph, adj=adj), TRAIN_STEPS)
                    loss = F.cross_entropy(logits, graph["y"])
                    if "_prior" in regime:
                        loss = loss + PRIOR_WEIGHT * density_kl(adj, graph, density)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                clip_grad_norm_(params, GRAD_CLIP)
                opt.step()
                loss_sum += float(loss.item()) * len(idx)
                hits += int((logits.argmax(dim=-1) == graph["y"]).sum().item())
                seen += len(idx)
        ev = evaluate(reader, solver, val, TRAIN_STEPS)
        probe = gradient_probe(reader, solver, train, probe_idx, through=through)
        history.append({"epoch": epoch, "train_loss": loss_sum / seen, "train_pipeline_acc": hits / seen,
                        "val_accuracy": ev["accuracy"], "val_auroc": ev["auroc"],
                        "val_exact_graphs": ev["reader"]["exact_graphs"], "val_edge_f1": ev["reader"]["f1"],
                        "val_closure_agreement": ev["reader"]["closure_agreement_all_pairs"],
                        "val_topology_bce": ev["reader"]["topology_bce"],
                        "val_edges_predicted_mean": ev["reader"]["edges_predicted_mean"],
                        "gradient_probe": probe})
        print(f"[{regime}/seed{seed}] epoch {epoch}/{reader_epochs}: loss {loss_sum / seen:.4f} "
              f"val acc {ev['accuracy']:.4f} auroc {ev['auroc']:.4f} | exact graphs {ev['reader']['exact_graphs']:.4f} "
              f"edge F1 {ev['reader']['f1']:.4f} closure {ev['reader']['closure_agreement_all_pairs']:.4f} "
              f"topology BCE {ev['reader']['topology_bce']:.4f} "
              f"edges {ev['reader']['edges_predicted_mean']:.1f}/{ev['reader']['edges_true_mean']:.1f} "
              f"| gradient {probe['abs_total']:.2e}: on query pair {probe['share_query_pair']:.2f}, "
              f"true edges {probe['share_true_edges']:.2f}",
              file=sys.stderr, flush=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{regime}_seed{seed}.pt"
    torch.save({"reader": reader.state_dict(), "solver": solver.state_dict(), "regime": regime, "seed": seed,
                "science_open": False}, path)
    fresh_reader, fresh_solver = build_reader().to(device), build_solver().to(device)
    state = torch.load(path, map_location="cpu", weights_only=True)
    fresh_reader.load_state_dict(state["reader"])
    fresh_solver.load_state_dict(state["solver"])
    final_val = evaluate(fresh_reader, fresh_solver, val, TRAIN_STEPS)
    out.update({
        "history": history,
        "train_seconds": time.perf_counter() - t0,
        "checkpoint_path": path.as_posix(),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rescore_matches_record": final_val["accuracy"] == history[-1]["val_accuracy"],
        "final": {
            "val": final_val,
            **{f"long_{s}": evaluate(fresh_reader, fresh_solver, ext, s) for s in LONG_STEPS},
        },
        "oracle": {
            "val": evaluate(fresh_reader, fresh_solver, val, TRAIN_STEPS, oracle=True),
            "long_16": evaluate(fresh_reader, fresh_solver, ext, 16, oracle=True),
        },
    })
    f = out["final"]
    if criteria == "exact":
        out["passes"] = (f["val"]["reader"]["exact_graphs"] >= PASS["exact_graphs"]
                         and f["long_16"]["reader"]["closure_agreement_all_pairs"] >= PASS["closure_agreement_long"]
                         and all(f[f"long_{s}"]["accuracy"] >= PASS["long_path_accuracy"] for s in LONG_STEPS))
    else:
        c = CRITERIA[criteria]
        out["passes"] = (f["val"]["reader"]["closure_agreement_all_pairs"] >= c["closure_agreement_val"]
                         and f["long_16"]["reader"]["closure_agreement_all_pairs"] >= c["closure_agreement_long"]
                         and all(f[f"long_{s}"]["accuracy"] >= c["long_path_accuracy"] for s in LONG_STEPS))
    return out


def summarise(runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for regime in ALL_REGIMES:
        rs = [r for r in runs if r["regime"] == regime]
        if not rs:
            continue
        def mean(key_fn):
            vals = [key_fn(r) for r in rs]
            return {"mean": statistics.fmean(vals), "min": min(vals), "max": max(vals)}
        out[regime] = {
            "seeds": [r["seed"] for r in rs],
            "passes": sum(r["passes"] for r in rs),
            "val_accuracy": mean(lambda r: r["final"]["val"]["accuracy"]),
            "val_auroc": mean(lambda r: r["final"]["val"]["auroc"]),
            "val_exact_graphs": mean(lambda r: r["final"]["val"]["reader"]["exact_graphs"]),
            "val_edge_f1": mean(lambda r: r["final"]["val"]["reader"]["f1"]),
            "val_topology_bce": mean(lambda r: r["final"]["val"]["reader"]["topology_bce"]),
            "val_edges_predicted_mean": mean(lambda r: r["final"]["val"]["reader"]["edges_predicted_mean"]),
            "val_edges_true_mean": mean(lambda r: r["final"]["val"]["reader"]["edges_true_mean"]),
            "long_closure_agreement": mean(lambda r: r["final"]["long_16"]["reader"]["closure_agreement_all_pairs"]),
            "val_closure_agreement": mean(lambda r: r["final"]["val"]["reader"]["closure_agreement_all_pairs"]),
            "val_closure_exact_graphs": mean(lambda r: r["final"]["val"]["reader"].get("closure_exact_graphs", 0.0)),
            "long_closure_exact_graphs": mean(
                lambda r: r["final"]["long_16"]["reader"].get("closure_exact_graphs", 0.0)),
            **{f"long_{s}_accuracy": mean(lambda r, s=s: r["final"][f"long_{s}"]["accuracy"]) for s in LONG_STEPS},
            **{f"long_{s}_auroc": mean(lambda r, s=s: r["final"][f"long_{s}"]["auroc"]) for s in LONG_STEPS},
            "untrained_val_accuracy": mean(lambda r: r["untrained_reader"]["val"]["accuracy"]),
            **{f"gradient_{k}_untrained": mean(lambda r, k=k: r["untrained_reader"]["gradient_probe"][k])
               for k in ("abs_total", "share_query_pair", "share_true_edges", "net_push_to_add")},
            **{f"gradient_{k}_final": mean(lambda r, k=k: r["history"][-1]["gradient_probe"][k])
               for k in ("abs_total", "share_query_pair", "share_true_edges", "net_push_to_add")},
            "errors_reader_caused_long_16": sum(r["final"]["long_16"]["errors"]["reader_caused"] for r in rs),
            "errors_solver_caused_long_16": sum(r["final"]["long_16"]["errors"]["solver_caused"] for r in rs),
        }
    return out


def merge(parts: Sequence[Path], out: Path) -> int:
    """Combine result files of one protocol (e.g. one per regime) into one."""
    loaded = [json.loads(Path(p).read_text(encoding="utf-8")) for p in parts]
    # Recorded only by parts that ran a prior regime or a dense regime; they must agree where present.
    conditional = ("edge_density_prior", "prior_weight", "dense_graphs_per_batch", "reinforce_samples",
                   "reinforce_baseline")

    def shared(art: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in art["protocol"].items() if k != "seeds" and k not in conditional}

    prior: dict[str, Any] = {}
    for path, art in zip(parts, loaded):
        if shared(art) != shared(loaded[0]) or art["science_open"] is not False:
            print(f"FAIL: {path} was run with a different protocol", file=sys.stderr)
            return 1
        for k in conditional:
            if k in art["protocol"] and prior.setdefault(k, art["protocol"][k]) != art["protocol"][k]:
                print(f"FAIL: {path} was run with a different {k}", file=sys.stderr)
                return 1
    runs = [r for art in loaded for r in art["runs"]]
    ids = [(r["regime"], r["seed"]) for r in runs]
    if len(ids) != len(set(ids)):
        print("FAIL: a run appears in more than one part", file=sys.stderr)
        return 1
    regimes = {k: v for art in loaded for k, v in art["regimes"].items()}
    mismatches = [f"{r['regime']}/seed{r['seed']}" for r in runs if not r["rescore_matches_record"]]
    artifact = {
        "science_open": False,
        "purpose": loaded[0]["purpose"],
        "regimes": {k: regimes[k] for k in ALL_REGIMES if k in regimes},
        "protocol": dict(shared(loaded[0]), **prior, seeds=sorted({r["seed"] for r in runs})),
        "runs": runs,
        "summary": summarise(runs),
        "self_audit_mismatches": mismatches,
        "merged_from": [Path(p).as_posix() for p in parts],
        "elapsed_seconds_by_part": [art["elapsed_seconds"] for art in loaded],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": not mismatches, "out": out.as_posix(), "self_audit_mismatches": mismatches,
                      "science_open": False}, sort_keys=True))
    return 1 if mismatches else 0


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Reader → anchored solver pipeline (MEASURE).")
    p.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--extended-data", type=Path, default=DEFAULT_EXTENDED)
    p.add_argument("--regimes", nargs="+", choices=list(ALL_REGIMES), default=list(REGIMES),
                   help="default: the three main regimes; the answers-only variants run on request")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--reader-epochs", type=int, default=READER_EPOCHS)
    p.add_argument("--solver-epochs", type=int, default=SOLVER_EPOCHS)
    p.add_argument("--reader-lr", type=float, default=READER_LR)
    p.add_argument("--through", choices=("probability", "logit"), default=THROUGH)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    p.add_argument("--no-verify", action="store_true", help="skip dataset verification (tests only)")
    p.add_argument("--criteria", choices=tuple(CRITERIA), default="exact",
                   help="pass criteria: exact edge set (default) or the true closure (answers-only studies)")
    p.add_argument("--merge", type=Path, nargs="+", default=None, metavar="PART",
                   help="combine result files of parallel runs into --out (no training)")
    args = p.parse_args(argv)
    if args.merge:
        return merge(args.merge, args.out)
    try:
        import torch
    except ImportError:
        print("FAIL: torch required", file=sys.stderr)
        return 2
    rows, ext_rows = load_jsonl(args.train_data), load_jsonl(args.extended_data)
    if not args.no_verify:
        n_val = sum(1 for r in rows if r["split"] == "val")
        issues = [i for ok, found in (
            verify_crossed(rows, n_total=len(rows), n_val=n_val, spec=CROSSED_ID_SPEC),
            verify_crossed(ext_rows, n_total=len(ext_rows), n_val=len(ext_rows), spec=CROSSED_EXTENDED_SPEC),
        ) for i in found]
        if issues:
            print(f"FAIL: dataset verify: {issues}", file=sys.stderr)
            return 1
    target = __import__("reachability_gen.run_stability", fromlist=["target_params"]).target_params()
    solver_params = sum(p.numel() for p in build_solver().parameters())
    if abs(solver_params / target - 1.0) > PARAM_TOL:
        print(f"FAIL parity: solver {solver_params} vs {target}", file=sys.stderr)
        return 1
    train = Data([r for r in rows if r["split"] == "train"], args.device)
    val = Data([r for r in rows if r["split"] == "val"], args.device)
    ext = Data(ext_rows, args.device)
    density = edge_density(train)
    artifact: dict[str, Any] = {
        "science_open": False,
        "purpose": "reader → anchored solver: can the graph come from the edge-list text?",
        "regimes": {k: ALL_REGIMES[k] for k in args.regimes},
        "protocol": {
            "reader": "question-blind, identity-free (one embedding for every node token), slot and clipped "
                      "relative-offset positions, hard 0/1 adjacency with a straight-through gradient",
            "reader_params": sum(p.numel() for p in build_reader().parameters()),
            "solver_params": solver_params,
            "solver_width": anchored_width(),
            "train_steps": TRAIN_STEPS,
            "solver_epochs": args.solver_epochs,
            "reader_epochs": args.reader_epochs,
            "reader_lr": args.reader_lr,
            "straight_through_answers_regimes": args.through,
            "solver_lr": LR,
            "long_steps": list(LONG_STEPS),
            "pass_criteria": CRITERIA[args.criteria],
            **({"criteria": args.criteria} if args.criteria != "exact" else {}),
            "gradient_probe_rows": PROBE_ROWS,
            **({"edge_density_prior": density, "prior_weight": PRIOR_WEIGHT}
               if any("_prior" in r for r in args.regimes) else {}),
            **({"dense_graphs_per_batch": DENSE_GRAPHS_PER_BATCH}
               if any(r.startswith("answers_dense") for r in args.regimes) else {}),
            **({"reinforce_samples": REINFORCE_SAMPLES, "reinforce_baseline": "leave-one-out (RLOO), "
                "advantages normalised per batch; log-probability per node pair"}
               if any(r.endswith("_reinforce") for r in args.regimes) else {}),
            "seeds": args.seeds,
            "batch_size": BATCH, "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "checkpoints_versioned": False,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "torch_version": torch.__version__,
        },
    }
    t0 = time.perf_counter()
    runs = [run(regime, seed, train, val, ext, device=args.device, ckpt_dir=args.ckpt_dir,
                reader_epochs=args.reader_epochs, solver_epochs=args.solver_epochs, reader_lr=args.reader_lr,
                through=args.through, density=density, criteria=args.criteria)
            for regime in args.regimes for seed in args.seeds]
    mismatches = [f"{r['regime']}/seed{r['seed']}" for r in runs if not r["rescore_matches_record"]]
    artifact.update({"runs": runs, "summary": summarise(runs), "self_audit_mismatches": mismatches,
                     "elapsed_seconds": time.perf_counter() - t0})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    for regime, s in artifact["summary"].items():
        long_acc = "/".join(f"{s[f'long_{k}_accuracy']['mean']:.3f}" for k in LONG_STEPS)
        print(f"{regime:15s} passes {s['passes']}/{len(s['seeds'])} | val acc {s['val_accuracy']['mean']:.4f} "
              f"exact graphs {s['val_exact_graphs']['mean']:.4f} | long closure {s['long_closure_agreement']['mean']:.4f} "
              f"| long acc {'/'.join(map(str, LONG_STEPS))}: {long_acc}", file=sys.stderr)
    print(json.dumps({"ok": not mismatches, "out": args.out.as_posix(), "self_audit_mismatches": mismatches,
                      "science_open": False}, sort_keys=True))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())

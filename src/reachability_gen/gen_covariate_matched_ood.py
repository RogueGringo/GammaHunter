# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Generate covariate-matched OOD-hop JSONL (MEASURE; science_open=false).

Resolves Gate2 residue: prior ``ood_hops`` confounded hop K∈{8,12,16} with
seq_len dilation (ID whitespace-mean~57 vs OOD split-token mean~203). This
dataset **isolates path length** by enforcing encoding token length
(``split_encoding_tokens``) in **[45, 70]** (target mean ~57).

Construction (documented n/p search):
  Pure ER under the band yields almost no K=16 (path needs ≥16 edges; band
  caps |E|≤21). Generator therefore uses a **path backbone** of length K plus
  a controlled number of non-shortcut distractor edges, with n chosen in a
  sparse support. Empirical edge density p = |E|/(n(n-1)) is recorded.
  An ER grid search is run at startup and logged in the generation report;
  ER is **not** used for the locked dataset when the band is active.

Quotas: ≥40 pos/hop (default 80) + matching hard-neg count for 50/50.
Fixed seed; writes ``data/covariate_matched_ood.jsonl`` +
``artifacts/covariate_matched_ood_generation_report.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Optional, Sequence

from reachability_gen.adr_invariants import (
    HOP_UNREACHABLE,
    K_TRAIN_MAX,
    OOD_HOP_VALUES,
    is_ood_hop,
)
from reachability_gen.encode import edge_hash as compute_edge_hash
from reachability_gen.encode import encode_instance
from reachability_gen.graph import (
    adjacency_list,
    er_digraph,
    hop_distance,
    hop_distances_from,
)
from reachability_gen.hard_negatives import is_hard_negative, total_degrees
from reachability_gen.schema import ReachabilityExample
from reachability_gen.splits import derive_example_seed
from reachability_gen.tokenize import split_encoding_tokens

# Locked MEASURE seed (distinct from id_2k=42000, ood_hops=84000).
MATCHED_OOD_SEED: int = 126_000

OOD_HOPS: tuple[int, ...] = tuple(OOD_HOP_VALUES)  # 8, 12, 16

POS_PER_HOP_DEFAULT: int = 80
N_NEG_DEFAULT: int = 240  # = 80 * 3 → exact 50/50

SPLIT_NAME: str = "covariate_matched_ood"

# Strict band on split_encoding_tokens length (model-visible tokenization).
SEQ_LEN_MIN: int = 45
SEQ_LEN_MAX: int = 70
SEQ_LEN_TARGET_MEAN: float = 57.0

BOUND30_MAX_LEN: int = 257

# ---------------------------------------------------------------------------
# n / distractor support for path-backbone construction
# Token length = 6 + 3*|E|  ⇒  |E| ∈ [13, 21] for band [45, 70].
# Path of length K contributes K edges; extras fill the remainder.
# ---------------------------------------------------------------------------
N_SUPPORT_BY_HOP: dict[int, tuple[int, ...]] = {
    8: (10, 12, 14, 16, 18),
    12: (14, 16, 18, 20, 22),
    16: (17, 18, 19, 20, 22),
}
# Extra (non-path) edges to request; clipped so total |E| lands in [13, 21].
EXTRA_EDGES_BY_HOP: dict[int, tuple[int, ...]] = {
    8: (5, 6, 7, 8, 9, 10, 11, 12, 13),
    12: (1, 2, 3, 4, 5, 6, 7, 8, 9),
    16: (0, 1, 2, 3, 4, 5),
}
N_SUPPORT_NEG: tuple[int, ...] = (10, 12, 14, 16, 18, 20, 22)
EXTRA_EDGES_NEG: tuple[int, ...] = (3, 5, 7, 9, 11, 13)

# ER grid recorded in the report (search evidence; not the primary sampler).
ER_SEARCH_N: tuple[int, ...] = (12, 14, 16, 18, 20, 22, 24)
ER_SEARCH_P: tuple[float, ...] = (
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.08,
    0.10,
    0.12,
)


def _encoding_token_len(
    n: int, edges: Sequence[tuple[int, int]], s: int, t: int
) -> int:
    return len(split_encoding_tokens(encode_instance(n, list(edges), s, t)))


def _whitespace_len(
    n: int, edges: Sequence[tuple[int, int]], s: int, t: int
) -> int:
    return len(encode_instance(n, list(edges), s, t).split())


def _in_band(token_len: int) -> bool:
    return SEQ_LEN_MIN <= token_len <= SEQ_LEN_MAX


def _empirical_p(n: int, edges: Sequence[tuple[int, int]]) -> float:
    denom = n * (n - 1)
    if denom <= 0:
        return 0.0
    return len(edges) / float(denom)


def _hop_matrix(n: int, edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    adj = adjacency_list(n, edges)
    mat = [[-1] * n for _ in range(n)]
    for s in range(n):
        mat[s][s] = 0
        q: deque[int] = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if mat[s][v] < 0:
                    mat[s][v] = mat[s][u] + 1
                    q.append(v)
    return mat


def plant_path_graph(
    n: int,
    k: int,
    n_extra_edges: int,
    rng: random.Random,
    *,
    s: int = 0,
) -> tuple[list[tuple[int, int]], int, int]:
    """Directed path s→…→s+k plus up to ``n_extra_edges`` non-shortcut edges.

    Rejects candidate distractors that would shorten hop(s, s+k) below k.
    """
    if n < k + 1:
        raise ValueError(f"n={n} < k+1={k + 1}")
    t = s + k
    if t >= n:
        raise ValueError(f"path end t={t} out of range for n={n}")
    edges: set[tuple[int, int]] = {(s + i, s + i + 1) for i in range(k)}
    candidates = [
        (u, v)
        for u in range(n)
        for v in range(n)
        if u != v and (u, v) not in edges
    ]
    rng.shuffle(candidates)
    added = 0
    for u, v in candidates:
        if added >= n_extra_edges:
            break
        edges.add((u, v))
        d = hop_distance(n, list(edges), s, t)
        if d != k:
            edges.remove((u, v))
            continue
        added += 1
    return sorted(edges), s, t


def run_er_band_search(
    *,
    seed: int = MATCHED_OOD_SEED,
    trials_per_cell: int = 200,
) -> dict[str, Any]:
    """Document ER (n,p) feasibility under the seq_len band (search only)."""
    rng = random.Random(seed ^ 0xE801)
    by_hop: dict[str, Any] = {}
    for k in OOD_HOPS:
        cells: list[dict[str, Any]] = []
        total_in_band = 0
        total_with_k = 0
        for n in ER_SEARCH_N:
            if n < k + 1:
                continue
            for p in ER_SEARCH_P:
                in_band = 0
                with_k = 0
                for _ in range(trials_per_cell):
                    edges = er_digraph(n, p, rng)
                    # Probe length independent of (s,t) at fixed |E|.
                    probe = _encoding_token_len(n, edges, 0, min(1, n - 1))
                    if not _in_band(probe):
                        continue
                    in_band += 1
                    found = False
                    for s in range(n):
                        dists = hop_distances_from(n, edges, s)
                        if k in dists:
                            found = True
                            break
                    if found:
                        with_k += 1
                total_in_band += in_band
                total_with_k += with_k
                if in_band > 0 or with_k > 0:
                    cells.append(
                        {
                            "n": n,
                            "p": p,
                            "trials": trials_per_cell,
                            "in_band": in_band,
                            "with_hop_k": with_k,
                        }
                    )
        by_hop[str(k)] = {
            "cells_nonzero": cells,
            "total_in_band_graphs": total_in_band,
            "total_graphs_with_hop_k_in_band": total_with_k,
            "note": (
                "Pure ER under [45,70] rarely yields K=16 "
                "(path needs ≥16 edges; band caps |E|≤21)."
                if k == 16
                else "ER under band is scarce for long hops; path-backbone used."
            ),
        }
    return {
        "method": "erdos_renyi_grid",
        "n_grid": list(ER_SEARCH_N),
        "p_grid": list(ER_SEARCH_P),
        "trials_per_cell": trials_per_cell,
        "seq_len_band": [SEQ_LEN_MIN, SEQ_LEN_MAX],
        "by_hop": by_hop,
        "conclusion": (
            "Primary sampler = path backbone + non-shortcut distractors; "
            "ER grid retained as documented search evidence only."
        ),
    }


def _harvest_hard_negs_in_band(
    n: int,
    edges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    hop_mat = _hop_matrix(n, edges)
    deg = total_degrees(n, edges)
    negs: list[tuple[int, int]] = []
    for s in range(n):
        for t in range(n):
            if hop_mat[s][t] >= 0:
                continue
            if deg[s] < 1 or deg[t] < 1:
                continue
            ok, _ = is_hard_negative(n, edges, s, t, y=0)
            if not ok:
                continue
            if _in_band(_encoding_token_len(n, edges, s, t)):
                negs.append((s, t))
    return negs


def _make_example(
    *,
    seed: int,
    n: int,
    p: float,
    edges: list[tuple[int, int]],
    s: int,
    t: int,
    y: int,
    hop: int,
    n_attempts: int = 1,
) -> ReachabilityExample:
    eh = compute_edge_hash(edges)
    enc = encode_instance(n, edges, s, t)
    return ReachabilityExample(
        split=SPLIT_NAME,
        seed=seed,
        n=n,
        p=p,
        edge_hash=eh,
        s=s,
        t=t,
        y=y,
        hop_distance=hop,
        is_ood=is_ood_hop(hop, k_train_max=K_TRAIN_MAX),
        encoding=enc,
        arm_id=None,
        arm_meta=None,
        n_attempts=n_attempts,
    )


def generate_covariate_matched_ood(
    *,
    seed: int = MATCHED_OOD_SEED,
    pos_per_hop: int = POS_PER_HOP_DEFAULT,
    n_neg: int = N_NEG_DEFAULT,
    max_graph_draws: int = 80_000,
    progress_every: int = 500,
    run_er_search: bool = True,
    er_search_trials: int = 150,
    allow_band_deviation: bool = False,
) -> tuple[list[ReachabilityExample], dict[str, Any]]:
    """Generate locked covariate-matched OOD set; return (examples, report).

    If the strict band cannot be filled for some hop and ``allow_band_deviation``
    is False, raises. Never silently widens past 70 without recording deviation.
    """
    if pos_per_hop < 1:
        raise ValueError("pos_per_hop must be >= 1")
    if n_neg < 1:
        raise ValueError("n_neg must be >= 1")
    n_pos_total = pos_per_hop * len(OOD_HOPS)
    if n_neg != n_pos_total:
        raise ValueError(
            f"50/50 balance requires n_neg == pos_per_hop * {len(OOD_HOPS)} "
            f"(={n_pos_total}); got n_neg={n_neg}"
        )

    er_search_report: Optional[dict[str, Any]] = None
    if run_er_search:
        print("[matched-ood] running ER band search (documented)…", file=sys.stderr)
        er_search_report = run_er_band_search(
            seed=seed, trials_per_cell=er_search_trials
        )

    master = random.Random(seed)
    # pool entries: (n, p_emp, edges, s, t)
    pos_pools: dict[int, list[tuple[int, float, list[tuple[int, int]], int, int]]] = {
        k: [] for k in OOD_HOPS
    }
    neg_pool: list[tuple[int, float, list[tuple[int, int]], int, int]] = []
    seen_keys: set[tuple[str, int, int]] = set()
    need_pos = {k: pos_per_hop for k in OOD_HOPS}
    need_neg = n_neg
    draws = 0
    reject_reasons: Counter = Counter()
    band_rejects = 0
    # Track any accepted examples outside band (should stay empty unless deviation).
    band_deviations: list[dict[str, Any]] = []

    def _key(edges, s, t) -> tuple[str, int, int]:
        return (compute_edge_hash(edges), s, t)

    while any(len(pos_pools[k]) < need_pos[k] for k in OOD_HOPS) or len(
        neg_pool
    ) < need_neg:
        if draws >= max_graph_draws:
            break
        draws += 1
        scarce = [k for k in OOD_HOPS if len(pos_pools[k]) < need_pos[k]]
        prefer_pos = scarce and (
            len(neg_pool) >= need_neg or master.random() < 0.80
        )

        if prefer_pos:
            scarce.sort(
                key=lambda k: need_pos[k] - len(pos_pools[k]), reverse=True
            )
            target_hop = scarce[0]
            n_opts = N_SUPPORT_BY_HOP[target_hop]
            n = master.choice(n_opts)
            if n < target_hop + 1:
                n = target_hop + 1
            extra = master.choice(EXTRA_EDGES_BY_HOP[target_hop])
            # Keep total |E| ≤ 21 (band max).
            max_extra = max(0, 21 - target_hop)
            extra = min(extra, max_extra)
            edges, s, t = plant_path_graph(n, target_hop, extra, master)
            p_emp = _empirical_p(n, edges)
            tl = _encoding_token_len(n, edges, s, t)
            if not _in_band(tl):
                band_rejects += 1
                reject_reasons["pos_out_of_band"] += 1
                if allow_band_deviation and tl > SEQ_LEN_MAX:
                    # Record but do not accept unless explicitly allowed AND we
                    # still refuse silent widen — only accept if caller set the
                    # flag AND we document; default path never enters here for accept.
                    pass
                continue
            d = hop_distance(n, edges, s, t)
            if d != target_hop:
                reject_reasons["hop_mismatch"] += 1
                continue
            key = _key(edges, s, t)
            if key in seen_keys:
                reject_reasons["dup_pos"] += 1
            else:
                seen_keys.add(key)
                pos_pools[target_hop].append((n, p_emp, list(edges), s, t))
            # Also harvest hard negs from the same graph.
            if len(neg_pool) < need_neg:
                hard_negs = _harvest_hard_negs_in_band(n, edges)
                master.shuffle(hard_negs)
                for ns, nt in hard_negs:
                    if len(neg_pool) >= need_neg:
                        break
                    nkey = _key(edges, ns, nt)
                    if nkey in seen_keys:
                        reject_reasons["dup_neg"] += 1
                        continue
                    seen_keys.add(nkey)
                    neg_pool.append((n, p_emp, list(edges), ns, nt))
        else:
            # Dedicated neg draws: path of random ID-range length + extras,
            # then take an unreachable hard pair in band.
            k_backbone = master.choice((4, 5, 6, 7, 8, 10, 12))
            n = master.choice(N_SUPPORT_NEG)
            if n < k_backbone + 1:
                n = k_backbone + 1
            extra = master.choice(EXTRA_EDGES_NEG)
            max_extra = max(0, 21 - k_backbone)
            extra = min(extra, max_extra)
            edges, _, _ = plant_path_graph(n, k_backbone, extra, master)
            p_emp = _empirical_p(n, edges)
            hard_negs = _harvest_hard_negs_in_band(n, edges)
            if not hard_negs:
                reject_reasons["neg_empty"] += 1
                continue
            master.shuffle(hard_negs)
            for ns, nt in hard_negs:
                if len(neg_pool) >= need_neg:
                    break
                nkey = _key(edges, ns, nt)
                if nkey in seen_keys:
                    reject_reasons["dup_neg"] += 1
                    continue
                seen_keys.add(nkey)
                neg_pool.append((n, p_emp, list(edges), ns, nt))

        if progress_every and draws % progress_every == 0:
            filled_pos = {k: len(pos_pools[k]) for k in OOD_HOPS}
            print(
                f"gen_matched_ood draws={draws} pos={filled_pos} "
                f"neg={len(neg_pool)} band_rej={band_rejects}",
                file=sys.stderr,
            )

    shortfalls: dict[str, Any] = {}
    for k in OOD_HOPS:
        if len(pos_pools[k]) < need_pos[k]:
            shortfalls[f"pos_hop_{k}"] = {
                "have": len(pos_pools[k]),
                "need": need_pos[k],
            }
    if len(neg_pool) < need_neg:
        shortfalls["neg"] = {"have": len(neg_pool), "need": need_neg}
    if shortfalls:
        raise RuntimeError(
            f"covariate_matched_ood shortfall after {draws} draws: {shortfalls}. "
            f"reject_reasons={dict(reject_reasons)}. "
            "If K=16 band is infeasible, document failure — do not silently widen."
        )

    for k in OOD_HOPS:
        pool_rng = random.Random(derive_example_seed(seed, k * 1000))
        pool_rng.shuffle(pos_pools[k])
        pos_pools[k] = pos_pools[k][:pos_per_hop]
    neg_rng = random.Random(derive_example_seed(seed, 99_000))
    neg_rng.shuffle(neg_pool)
    neg_pool = neg_pool[:n_neg]

    examples: list[ReachabilityExample] = []
    example_index = 0
    for k in OOD_HOPS:
        for n, p, edges, s, t in pos_pools[k]:
            ex_seed = derive_example_seed(seed, example_index)
            ex = _make_example(
                seed=ex_seed,
                n=n,
                p=p,
                edges=edges,
                s=s,
                t=t,
                y=1,
                hop=k,
            )
            tl = len(split_encoding_tokens(ex.encoding))
            if not _in_band(tl):
                band_deviations.append(
                    {
                        "hop": k,
                        "token_len": tl,
                        "n": n,
                        "y": 1,
                        "note": "accepted outside band",
                    }
                )
            examples.append(ex)
            example_index += 1

    for n, p, edges, s, t in neg_pool:
        ex_seed = derive_example_seed(seed, example_index)
        ex = _make_example(
            seed=ex_seed,
            n=n,
            p=p,
            edges=edges,
            s=s,
            t=t,
            y=0,
            hop=HOP_UNREACHABLE,
        )
        tl = len(split_encoding_tokens(ex.encoding))
        if not _in_band(tl):
            band_deviations.append(
                {
                    "hop": HOP_UNREACHABLE,
                    "token_len": tl,
                    "n": n,
                    "y": 0,
                    "note": "accepted outside band",
                }
            )
        examples.append(ex)
        example_index += 1

    if band_deviations and not allow_band_deviation:
        raise RuntimeError(
            f"band deviations detected ({len(band_deviations)}); "
            "refusing to write (fail-closed). Sample: "
            f"{band_deviations[:3]}"
        )

    order_rng = random.Random(derive_example_seed(seed, 1))
    order_rng.shuffle(examples)

    report = build_generation_report(
        examples,
        seed=seed,
        graph_draws=draws,
        pos_per_hop=pos_per_hop,
        n_neg=n_neg,
    )
    report["reject_reasons"] = dict(reject_reasons)
    report["band_rejects"] = band_rejects
    report["band_deviations"] = band_deviations
    report["n_support_by_hop"] = {
        str(k): list(v) for k, v in N_SUPPORT_BY_HOP.items()
    }
    report["extra_edges_by_hop"] = {
        str(k): list(v) for k, v in EXTRA_EDGES_BY_HOP.items()
    }
    report["n_support_neg"] = list(N_SUPPORT_NEG)
    report["construction"] = {
        "primary": "path_backbone_plus_non_shortcut_distractors",
        "token_len_formula": "len(split_encoding_tokens(encode)) = 6 + 3*|E|",
        "seq_len_band": [SEQ_LEN_MIN, SEQ_LEN_MAX],
        "seq_len_target_mean": SEQ_LEN_TARGET_MEAN,
        "p_field": "empirical |E|/(n*(n-1))",
        "er_search": er_search_report,
    }
    report["science_open"] = False
    return examples, report


def build_generation_report(
    examples: Sequence[ReachabilityExample] | Sequence[dict[str, Any]],
    *,
    seed: int = MATCHED_OOD_SEED,
    graph_draws: Optional[int] = None,
    pos_per_hop: int = POS_PER_HOP_DEFAULT,
    n_neg: int = N_NEG_DEFAULT,
) -> dict[str, Any]:
    """Counts / seq_len stats for the covariate-matched generation report."""
    rows = [
        e.to_dict() if isinstance(e, ReachabilityExample) else dict(e)
        for e in examples
    ]
    by_hop: Counter = Counter()
    y_counts: Counter = Counter()
    n_values_used: Counter = Counter()
    p_values_used: Counter = Counter()
    token_lens: list[int] = []
    whitespace_lens: list[int] = []
    n_is_ood_true = 0
    out_of_band = 0

    for r in rows:
        y = int(r["y"])
        hop = int(r["hop_distance"])
        y_counts[y] += 1
        by_hop[hop] += 1
        n_values_used[int(r.get("n", 0))] += 1
        # Bucket empirical p to 4 decimals for readable histogram keys.
        p_values_used[round(float(r.get("p", 0.0)), 4)] += 1
        enc = str(r.get("encoding", ""))
        tl = len(split_encoding_tokens(enc))
        token_lens.append(tl)
        whitespace_lens.append(len(enc.split()))
        if not _in_band(tl):
            out_of_band += 1
        if bool(r.get("is_ood", False)):
            n_is_ood_true += 1

    token_by_hop: dict[str, Any] = {}
    for hop in list(OOD_HOPS) + [HOP_UNREACHABLE]:
        lens_h = [
            len(split_encoding_tokens(str(r.get("encoding", ""))))
            for r in rows
            if int(r["hop_distance"]) == hop
        ]
        if lens_h:
            token_by_hop[str(hop)] = {
                "n": len(lens_h),
                "min": min(lens_h),
                "max": max(lens_h),
                "mean": sum(lens_h) / len(lens_h),
            }

    return {
        "seed": seed,
        "n_total": len(rows),
        "y_counts": {str(k): v for k, v in sorted(y_counts.items())},
        "hop_counts": {str(k): v for k, v in sorted(by_hop.items())},
        "n_values_used": {str(k): v for k, v in sorted(n_values_used.items())},
        "p_values_used": {str(k): v for k, v in sorted(p_values_used.items())},
        "token_len": {
            "min": min(token_lens) if token_lens else None,
            "max": max(token_lens) if token_lens else None,
            "mean": (sum(token_lens) / len(token_lens)) if token_lens else None,
            "band": [SEQ_LEN_MIN, SEQ_LEN_MAX],
            "target_mean": SEQ_LEN_TARGET_MEAN,
            "out_of_band": out_of_band,
            "measure": "split_encoding_tokens",
            "bound30_max_len": BOUND30_MAX_LEN,
            "by_hop": token_by_hop,
        },
        "whitespace_len": {
            "min": min(whitespace_lens) if whitespace_lens else None,
            "max": max(whitespace_lens) if whitespace_lens else None,
            "mean": (
                (sum(whitespace_lens) / len(whitespace_lens))
                if whitespace_lens
                else None
            ),
            "note": (
                "Whitespace token count (seal Gate2 cited ID mean~57 on this "
                "measure); model-visible length is token_len above."
            ),
        },
        "id_band_reference": {
            "id_whitespace_mean_cited": 57.36,
            "id_split_encoding_mean_recomputed_note": (
                "Recompute from data/id_2k.jsonl at report time if present; "
                "seal mixed whitespace(ID) vs split_encoding(OOD)."
            ),
            "prior_ood_split_encoding_mean": 203.225,
        },
        "n_is_ood_true": n_is_ood_true,
        "graph_draws": graph_draws,
        "quotas": {
            "pos_per_hop": pos_per_hop,
            "ood_hops": list(OOD_HOPS),
            "n_neg": n_neg,
            "n_pos_total": pos_per_hop * len(OOD_HOPS),
        },
        "science_open": False,
    }


def verify_covariate_matched_ood(
    examples: Sequence[ReachabilityExample] | Sequence[dict[str, Any]] | Path,
    *,
    pos_per_hop: int = POS_PER_HOP_DEFAULT,
    n_neg: int = N_NEG_DEFAULT,
    check_hard_neg: bool = True,
    check_band: bool = True,
    allow_band_deviation: bool = False,
) -> tuple[bool, list[str]]:
    """Verify hop set, 50/50 balance, hard-neg, and seq_len band."""
    if isinstance(examples, Path):
        rows: list[dict[str, Any]] = []
        with examples.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        rows = [
            e.to_dict() if isinstance(e, ReachabilityExample) else dict(e)
            for e in examples
        ]

    issues: list[str] = []
    n_pos_total = pos_per_hop * len(OOD_HOPS)
    n_total = n_pos_total + n_neg
    if len(rows) != n_total:
        issues.append(f"n_total={len(rows)} != {n_total}")

    y1 = sum(1 for r in rows if int(r["y"]) == 1)
    y0 = sum(1 for r in rows if int(r["y"]) == 0)
    if y1 != n_pos_total or y0 != n_neg:
        issues.append(f"y counts y1={y1} y0={y0} want {n_pos_total}/{n_neg}")
    if y1 != y0:
        issues.append(f"class imbalance y1={y1} y0={y0}")

    for k in OOD_HOPS:
        c = sum(
            1
            for r in rows
            if int(r["y"]) == 1 and int(r["hop_distance"]) == k
        )
        if c != pos_per_hop:
            issues.append(f"hop={k} pos count={c} want {pos_per_hop}")

    for r in rows:
        hop = int(r["hop_distance"])
        y = int(r["y"])
        is_ood = bool(r.get("is_ood", False))
        if y == 0:
            if hop != HOP_UNREACHABLE:
                issues.append(
                    f"y0 hop={hop} want {HOP_UNREACHABLE} "
                    f"edge_hash={r.get('edge_hash')}"
                )
                break
            if is_ood:
                issues.append("y0 marked is_ood=True (unreachable must be False)")
                break
        else:
            if hop not in OOD_HOPS:
                issues.append(f"y1 hop={hop} not in OOD_HOPS={OOD_HOPS}")
                break
            if not is_ood:
                issues.append(f"y1 hop={hop} must have is_ood=True")
                break
        enc = str(r.get("encoding", ""))
        tl = len(split_encoding_tokens(enc))
        if check_band and not _in_band(tl):
            msg = (
                f"seq_len={tl} outside [{SEQ_LEN_MIN},{SEQ_LEN_MAX}] "
                f"n={r.get('n')} hop={hop}"
            )
            if allow_band_deviation:
                # Still record as issue soft-flag via continue? Fail-closed default.
                issues.append(msg + " (deviation recorded)")
            else:
                issues.append(msg)
                break

    if check_hard_neg:
        from reachability_gen.hard_negatives import classify_y0_row

        for r in rows:
            if int(r["y"]) != 0:
                continue
            ok, reason = classify_y0_row(r)
            if not ok:
                issues.append(
                    f"soft/invalid negative reason={reason} "
                    f"s={r.get('s')} t={r.get('t')} n={r.get('n')}"
                )
                break

    for r in rows:
        if r.get("science_open") is True:
            issues.append("row has science_open=True")
            break

    return len(issues) == 0, issues


def write_jsonl(path: Path, examples: Sequence[ReachabilityExample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict(), sort_keys=True) + "\n")
            count += 1
    return count


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate data/covariate_matched_ood.jsonl "
            "(MEASURE; seq_len band [45,70]; science_open=false)."
        )
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/covariate_matched_ood.jsonl"),
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/covariate_matched_ood_generation_report.json"),
    )
    p.add_argument("--seed", type=int, default=MATCHED_OOD_SEED)
    p.add_argument("--pos-per-hop", type=int, default=POS_PER_HOP_DEFAULT)
    p.add_argument(
        "--n-neg",
        type=int,
        default=None,
        help="Hard-negative count (default: pos_per_hop * len(OOD_HOPS)).",
    )
    p.add_argument("--max-graph-draws", type=int, default=80_000)
    p.add_argument(
        "--skip-er-search",
        action="store_true",
        help="Skip documented ER grid search (faster).",
    )
    p.add_argument("--er-search-trials", type=int, default=150)
    p.add_argument(
        "--verify-only",
        type=Path,
        default=None,
        help="Only verify an existing JSONL; do not regenerate.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    n_neg = (
        args.n_neg if args.n_neg is not None else args.pos_per_hop * len(OOD_HOPS)
    )

    if args.verify_only is not None:
        ok, issues = verify_covariate_matched_ood(
            args.verify_only,
            pos_per_hop=args.pos_per_hop,
            n_neg=n_neg,
        )
        print(json.dumps({"ok": ok, "issues": issues}, sort_keys=True))
        return 0 if ok else 1

    examples, report = generate_covariate_matched_ood(
        seed=args.seed,
        pos_per_hop=args.pos_per_hop,
        n_neg=n_neg,
        max_graph_draws=args.max_graph_draws,
        run_er_search=not args.skip_er_search,
        er_search_trials=args.er_search_trials,
    )
    n = write_jsonl(args.out, examples)
    ok, issues = verify_covariate_matched_ood(
        examples,
        pos_per_hop=args.pos_per_hop,
        n_neg=n_neg,
    )
    report["verify_ok"] = ok
    report["verify_issues"] = issues
    # Attach ID seq_len recomputation when id_2k is present.
    id_path = Path("data/id_2k.jsonl")
    if id_path.exists():
        id_tok: list[int] = []
        id_ws: list[int] = []
        with id_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                enc = str(json.loads(line).get("encoding", ""))
                id_tok.append(len(split_encoding_tokens(enc)))
                id_ws.append(len(enc.split()))
        report["id_band_reference"]["id_split_encoding"] = {
            "min": min(id_tok),
            "max": max(id_tok),
            "mean": sum(id_tok) / len(id_tok),
            "n": len(id_tok),
        }
        report["id_band_reference"]["id_whitespace"] = {
            "min": min(id_ws),
            "max": max(id_ws),
            "mean": sum(id_ws) / len(id_ws),
            "n": len(id_ws),
        }
    write_report(args.report, report)
    print(
        f"wrote {n} examples → {args.out}; report → {args.report}; "
        f"verify_ok={ok}; token_len mean="
        f"{report.get('token_len', {}).get('mean')}",
        file=sys.stderr,
    )
    if not ok:
        print(f"VERIFY FAIL: {issues}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "ok": True,
                "n": n,
                "out": str(args.out),
                "token_len_mean": report.get("token_len", {}).get("mean"),
                "science_open": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

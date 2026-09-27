# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Generate held-out OOD-hop reachability JSONL for Gate 2 (MEASURE).

Constraints (science_open=False always):
  - Positives: hop_distance exactly in ADR-001 OOD_HOP_VALUES {8,12,16},
    balanced across buckets (default 80 / hop).
  - Negatives: ALL hard — deg(s)>=1, deg(t)>=1, unreachable; hop=-1.
  - Class balance 50/50 overall.
  - is_ood=True for positives (K>K_TRAIN_MAX); False for negatives.
  - Graphs large enough for K=16; encodings filtered to fit bound30
    pos-emb max_len (257) with headroom (MAX_TOKENS_ACCEPT=250).
  - Fixed seed; writes generation report under artifacts/.

MEASURE plumbing only — no science OPEN claims.
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
from reachability_gen.graph import adjacency_list, er_digraph
from reachability_gen.hard_negatives import is_hard_negative, total_degrees
from reachability_gen.schema import ReachabilityExample
from reachability_gen.splits import derive_example_seed
from reachability_gen.tokenize import split_encoding_tokens

# Locked MEASURE seed for regenerability of ood_hops.
OOD_HOPS_SEED: int = 84_000

OOD_HOPS: tuple[int, ...] = tuple(OOD_HOP_VALUES)  # 8, 12, 16

# Quotas (prereg: >=40 pos/hop + >=120 hard neg; prefer 80/hop + 240 neg)
POS_PER_HOP_DEFAULT: int = 80
N_NEG_DEFAULT: int = 240  # = 80 * 3 → exact 50/50

SPLIT_NAME: str = "ood_hops"

# Bound30 checkpoints use pos_emb max_len=257; reject encodings above headroom.
MAX_TOKENS_ACCEPT: int = 250
BOUND30_MAX_LEN: int = 257

# n/p support sized for long hops; sparse enough to keep |E| under token budget.
N_SUPPORT_BY_HOP: dict[int, tuple[int, ...]] = {
    8: (24, 28, 32, 40),
    12: (32, 40, 48, 56),
    16: (40, 48, 56, 64),
}
P_SUPPORT_BY_HOP: dict[int, tuple[float, ...]] = {
    8: (0.05, 0.06, 0.08, 0.10),
    12: (0.025, 0.03, 0.04, 0.05),
    16: (0.015, 0.018, 0.02, 0.025),
}
N_SUPPORT_NEG: tuple[int, ...] = (24, 28, 32, 40, 48, 56, 64)
P_SUPPORT_NEG: tuple[float, ...] = (0.02, 0.03, 0.04, 0.05, 0.06, 0.08)


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


def _encoding_token_len(n: int, edges: Sequence[tuple[int, int]], s: int, t: int) -> int:
    return len(split_encoding_tokens(encode_instance(n, list(edges), s, t)))


def _n_p_for_hop(hop: int, rng: random.Random) -> tuple[int, float]:
    n_opts = N_SUPPORT_BY_HOP.get(hop, N_SUPPORT_BY_HOP[16])
    p_opts = P_SUPPORT_BY_HOP.get(hop, P_SUPPORT_BY_HOP[16])
    n = rng.choice(n_opts)
    p = rng.choice(p_opts)
    if n < hop + 1:
        n = hop + 1
    return n, p


def _n_p_for_neg(rng: random.Random) -> tuple[int, float]:
    return rng.choice(N_SUPPORT_NEG), rng.choice(P_SUPPORT_NEG)


def _harvest_graph(
    n: int,
    edges: list[tuple[int, int]],
    *,
    want_hops: Optional[set[int]] = None,
    want_hard_neg: bool = True,
    max_tokens: int = MAX_TOKENS_ACCEPT,
) -> tuple[dict[int, list[tuple[int, int]]], list[tuple[int, int]]]:
    """Return (positives_by_hop, hard_neg_pairs); skip pairs exceeding token budget."""
    hop_mat = _hop_matrix(n, edges)
    deg = total_degrees(n, edges)
    pos: dict[int, list[tuple[int, int]]] = defaultdict(list)
    negs: list[tuple[int, int]] = []
    hop_filter = want_hops if want_hops is not None else set(OOD_HOPS)
    # Cheap length proxy: encoding length depends weakly on (s,t); check once
    # with a dummy query, then re-check accepted pairs.
    probe_len = _encoding_token_len(n, edges, 0, min(1, n - 1))
    if probe_len > max_tokens:
        return pos, negs

    for s in range(n):
        for t in range(n):
            k = hop_mat[s][t]
            if k in hop_filter:
                if _encoding_token_len(n, edges, s, t) <= max_tokens:
                    pos[k].append((s, t))
            elif want_hard_neg and k < 0 and deg[s] >= 1 and deg[t] >= 1:
                ok, _ = is_hard_negative(n, edges, s, t, y=0)
                if ok and _encoding_token_len(n, edges, s, t) <= max_tokens:
                    negs.append((s, t))
    return pos, negs


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


def generate_ood_hops(
    *,
    seed: int = OOD_HOPS_SEED,
    pos_per_hop: int = POS_PER_HOP_DEFAULT,
    n_neg: int = N_NEG_DEFAULT,
    max_graph_draws: int = 80_000,
    progress_every: int = 500,
    max_tokens: int = MAX_TOKENS_ACCEPT,
) -> tuple[list[ReachabilityExample], dict[str, Any]]:
    """Generate the locked OOD-hop dataset; return (examples, report)."""
    if pos_per_hop < 1:
        raise ValueError("pos_per_hop must be >= 1")
    if n_neg < 1:
        raise ValueError("n_neg must be >= 1")
    # Enforce 50/50 overall.
    n_pos_total = pos_per_hop * len(OOD_HOPS)
    if n_neg != n_pos_total:
        raise ValueError(
            f"50/50 balance requires n_neg == pos_per_hop * {len(OOD_HOPS)} "
            f"(={n_pos_total}); got n_neg={n_neg}"
        )

    master = random.Random(seed)
    pos_pools: dict[int, list[tuple[int, float, list[tuple[int, int]], int, int]]] = {
        k: [] for k in OOD_HOPS
    }
    neg_pool: list[tuple[int, float, list[tuple[int, int]], int, int]] = []
    seen_keys: set[tuple[str, int, int]] = set()
    need_pos = {k: pos_per_hop for k in OOD_HOPS}
    need_neg = n_neg
    draws = 0
    reject_reasons: Counter = Counter()

    def _key(edges, s, t) -> tuple[str, int, int]:
        return (compute_edge_hash(edges), s, t)

    while any(len(pos_pools[k]) < need_pos[k] for k in OOD_HOPS) or len(neg_pool) < need_neg:
        if draws >= max_graph_draws:
            break
        draws += 1
        scarce = [k for k in OOD_HOPS if len(pos_pools[k]) < need_pos[k]]
        if scarce and (len(neg_pool) >= need_neg or master.random() < 0.75):
            scarce.sort(key=lambda k: need_pos[k] - len(pos_pools[k]), reverse=True)
            target_hop = scarce[0]
            n, p = _n_p_for_hop(target_hop, master)
            want_hops: Optional[set[int]] = set(OOD_HOPS)
        else:
            n, p = _n_p_for_neg(master)
            want_hops = set(OOD_HOPS)

        edges = er_digraph(n, p, master)
        pos_by_hop, hard_negs = _harvest_graph(
            n,
            edges,
            want_hops=want_hops,
            want_hard_neg=True,
            max_tokens=max_tokens,
        )
        if not pos_by_hop and not hard_negs:
            reject_reasons["empty_or_overlong"] += 1

        for k, pairs in pos_by_hop.items():
            if k not in need_pos:
                continue
            master.shuffle(pairs)
            for s, t in pairs:
                if len(pos_pools[k]) >= need_pos[k]:
                    break
                key = _key(edges, s, t)
                if key in seen_keys:
                    reject_reasons["dup_pos"] += 1
                    continue
                seen_keys.add(key)
                pos_pools[k].append((n, p, list(edges), s, t))

        if len(neg_pool) < need_neg:
            master.shuffle(hard_negs)
            for s, t in hard_negs:
                if len(neg_pool) >= need_neg:
                    break
                key = _key(edges, s, t)
                if key in seen_keys:
                    reject_reasons["dup_neg"] += 1
                    continue
                seen_keys.add(key)
                neg_pool.append((n, p, list(edges), s, t))

        if progress_every and draws % progress_every == 0:
            filled_pos = {k: len(pos_pools[k]) for k in OOD_HOPS}
            print(
                f"gen_ood_hops draws={draws} pos={filled_pos} neg={len(neg_pool)}",
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
            f"ood_hops generation shortfall after {draws} draws: {shortfalls}"
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
            examples.append(
                _make_example(
                    seed=ex_seed,
                    n=n,
                    p=p,
                    edges=edges,
                    s=s,
                    t=t,
                    y=1,
                    hop=k,
                )
            )
            example_index += 1

    for n, p, edges, s, t in neg_pool:
        ex_seed = derive_example_seed(seed, example_index)
        examples.append(
            _make_example(
                seed=ex_seed,
                n=n,
                p=p,
                edges=edges,
                s=s,
                t=t,
                y=0,
                hop=HOP_UNREACHABLE,
            )
        )
        example_index += 1

    order_rng = random.Random(derive_example_seed(seed, 1))
    order_rng.shuffle(examples)

    report = build_generation_report(
        examples,
        seed=seed,
        graph_draws=draws,
        pos_per_hop=pos_per_hop,
        n_neg=n_neg,
        max_tokens=max_tokens,
    )
    report["reject_reasons"] = dict(reject_reasons)
    report["n_support_by_hop"] = {str(k): list(v) for k, v in N_SUPPORT_BY_HOP.items()}
    report["p_support_by_hop"] = {str(k): list(v) for k, v in P_SUPPORT_BY_HOP.items()}
    report["n_support_neg"] = list(N_SUPPORT_NEG)
    report["p_support_neg"] = list(P_SUPPORT_NEG)
    report["science_open"] = False
    return examples, report


def build_generation_report(
    examples: Sequence[ReachabilityExample] | Sequence[dict[str, Any]],
    *,
    seed: int = OOD_HOPS_SEED,
    graph_draws: Optional[int] = None,
    pos_per_hop: int = POS_PER_HOP_DEFAULT,
    n_neg: int = N_NEG_DEFAULT,
    max_tokens: int = MAX_TOKENS_ACCEPT,
) -> dict[str, Any]:
    """Counts per hop / class for the OOD generation report."""
    rows = [
        e.to_dict() if isinstance(e, ReachabilityExample) else dict(e)
        for e in examples
    ]
    by_hop: Counter = Counter()
    y_counts: Counter = Counter()
    n_values_used: Counter = Counter()
    p_values_used: Counter = Counter()
    token_lens: list[int] = []
    n_is_ood_true = 0

    for r in rows:
        y = int(r["y"])
        hop = int(r["hop_distance"])
        y_counts[y] += 1
        by_hop[hop] += 1
        n_values_used[int(r.get("n", 0))] += 1
        p_values_used[float(r.get("p", 0.0))] += 1
        enc = str(r.get("encoding", ""))
        token_lens.append(len(split_encoding_tokens(enc)))
        if bool(r.get("is_ood", False)):
            n_is_ood_true += 1

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
            "max_tokens_accept": max_tokens,
            "bound30_max_len": BOUND30_MAX_LEN,
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


def verify_ood_hops(
    examples: Sequence[ReachabilityExample] | Sequence[dict[str, Any]] | Path,
    *,
    pos_per_hop: int = POS_PER_HOP_DEFAULT,
    n_neg: int = N_NEG_DEFAULT,
    check_hard_neg: bool = True,
    max_tokens: int = MAX_TOKENS_ACCEPT,
) -> tuple[bool, list[str]]:
    """Verify ood_hops constraints. Returns ``(ok, list_of_issues)``."""
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
            if not is_ood_hop(hop):
                issues.append(f"is_ood_hop({hop}) unexpectedly False")
                break
        enc = str(r.get("encoding", ""))
        if len(split_encoding_tokens(enc)) > max_tokens:
            issues.append(
                f"encoding token_len exceeds max_tokens={max_tokens} "
                f"n={r.get('n')} hop={hop}"
            )
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
            "Generate data/ood_hops.jsonl (MEASURE Gate 2 plumbing; no science OPEN)."
        )
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/ood_hops.jsonl"),
        help="Output JSONL path (default: data/ood_hops.jsonl).",
    )
    p.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/ood_hops_generation_report.json"),
        help="Generation report JSON path.",
    )
    p.add_argument("--seed", type=int, default=OOD_HOPS_SEED)
    p.add_argument("--pos-per-hop", type=int, default=POS_PER_HOP_DEFAULT)
    p.add_argument(
        "--n-neg",
        type=int,
        default=None,
        help="Hard-negative count (default: pos_per_hop * len(OOD_HOPS) for 50/50).",
    )
    p.add_argument("--max-graph-draws", type=int, default=80_000)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS_ACCEPT)
    p.add_argument(
        "--verify-only",
        type=Path,
        default=None,
        help="Only verify an existing JSONL; do not regenerate.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    n_neg = args.n_neg if args.n_neg is not None else args.pos_per_hop * len(OOD_HOPS)

    if args.verify_only is not None:
        ok, issues = verify_ood_hops(
            args.verify_only,
            pos_per_hop=args.pos_per_hop,
            n_neg=n_neg,
            max_tokens=args.max_tokens,
        )
        print(json.dumps({"ok": ok, "issues": issues}, sort_keys=True))
        return 0 if ok else 1

    examples, report = generate_ood_hops(
        seed=args.seed,
        pos_per_hop=args.pos_per_hop,
        n_neg=n_neg,
        max_graph_draws=args.max_graph_draws,
        max_tokens=args.max_tokens,
    )
    n = write_jsonl(args.out, examples)
    ok, issues = verify_ood_hops(
        examples,
        pos_per_hop=args.pos_per_hop,
        n_neg=n_neg,
        max_tokens=args.max_tokens,
    )
    report["verify_ok"] = ok
    report["verify_issues"] = issues
    write_report(args.report, report)
    print(
        f"wrote {n} examples → {args.out}; report → {args.report}; "
        f"verify_ok={ok}",
        file=sys.stderr,
    )
    if not ok:
        print(f"VERIFY FAIL: {issues}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "n": n, "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

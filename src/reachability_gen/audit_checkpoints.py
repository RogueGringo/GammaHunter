# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Re-score saved best checkpoints with collapse diagnostics (MEASURE).

The rematch runners (fixed30, bound30) report ``best_val_acc`` from the best
epoch but every per-hop / drift / ``||z||`` table from the *last* epoch, and
size ``max_len`` from the first two rows, which cuts the ``QUERY s t`` tail
off 50/2000 id_2k rows. This CLI reloads each saved best checkpoint and
re-scores it on the exact val split with the runners' own pipeline — so the
re-scored accuracy must reproduce the recorded ``best_val_acc`` — then
reports, for that one model:

* per-hop accuracy on all rows (runner-comparable) and on answerable rows
  (query not truncated);
* drift (raw / LN / RMS, runner formulas), pooled ``||z_t||`` and the
  runners' damp/expand label;
* token coherence and mean per-token ``||z_t||`` by cycle (FF: by layer);
* propagated perturbation gain ``||Δz_t|| / ||Δz_0||``;
* τ scale vs state scale (Geo);
* collapse flags (:func:`reachability_gen.diagnostics.collapse_flags`).

It also reports how id_2k was built — distinct graphs, train/val graph
overlap and a query-blind baseline (majority train label of the graph) —
which is the floor every arm's accuracy should be read against, and, for each
extended-step set present in ``data/`` (:data:`EVAL_SETS`), how well it is
covered by the training vocabulary and graph sizes (:func:`eval_set_review`).

``science_open=false`` always.

Usage::

    python -m reachability_gen.audit_checkpoints
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from reachability_gen.adr_invariants import EPS_SIGMA
from reachability_gen.diagnostics import (
    FINAL_VS_BEST_GAP,
    OUTPUT_CONCENTRATION_RATIO,
    TOKEN_COLLAPSE_COS,
    TRAIN_ACC_DROP,
    collapse_flags,
    median_abs_deviation,
)
from reachability_gen.gen_id_2k import verify_id_2k
from reachability_gen.models.geometric import DEFAULT_RESIDUAL_ALPHA
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_id_2k_rematch_bound30 import (
    _drift_regime,
    _mean_traj,
    _n_heads,
    _split_train_val,
)
from reachability_gen.tokenize import (
    Vocab,
    build_vocab,
    encode_to_ids,
    max_token_len,
    required_max_len,
    split_encoding_tokens,
)

RUNS: dict[str, Path] = {
    "fixed30": Path("artifacts/id_2k_rematch_fixed30.json"),
    "bound30": Path("artifacts/id_2k_rematch_bound30.json"),
}
ARMS: tuple[str, ...] = ("ff", "geo", "loop")
DEFAULT_OUT = Path("artifacts/id_2k_checkpoint_audit.json")
# Extended-step evaluation sets reviewed for coverage when present locally.
EVAL_SETS: dict[str, Path] = {
    "ood_hops": Path("data/ood_hops.jsonl"),
    "covariate_matched_ood": Path("data/covariate_matched_ood.jsonl"),
}
FF_ID_SUMMARY = Path("artifacts/ff_id_train_summary.json")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _overall_acc(val_by_hop: dict[str, Any]) -> float:
    n = sum(int(v["n"]) for v in val_by_hop.values())
    return sum(float(v["acc_mean"]) * int(v["n"]) for v in val_by_hop.values()) / n


def runner_max_len(rows: list[dict[str, Any]], vocab: Vocab) -> int:
    """The rematch runners' sizing rule: first two train+val rows, +8, ≥64."""
    train, val = _split_train_val(rows)
    first_two = (train + val)[:2]
    return max(max_token_len((r["encoding"] for r in first_two), vocab) + 8, 64)


def build_model_from_state(
    arm_kind: str,
    state_dict: dict[str, Any],
    *,
    T: int,
    residual_alpha: float,
    pad_id: int,
) -> Any:
    """Rebuild the exact architecture a checkpoint was trained with.

    Widths, depth, MLP expansion, max_len, τ table and state bound are read
    from tensor shapes / keys; ``T`` and ``residual_alpha`` (not in the
    state dict) come from the run JSON. Loading is strict.
    """
    from reachability_gen.models.euclidean_loop import EuclideanLoop
    from reachability_gen.models.feedforward import FeedForward
    from reachability_gen.models.geometric import GeometricRecurrent

    vocab_size, d = (int(x) for x in state_dict["tok_emb.weight"].shape)
    common: dict[str, Any] = {
        "n_heads": _n_heads(d),
        "max_len": int(state_dict["pos_emb.weight"].shape[0]),
        "pad_id": pad_id,
    }
    if arm_kind == "ff":
        L = sum(
            1 for k in state_dict if k.startswith("blocks.") and k.endswith(".ln1.weight")
        )
        mlp_expansion = int(state_dict["blocks.0.mlp.0.weight"].shape[0]) // d
        model = FeedForward(vocab_size, d=d, L=L, mlp_expansion=mlp_expansion, **common)
    elif arm_kind in ("geo", "loop"):
        common.update(
            d=d,
            T=T,
            mlp_expansion=int(state_dict["phi.mlp.0.weight"].shape[0]) // d,
            residual_alpha=residual_alpha,
            apply_cycle_ln="cycle_ln.weight" in state_dict,
            apply_cycle_rmsnorm="cycle_rmsnorm.weight" in state_dict,
        )
        if arm_kind == "geo":
            use_tau = "tau_emb.weight" in state_dict
            max_T = int(state_dict["tau_emb.weight"].shape[0]) if use_tau else None
            model = GeometricRecurrent(vocab_size, use_tau=use_tau, max_T=max_T, **common)
        else:
            model = EuclideanLoop(vocab_size, **common)
    else:
        raise ValueError(f"unknown arm_kind={arm_kind!r}")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def score_model(
    model: Any,
    arm_kind: str,
    val: list[dict[str, Any]],
    vocab: Vocab,
    *,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Per-example metrics, evaluated exactly like the runners' val loop.

    Batches of one, padded to the model's ``max_len``; rows longer than that
    are truncated as in the original runs (``on_overflow="allow"``) and
    marked ``truncated``.
    """
    import torch
    import torch.nn.functional as F

    from reachability_gen.diagnostics import (
        logit_margins,
        mean_token_norm,
        perturbation_gain,
        token_coherence,
    )
    from reachability_gen.models.geometric import (
        drift_from_trajectory,
        mean_z_norms_from_trajectory,
    )
    from reachability_gen.train.ff_trainer import examples_to_batch

    recurrent = arm_kind != "ff"
    has_tau = recurrent and getattr(model, "tau_emb", None) is not None
    max_len = int(model.max_len)
    gen = torch.Generator().manual_seed(seed)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for ex in val:
            ids, mask, labels, _ = examples_to_batch(
                [ex], vocab, max_len=max_len, on_overflow="allow"
            )
            logits, _ = model(ids, mask)
            states = model.token_states(ids, mask)
            coh = [token_coherence(z, mask) for z in states]
            rec: dict[str, Any] = {
                "hop": int(ex["hop_distance"]),
                "truncated": len(encode_to_ids(ex["encoding"], vocab)) > max_len,
                "loss": float(F.cross_entropy(logits, labels).item()),
                "correct": float(int(logits.argmax(dim=-1).item()) == int(labels.item())),
                "margin": float(logit_margins(logits).item()),
                "token_cos_by_t": [float(c.item()) for c, _ in coh],
                "pooled_ratio_by_t": [float(r.item()) for _, r in coh],
                "token_norm_by_t": [float(mean_token_norm(z, mask).item()) for z in states],
            }
            if recurrent:
                traj = [model._pool(z, mask) for z in states]
                rec["drift"] = drift_from_trajectory(traj)
                rec["drift_ln"] = drift_from_trajectory(traj, apply_ln=True)
                rec["drift_rms"] = drift_from_trajectory(traj, apply_rmsnorm=True)
                rec["z_norm_by_t"] = mean_z_norms_from_trajectory(traj)
                noise = torch.randn(
                    (1, ids.shape[1], model.d), generator=gen
                ) * float(EPS_SIGMA)
                perturbed = model.token_states(ids, mask, context_noise=noise)
                rec["perturb_gain_by_t"] = perturbation_gain(states, perturbed, mask)
                rec["perturb_gain_rel_by_t"] = perturbation_gain(
                    states, perturbed, mask, relative=True
                )
            if has_tau:
                # Does adding τ_0 (same vector for every token) align tokens,
                # and does Φ amplify that? Compare Φ with and without τ_0.
                z0 = states[0]
                kpm = mask == 0
                tau0 = model.tau_emb.weight[0].view(1, 1, -1)
                rec["cos_z0_plus_tau0"] = float(token_coherence(z0 + tau0, mask)[0].item())
                rec["cos_phi_with_tau0"] = float(
                    token_coherence(model.phi(z0 + tau0, key_padding_mask=kpm), mask)[0].item()
                )
                rec["cos_phi_without_tau0"] = float(
                    token_coherence(model.phi(z0, key_padding_mask=kpm), mask)[0].item()
                )
            records.append(rec)
    return records


def _aggregate(records: list[dict[str, Any]], recurrent: bool) -> dict[str, Any]:
    answerable = [r for r in records if not r["truncated"]]
    margins = [r["margin"] for r in records]
    out: dict[str, Any] = {
        "n": len(records),
        "n_truncated": len(records) - len(answerable),
        "acc_mean": _mean([r["correct"] for r in records]),
        "acc_answerable": _mean([r["correct"] for r in answerable]),
        "loss_mean": _mean([r["loss"] for r in records]),
        "pred_pos_rate": _mean([float(m > 0) for m in margins]),
        "margin_mean": _mean(margins),
        "margin_sd": statistics.pstdev(margins) if len(margins) > 1 else 0.0,
        "token_cos_by_t": _mean_traj([r["token_cos_by_t"] for r in records]),
        "pooled_ratio_by_t": _mean_traj([r["pooled_ratio_by_t"] for r in records]),
        "token_norm_by_t": _mean_traj([r["token_norm_by_t"] for r in records]),
    }
    if recurrent:
        mean_traj = _mean_traj([r["drift"] for r in records])
        out.update(
            mean_drift_trajectory=mean_traj,
            mean_drift_trajectory_ln=_mean_traj([r["drift_ln"] for r in records]),
            mean_drift_trajectory_rms=_mean_traj([r["drift_rms"] for r in records]),
            mean_terminal_drift=_mean([r["drift"][-1] for r in records]),
            mean_z_norm_by_t=_mean_traj([r["z_norm_by_t"] for r in records]),
            perturb_gain_by_t=_mean_traj([r["perturb_gain_by_t"] for r in records]),
            perturb_gain_rel_by_t=_mean_traj(
                [r["perturb_gain_rel_by_t"] for r in records]
            ),
            damp_regime=_drift_regime(mean_traj),
        )
    return out


def audit_arm(
    arm_kind: str,
    block: dict[str, Any],
    val: list[dict[str, Any]],
    vocab: Vocab,
    *,
    residual_alpha: float,
    seed: int = 0,
) -> dict[str, Any]:
    """Re-score one arm's saved best checkpoint and compare to its run record."""
    import torch

    ckpt_path = Path(block["checkpoint_path"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = build_model_from_state(
        arm_kind,
        ckpt["state_dict"],
        T=int(block.get("T") or 0),
        residual_alpha=float(block.get("residual_alpha", residual_alpha)),
        pad_id=vocab.pad_id,
    )
    recurrent = arm_kind != "ff"
    records = score_model(model, arm_kind, val, vocab, seed=seed)

    by_hop = {
        str(k): _aggregate([r for r in records if r["hop"] == k], recurrent)
        for k in sorted({r["hop"] for r in records})
    }
    overall = _aggregate(records, recurrent)
    n_correct = int(sum(r["correct"] for r in records))
    recorded = float(block["best_val_acc"])
    margins = [r["margin"] for r in records]
    last_epoch_acc = _overall_acc(block["val_by_hop"])

    summary: dict[str, Any] = {
        "val_acc": overall["acc_mean"],
        "n_correct": n_correct,
        "val_acc_answerable": overall["acc_answerable"],
        "n_answerable": overall["n"] - overall["n_truncated"],
        "pred_pos_rate": overall["pred_pos_rate"],
        "margin_sd": overall["margin_sd"],
        "margin_mad": median_abs_deviation(margins),
        "token_cos_by_t": overall["token_cos_by_t"],
        "pooled_ratio_by_t": overall["pooled_ratio_by_t"],
        "token_norm_by_t": overall["token_norm_by_t"],
    }
    if recurrent:
        summary["mean_z_norm_by_t"] = overall["mean_z_norm_by_t"]
        summary["perturb_gain_by_t"] = overall["perturb_gain_by_t"]
        summary["perturb_gain_rel_by_t"] = overall["perturb_gain_rel_by_t"]
        summary["fraction_buckets_damping"] = _mean(
            [float(v["damp_regime"] == "damps") for v in by_hop.values()]
        )
    if getattr(model, "tau_emb", None) is not None:
        norms = torch.linalg.vector_norm(model.tau_emb.weight.detach(), dim=-1).tolist()
        summary["tau"] = {
            "norm_by_t": norms[: model.T],
            "untrained_rows": len(norms) - model.T,
            "untrained_row_norm_mean": _mean(norms[model.T :]),
            "mean_token_norm_t0": overall["token_norm_by_t"][0],
            "token_cos_z0": overall["token_cos_by_t"][0],
            "token_cos_z0_plus_tau0": _mean([r["cos_z0_plus_tau0"] for r in records]),
            "token_cos_phi_with_tau0": _mean([r["cos_phi_with_tau0"] for r in records]),
            "token_cos_phi_without_tau0": _mean(
                [r["cos_phi_without_tau0"] for r in records]
            ),
            "note": (
                "τ_t is added identically to every token; rows ≥T never receive "
                "gradient at T-cycle training (still at init)"
            ),
        }

    return {
        "arm": block["arm"],
        "checkpoint_path": ckpt_path.as_posix(),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "recorded_best_epoch": int(block["best_epoch"]),
        "recorded_best_val_acc": recorded,
        "rescored_val_acc": overall["acc_mean"],
        "rescore_matches_record": (
            n_correct == round(recorded * len(records))
            and int(ckpt["epoch"]) == int(block["best_epoch"])
            and ckpt["arm"] == block["arm"]
        ),
        "summary": summary,
        "val_by_hop": by_hop,
        "reported_last_epoch": {
            "epoch": int(block["epochs_run"]),
            "val_acc": last_epoch_acc,
            "acc_by_hop": {k: v["acc_mean"] for k, v in block["val_by_hop"].items()},
            "mean_z_norm_by_t": block.get("mean_z_norm_by_t"),
            "note": "as written by the runner: last epoch, not this checkpoint",
        },
        "checkpoint_flags": collapse_flags(
            final_token_cos=overall["token_cos_by_t"][-1], margins=margins
        ),
        "run_flags": collapse_flags(
            train_history=block.get("train_history"),
            best_val_acc=recorded,
            last_epoch_val_acc=last_epoch_acc,
        ),
        "science_open": False,
    }


def truncation_report(
    rows: list[dict[str, Any]], vocab: Vocab, max_len: int
) -> dict[str, Any]:
    """Which id_2k rows exceed ``max_len`` (their QUERY tokens were cut)."""
    lens = [len(encode_to_ids(r["encoding"], vocab)) for r in rows]
    over = [r for r, n in zip(rows, lens) if n > max_len]
    val = [r for r in rows if r.get("split") == "val"]
    val_over = Counter(int(r["hop_distance"]) for r in over if r.get("split") == "val")
    val_n = Counter(int(r["hop_distance"]) for r in val)
    return {
        "max_len_used_by_runs": max_len,
        "max_len_rule": "max(longest of first 2 train+val rows + 8, 64)",
        "max_token_len": max(lens),
        "required_max_len": required_max_len((r["encoding"] for r in rows), vocab),
        "n_rows": len(rows),
        "n_truncated": len(over),
        "truncated_by_split_label_hop": {
            f"{s}/y{y}/hop{h}": c
            for (s, y, h), c in sorted(
                Counter(
                    (r["split"], int(r["y"]), int(r["hop_distance"])) for r in over
                ).items()
            )
        },
        "truncated_by_n": {
            str(k): v for k, v in sorted(Counter(int(r["n"]) for r in over).items())
        },
        "val_by_hop": {
            str(h): {"n": val_n[h], "n_truncated": val_over.get(h, 0)}
            for h in sorted(val_n)
        },
        "note": (
            "pad_batch keeps the first max_len tokens, so every truncated row "
            "lost at least its QUERY target: the model never saw the locked "
            "encoding (ADR-001 §1) for these rows. Affects all arms equally."
        ),
    }


def dataset_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How many distinct graphs id_2k uses, and what the graph alone predicts.

    The query-blind baseline predicts, for each val row, the majority train
    label of its graph (``edge_hash``), ignoring ``s``/``t`` entirely.
    """
    train, val = _split_train_val(rows)
    rows_per_graph = Counter(r["edge_hash"] for r in rows)
    labels: dict[str, set[int]] = {}
    for r in rows:
        labels.setdefault(r["edge_hash"], set()).add(int(r["y"]))
    train_labels: dict[str, Counter] = {}
    for r in train:
        train_labels.setdefault(r["edge_hash"], Counter())[int(r["y"])] += 1
    val_graphs = {r["edge_hash"] for r in val}
    train_queries = {(r["edge_hash"], r["s"], r["t"]) for r in train}

    def _blind(r: dict[str, Any]) -> int:
        c = train_labels.get(r["edge_hash"])
        return c.most_common(1)[0][0] if c else 1

    hit: dict[int, list[float]] = {}
    for r in val:
        hit.setdefault(int(r["hop_distance"]), []).append(float(_blind(r) == int(r["y"])))
    counts = sorted(rows_per_graph.values())
    report_path = Path("artifacts/id_2k_generation_report.json")
    graph_draws = (
        json.loads(report_path.read_text()).get("graph_draws")
        if report_path.exists()
        else None
    )
    return {
        "graph_draws": graph_draws,
        "distinct_graphs": len(rows_per_graph),
        "rows_per_graph_max": counts[-1],
        "rows_per_graph_median": counts[len(counts) // 2],
        "single_label_graphs": sum(1 for v in labels.values() if len(v) == 1),
        "negative_graphs": len({r["edge_hash"] for r in rows if int(r["y"]) == 0}),
        "train_graphs": len(train_labels),
        "val_graphs": len(val_graphs),
        "val_graphs_seen_in_train": len(val_graphs & set(train_labels)),
        "val_queries_duplicated_in_train": sum(
            1 for r in val if (r["edge_hash"], r["s"], r["t"]) in train_queries
        ),
        "query_blind_baseline": {
            "rule": "predict the majority train label of the input graph; ignores s, t",
            "val_acc": _mean([x for xs in hit.values() for x in xs]),
            "acc_by_hop": {str(k): _mean(v) for k, v in sorted(hit.items())},
        },
        "note": (
            "gen_id_2k harvests every qualifying (s,t) pair of each sampled "
            "graph and splits train/val per example, not per graph: val asks "
            "new queries on training graphs, so accuracy mixes reachability "
            "with per-graph memorisation. Compare arm accuracy to the "
            "query-blind baseline, not to 0.5."
        ),
    }


def eval_set_review(
    train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Token and graph-size coverage of an evaluation set against training rows.

    Tokens never seen in training reach every arm as untrained embeddings, and
    graph sizes that differ by label let the ``N <n>`` token separate the
    classes; either confounds what the evaluation set can isolate.
    """
    seen = {tok for r in train_rows for tok in split_encoding_tokens(r["encoding"])}
    unseen: set[str] = set()
    by_label: dict[str, Any] = {}
    for y in (0, 1):
        rows = [r for r in eval_rows if int(r["y"]) == y]
        with_unseen = 0
        for r in rows:
            new = set(split_encoding_tokens(r["encoding"])) - seen
            if new:
                with_unseen += 1
                unseen |= new
        by_label[f"y{y}"] = {
            "n_rows": len(rows),
            "rows_with_unseen_tokens": with_unseen,
            "n_values": {
                str(k): v
                for k, v in sorted(Counter(int(r["n"]) for r in rows).items())
            },
            "distinct_graphs": len({r["edge_hash"] for r in rows}),
        }
    return {
        "train_n_values": sorted({int(r["n"]) for r in train_rows}),
        "unseen_tokens": sorted(
            unseen, key=lambda t: (not t.isdigit(), int(t) if t.isdigit() else 0, t)
        ),
        "by_label": by_label,
        "note": (
            "rows with unseen tokens feed untrained embeddings to every arm; "
            "label-dependent graph sizes let the size token separate classes. "
            "Results on this set mix hop length with vocabulary and size shift."
        ),
    }


def ff_id_train_note(vocab: Vocab) -> Optional[dict[str, Any]]:
    """Infer max_len of the scaffold FF ID-train run from its param count."""
    if not FF_ID_SUMMARY.exists():
        return None
    from reachability_gen.models.feedforward import FeedForward

    s = json.loads(FF_ID_SUMMARY.read_text())
    d, L = int(s["d"]), int(s["L"])
    non_emb = FeedForward(len(vocab), d=d, L=L, n_heads=_n_heads(d), max_len=1).non_embedding_param_count()
    implied = (int(s["param_count"]) - non_emb) // d - len(vocab)
    return {
        "artifact": FF_ID_SUMMARY.as_posix(),
        "param_count": int(s["param_count"]),
        "non_embedding_params": non_emb,
        "implied_max_len": implied,
        "val_acc_by_hop": {k: v["acc_mean"] for k, v in s["val_by_hop"].items()},
        "note": (
            "param_count = non-embedding + (vocab + max_len)·d implies max_len "
            f"= {implied}: the 64-token floor of the same first-rows rule, "
            "shorter than most n≥12 encodings, so most rows lost their QUERY "
            "tokens (consistent with 0.0 acc on every positive hop bucket)"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Re-score saved best checkpoints of the id_2k rematch runs with "
            "collapse diagnostics and truncation accounting (MEASURE; "
            "science_open=false)."
        )
    )
    p.add_argument("--data", type=Path, default=Path("data/id_2k.jsonl"))
    p.add_argument(
        "--runs", nargs="+", choices=sorted(RUNS), default=list(RUNS),
        help="Rematch runs whose saved checkpoints to audit",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--seed", type=int, default=0, help="Seed for perturbation-gain noise"
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch
    except ImportError:
        print("FAIL: torch required for checkpoint audit", file=sys.stderr)
        return 2
    if not args.data.exists():
        print(f"FAIL: missing {args.data} (do not regenerate)", file=sys.stderr)
        return 1

    t0 = time.perf_counter()
    rows = load_jsonl(args.data)
    ok, issues = verify_id_2k(rows)
    if not ok:
        print(f"FAIL: id_2k failed verify: {issues}", file=sys.stderr)
        return 1
    train, val = _split_train_val(rows)
    vocab = build_vocab()
    max_len = runner_max_len(rows, vocab)
    dataset = dataset_report(rows)
    truncation = truncation_report(rows, vocab, max_len)
    eval_reviews: dict[str, Optional[dict[str, Any]]] = {}
    for name, path in EVAL_SETS.items():
        if not path.exists():
            eval_reviews[name] = None
            continue
        review = eval_set_review(train, load_jsonl(path))
        review["data_path"] = path.as_posix()
        eval_reviews[name] = review
        pos, neg = review["by_label"]["y1"], review["by_label"]["y0"]
        print(
            f"[{name}] {len(review['unseen_tokens'])} tokens unseen in train; "
            f"unseen-token rows y1 {pos['rows_with_unseen_tokens']}/{pos['n_rows']}, "
            f"y0 {neg['rows_with_unseen_tokens']}/{neg['n_rows']}; distinct graphs "
            f"y1 {pos['distinct_graphs']}, y0 {neg['distinct_graphs']}",
            file=sys.stderr,
        )
    print(
        f"[dataset] {dataset['distinct_graphs']} distinct graphs; "
        f"{dataset['val_graphs_seen_in_train']}/{dataset['val_graphs']} val graphs "
        "also in train; query-blind baseline val_acc="
        f"{dataset['query_blind_baseline']['val_acc']:.4f}",
        file=sys.stderr,
    )
    print(
        f"[truncation] max_len={max_len} < longest={truncation['max_token_len']}: "
        f"{truncation['n_truncated']}/{truncation['n_rows']} rows lost QUERY tokens",
        file=sys.stderr,
    )

    runs_out: dict[str, Any] = {}
    mismatches: list[str] = []
    for run in args.runs:
        run_data = json.loads(RUNS[run].read_text())
        alpha = float(
            (run_data.get("drift_audit") or {}).get("residual_alpha", DEFAULT_RESIDUAL_ALPHA)
        )
        arms: dict[str, Any] = {}
        for arm in ARMS:
            a = audit_arm(arm, run_data[arm], val, vocab, residual_alpha=alpha, seed=args.seed)
            if not a["rescore_matches_record"]:
                mismatches.append(f"{run}/{arm}")
            arms[arm] = a
            s = a["summary"]
            fired = [
                k
                for flags in (a["checkpoint_flags"], a["run_flags"])
                for k, v in flags.items()
                if isinstance(v, dict) and v["flag"]
            ]
            print(
                f"[{run}/{arm}] ckpt ep{a['checkpoint_epoch']}: "
                f"recorded={a['recorded_best_val_acc']:.4f} "
                f"rescored={a['rescored_val_acc']:.4f} "
                f"({'match' if a['rescore_matches_record'] else 'MISMATCH'}) "
                f"answerable={s['val_acc_answerable']:.4f} "
                f"last_epoch={a['reported_last_epoch']['val_acc']:.4f} "
                f"token_cos_last={s['token_cos_by_t'][-1]:.3f} "
                f"flags={fired or 'none'}",
                file=sys.stderr,
            )
        runs_out[run] = {
            "run_json": RUNS[run].as_posix(),
            "residual_alpha": alpha,
            "arms": arms,
            "science_open": False,
        }

    artifact = {
        "science_open": False,
        "data_path": args.data.as_posix(),
        "purpose": (
            "Re-score each saved best checkpoint so every number describes one "
            "model; add collapse diagnostics and truncation accounting."
        ),
        "eval_protocol": (
            "runner val loop: batches of 1 padded to the checkpoint's max_len; "
            "truncation reproduced (on_overflow='allow') and reported"
        ),
        "torch_version": torch.__version__,
        "noise_seed": args.seed,
        "eps_sigma": EPS_SIGMA,
        "thresholds": {
            "token_collapse_cos": TOKEN_COLLAPSE_COS,
            "output_concentration_ratio": OUTPUT_CONCENTRATION_RATIO,
            "train_acc_drop": TRAIN_ACC_DROP,
            "last_epoch_below_best": FINAL_VS_BEST_GAP,
        },
        "metric_definitions": {
            "token_cos_by_t": "mean pairwise cosine between distinct real tokens of z_t (FF: residual stream after embeddings / each block)",
            "pooled_ratio_by_t": "||mean_i z_t,i|| / mean_i ||z_t,i|| (1 iff all tokens parallel)",
            "token_norm_by_t": "mean over real tokens of ||z_t,i||_2",
            "mean_z_norm_by_t": "runner metric: ||mean-pooled z_t||_2",
            "perturb_gain_by_t": "||Δz_t||_F / ||Δz_0||_F over real tokens for eps_sigma·N(0,1) added to c = tok_emb + pos_emb",
            "perturb_gain_rel_by_t": "same with each ||Δz_t|| divided by ||z_t|| (removes state-norm growth)",
            "acc_answerable": "accuracy excluding rows whose encoding exceeds max_len",
            "margin": "logit[y=1] - logit[y=0]",
        },
        "dataset": dataset,
        "truncation": truncation,
        "eval_set_reviews": eval_reviews,
        "other_affected_artifacts": [n for n in [ff_id_train_note(vocab)] if n],
        "runs": runs_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "ok": not mismatches,
                "out": args.out.as_posix(),
                "mismatches": mismatches,
                "elapsed_s": time.perf_counter() - t0,
                "science_open": False,
            },
            sort_keys=True,
        )
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())

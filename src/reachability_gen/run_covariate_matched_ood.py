# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Covariate-matched OOD eval on bound30 best ckpts (MEASURE; inference-only).

Zero-retrain Gate2 harness pointed at ``data/covariate_matched_ood.jsonl``
(seq_len band [45,70]; hop K∈{8,12,16} + hard negs). Isolates path length
from the Gate2 seq_len confound.

Eval modes:
  1. Fixed T=6 / L=2 all arms
  2. Dynamic T∈{8,12,16} Geo+Loop

Writes ``artifacts/id_2k_rematch_bound30_gate2_matched_ood.json`` with
``science_open: false`` always. Hard-neg collapse remains the primary
STOP/OPEN-candidate prereg rule.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from reachability_gen.adr_invariants import HOP_UNREACHABLE, OOD_HOP_VALUES
from reachability_gen.gen_covariate_matched_ood import (
    BOUND30_MAX_LEN,
    OOD_HOPS,
    SEQ_LEN_MAX,
    SEQ_LEN_MIN,
    verify_covariate_matched_ood,
)
from reachability_gen.overfit_ff import load_jsonl
from reachability_gen.run_ood_gate2 import (
    BOUND30_L,
    BOUND30_T_TRAIN,
    DYNAMIC_T_VALUES,
    DEFAULT_FF_CKPT,
    DEFAULT_GEO_CKPT,
    DEFAULT_ID_DATA,
    DEFAULT_LOOP_CKPT,
    DEFAULT_SUMMARY,
    _build_ff,
    _build_geo,
    _build_loop,
    _eval_ff,
    _eval_recurrent,
    _id_val_sanity,
    _interpret,
    _load_bound30_hparams,
    _load_ckpt,
)
from reachability_gen.tokenize import split_encoding_tokens

DEFAULT_MATCHED_DATA = Path("data/covariate_matched_ood.jsonl")
DEFAULT_GEN_REPORT = Path("artifacts/covariate_matched_ood_generation_report.json")
DEFAULT_OUT = Path("artifacts/id_2k_rematch_bound30_gate2_matched_ood.json")


def _dataset_stats(
    rows: list[dict[str, Any]],
    *,
    gen_report_path: Optional[Path] = None,
) -> dict[str, Any]:
    hop_counts = Counter(int(r["hop_distance"]) for r in rows)
    y_counts = Counter(int(r["y"]) for r in rows)
    n_vals = [int(r["n"]) for r in rows]
    p_vals = [float(r["p"]) for r in rows]
    token_lens = [
        len(split_encoding_tokens(str(r.get("encoding", "")))) for r in rows
    ]
    whitespace_lens = [len(str(r.get("encoding", "")).split()) for r in rows]
    out_of_band = sum(
        1 for tl in token_lens if not (SEQ_LEN_MIN <= tl <= SEQ_LEN_MAX)
    )
    stats: dict[str, Any] = {
        "path": str(DEFAULT_MATCHED_DATA),
        "n_total": len(rows),
        "y_counts": {str(k): v for k, v in sorted(y_counts.items())},
        "hop_counts": {str(k): v for k, v in sorted(hop_counts.items())},
        "n_range": [min(n_vals), max(n_vals)] if n_vals else None,
        "p_range": [min(p_vals), max(p_vals)] if p_vals else None,
        "ood_hops": list(OOD_HOPS),
        "seq_len_band": [SEQ_LEN_MIN, SEQ_LEN_MAX],
        "token_len": {
            "min": min(token_lens) if token_lens else None,
            "max": max(token_lens) if token_lens else None,
            "mean": (sum(token_lens) / len(token_lens)) if token_lens else None,
            "out_of_band": out_of_band,
            "measure": "split_encoding_tokens",
        },
        "whitespace_len": {
            "min": min(whitespace_lens) if whitespace_lens else None,
            "max": max(whitespace_lens) if whitespace_lens else None,
            "mean": (
                (sum(whitespace_lens) / len(whitespace_lens))
                if whitespace_lens
                else None
            ),
        },
        "verify_ok": True,
        "covariate_matched": True,
        "science_open": False,
    }
    if gen_report_path is not None and gen_report_path.exists():
        gen = json.loads(gen_report_path.read_text(encoding="utf-8"))
        stats["generation_report"] = str(gen_report_path)
        stats["construction"] = gen.get("construction")
        stats["id_band_reference"] = gen.get("id_band_reference")
        stats["quotas"] = gen.get("quotas")
        stats["seed"] = gen.get("seed")
        stats["n_support_by_hop"] = gen.get("n_support_by_hop")
        stats["extra_edges_by_hop"] = gen.get("extra_edges_by_hop")
    return stats


def run_matched_ood(
    *,
    matched_data: Path = DEFAULT_MATCHED_DATA,
    id_data: Path = DEFAULT_ID_DATA,
    ff_ckpt: Path = DEFAULT_FF_CKPT,
    geo_ckpt: Path = DEFAULT_GEO_CKPT,
    loop_ckpt: Path = DEFAULT_LOOP_CKPT,
    summary_path: Path = DEFAULT_SUMMARY,
    gen_report: Path = DEFAULT_GEN_REPORT,
    out_path: Path = DEFAULT_OUT,
    max_len: int = BOUND30_MAX_LEN,
    skip_id_sanity: bool = False,
) -> dict[str, Any]:
    """Run covariate-matched Gate-2-style OOD eval; return artifact dict."""
    from reachability_gen.tokenize import build_vocab
    from reachability_gen.train.ff_trainer import examples_to_batch

    if not matched_data.exists():
        raise FileNotFoundError(
            f"matched OOD dataset missing: {matched_data}. "
            "Run: python -m reachability_gen.gen_covariate_matched_ood"
        )
    for p in (ff_ckpt, geo_ckpt, loop_ckpt):
        if not p.exists():
            raise FileNotFoundError(f"checkpoint missing: {p}")

    rows = load_jsonl(matched_data)
    ok, issues = verify_covariate_matched_ood(rows)
    if not ok:
        raise RuntimeError(f"covariate_matched_ood verify failed: {issues}")

    hparams = _load_bound30_hparams(summary_path)
    vocab = build_vocab()
    _ = examples_to_batch(rows[:1], vocab, max_len=max_len)

    ff_model = _build_ff(len(vocab), vocab.pad_id, max_len, hparams)
    geo_model = _build_geo(len(vocab), vocab.pad_id, max_len, hparams)
    loop_model = _build_loop(len(vocab), vocab.pad_id, max_len, hparams)

    ff_meta = _load_ckpt(ff_model, ff_ckpt)
    geo_meta = _load_ckpt(geo_model, geo_ckpt)
    loop_meta = _load_ckpt(loop_model, loop_ckpt)

    ff_pos = int(ff_model.pos_emb.weight.shape[0])
    if ff_pos != max_len:
        print(
            f"WARN: ff pos_emb={ff_pos} != requested max_len={max_len}; "
            f"using checkpoint capacity {ff_pos}",
            file=sys.stderr,
        )
        max_len = ff_pos

    ff_pc = sum(p.numel() for p in ff_model.parameters() if p.requires_grad)
    geo_pc = sum(p.numel() for p in geo_model.parameters() if p.requires_grad)
    loop_pc = sum(p.numel() for p in loop_model.parameters() if p.requires_grad)

    run_id = f"gate2-matched-ood-{uuid.uuid4().hex[:10]}"

    print(
        f"[matched-ood] n={len(rows)}; band=[{SEQ_LEN_MIN},{SEQ_LEN_MAX}]; "
        f"fixed T={hparams['T']}/L={hparams['L']}; "
        f"dynamic T={list(DYNAMIC_T_VALUES)}; max_len={max_len}",
        file=sys.stderr,
    )

    ff_fixed = _eval_ff(
        ff_model,
        rows,
        vocab=vocab,
        max_len=max_len,
        run_id=f"{run_id}-ff-fixed",
        arm_name=f"ff-L{hparams['L']}-d{hparams['d']}",
        param_count=ff_pc,
        d=int(hparams["d"]),
        L=int(hparams["L"]),
    )
    print(
        f"[matched-ood] FF L={hparams['L']} "
        f"overall_acc={ff_fixed['overall_acc']:.4f}",
        file=sys.stderr,
    )

    geo_fixed = _eval_recurrent(
        geo_model,
        rows,
        vocab=vocab,
        max_len=max_len,
        T=int(hparams["T"]),
        run_id=f"{run_id}-geo-T{hparams['T']}",
        arm_name=(
            f"geo-T{hparams['T']}-d{hparams['d']}-mlp{hparams['mlp_expansion']}-tau"
        ),
        param_count=geo_pc,
        d=int(hparams["d"]),
        use_tau=True,
    )
    print(
        f"[matched-ood] Geo T={hparams['T']} "
        f"overall_acc={geo_fixed['overall_acc']:.4f}",
        file=sys.stderr,
    )

    loop_fixed = _eval_recurrent(
        loop_model,
        rows,
        vocab=vocab,
        max_len=max_len,
        T=int(hparams["T"]),
        run_id=f"{run_id}-loop-T{hparams['T']}",
        arm_name=(
            f"loop-T{hparams['T']}-d{hparams['d']}-mlp{hparams['mlp_expansion']}"
        ),
        param_count=loop_pc,
        d=int(hparams["d"]),
        use_tau=False,
    )
    print(
        f"[matched-ood] Loop T={hparams['T']} "
        f"overall_acc={loop_fixed['overall_acc']:.4f}",
        file=sys.stderr,
    )

    fixed = {"ff": ff_fixed, "geo": geo_fixed, "loop": loop_fixed}

    geo_dyn: dict[str, Any] = {}
    loop_dyn: dict[str, Any] = {}
    for T in DYNAMIC_T_VALUES:
        geo_dyn[str(T)] = _eval_recurrent(
            geo_model,
            rows,
            vocab=vocab,
            max_len=max_len,
            T=T,
            run_id=f"{run_id}-geo-T{T}",
            arm_name=(
                f"geo-T{T}-d{hparams['d']}-mlp{hparams['mlp_expansion']}-tau"
            ),
            param_count=geo_pc,
            d=int(hparams["d"]),
            use_tau=True,
        )
        print(
            f"[matched-ood] Geo T={T} "
            f"overall_acc={geo_dyn[str(T)]['overall_acc']:.4f}",
            file=sys.stderr,
        )
        loop_dyn[str(T)] = _eval_recurrent(
            loop_model,
            rows,
            vocab=vocab,
            max_len=max_len,
            T=T,
            run_id=f"{run_id}-loop-T{T}",
            arm_name=(
                f"loop-T{T}-d{hparams['d']}-mlp{hparams['mlp_expansion']}"
            ),
            param_count=loop_pc,
            d=int(hparams["d"]),
            use_tau=False,
        )
        print(
            f"[matched-ood] Loop T={T} "
            f"overall_acc={loop_dyn[str(T)]['overall_acc']:.4f}",
            file=sys.stderr,
        )

    dynamic = {
        "geo": geo_dyn,
        "loop": loop_dyn,
        "T_values": list(DYNAMIC_T_VALUES),
    }

    def _matrix(arm_dyn: dict[str, Any]) -> dict[str, Any]:
        mat: dict[str, dict[str, float]] = {}
        for hop in list(OOD_HOPS) + [HOP_UNREACHABLE]:
            row: dict[str, float] = {}
            for T in DYNAMIC_T_VALUES:
                cell = arm_dyn[str(T)]["by_hop"].get(str(hop), {})
                row[str(T)] = float(cell.get("acc_mean", float("nan")))
            mat[str(hop)] = row
        return mat

    hop_T_acc_geo = _matrix(geo_dyn)
    hop_T_acc_loop = _matrix(loop_dyn)

    id_sanity: dict[str, Any] = {"skipped": True, "science_open": False}
    if not skip_id_sanity and id_data.exists():
        id_rows = load_jsonl(id_data)
        id_sanity = _id_val_sanity(
            ff_model=ff_model,
            geo_model=geo_model,
            loop_model=loop_model,
            id_rows=id_rows,
            vocab=vocab,
            max_len=max_len,
            hparams=hparams,
        )
        print(
            f"[matched-ood] ID val sanity: "
            f"FF={id_sanity.get('ff_overall_acc')} "
            f"Geo={id_sanity.get('geo_overall_acc')} "
            f"Loop={id_sanity.get('loop_overall_acc')}",
            file=sys.stderr,
        )

    interpretation = _interpret(fixed, dynamic)
    # Annotate that this is the covariate-matched variant of Gate2 STOP rules.
    interpretation = dict(interpretation)
    interpretation["variant"] = "covariate_matched_ood"
    interpretation["prior_gate2_stop"] = (
        "Prior Gate2 STOP (ood_hops, seq_len mean~203) confounded hop with "
        "length; this run holds seq_len in ID band [45,70]."
    )
    # Evidence note: matched seq_len *inverts* the prior Gate2 failure mode.
    snap = interpretation.get("metrics_snapshot", {})
    ff_neg = float(snap.get("ff_hardneg_acc", float("nan")))
    geo_neg = float(snap.get("geo_T6_hardneg_acc", float("nan")))
    loop_neg = float(snap.get("loop_T6_hardneg_acc", float("nan")))
    ff_pos = float(snap.get("ff_ood_pos_acc_mean", float("nan")))
    geo_pos = float(snap.get("geo_T6_ood_pos_acc_mean", float("nan")))
    loop_pos = float(snap.get("loop_T6_ood_pos_acc_mean", float("nan")))
    inverted = (
        ff_neg >= 0.5
        and loop_neg >= 0.5
        and ff_pos < 0.55
        and loop_pos < 0.55
    )
    interpretation["vs_prior_gate2"] = {
        "prior_pattern": (
            "hard-neg acc≈0 all arms; OOD positives high (incl. FF); "
            "overall~chance → STOP (all-positive bias under long seq_len)"
        ),
        "matched_pattern": (
            f"hard-neg FF={ff_neg:.3f}/Geo={geo_neg:.3f}/Loop={loop_neg:.3f}; "
            f"OOD-pos mean FF={ff_pos:.3f}/Geo={geo_pos:.3f}/Loop={loop_pos:.3f}"
        ),
        "inverted_failure_mode": inverted,
        "fail_closed_one_liner_vs_prior": (
            "STOP still: matched-seq_len OOD inverts Gate2 (hard-neg holds, "
            "long-hop positives collapse for FF/Loop; Geo weak) — length-gen "
            "not licensed; prior all-positive bias was seq_len-confounded; "
            "science_open=false"
            if inverted
            else (
                "Document tables vs prior Gate2 STOP; hard-neg collapse still "
                "primary; science_open=false"
            )
        ),
        "science_open": False,
    }
    # Prefer the matched vs-prior one-liner when the inverted pattern holds,
    # while keeping the raw prereg framing from shared _interpret.
    if inverted:
        interpretation["fail_closed_one_liner"] = interpretation[
            "vs_prior_gate2"
        ]["fail_closed_one_liner_vs_prior"]
        interpretation["framing"] = "STOP_INVERTED_VS_PRIOR_GATE2"
    interpretation["science_open"] = False

    dataset = _dataset_stats(rows, gen_report_path=gen_report)
    dataset["path"] = str(matched_data)

    artifact: dict[str, Any] = {
        "gate": "gate2_covariate_matched_ood",
        "cycle": "CYCLE_COVARIATE_MATCHED_OOD",
        "mode": "MEASURE",
        "science_open": False,
        "residue_of": "SESSION-SEAL-GATE2 (ccedd28)",
        "dataset": dataset,
        "checkpoints": {
            "ff": ff_meta,
            "geo": geo_meta,
            "loop": loop_meta,
        },
        "hparams": {
            **hparams,
            "note": (
                "Exact bound30 arch: RMSNorm on, residual α=0.5, "
                "Geo τ on, Loop τ off; frozen best ckpts (no retrain)."
            ),
        },
        "param_counts": {"ff": ff_pc, "geo": geo_pc, "loop": loop_pc},
        "max_len": max_len,
        "id_val_sanity": id_sanity,
        "fixed_depth": {
            "note": f"FF at trained L={BOUND30_L}; Geo/Loop at trained T={BOUND30_T_TRAIN}",
            "ff": ff_fixed,
            "geo": geo_fixed,
            "loop": loop_fixed,
        },
        "dynamic_unroll": {
            "note": "Geo/Loop only; T in {8,12,16}; test extra test-time cycles",
            "T_values": list(DYNAMIC_T_VALUES),
            "geo": geo_dyn,
            "loop": loop_dyn,
            "hop_T_acc_matrix_geo": hop_T_acc_geo,
            "hop_T_acc_matrix_loop": hop_T_acc_loop,
        },
        "prereg": interpretation,
        "interpretation_stop": interpretation,
        "comparison_to_prior_gate2": {
            "prior_artifact": "artifacts/id_2k_rematch_bound30_gate2_ood.json",
            "prior_seq_len_mean": 203.225,
            "matched_seq_len_band": [SEQ_LEN_MIN, SEQ_LEN_MAX],
            "matched_seq_len_mean": dataset["token_len"]["mean"],
            "note": (
                "Hard-neg collapse still primary STOP rule; positives alone "
                "do not license OPEN. science_open=false."
            ),
            "science_open": False,
        },
    }

    def _assert_no_open(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            if obj.get("science_open") is True:
                raise AssertionError(f"science_open=True at {path}")
            for k, v in obj.items():
                _assert_no_open(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _assert_no_open(v, f"{path}[{i}]")

    _assert_no_open(artifact)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"[matched-ood] wrote {out_path}", file=sys.stderr)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Covariate-matched OOD eval on bound30 best ckpts "
            "(MEASURE; inference-only; science_open=false)."
        )
    )
    p.add_argument("--matched-data", type=Path, default=DEFAULT_MATCHED_DATA)
    p.add_argument("--id-data", type=Path, default=DEFAULT_ID_DATA)
    p.add_argument("--ff-ckpt", type=Path, default=DEFAULT_FF_CKPT)
    p.add_argument("--geo-ckpt", type=Path, default=DEFAULT_GEO_CKPT)
    p.add_argument("--loop-ckpt", type=Path, default=DEFAULT_LOOP_CKPT)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--gen-report", type=Path, default=DEFAULT_GEN_REPORT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-len", type=int, default=BOUND30_MAX_LEN)
    p.add_argument("--skip-id-sanity", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        artifact = run_matched_ood(
            matched_data=args.matched_data,
            id_data=args.id_data,
            ff_ckpt=args.ff_ckpt,
            geo_ckpt=args.geo_ckpt,
            loop_ckpt=args.loop_ckpt,
            summary_path=args.summary,
            gen_report=args.gen_report,
            out_path=args.out,
            max_len=args.max_len,
            skip_id_sanity=args.skip_id_sanity,
        )
    except Exception as exc:
        print(f"matched-ood FAILED: {exc}", file=sys.stderr)
        return 1
    framing = artifact.get("prereg", {}).get("framing", "?")
    one_liner = artifact.get("prereg", {}).get("fail_closed_one_liner", "")
    print(
        json.dumps(
            {
                "ok": True,
                "out": str(args.out),
                "framing": framing,
                "fail_closed": one_liner,
                "science_open": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

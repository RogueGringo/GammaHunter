# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Immutable runtime constants frozen by ADR-001 (metrics & compute).

Import these — do not redefine — from flops, param_match, harness, and generate.
Changing a value here is an ADR revision, not a silent tweak.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Param parity (non-embedding) among Geo / FF / Loop
# ---------------------------------------------------------------------------
PARAM_TOL: float = 0.05  # ±5%

# ---------------------------------------------------------------------------
# Hop stratification (shortest-path hop distance K)
# ---------------------------------------------------------------------------
K_TRAIN_MAX: int = 6
ID_HOP_MIN: int = 2
ID_HOP_MAX: int = 6  # inclusive; ID hops are K in [ID_HOP_MIN, ID_HOP_MAX]
OOD_HOP_VALUES: tuple[int, ...] = (8, 12, 16)
# Unreachable (y=0): sentinel hop distance written to JSONL. ADR also allows null.
HOP_UNREACHABLE: int = -1

# ---------------------------------------------------------------------------
# Perturbation diagnostic
# ---------------------------------------------------------------------------
EPS_SIGMA: float = 1e-4  # default sigma for perturbation_delta

# ---------------------------------------------------------------------------
# F_block schematic coefficients (mlp_expansion=4, MAC=2, no bias/norm FLOPs)
# F_block(m, d) = 24 * m * d^2 + 4 * m^2 * d
# ---------------------------------------------------------------------------
MLP_EXPANSION: int = 4
MAC_FLOPS: int = 2
F_BLOCK_COEFF_MD2: int = 24  # 8 + 4 * MLP_EXPANSION when expansion=4
F_BLOCK_COEFF_M2D: int = 4


def f_block_checksum(m: int, d: int) -> int:
    """Canonical ADR F_block formula — single source for runtime asserts."""
    return int(F_BLOCK_COEFF_MD2 * m * d * d + F_BLOCK_COEFF_M2D * m * m * d)


def is_ood_hop(hop_distance: int, *, k_train_max: int = K_TRAIN_MAX) -> bool:
    """True iff hop_distance is a defined positive hop exceeding the train max.

    Unreachable (HOP_UNREACHABLE / negative) is never marked hop-OOD.
    """
    if hop_distance < 0:
        return False
    return hop_distance > k_train_max


def assert_f_block_matches_implementation(impl_fn) -> None:
    """Raise AssertionError if ``impl_fn(m,d)`` disagrees with ADR checksum."""
    for m, d in ((1, 1), (8, 32), (16, 64), (48, 128)):
        expected = f_block_checksum(m, d)
        got = int(impl_fn(m, d))
        if got != expected:
            raise AssertionError(
                f"F_block ADR checksum mismatch at m={m}, d={d}: "
                f"impl={got} adr={expected}"
            )


# Encoding lock pointer (implementation lives in encode.py).
ENCODING_SCHEME: str = "canonical_sorted_edge_list_plus_QUERY"
ENCODING_MODULE: str = "reachability_gen.encode"

# Status flags — harness / checkers must never self-stamp science OPEN.
SCIENCE_OPEN_DEFAULT: bool = False


# ---------------------------------------------------------------------------
# RunMetricRecord required keys (user TypedDict — ADR-001 §7)
# ---------------------------------------------------------------------------
REQUIRED_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "run_id",
        "seed",
        "arm",
        "step",
        "epoch",
        "param_count",
        "d_model",
        "seq_len",
        "cycles_or_depth",
        "tokens_decoded",
        "cumulative_flops",
        "hop_distance",
        "is_ood",
        "loss",
        "accuracy",
        "drift_trajectory",
        "terminal_drift",
        "perturbation_delta",
    }
)

# Renamed / removed from the previous scaffold schema (must not appear as
# required keys; emitters must not write these names).
LEGACY_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "z_drift_series",
        "T_or_L_or_K",
        "params",
        "flops",
    }
)


def validate_metric_record(record: dict) -> None:
    """Raise ``ValueError`` if required ADR-001 metric keys are missing.

    Also rejects ``science_open=True`` when that optional hygiene key is present
    (harness / scaffolding must never self-stamp OPEN).
    """
    missing = REQUIRED_METRIC_KEYS - set(record.keys())
    if missing:
        raise ValueError(f"RunMetricRecord missing keys: {sorted(missing)}")
    legacy = LEGACY_METRIC_KEYS & set(record.keys())
    if legacy:
        raise ValueError(
            f"RunMetricRecord still uses legacy key names: {sorted(legacy)}"
        )
    if record.get("science_open") is True:
        raise ValueError(
            "science_open must not be self-stamped True by the harness (ADR-001)"
        )

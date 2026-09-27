# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Train / val / test / ood_size split specs with fixed seed tables.

Train/val/test share the same n support {8, 12, 16}.
ood_size uses larger n {24, 32}.
p support is shared: {0.15, 0.25, 0.35}.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

# Spec-locked size / density grids.
TRAIN_N: tuple[int, ...] = (8, 12, 16)
OOD_N: tuple[int, ...] = (24, 32)
P_VALUES: tuple[float, ...] = (0.15, 0.25, 0.35)

# Fixed master seeds per split (do not change casually — regenerability).
SEED_TABLE: Mapping[str, int] = {
    "train": 10_001,
    "val": 20_002,
    "test": 30_003,
    "ood_size": 40_004,
}

SPLITS: tuple[str, ...] = ("train", "val", "test", "ood_size")


@dataclass(frozen=True)
class SplitSpec:
    name: str
    seed: int
    n_values: tuple[int, ...]
    p_values: tuple[float, ...] = P_VALUES

    def n_grid(self) -> tuple[int, ...]:
        return self.n_values


def get_split_spec(name: str) -> SplitSpec:
    if name not in SEED_TABLE:
        raise KeyError(f"unknown split {name!r}; expected one of {list(SEED_TABLE)}")
    n_values = OOD_N if name == "ood_size" else TRAIN_N
    return SplitSpec(name=name, seed=SEED_TABLE[name], n_values=n_values)


def all_split_specs() -> list[SplitSpec]:
    return [get_split_spec(s) for s in SPLITS]


def derive_example_seed(split_seed: int, example_index: int) -> int:
    """Deterministic per-example seed from split master seed + index."""
    # Keep in 32-bit unsigned range for Random.seed portability.
    return (split_seed * 1_000_003 + example_index * 97) & 0xFFFFFFFF

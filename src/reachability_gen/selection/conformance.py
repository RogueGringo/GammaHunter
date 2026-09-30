# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""The conformance battery every selection kernel must pass.

A backend is any callable ``(z, mask) -> p`` over the last dimension. The
battery draws seeded rows (float64, about a quarter of positions masked, never a
whole row) and checks:

* ``simplex``: p ≥ 0 and each row sums to 1;
* ``mask_exact``: masked positions are exactly 0;
* ``shift_invariance``: adding a constant to a row leaves p unchanged;
* ``permutation_invariance``: permuting positions permutes p the same way;
* ``order_preservation``: z_i > z_j implies p_i ≥ p_j;
* ``sub_threshold_invariance`` (α > 1 only): appending positions far below the
  threshold leaves p unchanged and gives them 0;
* ``certificate`` (α > 1 only): every row passes the framework-free KKT check;
* ``gradient_agreement``: the backward pass matches float64 finite differences;
* ``zero_gradient_off_support`` (α > 1 only): the gradient is 0 wherever p is 0.

A kernel passes when every check that applies to its α passes.

The battery works in float64 on the CPU. The attention it serves runs in
float32 (on the GPU in stage B), so ``float32_agreement`` also compares each
reference backend's float32 output, on the CPU and on CUDA when present, with
its float64 output on the same rows: values within 1e-5, rows summing to 1
within 1e-5, masked positions exactly 0, and the same support except at
positions whose weight is at most 1e-5 in the other precision. It does so on
the battery's short rows and on rows of 2,048 positions, longer than any
stage-B rendering (the longest, a validation graph at 4 distractor sentences
per edge, has 1,764 tokens), with scores spread (standard deviation 2) and
nearly tied (standard deviation 1e-3, as attention scores are near
initialisation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import torch

from .certificate import CERTIFIERS
from .spec import NORMALIZERS

ATOL: float = 1e-9
FLOAT32_TOL: float = 1e-5


def _rows(seed: int, rows: int, length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Generator]:
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(rows, length, generator=gen, dtype=torch.float64) * 2
    mask = torch.rand(rows, length, generator=gen) > 0.25
    mask[:, 0] = True
    return z, mask, gen


def run_battery(fn: Callable[..., torch.Tensor], kind: str, *, seed: int = 0, rows: int = 64,
                length: int = 12) -> dict[str, Any]:
    """Every applicable check for backend ``fn`` implementing normalizer ``kind``; returns a report."""
    spec = NORMALIZERS[kind]
    z, mask, gen = _rows(seed, rows, length)
    with torch.no_grad():
        p = fn(z, mask)
        checks: dict[str, bool] = {
            "simplex": bool((p >= 0).all()) and bool(torch.allclose(p.sum(-1), torch.ones(rows, dtype=p.dtype),
                                                                     atol=ATOL)),
            "mask_exact": bool((p[~mask] == 0).all()),
            "shift_invariance": bool(torch.allclose(fn(z + 3.7, mask), p, atol=ATOL)),
        }
        perm = torch.randperm(length, generator=gen)
        checks["permutation_invariance"] = bool(torch.allclose(fn(z[:, perm], mask[:, perm]), p[:, perm], atol=ATOL))
        zi, zj = z[:, :, None], z[:, None, :]
        both = mask[:, :, None] & mask[:, None, :]
        checks["order_preservation"] = bool(((p[:, :, None] >= p[:, None, :] - ATOL) | ~(zi > zj) | ~both).all())
        if spec.sparse:
            low = z.masked_fill(~mask, float("inf")).min(-1, keepdim=True).values - 10.0  # far below the threshold
            wide = fn(torch.cat([z, low.expand(rows, 5)], -1), torch.cat([mask, torch.ones(rows, 5, dtype=torch.bool)], -1))
            checks["sub_threshold_invariance"] = bool(torch.allclose(wide[:, :length], p, atol=ATOL)) and bool(
                (wide[:, length:] == 0).all())
            certify = CERTIFIERS[kind]
            certs = [certify(z[r].tolist(), p[r].tolist(), mask[r].tolist()) for r in range(rows)]
            checks["certificate"] = all(c.passed for c in certs)
    small = z[:8].clone().requires_grad_(True)
    small_mask = mask[:8]
    try:
        checks["gradient_agreement"] = bool(torch.autograd.gradcheck(
            lambda t: fn(t, small_mask), (small,), eps=1e-6, atol=1e-5, rtol=1e-4, raise_exception=False))
    except RuntimeError:
        checks["gradient_agreement"] = False
    if spec.sparse:
        zg = z.clone().requires_grad_(True)
        out = fn(zg, mask)
        weights = torch.randn(out.shape, generator=gen, dtype=out.dtype)
        (out * weights).sum().backward()
        dead = (out.detach() == 0)
        checks["zero_gradient_off_support"] = bool((zg.grad[dead] == 0).all())
    return {"kind": kind, "alpha": spec.alpha, "checks": checks, "passed": all(checks.values()),
            "rows": rows, "length": length, "seed": seed}


FLOAT32_CASES: tuple[tuple[str, int, int, float], ...] = (
    ("short", 64, 12, 2.0), ("long_spread", 8, 2048, 2.0), ("long_near_tie", 8, 2048, 1e-3))


def float32_agreement(fn: Callable[..., torch.Tensor], *, device: str = "cpu") -> dict[str, Any]:
    """Float32 output on ``device`` against float64 on the CPU, for every case of ``FLOAT32_CASES``."""
    cases = {name: _float32_case(fn, device=device, seed=i + 1, rows=rows, length=length, scale=scale)
             for i, (name, rows, length, scale) in enumerate(FLOAT32_CASES)}
    return {"device": device, "cases": cases, "passed": all(c["passed"] for c in cases.values())}


def _float32_case(fn: Callable[..., torch.Tensor], *, device: str, seed: int, rows: int, length: int,
                  scale: float) -> dict[str, Any]:
    z, mask, _ = _rows(seed, rows, length)
    z = z / 2 * scale  # _rows draws with standard deviation 2
    with torch.no_grad():
        p64 = fn(z, mask)
        p32 = fn(z.float().to(device), mask.to(device)).double().cpu()
    checks = {
        "values": bool((p32 - p64).abs().max() <= FLOAT32_TOL),
        "rows_sum_to_one": bool(((p32.sum(-1) - 1).abs() <= FLOAT32_TOL).all()),
        "mask_exact": bool((p32[~mask] == 0).all()),
        "support": bool((((p32 > 0) == (p64 > 0)) | ((p32 <= FLOAT32_TOL) & (p64 <= FLOAT32_TOL))).all()),
    }
    return {"rows": rows, "length": length, "score_sd": scale, "checks": checks, "passed": all(checks.values()),
            "max_abs_difference": float((p32 - p64).abs().max())}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the battery on every reference backend and on the broken one; fail unless each lands as expected."""
    from .reference import BACKENDS, broken_sparsemax

    p = argparse.ArgumentParser(description="Conformance battery for the selection normalizers.")
    p.add_argument("--out", type=Path, default=Path("artifacts/selection_conformance.json"))
    args = p.parse_args(argv)
    reports = {kind: run_battery(fn, kind) for kind, fn in BACKENDS.items()}
    broken = run_battery(broken_sparsemax, "sparsemax")
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    float32 = {kind: {d: float32_agreement(fn, device=d) for d in devices} for kind, fn in BACKENDS.items()}
    ok = (all(r["passed"] for r in reports.values()) and not broken["passed"]
          and all(r["passed"] for per in float32.values() for r in per.values()))
    result = {"science_open": False, "reference_backends": reports, "broken_backend": broken,
              "float32_agreement": float32, "torch_version": torch.__version__,
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "expected": "every reference backend passes, in float64 and in float32 agreement, and the broken "
                          "backend fails", "ok": ok}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": ok, "out": args.out.as_posix(), "science_open": False}, sort_keys=True))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

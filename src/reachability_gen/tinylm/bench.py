# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Training throughput of ``TinyLM``: full optimiser steps timed on the device, not one forward pass.

Each measured step is a forward pass, the loss, a backward pass, gradient
clipping and an AdamW step, on random byte sequences, after warm-up steps that
are not timed. Reported per configuration: parameters, milliseconds per step,
training tokens per second and peak device memory.

Usage::

    python -m reachability_gen.tinylm.bench --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Optional, Sequence

import torch

from .model import TinyConfig, TinyLM
from .tokenizer import VOCAB_SIZE

CONFIGS: dict[str, TinyConfig] = {
    "xs": TinyConfig(d=128, layers=4, heads=4, mlp_hidden=320),
    "s": TinyConfig(d=256, layers=6, heads=8, mlp_hidden=704),
    "m": TinyConfig(d=384, layers=8, heads=6, mlp_hidden=1024),
    "l": TinyConfig(d=512, layers=12, heads=8, mlp_hidden=1408),
}


def throughput(cfg: TinyConfig, *, batch: int, seq: int, steps: int, warmup: int, device: str,
               bf16: bool = True, seed: int = 0) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = TinyLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    data = torch.randint(0, 256, (batch, seq + 1), device=device)
    use_amp = bf16 and device == "cuda"

    def step() -> float:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = model.loss(data)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return loss.item()

    for _ in range(warmup):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    seconds = (time.perf_counter() - t0) / steps
    return {"params": model.num_params(), "batch": batch, "seq": seq, "bf16": use_amp,
            "ms_per_step": 1000 * seconds, "tokens_per_second": batch * seq / seconds,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None}


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="TinyLM training throughput (full optimiser steps).")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = p.parse_args(argv)
    out = {}
    for name in args.configs:
        out[name] = {"config": CONFIGS[name].as_dict(), **throughput(
            CONFIGS[name], batch=args.batch, seq=args.seq, steps=args.steps, warmup=args.warmup, device=args.device)}
        r = out[name]
        print(f"[{name}] {r['params'] / 1e6:.2f}M params: {r['ms_per_step']:.1f} ms/step, "
              f"{r['tokens_per_second'] / 1e3:.0f}k tokens/s"
              + (f", peak {r['peak_memory_gib']:.2f} GiB" if r["peak_memory_gib"] is not None else ""),
              file=sys.stderr, flush=True)
    print(json.dumps({"vocab_size": VOCAB_SIZE, "results": out}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

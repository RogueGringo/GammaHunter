# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Reading one sentence into its edge with the byte LM: the prompt is the sentence, the answer "u>v" or "none".

An example is BOS + sentence + SEP + answer + EOS; the loss counts only the
answer bytes and EOS, so the model learns to read, not to recite sentences.
Decoding is greedy and batched over prompts of equal length (no padding ever
enters attention), for at most ``MAX_ANSWER`` + 1 tokens; the answer is the
bytes before the first EOS. A reply with no EOS in that budget, or with any
other special token before it, is malformed and never matches a valid answer,
so scoring is exact. Wrong answers are sorted into reversed edges, "none",
malformed replies and other answers.
"""

from __future__ import annotations

import math
import random
import re
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .grammar import NONE, sample
from .model import TinyLM
from .tokenizer import BOS, EOS, PAD, SEP, decode, encode

MAX_ANSWER: int = 8  # "47>12" is 5 bytes, "none" 4
MALFORMED = "<malformed>"
EDGE = re.compile(r"(\d+)>(\d+)")


def prompt_ids(sentence: str) -> list[int]:
    return [BOS] + encode(sentence, bos=False, eos=False) + [SEP]


def example_ids(sentence: str, ans: str) -> tuple[list[int], int]:
    """The example's ids and the index of its first answer byte."""
    prompt = prompt_ids(sentence)
    return prompt + encode(ans, bos=False, eos=False) + [EOS], len(prompt)


def collate(examples: Sequence[tuple[str, str]], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-padded ids ``[B, L]`` and a ``[B, L-1]`` mask over the targets that are answer bytes or EOS."""
    rows = [example_ids(s, a) for s, a in examples]
    length = max(len(ids) for ids, _ in rows)
    ids = torch.full((len(rows), length), PAD, dtype=torch.long)
    mask = torch.zeros((len(rows), length - 1), dtype=torch.bool)
    for b, (row, start) in enumerate(rows):
        ids[b, : len(row)] = torch.tensor(row)
        mask[b, start - 1 : len(row) - 1] = True  # the logits at t predict token t + 1
    return ids.to(device), mask.to(device)


def answer_loss(model: TinyLM, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = model(ids[:, :-1]).float()
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), reduction="none")
    return (losses * mask.reshape(-1)).sum() / mask.sum()


@torch.no_grad()
def greedy_answers(model: TinyLM, sentences: Sequence[str], device: str, batch: int = 512) -> list[str]:
    """The model's answer to each sentence (greedy; prompts batched by length, so no padding is attended)."""
    was = model.training
    model.eval()
    prompts = [prompt_ids(s) for s in sentences]
    out: list[str] = [""] * len(prompts)
    by_length: dict[int, list[int]] = {}
    for i, p in enumerate(prompts):
        by_length.setdefault(len(p), []).append(i)
    for idx in by_length.values():
        for start in range(0, len(idx), batch):
            group = idx[start : start + batch]
            ids = torch.tensor([prompts[i] for i in group], device=device)
            done = torch.zeros(len(group), dtype=torch.bool, device=device)
            generated = []
            for _ in range(MAX_ANSWER + 1):
                nxt = model(ids)[:, -1].float().argmax(-1)
                nxt = torch.where(done, torch.full_like(nxt, EOS), nxt)
                generated.append(nxt)
                done |= nxt == EOS
                ids = torch.cat([ids, nxt[:, None]], dim=1)
                if bool(done.all()):
                    break
            gen = torch.stack(generated, dim=1).tolist()
            for row, i in zip(gen, group):
                if EOS not in row or any(t >= 256 for t in row[: row.index(EOS)]):
                    out[i] = MALFORMED  # no EOS in the budget, or a special token before it
                else:
                    out[i] = decode(row[: row.index(EOS)])
    model.train(was)
    return out


def classify(got: str, gold: str) -> str:
    """"correct", or the kind of error: reversed edge, "none", malformed reply, or another answer."""
    if got == gold:
        return "correct"
    if got == MALFORMED or (got != NONE and not EDGE.fullmatch(got)):
        return "malformed"
    if got == NONE:
        return "none"
    m, g = EDGE.fullmatch(got), EDGE.fullmatch(gold)
    if m and g and (m.group(1), m.group(2)) == (g.group(2), g.group(1)):
        return "reversed"
    return "other"


def score(model: TinyLM, rows: Sequence[tuple[str, str]], device: str) -> dict[str, Any]:
    """Exact-match accuracy over ``rows`` and the count of each kind of error."""
    got = greedy_answers(model, [s for s, _ in rows], device)
    kinds = [classify(g, a) for g, (_, a) in zip(got, rows)]
    errors = {k: kinds.count(k) for k in ("reversed", "none", "malformed", "other")}
    return {"accuracy": kinds.count("correct") / len(rows), "errors": errors}


def accuracy(model: TinyLM, rows: Sequence[tuple[str, str]], device: str) -> float:
    return score(model, rows, device)["accuracy"]


def train(model: TinyLM, frames: tuple[str, ...], *, steps: int, batch: int, lr: float, warmup: int,
          weight_decay: float, clip: float, distractor_rate: float, data_seed: str, device: str,
          betas: tuple[float, float] = (0.9, 0.95), log_every: int = 50) -> dict[str, Any]:
    """AdamW with linear warm-up and cosine decay; a fresh batch of sampled sentences every step."""
    rng = random.Random(data_seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    use_amp = device == "cuda"
    losses: list[float] = []
    t0 = time.perf_counter()
    model.train()
    for step in range(steps):
        scale = (step + 1) / warmup if step < warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) /
                                                                              max(1, steps - warmup)))
        for group in opt.param_groups:
            group["lr"] = lr * scale
        examples = [sample(frames, rng, distractor_rate)[:2] for _ in range(batch)]
        ids, mask = collate(examples, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss = answer_loss(model, ids, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        if step % log_every == 0 or step == steps - 1:
            losses.append(loss.item())
    return {"losses_every": log_every, "losses": losses, "seconds": time.perf_counter() - t0}

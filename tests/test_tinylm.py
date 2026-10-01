# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Tests for the from-scratch small language model: byte tokens, the decoder, and a tiny training run."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from reachability_gen.tinylm.model import TinyConfig, TinyLM, apply_rotary, rotary_tables  # noqa: E402
from reachability_gen.tinylm.tokenizer import BOS, EOS, PAD, VOCAB_SIZE, decode, encode  # noqa: E402

TINY = TinyConfig(d=64, layers=2, heads=4, mlp_hidden=192, max_len=256)


def test_byte_tokens_round_trip_and_never_meet_an_unseen_token():
    text = "Starting at 7, one step takes you to 2. Café → 3."
    ids = encode(text)
    assert ids[0] == BOS and ids[-1] == EOS and all(0 <= i < VOCAB_SIZE for i in ids)
    assert decode(ids) == text and decode(encode(text, bos=False, eos=False)) == text
    assert decode([BOS, 72, 105, EOS], show_specials=True) == "<bos>Hi<eos>"
    assert all(i < 256 for i in encode("an unseen word: zyzzyva", bos=False, eos=False))


def test_rotary_positions_are_relative():
    cos, sin = rotary_tables(16, 64, 10_000.0)
    q, k = torch.randn(16), torch.randn(16)

    def score(i, j):
        return float(apply_rotary(q[None], cos[i:i + 1], sin[i:i + 1])[0]
                     @ apply_rotary(k[None], cos[j:j + 1], sin[j:j + 1])[0])

    assert abs(score(5, 2) - score(40, 37)) < 1e-4  # the same offset scores the same anywhere


def test_decoder_is_causal_and_shaped():
    torch.manual_seed(0)
    model = TinyLM(TINY).eval()
    ids = torch.tensor([encode("abcdefgh")])
    logits = model(ids)
    assert logits.shape == (1, ids.shape[1], VOCAB_SIZE)
    changed = ids.clone()
    changed[0, 6] = ord("z")
    with torch.no_grad():
        same = model(changed)
    assert torch.allclose(logits[0, :6], same[0, :6], atol=1e-5)  # earlier positions cannot see position 6
    assert not torch.allclose(logits[0, 6:], same[0, 6:])
    assert model.embed.weight is not None and model.num_params() == sum(p.numel() for p in model.parameters())


def test_padding_is_ignored_by_the_loss():
    torch.manual_seed(0)
    model = TinyLM(TINY)
    ids = torch.tensor([encode("abc")])
    padded = torch.cat([ids, torch.full((1, 5), PAD)], dim=1)
    assert torch.allclose(model.loss(ids), model.loss(padded), atol=1e-6)


def test_tiny_model_learns_a_sentence_and_recites_it():
    torch.manual_seed(0)
    model = TinyLM(TINY)
    text = "0 -> 3. 3 points to 1. Starting at 1, one step takes you to 2."
    ids = torch.tensor([encode(text)])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        loss = model.loss(ids)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    out = model.generate(torch.tensor([[BOS]]), max_new=len(ids[0]))
    assert decode(out[0]) == text


def test_bench_runs_on_cpu():
    from reachability_gen.tinylm.bench import throughput

    r = throughput(TINY, batch=2, seq=32, steps=2, warmup=1, device="cpu")
    assert r["tokens_per_second"] > 0 and r["params"] > 0

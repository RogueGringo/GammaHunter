# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Natural-language renderings of the reachability graphs (MEASURE, fixed before any run).

Each edge u→v becomes one sentence from a template split: the training
templates, or held-out templates whose wording never appears in training.
Both splits contain a template that names the target first, so word order
alone does not give the direction. About a quarter as many distractor
sentences as edges mention single nodes. Sentences are shuffled with a seed
fixed per graph and split; the question is never rendered.

Also a word tokenizer for readers trained from scratch: node numbers share
one kind (NODE, grouped by number only, as in ``models.reader``), words
get vocabulary ids from the training templates, and unseen words map to UNK.

``science_open=false`` always.
"""

from __future__ import annotations

import random
import re
from typing import Sequence

import torch

from reachability_gen.models.reader import NODE, EdgeListTokens

TRAIN_TEMPLATES: tuple[str, ...] = (
    "{u} -> {v}.",
    "There is an edge from {u} to {v}.",
    "{u} points to {v}.",
    "From {u} you can go directly to {v}.",
    "{v} is directly reachable from {u}.",
)
HELDOUT_TEMPLATES: tuple[str, ...] = (
    "{u} leads to {v}.",
    "A link runs from {u} to {v}.",
    "Starting at {u}, one step takes you to {v}.",
    "{v} can be entered straight from {u}.",
)
TEMPLATES: dict[str, tuple[str, ...]] = {"train": TRAIN_TEMPLATES, "heldout": HELDOUT_TEMPLATES}
DISTRACTORS: tuple[str, ...] = ("Node {w} is marked.", "{w} is one of the nodes.")
DISTRACTOR_RATE: float = 0.25
SEED: int = 2026
UNK: int = 1  # token kinds: 0 = NODE, 1 = UNK, 2.. = words
WORD = re.compile(r"\d+|->|[A-Za-z]+|[.,]")


def render(n: int, edges: Sequence[tuple[int, int]], split: str, key: str) -> str:
    """The graph as shuffled sentences from one template split (deterministic per ``key`` and split)."""
    rng = random.Random(f"{SEED}/{key}/{split}")
    sentences = [rng.choice(TEMPLATES[split]).format(u=u, v=v) for u, v in edges]
    sentences += [rng.choice(DISTRACTORS).format(w=rng.randrange(n))
                  for _ in range(int(round(DISTRACTOR_RATE * len(edges))))]
    rng.shuffle(sentences)
    return " ".join(sentences)


def build_vocab(templates: Sequence[str] = TRAIN_TEMPLATES + DISTRACTORS) -> dict[str, int]:
    """Word ids from the training templates and the distractors (node numbers excluded)."""
    words = sorted({w.lower() for t in templates for w in WORD.findall(t.replace("{u}", "0").replace("{v}", "0")
                                                                        .replace("{w}", "0")) if not w.isdigit()})
    return {w: i + 2 for i, w in enumerate(words)}


def word_tokens(text: str, n: int, vocab: dict[str, int]) -> EdgeListTokens:
    """Reader tokens for a rendering: NODE (grouped by number) or a vocabulary id (UNK if unseen)."""
    kinds, symbol = [], []
    for w in WORD.findall(text):
        if w.isdigit():
            kinds.append(NODE)
            symbol.append(int(w))
        else:
            kinds.append(vocab.get(w.lower(), UNK))
            symbol.append(-1)
    return EdgeListTokens(torch.tensor(kinds), torch.zeros(len(kinds), dtype=torch.long), torch.tensor(symbol), n)


__all__ = ["DISTRACTORS", "HELDOUT_TEMPLATES", "TEMPLATES", "TRAIN_TEMPLATES", "UNK", "build_vocab", "render",
           "word_tokens"]

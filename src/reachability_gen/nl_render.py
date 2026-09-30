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

Two further splits, fixed before any run that uses them:

* ``diverse``: the five training templates plus 46 more wordings. A quarter of
  them name the target first, so word order does not give the direction, and
  "by" introduces the source in some templates and the target in another.
  None contains an open-class word of the held-out or novel templates
  (``FORBIDDEN_WORDS``).
* ``novel``: four held-out templates built from constructions the diverse set
  lacks (an imperative, a conditional, a locative inversion and an "endpoint
  of" noun phrase); no diverse template matches their skeletons (``skeleton``).

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
DIVERSE_TEMPLATES: tuple[str, ...] = TRAIN_TEMPLATES + (
    # source first: a verb, usually with a preposition
    "{u} goes to {v}.",
    "{u} points at {v}.",
    "{u} moves to {v}.",
    "{u} passes to {v}.",
    "{u} feeds into {v}.",
    "{u} opens onto {v}.",
    "{u} heads to {v}.",
    "{u} sends an edge to {v}.",
    "{u} sends an arrow to {v}.",
    "{u} has an edge to {v}.",
    "{u} has an arrow pointing to {v}.",
    "{u} has a direct edge to {v}.",
    "{u} has an outgoing edge to {v}.",
    "{u} reaches {v} directly.",
    "{u} feeds {v}.",
    "{u} targets {v}.",
    "{u} is directly followed by {v}.",
    # source first: "there is" and noun subjects
    "There is an arrow from {u} to {v}.",
    "There is an arc from {u} to {v}.",
    "There is a connection from {u} to {v}.",
    "There is a direct edge from {u} to {v}.",
    "There is a pointer from {u} to {v}.",
    "There is a transition from {u} to {v}.",
    "There is a channel from {u} to {v}.",
    "There exists an edge from {u} to {v}.",
    "An edge goes from {u} to {v}.",
    "An arrow points from {u} to {v}.",
    "An arc extends from {u} to {v}.",
    "A connection goes directly from {u} to {v}.",
    "A pointer goes from {u} to {v}.",
    # source first: a fronted "from"
    "From {u} there is an edge to {v}.",
    "From {u}, an arrow points to {v}.",
    "From {u} you can move directly to {v}.",
    "From {u} one can get directly to {v}.",
    # target first
    "{v} can be reached directly from {u}.",
    "{v} can be accessed directly from {u}.",
    "{v} can be visited directly from {u}.",
    "{v} is reached directly from {u}.",
    "{v} is pointed to by {u}.",
    "{v} is fed by {u}.",
    "{v} is fed directly from {u}.",
    "{v} receives an edge from {u}.",
    "{v} gets an arrow from {u}.",
    "{v} has an incoming edge from {u}.",
    "{v} follows {u} directly.",
    "To {v} there is an edge from {u}.",
)
NOVEL_TEMPLATES: tuple[str, ...] = (
    "Hop out of {u} and you land on {v}.",
    "If you are at {u}, {v} is one hop away.",
    "Into {v} flows a current from {u}.",
    "{v} is the endpoint of a wire out of {u}.",
)
# Open-class words of the held-out and novel templates, with their inflections: never in a training template.
FORBIDDEN_WORDS: frozenset[str] = frozenset({
    "lead", "leads", "leading", "led", "link", "links", "linked", "linking", "run", "runs", "running", "ran",
    "start", "starts", "starting", "started", "step", "steps", "take", "takes", "taking", "took", "taken",
    "enter", "enters", "entered", "entering", "entry", "straight",
    "hop", "hops", "hopping", "hopped", "land", "lands", "landing", "landed", "away", "flow", "flows",
    "flowing", "flowed", "current", "currents", "endpoint", "endpoints", "wire", "wires", "wired",
})
TEMPLATES: dict[str, tuple[str, ...]] = {"train": TRAIN_TEMPLATES, "heldout": HELDOUT_TEMPLATES,
                                         "diverse": DIVERSE_TEMPLATES, "novel": NOVEL_TEMPLATES}
DISTRACTORS: tuple[str, ...] = ("Node {w} is marked.", "{w} is one of the nodes.")
DISTRACTOR_RATE: float = 0.25
SEED: int = 2026
UNK: int = 1  # token kinds: 0 = NODE, 1 = UNK, 2.. = words
WORD = re.compile(r"\d+|->|[A-Za-z]+|[.,]")


def render(n: int, edges: Sequence[tuple[int, int]], split: str, key: str, rate: float = DISTRACTOR_RATE) -> str:
    """The graph as shuffled sentences from one template split (deterministic per ``key`` and split).

    ``rate`` distractor sentences per edge; the default is the rate of every earlier study.
    """
    rng = random.Random(f"{SEED}/{key}/{split}")
    sentences = [rng.choice(TEMPLATES[split]).format(u=u, v=v) for u, v in edges]
    sentences += [rng.choice(DISTRACTORS).format(w=rng.randrange(n))
                  for _ in range(int(round(rate * len(edges))))]
    rng.shuffle(sentences)
    return " ".join(sentences)


def skeleton(template: str) -> re.Pattern:
    """``template`` with each of its ``FORBIDDEN_WORDS`` replaced by a word wildcard (full-match pattern)."""
    parts = re.split(r"([A-Za-z]+)", template)
    return re.compile("".join(r"[A-Za-z]+" if p.isalpha() and p.lower() in FORBIDDEN_WORDS else re.escape(p)
                              for p in parts))


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


__all__ = ["DISTRACTORS", "DIVERSE_TEMPLATES", "FORBIDDEN_WORDS", "HELDOUT_TEMPLATES", "NOVEL_TEMPLATES", "TEMPLATES",
           "TRAIN_TEMPLATES", "UNK", "build_vocab", "render", "skeleton", "word_tokens"]

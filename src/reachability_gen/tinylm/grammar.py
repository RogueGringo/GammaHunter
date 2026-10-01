# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""A grammar of edge sentences: constructions ("frames") built from shared word lists.

Every frame states one directed edge u → v, in lower case. Frames differ in
structure; their words come from shared slot lists (verbs, nouns, prepositions),
so a frame held out from training is new in structure only once the whole pool
has been trained: every word and every byte it uses occurs in the pool (tested).
Each frame records its role order: "S" when the source's number comes first in
the sentence, "T" when the target's does.

Four controls read no structure: the order heuristic (the first number is the
source) reads every S frame and no T frame; the mirror heuristic (the first
number is the target) the reverse; the marker heuristic reads roles from
familiar local markers, prepositions first ("from N" source, "to N" or "into N"
target) and then the subject position ("N" before a source verb: source), and by
construction reads every frame here, held-out frames included; the
subject-first variant, which trusts the subject position before the
prepositions, reads every frame except locative inversion, where the target
stands before the verb. A held-out frame read correctly is therefore read at
least as well as familiar local markers allow; this grammar cannot show reading
beyond them.

The pool holds 12 frames, 6 S and 6 T. Held out (fixed): ``participial`` (S; a
participial opening like the one item 20's readers did not read, its phrase
"starting at" occurring in the pool), ``cleft_target``, ``locative_inversion``
and ``relative_target`` (T).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

N_NODES: int = 48
SV: tuple[str, ...] = ("leads", "goes", "points", "moves", "passes", "runs")  # source subject, then "to"
PV: tuple[str, ...] = ("is reached", "is entered", "can be reached", "can be entered", "is fed")  # target subject
IV: tuple[str, ...] = ("go", "move", "pass", "run")  # bare verbs ("go to ...", "you can go to ...")
NOUNS: tuple[str, ...] = ("edge", "arrow", "link", "path", "pointer", "road")
TO_INTO: tuple[str, ...] = ("to", "into")
DISTRACTORS: tuple[str, ...] = ("node {w} is marked.", "{w} is one of the nodes.")
NONE = "none"


@dataclass(frozen=True)
class Frame:
    name: str
    order: str  # "S": the source's number comes first; "T": the target's does
    pattern: str


_FRAMES = (
    # pool, source first
    Frame("svo", "S", "{u} {sv} {to} {v}."),
    Frame("existential_participle", "S", "there is {a_n} starting at {u} that {sv} to {v}."),
    Frame("cleft_source", "S", "it is {u} that {sv} to {v}."),
    Frame("relative_source", "S", "the node that {u} {sv} to is {v}."),
    Frame("fronted_source", "S", "from {u}, {a_n} {sv} to {v}."),
    Frame("conditional_source", "S", "if you are at {u}, you can {iv} to {v}."),
    # pool, target first
    Frame("passive", "T", "{v} {pv} from {u}."),
    Frame("fronted_target", "T", "to {v}, {a_n} {sv} from {u}."),
    Frame("existential_target", "T", "there is {a_n} to {v} from {u}."),
    Frame("noun_target", "T", "{a_n} {sv} to {v} from {u}."),
    Frame("imperative_target", "T", "{iv} to {v} from {u}."),
    Frame("conditional_target", "T", "you can {iv} to {v} if you are at {u}."),
    # held out
    Frame("participial", "S", "starting at {u}, {a_n} {sv} to {v}."),
    Frame("cleft_target", "T", "it is {v} that {u} {sv} to."),
    Frame("locative_inversion", "T", "into {v} {sv} {a_n} from {u}."),
    Frame("relative_target", "T", "{v} is the node that {u} {sv} to."),
)
FRAMES: dict[str, Frame] = {f.name: f for f in _FRAMES}
HELD_OUT: tuple[str, ...] = ("participial", "cleft_target", "locative_inversion", "relative_target")
POOL: tuple[str, ...] = tuple(f.name for f in _FRAMES if f.name not in HELD_OUT)
_SLOTS: dict[str, tuple[str, ...]] = {"sv": SV, "pv": PV, "iv": IV, "to": TO_INTO}


def _article(noun: str) -> str:
    return f"{'an' if noun[0] in 'aeiou' else 'a'} {noun}"


def realize(frame: Frame, u: int, v: int, rng: random.Random) -> str:
    """The sentence of ``frame`` for the edge u → v, its words drawn from the slot lists."""
    return frame.pattern.format(u=u, v=v, sv=rng.choice(SV), pv=rng.choice(PV), to=rng.choice(TO_INTO),
                                iv=rng.choice(IV), a_n=_article(rng.choice(NOUNS)))


def answer(u: int, v: int) -> str:
    return f"{u}>{v}"


def sample(frames: tuple[str, ...], rng: random.Random, distractor_rate: float) -> tuple[str, str, str]:
    """(sentence, answer, frame name or "distractor"): an edge in one of ``frames``, or a distractor sentence."""
    if rng.random() < distractor_rate:
        return rng.choice(DISTRACTORS).format(w=rng.randrange(N_NODES)), NONE, "distractor"
    u, v = rng.sample(range(N_NODES), 2)
    name = rng.choice(frames)
    return realize(FRAMES[name], u, v, rng), answer(u, v), name


def _numbers(sentence: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", sentence)]


def order_heuristic(sentence: str) -> str:
    """Control: the first number is the source, the second the target; one number means none."""
    nums = _numbers(sentence)
    return answer(nums[0], nums[1]) if len(nums) >= 2 else NONE


def mirror_heuristic(sentence: str) -> str:
    """Control: the first number is the target, the second the source; one number means none."""
    nums = _numbers(sentence)
    return answer(nums[1], nums[0]) if len(nums) >= 2 else NONE


def _marker(sentence: str, *, prepositions_first: bool) -> str:
    """Roles from local markers: "from N" source, "to N"/"into N" target, "N" before a source verb source.

    Each number takes the role of its first marker in the chosen priority; a
    role on either number decides the answer (the first number's role first);
    without one, the order heuristic's answer.
    """
    toks = re.findall(r"\d+|[a-z]+", sentence.lower())
    nums = [(i, int(t)) for i, t in enumerate(toks) if t.isdigit()]
    if len(nums) < 2:
        return NONE
    roles = {}
    for i, n in nums[:2]:
        prev = toks[i - 1] if i > 0 else ""
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        by_preposition = "source" if prev == "from" else "target" if prev in TO_INTO else None
        by_subject = "source" if nxt in SV else None
        role = (by_preposition or by_subject) if prepositions_first else (by_subject or by_preposition)
        if role:
            roles[n] = role
    a, b = nums[0][1], nums[1][1]
    if roles.get(a) == "source" or roles.get(b) == "target":
        return answer(a, b)
    if roles.get(a) == "target" or roles.get(b) == "source":
        return answer(b, a)
    return answer(a, b)


def marker_heuristic(sentence: str) -> str:
    """Control: roles from local markers, prepositions first, then the subject position."""
    return _marker(sentence, prepositions_first=True)


def subject_marker_heuristic(sentence: str) -> str:
    """Control: roles from local markers, the subject position first, then prepositions."""
    return _marker(sentence, prepositions_first=False)


def words(text: str) -> set[str]:
    """Alphabetic words (numbers and punctuation left out)."""
    return set(re.findall(r"[a-z]+", text))


def frame_words(name: str) -> set[str]:
    """Every word ``name`` can use: its fixed words and every word of every slot it fills."""
    pattern = FRAMES[name].pattern
    out = words(re.sub(r"\{[^}]*\}", " ", pattern))
    slots = set(re.findall(r"\{([^}]*)\}", pattern))
    for slot in slots & set(_SLOTS):
        for item in _SLOTS[slot]:
            out |= words(item)
    if "a_n" in slots:
        out |= set(NOUNS) | {"a", "an"}
    return out


def frame_chars(name: str) -> set[str]:
    """Every character (byte) ``name`` can produce: its fixed text, its slot fillers, digits and the space."""
    pattern = FRAMES[name].pattern
    out = set(re.sub(r"\{[^}]*\}", " ", pattern)) | set("0123456789 ")
    for w in frame_words(name):
        out |= set(w)
    return out


def skeleton(name: str) -> str:
    """The frame's structure with every slot abstracted: numbers to N, word slots to their kind."""
    s = FRAMES[name].pattern
    for slot, kind in (("{u}", "N"), ("{v}", "N"), ("{sv}", "VERB"), ("{pv}", "PASSIVE"), ("{iv}", "VERB"),
                       ("{to}", "PREP"), ("{a_n}", "NOUNPHRASE")):
        s = s.replace(slot, kind)
    return s


def eval_sets(per_frame: int, seed: int) -> dict[str, list[tuple[str, str]]]:
    """Fixed evaluation sentences: ``per_frame`` for every frame and as many distractors, (sentence, answer)."""
    out: dict[str, list[tuple[str, str]]] = {}
    for name in FRAMES:
        rng = random.Random(f"eval/{seed}/{name}")
        rows = []
        for _ in range(per_frame):
            u, v = rng.sample(range(N_NODES), 2)
            rows.append((realize(FRAMES[name], u, v, rng), answer(u, v)))
        out[name] = rows
    rng = random.Random(f"eval/{seed}/distractor")
    out["distractor"] = [(rng.choice(DISTRACTORS).format(w=rng.randrange(N_NODES)), NONE) for _ in range(per_frame)]
    return out

# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Byte tokens: a string is its UTF-8 bytes (0–255), plus four special tokens.

There is no vocabulary to learn, so no input token is ever unseen: a word that
never occurred in training is still a sequence of familiar bytes.
"""

from __future__ import annotations

from typing import Iterable

PAD, BOS, EOS, SEP = 256, 257, 258, 259
VOCAB_SIZE = 260
SPECIALS = {PAD: "<pad>", BOS: "<bos>", EOS: "<eos>", SEP: "<sep>"}


def encode(text: str, *, bos: bool = True, eos: bool = True) -> list[int]:
    """The UTF-8 bytes of ``text``, optionally between BOS and EOS."""
    ids = list(text.encode("utf-8"))
    return ([BOS] if bos else []) + ids + ([EOS] if eos else [])


def decode(ids: Iterable[int], *, show_specials: bool = False) -> str:
    """Bytes back to text (invalid UTF-8 replaced); special tokens dropped unless shown."""
    out, buf = [], bytearray()
    for i in ids:
        i = int(i)
        if i < 256:
            buf.append(i)
            continue
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
            buf = bytearray()
        if show_specials:
            out.append(SPECIALS.get(i, f"<{i}>"))
    if buf:
        out.append(buf.decode("utf-8", errors="replace"))
    return "".join(out)

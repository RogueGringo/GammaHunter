# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Locked v0 encoding: canonical edge-list serialization + query markers.

Why edge-list, not adjacency-list?
----------------------------------
- Canonical total order: sorting (u,v) pairs gives a unique string for a given
  directed simple graph, independent of generation order. That makes edge_hash
  stable and comparable across runs.
- Compact for sparse ER digraphs (expected edges ≈ p * n * (n-1)): listing
  only present edges avoids padding absent neighbors.
- Token-friendly for sequence models in later work: a flat list of edge tokens
  plus explicit QUERY s t markers; no adjacency-row separators to invent.
- Deterministic: same edges + same (s,t) → same encoding string always.

NO scratchpad / chain-of-thought is included in this encoding.
"""

from __future__ import annotations

import hashlib
from typing import Sequence


def canonical_edge_list(edges: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return a sorted copy of the edge list (u ascending, then v ascending)."""
    return sorted((int(u), int(v)) for u, v in edges)


def edge_hash(edges: Sequence[tuple[int, int]]) -> str:
    """Stable hash of the canonical edge list (sha256 hex of encoding bytes)."""
    canon = canonical_edge_list(edges)
    payload = edges_to_bytes(canon)
    return hashlib.sha256(payload).hexdigest()


def edges_to_bytes(edges: Sequence[tuple[int, int]]) -> bytes:
    """Deterministic byte serialization of a canonical edge list."""
    # Format: "u,v;u,v;..." — empty graph → empty bytes.
    if not edges:
        return b""
    return ";".join(f"{u},{v}" for u, v in edges).encode("ascii")


def encode_instance(
    n: int,
    edges: Sequence[tuple[int, int]],
    s: int,
    t: int,
) -> str:
    """Locked v0 string encoding.

    Format:
        N <n> EDGES <u>,<v> <u>,<v> ... QUERY <s> <t>

    Node ids are 0..n-1. Edges are the sorted list of (u,v). No scratchpad.
    """
    if not (0 <= s < n and 0 <= t < n):
        raise ValueError(f"query ({s},{t}) out of range for n={n}")
    canon = canonical_edge_list(edges)
    edge_tokens = " ".join(f"{u},{v}" for u, v in canon)
    if edge_tokens:
        return f"N {n} EDGES {edge_tokens} QUERY {s} {t}"
    return f"N {n} EDGES QUERY {s} {t}"


def parse_instance(encoding: str) -> tuple[int, list[tuple[int, int]], int, int]:
    """Parse a locked v0 encoding string back to ``(n, edges, s, t)``.

    Inverse of :func:`encode_instance`. Raises ``ValueError`` on malformed input.
    """
    parts = str(encoding).split()
    if len(parts) < 5 or parts[0] != "N" or parts[2] != "EDGES":
        raise ValueError(f"malformed encoding (expected N … EDGES … QUERY …): {encoding!r}")
    try:
        n = int(parts[1])
    except ValueError as e:
        raise ValueError(f"malformed n in encoding: {encoding!r}") from e
    try:
        q_idx = parts.index("QUERY")
    except ValueError as e:
        raise ValueError(f"missing QUERY marker in encoding: {encoding!r}") from e
    if q_idx + 2 >= len(parts):
        raise ValueError(f"incomplete QUERY in encoding: {encoding!r}")
    edge_tokens = parts[3:q_idx]
    edges: list[tuple[int, int]] = []
    for tok in edge_tokens:
        if "," not in tok:
            raise ValueError(f"bad edge token {tok!r} in encoding: {encoding!r}")
        a, b = tok.split(",", 1)
        edges.append((int(a), int(b)))
    s = int(parts[q_idx + 1])
    t = int(parts[q_idx + 2])
    return n, edges, s, t

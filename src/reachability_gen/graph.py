# Copyright (c) 2026 B.Jones. All rights reserved.
# Proprietary and confidential. See LICENSE.
# SPDX-License-Identifier: LicenseRef-Proprietary

"""Erdős–Rényi directed graphs and reachability labels.

Directed simple graphs: Bernoulli(p) on ordered pairs (u,v) with u != v
(no self-loops). Label y = 1 iff target t is reachable from source s.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Sequence


def er_digraph(n: int, p: float, rng) -> list[tuple[int, int]]:
    """Sample an ER digraph; return a sorted edge list with no self-loops.

    For every ordered pair (u, v) with u != v, include the edge independently
    with probability p. Node ids are 0..n-1.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")

    edges: list[tuple[int, int]] = []
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if rng.random() < p:
                edges.append((u, v))
    edges.sort()
    return edges


def adjacency_list(n: int, edges: Sequence[tuple[int, int]]) -> list[list[int]]:
    """Build outgoing adjacency lists from an edge list."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        if u == v:
            raise ValueError(f"self-loop forbidden: ({u},{v})")
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"edge ({u},{v}) out of range for n={n}")
        adj[u].append(v)
    return adj


def reachable_bfs(n: int, edges: Sequence[tuple[int, int]], s: int, t: int) -> bool:
    """True iff t is reachable from s via directed BFS (including s == t)."""
    if s == t:
        return True
    adj = adjacency_list(n, edges)
    seen = [False] * n
    q: deque[int] = deque([s])
    seen[s] = True
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not seen[v]:
                if v == t:
                    return True
                seen[v] = True
                q.append(v)
    return False


def reachable_dfs(n: int, edges: Sequence[tuple[int, int]], s: int, t: int) -> bool:
    """True iff t is reachable from s via directed DFS (including s == t)."""
    if s == t:
        return True
    adj = adjacency_list(n, edges)
    seen = [False] * n

    def dfs(u: int) -> bool:
        if u == t:
            return True
        seen[u] = True
        for v in adj[u]:
            if not seen[v] and dfs(v):
                return True
        return False

    return dfs(s)


def has_self_loops(edges: Iterable[tuple[int, int]]) -> bool:
    return any(u == v for u, v in edges)


def sample_query(n: int, rng) -> tuple[int, int]:
    """Sample ordered query pair (s, t); s and t may be equal."""
    s = rng.randrange(n)
    t = rng.randrange(n)
    return s, t


def hop_distance(
    n: int,
    edges: Sequence[tuple[int, int]],
    s: int,
    t: int,
) -> int | None:
    """Shortest-path hop distance from s to t, or None if unreachable.

    ``s == t`` returns 0 (zero hops; still reachable / y=1 under the label rule).
    """
    if not (0 <= s < n and 0 <= t < n):
        raise ValueError(f"query ({s},{t}) out of range for n={n}")
    if s == t:
        return 0
    adj = adjacency_list(n, edges)
    dist = [-1] * n
    dist[s] = 0
    q: deque[int] = deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                if v == t:
                    return dist[v]
                q.append(v)
    return None


def hop_distances_from(
    n: int,
    edges: Sequence[tuple[int, int]],
    s: int,
) -> list[int]:
    """BFS distances from ``s``; unreachable nodes get -1."""
    if not (0 <= s < n):
        raise ValueError(f"source {s} out of range for n={n}")
    adj = adjacency_list(n, edges)
    dist = [-1] * n
    dist[s] = 0
    q: deque[int] = deque([s])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist

"""
The methods Experiment 1 compares, each reduced to one map g(k, l, t): RT bins in batch k → RT bins
in batch l, for every ordered batch pair (direct anchors or not).

  joint              jointalign.fit_joint on all kept anchors
  star               fit_joint on the kept anchors that touch the hub; other maps go through the hub
                     (batches with no anchor to the hub stay at identity)
  composed pairwise  jointalign.fit_pairwise per kept edge, composed along the shortest path in the
                     kept graph (ties: the path with the most anchors)
  composed pchip     jointalign.pchip_pairwise per kept edge, composed the same way

Every λ-method uses the same gauge (batch 0) and jointalign's robust reweighting.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
from jointalign import Anchors, JointWarp, fit_joint, fit_pairwise, pchip_pairwise

from trustalign.graphs import Edge, adjacency

MapFn = Callable[[int, int, np.ndarray], np.ndarray]
GAUGE = 0


@dataclass
class Fitted:
    map: MapFn
    model: JointWarp | None = None                 # the joint-type fit, for weights and scales
    used: np.ndarray | None = None                 # mask of the anchors the model was fitted to
    paths: dict[tuple[int, int], list[int]] = field(default_factory=dict)


def joint(anchors: Anchors, K: int, lam: float) -> Fitted:
    m = fit_joint(anchors, K, lam=lam, gauge=GAUGE)
    return Fitted(m.map, m, np.ones(len(anchors), bool))


def star(anchors: Anchors, K: int, lam: float, hub: int) -> Fitted:
    touch = (anchors.ba == hub) | (anchors.bb == hub)
    m = fit_joint(anchors.subset(touch), K, lam=lam, gauge=GAUGE)
    return Fitted(m.map, m, touch)


def shortest_paths(K: int, edges: set[Edge], n_pair: dict[Edge, int]) -> dict[tuple[int, int], list[int]]:
    """For every ordered (k, l) connected in the graph: the fewest-hop path, ties broken by the
    largest total number of anchors on its edges, then by the lowest batch indices."""
    adj = adjacency(K, edges)
    out: dict[tuple[int, int], list[int]] = {}
    for src in range(K):
        hops, best, prev = {src: 0}, {src: 0}, {src: -1}
        frontier = [src]
        while frontier:
            nxt: list[int] = []
            for u in frontier:
                for v in adj[u]:
                    score = best[u] + n_pair[(min(u, v), max(u, v))]
                    if v not in hops:
                        hops[v], best[v], prev[v] = hops[u] + 1, score, u
                        nxt.append(v)
                    elif hops[v] == hops[u] + 1 and score > best[v]:
                        best[v], prev[v] = score, u
            frontier = sorted(nxt)
        for dst in hops:
            if dst != src:
                path, v = [dst], dst
                while prev[v] != -1:
                    v = prev[v]
                    path.append(v)
                out[(src, dst)] = path[::-1]
    return out


def _compose(edge_maps: dict[tuple[int, int], Callable[[np.ndarray], np.ndarray]],
             paths: dict[tuple[int, int], list[int]]) -> MapFn:
    def g(k: int, l: int, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if k == l:
            return t
        if (k, l) not in paths:
            raise ValueError(f'batches {k} and {l} are not connected')
        path = paths[(k, l)]
        for u, v in pairwise(path):
            t = edge_maps[(u, v)](t)
        return t
    return g


def _counts(anchors: Anchors, edges: set[Edge]) -> dict[Edge, int]:
    return {(a, b): int(np.sum((anchors.ba == a) & (anchors.bb == b))) for a, b in edges}


def _edge_map(m: JointWarp, k: int, l: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda t: m.map(k, l, t)


def composed_pairwise(anchors: Anchors, K: int, lam: float, edges: set[Edge]) -> Fitted:
    maps: dict[tuple[int, int], Callable[[np.ndarray], np.ndarray]] = {}
    for a, b in edges:
        m = fit_pairwise(anchors, a, b, K, lam=lam, gauge=0)
        maps[(a, b)], maps[(b, a)] = _edge_map(m, a, b), _edge_map(m, b, a)
    paths = shortest_paths(K, edges, _counts(anchors, edges))
    return Fitted(_compose(maps, paths), paths=paths)


def composed_pchip(anchors: Anchors, K: int, edges: set[Edge]) -> Fitted:
    maps = {}
    for a, b in edges:
        maps[(a, b)] = pchip_pairwise(anchors, a, b)
        maps[(b, a)] = pchip_pairwise(anchors, b, a)
    paths = shortest_paths(K, edges, _counts(anchors, edges))
    return Fitted(_compose(maps, paths), paths=paths)


def fitters(K: int, edges: set[Edge], hub: int) -> dict[str, Callable[[Anchors, float], Fitted]]:
    """Each method as fit(anchors, lam) for one condition (composed pchip ignores lam)."""
    return {
        'Joint': lambda anc, lam: joint(anc, K, lam),
        'Star': lambda anc, lam: star(anc, K, lam, hub),
        'Composed pairwise': lambda anc, lam: composed_pairwise(anc, K, lam, edges),
        'Composed PCHIP': lambda anc, lam: composed_pchip(anc, K, edges),
    }


LAMBDA_FREE = {'Composed PCHIP'}

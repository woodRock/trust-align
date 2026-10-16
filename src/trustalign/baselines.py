"""
The methods Experiment 1 compares, each reduced to one map g(k, l, t): RT bins in batch k → RT bins
in batch l, for every ordered batch pair (direct anchors or not).

  joint              jointalign.fit_joint on all kept anchors
  star               fit_joint on the kept anchors that touch the hub; other maps go through the hub
                     (batches with no anchor to the hub stay at identity)
  composed pairwise  jointalign.fit_pairwise per kept edge, composed along the shortest path in the
                     kept graph (ties: the path with the most anchors)
  composed pchip     jointalign.pchip_pairwise per kept edge, composed the same way

Every λ-method uses the same gauge (batch 0) and jointalign's robust reweighting. Shortest paths,
their composition and composed PCHIP are alignbench.maps' (bit-identical to this module's 0.1.0). `solver` picks
fit_joint's bounded least-squares solver: 'bvls' (jointalign's default, paper one's) or 'active_set'
(jointalign ≥ 1.1.0; the same optimum, much faster for K ≥ 10).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from alignbench.maps import compose_paths as _compose
from alignbench.maps import composed_pchip as _composed_pchip
from alignbench.maps import edge_counts as _counts
from alignbench.maps import shortest_paths
from jointalign import Anchors, JointWarp, fit_joint, fit_pairwise

from trustalign.graphs import Edge

MapFn = Callable[[int, int, np.ndarray], np.ndarray]
GAUGE = 0


@dataclass
class Fitted:
    map: MapFn
    model: JointWarp | None = None                 # the joint-type fit, for weights and scales
    used: np.ndarray | None = None                 # mask of the anchors the model was fitted to
    paths: dict[tuple[int, int], list[int]] = field(default_factory=dict)


def joint(anchors: Anchors, K: int, lam: float, solver: str = 'bvls') -> Fitted:
    m = fit_joint(anchors, K, lam=lam, gauge=GAUGE, solver=solver)
    return Fitted(m.map, m, np.ones(len(anchors), bool))


def star(anchors: Anchors, K: int, lam: float, hub: int, solver: str = 'bvls') -> Fitted:
    touch = (anchors.ba == hub) | (anchors.bb == hub)
    m = fit_joint(anchors.subset(touch), K, lam=lam, gauge=GAUGE, solver=solver)
    return Fitted(m.map, m, touch)


def _edge_map(m: JointWarp, k: int, l: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda t: m.map(k, l, t)


def composed_pairwise(anchors: Anchors, K: int, lam: float, edges: set[Edge], solver: str = 'bvls') -> Fitted:
    maps: dict[tuple[int, int], Callable[[np.ndarray], np.ndarray]] = {}
    for a, b in edges:
        m = fit_pairwise(anchors, a, b, K, lam=lam, gauge=0, solver=solver)
        maps[(a, b)], maps[(b, a)] = _edge_map(m, a, b), _edge_map(m, b, a)
    paths = shortest_paths(K, edges, _counts(anchors, edges))
    return Fitted(_compose(maps, paths), paths=paths)


def composed_pchip(anchors: Anchors, K: int, edges: set[Edge]) -> Fitted:
    g, paths = _composed_pchip(anchors, K, edges)
    return Fitted(g, paths=paths)


def fitters(K: int, edges: set[Edge], hub: int,
            solver: str = 'bvls') -> dict[str, Callable[[Anchors, float], Fitted]]:
    """Each method as fit(anchors, lam) for one condition (composed pchip ignores lam)."""
    return {
        'Joint': lambda anc, lam: joint(anc, K, lam, solver),
        'Star': lambda anc, lam: star(anc, K, lam, hub, solver),
        'Composed pairwise': lambda anc, lam: composed_pairwise(anc, K, lam, edges, solver),
        'Composed PCHIP': lambda anc, lam: composed_pchip(anc, K, edges),
    }


LAMBDA_FREE = {'Composed PCHIP'}

"""
Error against the truth, held-out anchor error, λ selection, cycle inconsistency, rejection rates.

λ selection is paper one's rule (joint-align scripts/30_multibatch_align.py): RT blocks of 12 bins,
block b in fold b mod 5; an anchor is held out if either of its times falls in the fold and used for
training only if neither does; the score is the median |held-out error|. Neither TIC r nor the
synthetic truth is ever used to choose anything.
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import permutations

import numpy as np
from jointalign import Anchors
from jointalign.synthetic import BIN_MIN, TIME_AXIS

from trustalign.baselines import Fitted, MapFn
from trustalign.graphs import Edge
from trustalign.worlds import World

RT_BLOCK = 12
N_FOLDS = 5
LAMBDAS = (0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)


def rt_folds(anchors: Anchors) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, test) masks of paper one's RT-blocked folds."""
    blk = lambda t: np.floor(t / RT_BLOCK).astype(int) % N_FOLDS
    ba, bb = blk(anchors.ta), blk(anchors.tb)
    return [((ba != f) & (bb != f), (ba == f) | (bb == f)) for f in range(N_FOLDS)]


def anchor_errors(g: MapFn, anchors: Anchors) -> np.ndarray:
    """Signed error in batch-bb bins of predicting every anchor's tb from its ta."""
    err = np.empty(len(anchors))
    for a, b in {(int(a), int(b)) for a, b in zip(anchors.ba, anchors.bb)}:
        m = (anchors.ba == a) & (anchors.bb == b)
        err[m] = g(a, b, anchors.ta[m]) - anchors.tb[m]
    return err


def select_lambda(fit: Callable[[Anchors, float], Fitted], anchors: Anchors,
                  lambdas=LAMBDAS) -> tuple[float, dict[float, float]]:
    """The λ with the lowest median |held-out error| (bins) over the RT-blocked folds, and every
    λ's score. Ties go to the smaller λ."""
    folds = rt_folds(anchors)
    scores = {}
    for lam in lambdas:
        errs = [anchor_errors(fit(anchors.subset(tr), lam).map, anchors.subset(te)) for tr, te in folds]
        scores[lam] = float(np.median(np.abs(np.concatenate(errs))))
    return min(lambdas, key=lambda l: scores[l]), scores


def inside_mask(world: World, k: int, l: int, t_min: np.ndarray) -> np.ndarray:
    """Times in batch k where compounds exist in both batches: t inside batch k's observed compound
    span and its true image inside batch l's (joint-align's test_script30_recovers_true_drifts,
    restricted to compounds each batch actually records)."""
    lo_k, hi_k = world.observed_span(k)
    lo_l, hi_l = world.observed_span(l)
    true = world.true_map(k, l, t_min)
    return (t_min >= lo_k) & (t_min <= hi_k) & (true >= lo_l) & (true <= hi_l)


def map_errors(world: World, g: MapFn, pairs=None) -> dict[tuple[int, int], np.ndarray]:
    """|ĝ_k→l(t) − g*_k→l(t)| in minutes, at every RT bin inside the mask, for every ordered pair
    (or only the ordered pairs given)."""
    out = {}
    for k, l in (permutations(range(world.K), 2) if pairs is None else pairs):
        t = TIME_AXIS[inside_mask(world, k, l, TIME_AXIS)]
        out[(k, l)] = np.abs(g(k, l, t / BIN_MIN) * BIN_MIN - world.true_map(k, l, t))
    return out


def summarise(errs: dict[tuple[int, int], np.ndarray], direct: set[Edge]) -> dict[str, float]:
    """Median and 95th percentile over t and pairs: all pairs, pairs with direct anchors, without."""
    def stats(sel) -> tuple[float, float]:
        v = np.concatenate([errs[p] for p in sel]) if sel else np.empty(0)
        return (float(np.median(v)), float(np.percentile(v, 95))) if len(v) else (np.nan, np.nan)
    with_d = [p for p in errs if (min(p), max(p)) in direct]
    without = [p for p in errs if (min(p), max(p)) not in direct]
    out = {}
    for name, sel in (('all', list(errs)), ('direct', with_d), ('indirect', without)):
        out[f'median_{name}'], out[f'p95_{name}'] = stats(sel)
    return out


def cycle_inconsistency(world: World, g: MapFn) -> tuple[float, float]:
    """Median and max over ordered triplets (k, m, l) of the mean |g_m→l(g_k→m(t)) − g_k→l(t)|
    (minutes) over t inside the k → l mask. Zero by construction for a joint-type model."""
    vals = []
    for k, m, l in permutations(range(world.K), 3):
        t = TIME_AXIS[inside_mask(world, k, l, TIME_AXIS)] / BIN_MIN
        if len(t):
            vals.append(float(np.mean(np.abs(g(m, l, g(k, m, t)) - g(k, l, t)))) * BIN_MIN)
    return (float(np.median(vals)), float(np.max(vals))) if vals else (np.nan, np.nan)


def rejection(weights: np.ndarray, bad: np.ndarray) -> tuple[float, float]:
    """Share of corrupted anchors given weight 0 (recall) and of clean anchors given weight 0."""
    recall = float(np.mean(weights[bad] == 0)) if bad.any() else np.nan
    false = float(np.mean(weights[~bad] == 0)) if (~bad).any() else np.nan
    return recall, false

"""
Error against the truth, held-out anchor error, λ selection, cycle inconsistency, rejection rates.

λ selection is paper one's rule (joint-align scripts/01_multibatch_align.py): RT blocks of 12 bins,
block b in fold b mod 5; an anchor is held out if either of its times falls in the fold and used for
training only if neither does; the score is the median |held-out error|. Neither TIC r nor the
synthetic truth is ever used to choose anything. All of it is alignbench.metrics (anchors, truth)
since the alignbench migration, bit-identical; the World-taking functions delegate via World.study.
"""
from __future__ import annotations

from alignbench.metrics import truth as _truth
from alignbench.metrics.anchors import LAMBDAS, N_FOLDS, RT_BLOCK, anchor_errors, rt_folds, select_lambda  # noqa: F401
from alignbench.metrics.truth import rejection, summarise  # noqa: F401

from trustalign.baselines import MapFn
from trustalign.worlds import World


def inside_mask(world: World, k: int, l: int, t_min):
    """Times in batch k where compounds exist in both batches: t inside batch k's observed compound
    span and its true image inside batch l's (alignbench.metrics.truth.inside_mask)."""
    return _truth.inside_mask(world.study, k, l, t_min)


def map_errors(world: World, g: MapFn, pairs=None):
    """|ĝ_k→l(t) − g*_k→l(t)| in minutes, at every RT bin inside the mask, for every ordered pair
    (or only the ordered pairs given) (alignbench.metrics.truth.map_errors)."""
    return _truth.map_errors(world.study, g, pairs)


def cycle_inconsistency(world: World, g: MapFn) -> tuple[float, float]:
    """Median and max over ordered triplets of the mean cycle error (minutes) inside the k → l mask
    (alignbench.metrics.truth.cycle_inconsistency). Zero by construction for a joint-type model."""
    return _truth.cycle_inconsistency(world.study, g)

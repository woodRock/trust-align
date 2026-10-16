"""
Random K-batch synthetic worlds with known drift (Experiment 1).

Since the alignbench migration the generator is alignbench.datasets.synthetic (draw_specs,
make_world_k; dataset id 'synthetic:k<K>'), which reproduces this module's worlds bit for bit;
World keeps its fields and methods, and World.study gives the same world as an alignbench Study.

Each batch k gets an independent random drift d_k and its own sample count and acquisition start;
the compound library, peak shapes and noise are jointalign.synthetic's (make_world, in memory). The
truth dictionary is make_world's, so jointalign.synthetic.true_map gives the exact correspondence
between any two batches. Batch 0 is the gauge reference for every method; its drift is random too
(maps between batches do not depend on where the latent frame sits).

Drift families:
  'i'   paper-one family (synthetic.BatchSpec: shift + smooth ramp + sinusoidal wiggle).
        Ranges agreed with Jesse (2026-10-14): shift ±3 min, ramp ±3 min, wiggle 0–0.3 min,
        phase uniform on [0, 2π).
  'ii'  out-of-family (piecewise-linear with breakpoints, local bumps): not implemented yet; its
        ranges are still to be decided.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from alignbench.data import Study
from alignbench.datasets import synthetic as syn
from alignbench.datasets.synthetic import FAMILY_I, N_SAMPLES, STARTS, draw_specs  # noqa: F401


@dataclass
class World:
    names: list[str]
    specs: tuple[syn.BatchSpec, ...]
    chromas: list[np.ndarray]     # chromas[k]: (n_k, T, M) float32
    tic_raws: list[np.ndarray]    # tic_raws[k]: (n_k, T) float32
    truth: dict                   # the synthetic truth dict (latent grid, drifts, compound τ)
    seed: int

    @property
    def K(self) -> int:
        return len(self.names)

    @property
    def sizes(self) -> list[int]:
        return [s.n for s in self.specs]

    @cached_property
    def study(self) -> Study:
        """The same world as an alignbench Study (with exact truth), for alignbench's metrics."""
        return syn.to_study(self.specs, self.chromas, self.tic_raws, self.truth, f'synthetic:k{self.K}')

    def true_map(self, k: int, l: int, t_min: np.ndarray) -> np.ndarray:
        """Exact retention time (min) in batch l of whatever elutes at t_min in batch k."""
        return syn.true_map(self.truth, self.names[k], self.names[l], t_min)

    def observed_span(self, k: int) -> tuple[float, float]:
        """Earliest and latest elution time (min) of the compounds batch k actually records:
        inside the run and after its acquisition start (jitter-free positions)."""
        truth: syn.SyntheticTruth = self.study.truth  # type: ignore[assignment]
        return truth.observed_span(k)


def make(K: int, seed: int, family: str = 'i', **world_kw) -> World:
    """One world. Every random draw comes from default_rng(seed): first the batch specs, then the
    seed handed to the synthetic generator (library, compound times, samples)."""
    study = syn.make_world_k(K, seed, family, **world_kw)
    truth: syn.SyntheticTruth = study.truth  # type: ignore[assignment]
    w = World([s.name for s in truth.specs], truth.specs, [b.chromas for b in study.batches],
              [b.tic_raw for b in study.batches], truth.truth, seed)
    w.__dict__['study'] = study  # the cached_property, already known
    return w

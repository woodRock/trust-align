"""
Random K-batch synthetic worlds with known drift (Experiment 1).

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
from typing import Any

import numpy as np
from jointalign import synthetic as syn

FAMILY_I: dict[str, Any] = dict(shift=3.0, ramp=3.0, wiggle=(0.0, 0.3))
N_SAMPLES = (2, 8)            # samples per batch, inclusive
STARTS = (3.0, 5.0)           # acquisition start (min)


@dataclass
class World:
    names: list[str]
    specs: tuple[syn.BatchSpec, ...]
    chromas: list[np.ndarray]     # chromas[k]: (n_k, T, M) float32
    tic_raws: list[np.ndarray]    # tic_raws[k]: (n_k, T) float32
    truth: dict                   # jointalign.synthetic truth (latent grid, drifts, compound τ)
    seed: int

    @property
    def K(self) -> int:
        return len(self.names)

    @property
    def sizes(self) -> list[int]:
        return [s.n for s in self.specs]

    def true_map(self, k: int, l: int, t_min: np.ndarray) -> np.ndarray:
        """Exact retention time (min) in batch l of whatever elutes at t_min in batch k."""
        return syn.true_map(self.truth, self.names[k], self.names[l], t_min)

    def observed_span(self, k: int) -> tuple[float, float]:
        """Earliest and latest elution time (min) of the compounds batch k actually records:
        inside the run and after its acquisition start (jitter-free positions)."""
        tau = np.asarray(self.truth['compound_tau_min'])
        rt = tau + self.specs[k].drift(tau)
        seen = rt[(rt >= self.specs[k].start) & (rt < syn.RUN_MIN)]
        return float(seen.min()), float(seen.max())


def draw_specs(K: int, rng: np.random.Generator, family: str = 'i') -> tuple[syn.BatchSpec, ...]:
    """K random batch specifications from one drift family."""
    if family != 'i':
        raise NotImplementedError(f'drift family {family!r}: only the paper-one family "i" exists yet')
    specs = []
    for k in range(K):
        specs.append(syn.BatchSpec(
            name=f'batch_{k:02d}',
            n=int(rng.integers(N_SAMPLES[0], N_SAMPLES[1] + 1)),
            shift=float(rng.uniform(-FAMILY_I['shift'], FAMILY_I['shift'])),
            ramp=float(rng.uniform(-FAMILY_I['ramp'], FAMILY_I['ramp'])),
            wiggle=float(rng.uniform(*FAMILY_I['wiggle'])),
            phase=float(rng.uniform(0.0, 2 * np.pi)),
            start=float(rng.choice(STARTS)),
        ))
    return tuple(specs)


def make(K: int, seed: int, family: str = 'i', **world_kw) -> World:
    """One world. Every random draw comes from default_rng(seed): first the batch specs, then the
    seed handed to jointalign.synthetic.make_world (library, compound times, samples)."""
    rng = np.random.default_rng(seed)
    specs = draw_specs(K, rng, family)
    chromas, tic_raws, truth = syn.make_world(batches=specs, seed=int(rng.integers(2**31)), **world_kw)
    return World([s.name for s in specs], specs, chromas, tic_raws, truth, seed)

"""
13_full_sparsity.py — Experiment 1 full grid: when does the joint model beat a star fit and composed
pairwise fits, and does its advantage grow as anchors thin, corruption rises and K grows?

Research question (README, Experiment 1): when does joint fitting beat a star design and composed
pairwise fits? The grid and analyses below were fixed (committed) before the full run; every cell is
reported.

Grid:
  main     K ∈ {5, 10} × topology {complete, star, weak_star, chain, ring, density0.4}
           × budget (anchors kept per batch pair) {10, 30, 100} × ε {0, 0.15, 0.30}
  K slice  K = 20 × {complete, chain, density0.4} × budget 30 × ε {0, 0.15}
Seeds: 3 tuning worlds and 20 test worlds per K, disjoint. λ is chosen per setting and method by
the median over the tuning worlds of paper one's RT-blocked held-out anchor error, then used
unchanged on every test world; the test worlds are never used to choose anything, and the truth is
used only to score. A fixed λ = 10 is reported as a sensitivity check.

Solver: jointalign's active-set solver (fit_joint(solver='active_set'), joint-align ≥ 1.1.0): the
same optimum as paper one's BVLS (fish oil results identical; warps within ~1e-12 bins), 20–350×
faster at K = 10–20, falling back to BVLS whenever its KKT check fails.

Methods: Joint; Star through batch 0 (the reference) and Star through the weak hub (fewest samples),
each scored only on the pairs it reaches and compared with the joint model on those same pairs;
Composed pairwise; Composed PCHIP. Every method in a cell sees the same anchors, so comparisons are
paired by test world.

Analyses (the report's sections), in paper one's style: medians over test worlds, paired two-sided
Wilcoxon signed-rank tests with Holm correction, and seed-bootstrap 95 % CIs.
  RQ1a  Well-connected data: complete graph, budget 100 (or every anchor, where a pair has fewer):
        how far is a star through the reference from the joint fit? Median Δ with its CI.
  RQ1b  Weak hub (complete graph, star through the weak hub), weak_star, chain, ring: joint vs star
        and composed pairwise, on pairs without direct anchors; how the gap moves with budget and ε
        (slopes with CIs).
  RQ1c  Many batches: joint − composed pairwise against K (5, 10, 20) at budget 30; density0.4
        keeps the edge density fixed across K.
Also: breakdown rates (median error > 0.3 min) with Wilson intervals, robust-reweighting recall,
cycle inconsistency, the complete-graph sanity check and the fixed-λ sensitivity.

Runs in parallel (one task = one world × one topology) and checkpoints every task under
results/full_sparsity_cache/, so an interrupted run resumes where it stopped. --timing runs a small
sample of cells instead and projects the full grid's run time.

Output: results/full_sparsity.txt, results/full_sparsity.npz (not in git); with --timing,
results/full_sparsity_timing.txt.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')        # one BLAS thread per worker: the workers share the cores

import argparse
import pickle
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from itertools import permutations, product
from pathlib import Path

import numpy as np
from jointalign import harvest_anchors
from scipy.stats import wilcoxon

from trustalign import baselines as bl
from trustalign import graphs, metrics, worlds
from trustalign.provenance import header

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'

MAIN = dict(Ks=(5, 10), topologies=('complete', 'star', 'weak_star', 'chain', 'ring', 'density0.4'),
            budgets=(10, 30, 100), epsilons=(0.0, 0.15, 0.30))
SLICE = dict(Ks=(20,), topologies=('complete', 'chain', 'density0.4'), budgets=(30,), epsilons=(0.0, 0.15))
SOLVER = 'active_set'
TUNE_SEEDS = (0, 1, 2)
TEST_SEEDS = tuple(range(3, 23))
FIXED_LAM = 10.0
TOL_MIN = 0.15            # paper one's tolerance (sanity check)
BREAK_MIN = 0.3           # a test world is "broken" for a method if its median map error exceeds this
ALPHA = 0.05
N_BOOT = 2000
METHODS = ('Joint', 'Star', 'Star (weak hub)', 'Composed pairwise', 'Composed PCHIP')
STARS = {'Star': 'Joint on star pairs', 'Star (weak hub)': 'Joint on weak-hub pairs'}


def settings(grid: dict) -> list[tuple]:
    return list(product(grid['Ks'], grid['topologies'], grid['budgets'], grid['epsilons']))


def full_grid() -> list[tuple]:
    return settings(MAIN) + settings(SLICE)


def world_seed(K: int, seed: int) -> int:
    return 130_000 + 1000 * K + seed          # disjoint from the pilot's and the draft run's worlds


def cell_rng(K: int, seed: int, topology: str, budget: int, eps: float) -> np.random.Generator:
    """The topology, thinning and corruption draws of one cell, independent of every other cell."""
    return np.random.default_rng([13, K, seed, graphs.TOPOLOGIES.index(topology), budget, round(eps * 1000)])


@lru_cache(maxsize=2)
def world_and_anchors(K: int, seed: int, world_kw: tuple = ()):
    """One world and every cross-batch anchor (cached per worker: consecutive tasks often share it)."""
    t0 = time.perf_counter()
    world = worlds.make(K, world_seed(K, seed), **dict(world_kw))
    anchors = harvest_anchors(world.chromas, world.tic_raws)
    return world, anchors, time.perf_counter() - t0


def prepare(world, anchors, setting, seed):
    """The kept, corrupted anchors of one cell, its edges, and each star's hub and reachable pairs."""
    K, topology, budget, eps = setting
    rng = cell_rng(K, seed, topology, budget, eps)
    edges = graphs.edges_for(topology, K, world.sizes, rng)
    kept = anchors.subset(graphs.thin(anchors, edges, budget, rng))
    kept, bad = graphs.corrupt(kept, eps, rng)
    hubs = {'Star': 0, 'Star (weak hub)': graphs.weak_hub(world.sizes)}
    reach = {}
    for m, h in hubs.items():
        r = {h} | {b for a, b in edges if a == h} | {a for a, b in edges if b == h}
        reach[m] = list(permutations(sorted(r), 2))
    return edges, kept, bad, hubs, reach


def fitters(K, edges, hubs):
    fit = bl.fitters(K, edges, hubs['Star'], SOLVER)
    fit['Star (weak hub)'] = lambda anc, lam: bl.star(anc, K, lam, hubs['Star (weak hub)'], SOLVER)
    return {m: fit[m] for m in METHODS}


def tune_cell(world, anchors, setting, seed, lambdas) -> dict:
    """Every λ's held-out anchor score, per λ-method, on one tuning world."""
    edges, kept, _, hubs, _ = prepare(world, anchors, setting, seed)
    fit = fitters(world.K, edges, hubs)
    t0 = time.perf_counter()
    scores = {m: metrics.select_lambda(fit[m], kept, lambdas)[1] for m in METHODS if m not in bl.LAMBDA_FREE}
    return {'scores': scores, 'seconds': time.perf_counter() - t0, 'n_anchors': len(kept), 'n_edges': len(edges),
            'pair_counts': pair_counts(kept, edges)}


def pair_counts(anchors, edges) -> np.ndarray:
    return np.array([int(np.sum((anchors.ba == a) & (anchors.bb == b))) for a, b in sorted(edges)])


def test_cell(world, anchors, setting, seed, lams: dict[str, float]) -> dict:
    """Fit and score every method on one test world at the tuned λ (and at the fixed λ)."""
    t_start = time.perf_counter()
    edges, kept, bad, hubs, reach = prepare(world, anchors, setting, seed)
    fit = fitters(world.K, edges, hubs)
    out: dict = {'n_anchors': len(kept), 'n_edges': len(edges), 'weak_hub': hubs['Star (weak hub)'],
                 'coverage': {m: len(p) / (world.K * (world.K - 1)) for m, p in reach.items()}}
    for name in METHODS:
        variants = [('cv', lams[name])] + ([] if name in bl.LAMBDA_FREE else [('fixed', FIXED_LAM)])
        for tag, lam in variants:
            t0 = time.perf_counter()
            f = fit[name](kept, lam)
            secs = time.perf_counter() - t0
            errs = metrics.map_errors(world, f.map, reach.get(name))
            row = {'lam': lam, 'seconds': secs, **metrics.summarise(errs, edges),
                   'max_all': float(max(e.max() for e in errs.values() if len(e)))}
            if f.model is not None and f.used is not None:
                row['recall'], row['false_reject'] = metrics.rejection(f.model.weights, bad[f.used])
            if name.startswith('Composed') and tag == 'cv':
                row['cycle_median'], row['cycle_max'] = metrics.cycle_inconsistency(world, f.map)
            if name == 'Joint':
                for star, key in STARS.items():
                    out[(key, tag)] = metrics.summarise(metrics.map_errors(world, f.map, reach[star]), edges)
            out[(name, tag)] = row
    out['seconds'] = time.perf_counter() - t_start
    return out


def run_task(phase: str, K: int, seed: int, topology: str, cells: list[tuple], lambdas, lams_by, world_kw):
    """One world × one topology: every (budget, ε) cell of it, tuning or testing."""
    world, anchors, build = world_and_anchors(K, seed, world_kw)
    out = {}
    for setting in cells:
        if phase == 'tune':
            out[setting] = tune_cell(world, anchors, setting, seed, lambdas)
        else:
            out[setting] = test_cell(world, anchors, setting, seed, lams_by[setting])
    return {'phase': phase, 'K': K, 'seed': seed, 'topology': topology, 'cells': out, 'build_seconds': build,
            'n_harvested': len(anchors), 'maxrss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}


def tasks_for(grid: list[tuple], seeds) -> list[tuple]:
    """(K, seed, topology, [settings]) tasks, the most expensive (largest K) first."""
    by: dict[tuple, list] = {}
    for s in grid:
        for seed in seeds:
            by.setdefault((s[0], seed, s[1]), []).append(s)
    return sorted(((K, seed, t, cells) for (K, seed, t), cells in by.items()), key=lambda x: (-x[0], x[1], x[2]))


def run_phase(phase, tasks, workers, cache: Path | None, lambdas, lams_by, world_kw) -> list[dict]:
    """Run every task, reusing checkpointed results; each finished task is checkpointed at once."""
    done, todo = [], []
    for K, seed, topo, cells in tasks:
        f = cache / f'{phase}_K{K}_s{seed}_{topo}.pkl' if cache else None
        if f is not None and f.exists():
            r = pickle.loads(f.read_bytes())
            if set(r['cells']) == set(cells):
                done.append(r)
                continue
        todo.append((K, seed, topo, cells, f))
    if done:
        print(f'{phase}: {len(done)} tasks reused from {cache}', flush=True)
    t0 = time.perf_counter()

    def finish(r, f):
        if f is not None:
            f.write_bytes(pickle.dumps(r))
        done.append(r)
        n = len(done)
        print(f'{phase} K={r["K"]} seed={r["seed"]} {r["topology"]}: '
              f'{sum(c["seconds"] for c in r["cells"].values()):.0f} s ({n}/{len(tasks)}, '
              f'{time.perf_counter() - t0:.0f} s elapsed)', flush=True)

    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(run_task, phase, K, s, t, c, lambdas, lams_by, world_kw): f
                    for K, s, t, c, f in todo}
            for fut in as_completed(futs):
                finish(fut.result(), futs[fut])
    else:
        for K, s, t, c, f in todo:
            finish(run_task(phase, K, s, t, c, lambdas, lams_by, world_kw), f)
    return done


def choose_lambdas(tuned: list[dict], grid: list[tuple]) -> tuple[dict, dict]:
    """Per setting and λ-method: the λ with the lowest median tuning score (ties → smaller λ)."""
    scores: dict = {}
    for r in tuned:
        for setting, c in r['cells'].items():
            for m, sc in c['scores'].items():
                scores.setdefault((setting, m), []).append(sc)
    lams_by, med_scores = {}, {}
    for setting in grid:
        lams_by[setting] = dict.fromkeys(bl.LAMBDA_FREE, np.nan)
        for m in METHODS:
            if m in bl.LAMBDA_FREE:
                continue
            per = scores[(setting, m)]
            med = {lam: float(np.median([sc[lam] for sc in per])) for lam in per[0]}
            lams_by[setting][m] = min(sorted(med), key=lambda lam: med[lam])
            med_scores[(setting, m)] = med
    return lams_by, med_scores


def main(grid: list[tuple] | None = None, tune_seeds=TUNE_SEEDS, test_seeds=TEST_SEEDS, workers: int = 6,
         results: Path = RESULTS, lambdas=metrics.LAMBDAS, world_kw: dict | None = None,
         cache: Path | None = None, n_boot: int = N_BOOT) -> dict:
    grid = full_grid() if grid is None else grid
    cache = results / 'full_sparsity_cache' if cache is None else cache
    cache.mkdir(parents=True, exist_ok=True)
    wkw = tuple(sorted((world_kw or {}).items()))
    t0 = time.perf_counter()
    tuned = run_phase('tune', tasks_for(grid, tune_seeds), workers, cache, lambdas, None, wkw)
    lams_by, med_scores = choose_lambdas(tuned, grid)
    tested = run_phase('test', tasks_for(grid, test_seeds), workers, cache, lambdas, lams_by, wkw)
    wall = time.perf_counter() - t0
    cells = {(*setting, r['seed']): c for r in tested for setting, c in r['cells'].items()}
    lines = report(cells, grid, sorted(test_seeds), lams_by, tuned, tested, wall, workers, n_boot)
    results.mkdir(parents=True, exist_ok=True)
    (results / 'full_sparsity.txt').write_text('\n'.join(lines) + '\n')
    np.savez(results / 'full_sparsity.npz', **flatten(cells))
    print('\n'.join(lines))
    return {'cells': cells, 'lams': lams_by, 'scores': med_scores}


# ── statistics ────────────────────────────────────────────────────────────────

def stat(cells, setting, seeds, method, key='median_all', tag='cv') -> np.ndarray:
    return np.array([cells[(*setting, s)][(method, tag)][key] for s in seeds])


def signed_rank(d: np.ndarray, alternative: str = 'two-sided') -> float:
    """Wilcoxon signed-rank p for the median of d differing from zero (nan if all zero)."""
    d = d[np.isfinite(d)]
    if not len(d) or not np.any(d != 0):
        return np.nan
    return float(wilcoxon(d, alternative=alternative).pvalue)


def median_ci(d: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Median of d and its 95 % CI from resampling test worlds with replacement."""
    d = d[np.isfinite(d)]
    if not len(d):
        return np.nan, np.nan, np.nan
    boots = np.median(d[rng.integers(0, len(d), (n_boot, len(d)))], axis=1)
    return float(np.median(d)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def holm(ps: list[float]) -> list[float]:
    """Holm-adjusted p-values (nan stays nan and is not counted)."""
    idx = [i for i, p in enumerate(ps) if np.isfinite(p)]
    order = sorted(idx, key=lambda i: ps[i])
    adj = [np.nan] * len(ps)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * ps[i]))
        adj[i] = running
    return adj


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def slope_ci(groups: dict, n_boot: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares slopes of y on the columns of x, and their 95 % CIs from resampling test worlds
    with replacement within each group. groups: {group: (x rows (n, p), y (n,))}."""
    def fit(xs, ys):
        X = np.column_stack([np.ones(len(ys)), np.vstack(xs)])
        ok = np.isfinite(ys)
        return np.linalg.lstsq(X[ok], ys[ok], rcond=None)[0][1:]
    xs = [g[0] for g in groups.values()]
    ys = [g[1] for g in groups.values()]
    est = fit(xs, np.concatenate(ys))
    boots = []
    for _ in range(n_boot):
        picks = [rng.integers(0, len(y), len(y)) for y in ys]
        boots.append(fit([x[p] for x, p in zip(xs, picks)], np.concatenate([y[p] for y, p in zip(ys, picks)])))
    b = np.array(boots)
    return est, np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)


# ── report ────────────────────────────────────────────────────────────────────

def flatten(cells: dict) -> dict[str, np.ndarray]:
    """One array per (method, λ variant, metric): rows = cells in key order, plus the cell keys."""
    keys = sorted(cells, key=str)
    out: dict[str, np.ndarray] = {'cell_keys': np.array([str(k) for k in keys])}
    for name, tag in product((*METHODS, *STARS.values()), ('cv', 'fixed')):
        rows = [cells[k].get((name, tag)) for k in keys]
        if all(r is None for r in rows):
            continue
        for metric in sorted({m for r in rows if r for m in r}):
            out[f'{name}|{tag}|{metric}'] = np.array([r.get(metric, np.nan) if r else np.nan for r in rows])
    out['n_anchors'] = np.array([cells[k]['n_anchors'] for k in keys])
    out['n_edges'] = np.array([cells[k]['n_edges'] for k in keys])
    return out


def fmt_p(p: float) -> str:
    return '—' if not np.isfinite(p) else f'{p:.3f}' if p >= 0.001 else '<0.001'


def h2_rows(grid) -> list[tuple]:
    """(setting, comparison label, star method or None): RQ1b's comparisons."""
    rows = []
    for s in grid:
        topo = s[1]
        if topo == 'complete':
            rows.append((s, 'vs Star (weak hub), its pairs', 'Star (weak hub)'))
        if topo in ('weak_star', 'chain', 'ring'):
            star = 'Star (weak hub)' if topo == 'weak_star' else 'Star'
            rows.append((s, f'vs {star}, indirect pairs', star))
            rows.append((s, 'vs composed pairwise, indirect', None))
    return rows


def h2_diff(cells, setting, seeds, star) -> np.ndarray:
    """Per test world: joint minus competitor (min) for one RQ1b comparison."""
    if star is None:
        return (stat(cells, setting, seeds, 'Joint', 'median_indirect')
                - stat(cells, setting, seeds, 'Composed pairwise', 'median_indirect'))
    key = 'median_all' if setting[1] == 'complete' else 'median_indirect'
    return stat(cells, setting, seeds, STARS[star], key) - stat(cells, setting, seeds, star, key)


def report(cells, grid, seeds, lams_by, tuned, tested, wall, workers, n_boot) -> list[str]:
    n = len(seeds)
    rng = np.random.default_rng(13)
    Ks = sorted({s[0] for s in grid})
    L = [__doc__.strip().splitlines()[0], '', *header(), '',
         f'{len(grid)} settings; tuning worlds {sorted({r["seed"] for r in tuned})}, '
         f'test worlds {seeds[0]}–{seeds[-1]} ({n}); '
         f'drift family (i): shift ±{worlds.FAMILY_I["shift"]} min, ramp ±{worlds.FAMILY_I["ramp"]} min, wiggle '
         f'{worlds.FAMILY_I["wiggle"][0]}–{worlds.FAMILY_I["wiggle"][1]} min; samples per batch '
         f'{worlds.N_SAMPLES[0]}–{worlds.N_SAMPLES[1]}.',
         'Map error = |ĝ − g*| (min) over every RT bin where compounds exist in both batches. Per test world: the '
         'median over pairs and times; tables give the median over test worlds. "indirect" = pairs without '
         'kept anchors between them. Stars are scored on the pairs they reach. '
         f'Solver: fit_joint(solver={SOLVER!r}).', '']

    L += ['=' * 124, '1. Median map error (min), all pairs, λ tuned on the tuning worlds (λ in brackets)',
          f'   {"K":>2} {"topology":<10} {"budget":>6} {"ε":>5} ' + ' '.join(f'{m:>19}' for m in METHODS)]
    for s in grid:
        vals = []
        for m in METHODS:
            v = float(np.nanmedian(stat(cells, s, seeds, m)))
            lam = lams_by[s][m]
            vals.append(f'{v:.3f}' + ('' if np.isnan(lam) else f' [{lam:g}]'))
        L.append(f'   {s[0]:>2} {s[1]:<10} {s[2]:>6} {s[3]:>5} ' + ' '.join(f'{v:>19}' for v in vals))
    L += ['', '=' * 124, '2. Pairs without direct anchors: median map error (min)',
          f'   {"K":>2} {"topology":<10} {"budget":>6} {"ε":>5} '
          + ' '.join(f'{m:>19}' for m in ('Joint', 'Composed pairwise', 'Composed PCHIP'))]
    for s in grid:
        if s[1] == 'complete':
            continue
        vals = [float(np.nanmedian(stat(cells, s, seeds, m, 'median_indirect')))
                for m in ('Joint', 'Composed pairwise', 'Composed PCHIP')]
        L.append(f'   {s[0]:>2} {s[1]:<10} {s[2]:>6} {s[3]:>5} ' + ' '.join(f'{v:>19.3f}' for v in vals))

    L += ['', '=' * 124, '3. RQ1a — well-connected data: complete graph, budget 100 (every anchor where a '
          'pair has fewer). Δ = joint − star through the reference (min) per test world; median [95 % CI], range, '
          'two-sided Wilcoxon p; anchors per batch pair on the tuning worlds (min / median)',
          f'   {"K":>2} {"ε":>5} {"median Δ [95% CI]":>28} {"[min Δ, max Δ]":>20} {"anchors per pair":>17} '
          f'{"p":>7}']
    for s in grid:
        if s[1] != 'complete' or s[2] != 100:
            continue
        d = stat(cells, s, seeds, STARS['Star']) - stat(cells, s, seeds, 'Star')
        med, lo, hi = median_ci(d, n_boot, rng)
        pc = np.concatenate([c['pair_counts'] for r in tuned for st, c in r['cells'].items() if st == s])
        L.append(f'   {s[0]:>2} {s[3]:>5} {f"{med:+.4f} [{lo:+.4f}, {hi:+.4f}]":>28} '
                 f'{f"[{np.nanmin(d):+.4f}, {np.nanmax(d):+.4f}]":>20} '
                 f'{f"{pc.min()} / {int(np.median(pc))}":>17} {fmt_p(signed_rank(d)):>7}')

    rows = h2_rows(grid)
    diffs = [h2_diff(cells, s, seeds, star) for s, _, star in rows]
    ps = [signed_rank(d) for d in diffs]
    adj = holm(ps)
    L += ['', '=' * 124, f'4. RQ1b — weak hub, chain, ring. Δ = joint − competitor (min) per test world; '
          f'two-sided Wilcoxon, Holm over all {len(rows)} comparisons; wins = test worlds where joint is lower',
          f'   {"K":>2} {"topology":<10} {"budget":>6} {"ε":>5} {"comparison":<32} {"median Δ":>9} '
          f'{"wins":>6} {"p":>7} {"Holm p":>7}']
    for (s, label, _), d, p, pa in zip(rows, diffs, ps, adj):
        ok = d[np.isfinite(d)]
        med = np.median(ok) if len(ok) else np.nan
        L.append(f'   {s[0]:>2} {s[1]:<10} {s[2]:>6} {s[3]:>5} {label:<32} {med:>+9.4f} '
                 f'{int(np.sum(ok < 0)):>3}/{len(ok):<2} {fmt_p(p):>7} {fmt_p(pa):>7}'
                 + ('  *' if np.isfinite(pa) and pa < ALPHA else ''))
    L += ['', '   RQ1b trend: Δ regressed on log10(budget) and ε within each (K, topology, comparison). Δ < 0 '
          'favours joint, so "the gap grows as the budget falls" = positive log-budget slope and "grows as ε '
          'rises" = negative ε slope. 95 % CIs from resampling test worlds.',
          f'   {"K":>2} {"topology":<10} {"comparison":<32} {"slope log10 budget [95% CI]":>32} '
          f'{"slope ε [95% CI]":>32}']
    trend: dict = {}
    for (s, label, _), d in zip(rows, diffs):
        trend.setdefault((s[0], s[1], label), {})[(s[2], s[3])] = d
    for (K, topo, label), by in trend.items():
        if len({b for b, _ in by}) < 2 and len({e for _, e in by}) < 2:
            continue
        groups = {k: (np.tile([np.log10(k[0]), k[1]], (len(v), 1)), v) for k, v in by.items()}
        est, lo, hi = slope_ci(groups, n_boot, rng)
        L.append(f'   {K:>2} {topo:<10} {label:<32} {f"{est[0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}]":>32} '
                 f'{f"{est[1]:+.4f} [{lo[1]:+.4f}, {hi[1]:+.4f}]":>32}')

    L += ['', '=' * 124, '5. RQ1c — many batches: joint − composed pairwise (median map error, all pairs) vs K '
          'at budget 30; slope in log2 K with a 95 % CI from resampling test worlds within each K (Δ < 0 '
          'favours joint, so a growing advantage = negative slope)',
          f'   {"topology":<10} {"ε":>5} ' + ' '.join(f'{"K=" + str(K):>9}' for K in Ks)
          + f' {"slope [95% CI]":>30}']
    for topo, eps in product(SLICE['topologies'], SLICE['epsilons']):
        sets = [(K, topo, 30, eps) for K in Ks if (K, topo, 30, eps) in grid]
        if len(sets) < 2:
            continue
        ds = {s: stat(cells, s, seeds, 'Joint') - stat(cells, s, seeds, 'Composed pairwise') for s in sets}
        groups = {s: (np.full((len(d), 1), np.log2(s[0])), d) for s, d in ds.items()}
        est, lo, hi = slope_ci(groups, n_boot, rng)
        med = {s[0]: float(np.median(d)) for s, d in ds.items()}
        L.append(f'   {topo:<10} {eps:>5} ' + ' '.join(f'{med[K]:>+9.4f}' if K in med else f'{"—":>9}' for K in Ks)
                 + f' {f"{est[0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}]":>30}')
    L.append('   (density0.4 keeps the edge density fixed across K; the chain\'s density falls as 2/K.)')

    L += ['', '=' * 124, f'6. Breakdown: share of test worlds with median map error > {BREAK_MIN} min '
          '[Wilson 95 % CI], ε > 0',
          f'   {"K":>2} {"topology":<10} {"budget":>6} {"ε":>5} {"Joint":>18} {"Composed pairwise":>18}']
    for s in grid:
        if s[3] == 0:
            continue
        out = []
        for m in ('Joint', 'Composed pairwise'):
            v = stat(cells, s, seeds, m)
            k = int(np.sum(v > BREAK_MIN))
            lo, hi = wilson(k, n)
            out.append(f'{k / n:.2f} [{lo:.2f}, {hi:.2f}]')
        L.append(f'   {s[0]:>2} {s[1]:<10} {s[2]:>6} {s[3]:>5} {out[0]:>18} {out[1]:>18}')

    L += ['', '=' * 124, '7. Robust reweighting, ε > 0 (median over test worlds): share of corrupted anchors '
          'given weight 0 (recall) / clean anchors given weight 0 (false)',
          f'   {"K":>2} {"topology":<10} {"budget":>6} {"ε":>5} {"Joint recall":>13} {"Joint false":>12} '
          f'{"Star recall":>12} {"Star false":>11}']
    for s in grid:
        if s[3] == 0:
            continue
        v = [float(np.nanmedian(stat(cells, s, seeds, m, k))) for m in ('Joint', 'Star')
             for k in ('recall', 'false_reject')]
        L.append(f'   {s[0]:>2} {s[1]:<10} {s[2]:>6} {s[3]:>5} {v[0]:>13.3f} {v[1]:>12.3f} {v[2]:>12.3f} '
                 f'{v[3]:>11.3f}')

    L += ['', '=' * 124, '8. Cycle inconsistency of the composed maps (median over test worlds of the median over '
          'ordered triplets, min); joint-type models are 0 by construction',
          f'   {"K":>2} {"topology":<10} {"budget":>6} {"ε":>5} {"pairwise":>9} {"PCHIP":>9}']
    for s in grid:
        v = [float(np.nanmedian(stat(cells, s, seeds, m, 'cycle_median')))
             for m in ('Composed pairwise', 'Composed PCHIP')]
        L.append(f'   {s[0]:>2} {s[1]:<10} {s[2]:>6} {s[3]:>5} {v[0]:>9.4f} {v[1]:>9.4f}')

    L += ['', '=' * 124, f'9. Sanity check: complete graph, ε = 0, joint model, max |error| ≤ {TOL_MIN} min at '
          'every scored point (share of test worlds passing)']
    for s in grid:
        if s[1] == 'complete' and s[3] == 0:
            v = stat(cells, s, seeds, 'Joint', 'max_all')
            L.append(f'   K={s[0]:>2} budget={s[2]:>3}: {int(np.sum(v <= TOL_MIN))}/{n} pass; median max '
                     f'{np.median(v):.3f} min, worst {v.max():.3f} min')

    lam_methods = [m for m in METHODS if m not in bl.LAMBDA_FREE]
    L += ['', '=' * 124, f'10. Sensitivity: fixed λ = {FIXED_LAM:g} (median map error, all pairs)',
          f'   {"K":>2} {"topology":<10} {"budget":>6} {"ε":>5} ' + ' '.join(f'{m:>19}' for m in lam_methods)]
    for s in grid:
        vals = [float(np.nanmedian(stat(cells, s, seeds, m, tag='fixed'))) for m in lam_methods]
        L.append(f'   {s[0]:>2} {s[1]:<10} {s[2]:>6} {s[3]:>5} ' + ' '.join(f'{v:>19.3f}' for v in vals))

    L += ['', '=' * 124, '11. Run time']
    for K in Ks:
        tu = [c['seconds'] for r in tuned if r['K'] == K for c in r['cells'].values()]
        te = [c['seconds'] for r in tested if r['K'] == K for c in r['cells'].values()]
        b = [r['build_seconds'] for r in (*tuned, *tested) if r['K'] == K]
        L.append(f'   K={K:>2}: world + harvest {np.mean(b):.1f} s; tuning cell {np.mean(tu):.1f} s; test cell '
                 f'{np.mean(te):.1f} s (means)')
    rss = max(r['maxrss_mb'] for r in (*tuned, *tested))
    L += [f'   {workers} workers; peak worker memory {rss:.0f} MB; wall time {wall:.0f} s', '']
    return L


# ── timing sample ─────────────────────────────────────────────────────────────

def timing_probes() -> list[tuple]:
    """The chain and the complete graph (fewest and most edges) at every (K, budget) of the full grid,
    plus density0.4 at K = 20 as a check on the interpolation; all at ε = 0.15."""
    kb = sorted({(s[0], s[2]) for s in full_grid()})
    probes = [(K, t, b, 0.15) for K, b in kb for t in ('complete', 'chain')]
    return probes + [(20, 'density0.4', 30, 0.15)]


def n_edges_of(K: int, topology: str) -> int:
    if topology == 'complete':
        return K * (K - 1) // 2
    if topology in ('star', 'weak_star', 'chain'):
        return K - 1
    if topology == 'ring':
        return K
    if topology.startswith('density'):
        return max(round(float(topology[7:]) * K * (K - 1) / 2), K - 1)
    raise ValueError(topology)


def timing_main(workers: int = 6, results: Path = RESULTS, lambdas=metrics.LAMBDAS,
                world_kw: dict | None = None) -> list[str]:
    """Time one tuning cell and one test cell per probe and project the full grid's cost."""
    probes = timing_probes()
    wkw = tuple(sorted((world_kw or {}).items()))
    t0 = time.perf_counter()
    tuned = run_phase('tune', tasks_for(probes, TUNE_SEEDS[:1]), workers, None, lambdas, None, wkw)
    lams_by, _ = choose_lambdas(tuned, probes)
    tested = run_phase('test', tasks_for(probes, TEST_SEEDS[:1]), workers, None, lambdas, lams_by, wkw)
    wall = time.perf_counter() - t0
    tu = {s: c for r in tuned for s, c in r['cells'].items()}
    te = {s: c for r in tested for s, c in r['cells'].items()}
    build = {}
    for r in (*tuned, *tested):
        build.setdefault(r['K'], []).append(r['build_seconds'])
    harvested = {r['K']: r['n_harvested'] for r in tuned}
    rss = max(r['maxrss_mb'] for r in (*tuned, *tested))

    def cost(K, topology, budget, which):
        """Seconds per cell, linear in the number of edges between the chain and the complete graph."""
        lo, hi = (K, 'chain', budget, 0.15), (K, 'complete', budget, 0.15)
        src = tu if which == 'tune' else te
        e0, e1 = K - 1, K * (K - 1) // 2
        frac = (n_edges_of(K, topology) - e0) / (e1 - e0)
        return src[lo]['seconds'] + (src[hi]['seconds'] - src[lo]['seconds']) * frac

    L = ['13_full_sparsity.py --timing — run-time sample for the Experiment 1 full grid', '', *header(), '',
         f'One tuning world (seed {TUNE_SEEDS[0]}) and one test world (seed {TEST_SEEDS[0]}) per probe, ε = 0.15; '
         f'{workers} workers; wall time {wall:.0f} s; peak worker memory {rss:.0f} MB.', '',
         '1. Measured',
         f'   {"K":>2} {"topology":<10} {"budget":>6} {"edges":>5} {"anchors":>8} {"min/median per pair":>20} '
         f'{"tune s":>8} {"test s":>8}']
    for s in probes:
        pc = tu[s]['pair_counts']
        L.append(f'   {s[0]:>2} {s[1]:<10} {s[2]:>6} {tu[s]["n_edges"]:>5} {tu[s]["n_anchors"]:>8} '
                 f'{f"{pc.min()} / {int(np.median(pc))}":>20} {tu[s]["seconds"]:>8.1f} {te[s]["seconds"]:>8.1f}')
    L += ['', '   World + harvest per world: ' + '; '.join(
        f'K={K}: {np.mean(v):.1f} s ({harvested[K]:,} anchors harvested)' for K, v in sorted(build.items()))]
    s = (20, 'density0.4', 30, 0.15)
    L += [f'   Interpolation check, K=20 density0.4: measured tune {tu[s]["seconds"]:.0f} s / test '
          f'{te[s]["seconds"]:.0f} s, interpolated {cost(*s[:3], "tune"):.0f} s / '
          f'{cost(*s[:3], "test"):.0f} s', '']

    L += ['2. Projection for the full grid '
          f'({len(TUNE_SEEDS)} tuning + {len(TEST_SEEDS)} test worlds per setting)',
          f'   {"part":<26} {"settings":>8} {"tasks":>6} {"CPU h":>7} {"longest task s":>15}']
    total, longest = 0.0, 0.0
    for name, g in (('main grid, K = 5', [x for x in settings(MAIN) if x[0] == 5]),
                    ('main grid, K = 10', [x for x in settings(MAIN) if x[0] == 10]),
                    ('K slice, K = 20', settings(SLICE))):
        K = g[0][0]
        cpu, tasks, worst = 0.0, 0, 0.0
        for phase, seeds in (('tune', TUNE_SEEDS), ('test', TEST_SEEDS)):
            for _, _, topo, cells in tasks_for(g, seeds):
                secs = np.mean(build[K]) + sum(cost(K, topo, b, phase) for _, _, b, _ in cells)
                cpu += secs
                tasks += 1
                worst = max(worst, secs)
        total += cpu
        longest = max(longest, worst)
        L.append(f'   {name:<26} {len(g):>8} {tasks:>6} {cpu / 3600:>7.2f} {worst:>15.0f}')
    est = max(total / workers, longest)
    L += [f'   total {total / 3600:.1f} CPU h → about {est / 3600:.1f} h of wall time on {workers} workers '
          f'(CPU h / workers; the tuning phase must finish before testing starts, so add the tail of its '
          f'longest task, {longest / 60:.0f} min at most).', '']
    results.mkdir(parents=True, exist_ok=True)
    (results / 'full_sparsity_timing.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return L


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--timing', action='store_true', help='time a sample of cells and project the full run')
    ap.add_argument('--results', type=Path, default=RESULTS)
    ap.add_argument('--device', default='cpu', choices=['cpu'], help='CPU only (bit-reproducible)')
    a = ap.parse_args()
    if a.timing:
        timing_main(workers=a.workers, results=a.results)
    else:
        main(workers=a.workers, results=a.results)

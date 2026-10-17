"""
03_diagnose_lambda.py — How much of Experiment 1's breakdown comes from choosing λ?

Diagnostic, not an experiment. In the full grid (scripts/02_full_sparsity.py) the joint model broke
down on sparse, corrupted graphs, and its λ (tuned on 3 separate worlds) sometimes did far worse than
the fixed λ = 10 of the sensitivity check. This script separates the λ rule from the model:

  1. From script 02's checkpoints (no fits): tuned vs fixed λ per setting, and the tuning worlds' CV
     curves — which λ each tuning world prefers and how flat the curve is.
  2. Refits of script 02's exact test cells (same worlds, same anchors) at every λ of the grid, for
     the joint model and composed pairwise, plus paper one's rule applied to each test world itself
     (λ by RT-blocked CV on that world's own anchors, as a user would). Compared rules:
        tuned      script 02's λ (median CV score over the tuning worlds)
        λ = 10     the fixed sensitivity λ
        own CV     CV on the test world's own anchors (joint only)
        best λ     the single λ with the lowest median error over the setting's test worlds
        per world  the best λ for each test world separately
     "best λ" and "per world" use the truth, so they are bounds on what any λ rule could reach, never
     rules: if the joint model breaks down even per world, the failure is the model's, not the λ's.

Settings: ε ∈ {0, 0.15, 0.30} on weak_star, chain, ring and density0.4 at K = 5 and 10 (--all: every
setting of the full grid). Tasks are checkpointed under results/diagnose_lambda_cache/.

Output: results/diagnose_lambda.txt
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')        # one BLAS thread per worker: the workers share the cores

import argparse
import importlib.util
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from trustalign import metrics
from trustalign.provenance import header

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'

SUBSET = dict(Ks=(5, 10), topologies=('weak_star', 'chain', 'ring', 'density0.4'), budgets=(10, 30, 100),
              epsilons=(0.0, 0.15, 0.30))
METHODS = ('Joint', 'Composed pairwise')
RULES = ('tuned', 'λ = 10', 'own CV', 'best λ', 'per world')


def _script(name: str):
    spec = importlib.util.spec_from_file_location(Path(name).stem, ROOT / 'scripts' / name)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_S13 = None


def s13():
    """Script 02, loaded once per process."""
    global _S13
    if _S13 is None:
        _S13 = _script('02_full_sparsity.py')
    return _S13


def sweep_cell(world, anchors, setting, seed, lambdas) -> dict:
    """Every λ's map error for the joint model and composed pairwise on one test cell, and the λ that
    CV on the cell's own anchors picks for the joint model."""
    m = s13()
    t0 = time.perf_counter()
    edges, kept, _, hubs, _ = m.prepare(world, anchors, setting, seed)
    fit = m.fitters(world.K, edges, hubs)
    out: dict = {}
    for name in METHODS:
        out[name] = {}
        for lam in lambdas:
            s = metrics.summarise(metrics.map_errors(world, fit[name](kept, lam).map), edges)
            out[name][lam] = (s['median_all'], s['median_indirect'])
    out['own_cv'] = metrics.select_lambda(fit['Joint'], kept, lambdas)[0]
    out['seconds'] = time.perf_counter() - t0
    return out


def run_task(K: int, seed: int, topology: str, cells: list[tuple], lambdas, world_kw: tuple):
    world, anchors, _ = s13().world_and_anchors(K, seed, world_kw)
    return {'K': K, 'seed': seed, 'topology': topology,
            'cells': {s: sweep_cell(world, anchors, s, seed, lambdas) for s in cells}}


def run_sweep(grid, seeds, workers, cache: Path, lambdas, world_kw) -> dict:
    """{(setting, seed): sweep} for every cell, reusing checkpointed tasks."""
    tasks = s13().tasks_for(grid, seeds)
    done, todo = [], []
    for K, seed, topo, cells in tasks:
        f = cache / f'sweep_K{K}_s{seed}_{topo}.pkl'
        if f.exists():
            r = pickle.loads(f.read_bytes())
            if set(r['cells']) == set(cells):
                done.append(r)
                continue
        todo.append((K, seed, topo, cells, f))
    t0 = time.perf_counter()

    def finish(r, f):
        f.write_bytes(pickle.dumps(r))
        done.append(r)
        print(f'sweep K={r["K"]} seed={r["seed"]} {r["topology"]} ({len(done)}/{len(tasks)}, '
              f'{time.perf_counter() - t0:.0f} s elapsed)', flush=True)

    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(run_task, K, s, t, c, lambdas, world_kw): f for K, s, t, c, f in todo}
            for fut in as_completed(futs):
                finish(fut.result(), futs[fut])
    else:
        for K, s, t, c, f in todo:
            finish(run_task(K, s, t, c, lambdas, world_kw), f)
    return {(setting, r['seed']): c for r in done for setting, c in r['cells'].items()}


def rule_errors(sweep: dict, setting, seeds, method: str, tuned_lam: float,
                key: int = 0) -> dict[str, np.ndarray]:
    """Per test world, the map error (key 0: all pairs, 1: indirect) under each λ rule."""
    cells = [sweep[(setting, s)][method] for s in seeds]
    lams = sorted(cells[0])
    E = np.array([[c[lam][key] for lam in lams] for c in cells])          # (worlds, λ)
    with np.errstate(all='ignore'):
        best = lams[int(np.nanargmin(np.nanmedian(E, axis=0)))] if np.isfinite(E).any() else lams[0]
    out = {'tuned': E[:, lams.index(tuned_lam)], 'λ = 10': E[:, lams.index(10.0)],
           'best λ': E[:, lams.index(best)],
           'per world': np.array([np.nanmin(r) if np.isfinite(r).any() else np.nan for r in E])}
    if method == 'Joint':
        own = [sweep[(setting, s)] for s in seeds]
        out['own CV'] = np.array([c['Joint'][c['own_cv']][key] for c in own])
    out['_best_lam'] = np.array([best])
    return out


def main(grid: list[tuple] | None = None, workers: int = 6, results: Path = RESULTS,
         lambdas=metrics.LAMBDAS, world_kw: dict | None = None, s13_cache: Path | None = None,
         cache: Path | None = None, tune_seeds=None, test_seeds=None) -> list[str]:
    m = s13()
    tune_seeds = m.TUNE_SEEDS if tune_seeds is None else tune_seeds
    test_seeds = m.TEST_SEEDS if test_seeds is None else test_seeds
    grid = [s for s in m.full_grid() if s in set(m.settings(SUBSET))] if grid is None else grid
    s13_cache = results / 'full_sparsity_cache' if s13_cache is None else s13_cache
    cache = results / 'diagnose_lambda_cache' if cache is None else cache
    cache.mkdir(parents=True, exist_ok=True)
    wkw = tuple(sorted((world_kw or {}).items()))
    t0 = time.perf_counter()
    # Script 02's results for these settings; every task must already be checkpointed (no fits here).
    for phase, seeds in (('tune', tune_seeds), ('test', test_seeds)):
        missing = [t for t in m.tasks_for(grid, seeds)
                   if not (s13_cache / f'{phase}_K{t[0]}_s{t[1]}_{t[2]}.pkl').exists()]
        if missing:
            raise FileNotFoundError(f'{len(missing)} script-13 {phase} checkpoints missing in {s13_cache}; '
                                    'run scripts/02_full_sparsity.py first')
    tuned = m.run_phase('tune', m.tasks_for(grid, tune_seeds), 1, s13_cache, lambdas, None, wkw)
    lams_by, _ = m.choose_lambdas(tuned, grid)
    tested = m.run_phase('test', m.tasks_for(grid, test_seeds), 1, s13_cache, lambdas, lams_by, wkw)
    full = {(*s, r['seed']): c for r in tested for s, c in r['cells'].items()}
    sweep = run_sweep(grid, test_seeds, workers, cache, lambdas, wkw)
    wall = time.perf_counter() - t0
    lines = report(grid, sorted(test_seeds), tuned, lams_by, full, sweep, lambdas, wall, workers)
    results.mkdir(parents=True, exist_ok=True)
    (results / 'diagnose_lambda.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return lines


def report(grid, seeds, tuned, lams_by, full, sweep, lambdas, wall, workers) -> list[str]:
    m = s13()
    n, brk = len(seeds), m.BREAK_MIN
    nb = lambda v: int(np.sum(v > brk))
    L = [__doc__.strip().splitlines()[0], '', *header(), '',
         f'{len(grid)} settings × {n} test worlds of scripts/02_full_sparsity.py (same worlds and anchors); '
         f'λ grid {[f"{x:g}" for x in lambdas]}. Map error: median over pairs and times per test world (min); '
         f'tables give the median over test worlds and, in brackets, the test worlds broken down (> {brk} min). '
         'The "best λ" and "per world" columns use the truth: bounds, not rules.', '']

    # 0. The refits reproduce script 02.
    dev = []
    for s in grid:
        for seed in seeds:
            cell, sw = full[(*s, seed)], sweep[(s, seed)]
            for name in METHODS:
                for tag in ('cv', 'fixed'):
                    lam = cell[(name, tag)]['lam']
                    dev.append(abs(sw[name][lam][0] - cell[(name, tag)]['median_all']))
    L += ['=' * 118, '0. Check: the refits reproduce script 02\'s map errors at its tuned and fixed λ: '
          f'max |difference| {np.nanmax(dev):.2e} min over {len(dev)} fits', '']

    # 1. Tuned vs fixed λ, and the tuning worlds' CV curves (from the checkpoints).
    scores: dict = {}
    for r in tuned:
        for s, c in r['cells'].items():
            scores.setdefault(s, []).append(c['scores']['Joint'])
    L += ['=' * 118, '1. Joint model: tuned λ vs λ = 10 (median error [broken]), and the tuning worlds\' CV '
          'curves: the λ each tuning world prefers, and the curve\'s relative range (max − min) / min of the '
          'median held-out error',
          f'   {"K":>2} {"topology":<10} {"budget":>6} {"ε":>5} {"tuned λ":>8} {"tuned":>12} {"λ = 10":>12} '
          f'{"tuning worlds prefer":>22} {"curve range":>12}']
    worse = 0
    for s in grid:
        tl = lams_by[s]['Joint']
        t = np.array([full[(*s, x)][('Joint', 'cv')]['median_all'] for x in seeds])
        f = np.array([full[(*s, x)][('Joint', 'fixed')]['median_all'] for x in seeds])
        worse += np.median(t) > np.median(f)
        prefs = [min(sorted(sc), key=lambda lam, sc=sc: sc[lam]) for sc in scores[s]]
        rng = np.median([(max(sc.values()) - min(sc.values())) / max(min(sc.values()), 1e-12) for sc in scores[s]])
        L.append(f'   {s[0]:>2} {s[1]:<10} {s[2]:>6} {s[3]:>5} {tl:>8g} {f"{np.median(t):.3f} [{nb(t):>2}]":>12} '
                 f'{f"{np.median(f):.3f} [{nb(f):>2}]":>12} {", ".join(f"{p:g}" for p in prefs):>22} {rng:>12.2f}')
    L.append(f'   tuned λ has the higher median error in {worse} of {len(grid)} settings.')

    # 2. Every λ rule, joint and composed pairwise.
    L += ['', '=' * 118, '2. Median map error, all pairs [test worlds broken], under each λ rule',
          f'   {"K":>2} {"topology":<10} {"budget":>6} {"ε":>5} {"method":<18} '
          + ' '.join(f'{r:>13}' for r in RULES) + f' {"best λ":>7}']
    totals: dict = {}
    for s in grid:
        for name in METHODS:
            e = rule_errors(sweep, s, seeds, name, lams_by[s][name])
            cells = []
            for rule in RULES:
                if rule in e:
                    v = e[rule]
                    cells.append(f'{np.nanmedian(v):.3f} [{nb(v):>2}]')
                    if s[3] > 0:
                        totals[(name, rule)] = totals.get((name, rule), 0) + nb(v)
                else:
                    cells.append('—')
            L.append(f'   {s[0]:>2} {s[1]:<10} {s[2]:>6} {s[3]:>5} {name:<18} '
                     + ' '.join(f'{c:>13}' for c in cells) + f' {e["_best_lam"][0]:>7g}')
        L.append('')

    # 3. Summary over the corrupted settings.
    corrupted = [s for s in grid if s[3] > 0]
    L += ['=' * 118, f'3. Broken test worlds (> {brk} min) summed over the {len(corrupted)} corrupted settings '
          f'({len(corrupted) * n} test worlds)',
          f'   {"method":<18} ' + ' '.join(f'{r:>10}' for r in RULES)]
    for name in METHODS:
        L.append(f'   {name:<18} ' + ' '.join(f'{totals.get((name, r), "—"):>10}' for r in RULES))
    joint_bound = {s: nb(rule_errors(sweep, s, seeds, 'Joint', lams_by[s]['Joint'])['per world'])
                   for s in corrupted}
    still = [s for s, k in joint_bound.items() if k >= 2]
    L += ['', f'   Settings where the joint model still breaks down in ≥ 2 test worlds with the best λ per world: '
          f'{len(still)} of {len(corrupted)}']
    for s in still:
        pw = rule_errors(sweep, s, seeds, 'Composed pairwise', lams_by[s]['Composed pairwise'])['per world']
        L.append(f'     K={s[0]:>2} {s[1]:<10} budget {s[2]:>3} ε {s[3]}: joint {joint_bound[s]}/{n} broken, '
                 f'composed pairwise (its best λ per world) {nb(pw)}/{n}')
    secs = [sweep[(s, x)]['seconds'] for s in grid for x in seeds]
    L += ['', '=' * 118, f'4. Run time: {np.mean(secs):.1f} s per test cell (mean); {workers} workers; '
          f'wall time {wall:.0f} s', '']
    return L


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--all', action='store_true', help='every setting of the full grid, not just the subset')
    ap.add_argument('--results', type=Path, default=RESULTS)
    ap.add_argument('--device', default='cpu', choices=['cpu'], help='CPU only (bit-reproducible)')
    a = ap.parse_args()
    main(grid=s13().full_grid() if a.all else None, workers=a.workers, results=a.results)

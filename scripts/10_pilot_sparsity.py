"""
10_pilot_sparsity.py — Experiment 1 pilot: when does the joint model beat a star fit and composed
pairwise fits? A small grid, to size the full one and to sanity-check the pipeline.

Grid (Handover 2, §5): K ∈ {5, 10}; topologies complete, star, chain; budget (anchors kept per batch
pair) ∈ {20, all}; corruption ε ∈ {0, 0.15}; 3 seeds; drift family (i).

Per world: anchors harvested once on every cross-batch sample pair (jointalign.harvest_anchors);
each cell keeps the anchors of its topology, subsamples each pair to the budget, then corrupts a
fraction ε. λ is chosen per cell and method by RT-blocked held-out anchor error at seed 0 and reused
for the other seeds; a fixed λ = 10 is reported as a sensitivity check. The truth is used only to
score.

Sanity check: on complete graphs with ε = 0, the joint model must recover the true maps to within
0.15 min (paper one's tolerance) at every scored time point.

Output: results/pilot_sparsity.txt (report), results/pilot_sparsity.npz (every cell, not in git).
"""
from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path

import numpy as np
from jointalign import harvest_anchors

from trustalign import baselines as bl
from trustalign import graphs, metrics, worlds
from trustalign.provenance import header

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'

KS = (5, 10)
TOPOLOGIES = ('complete', 'star', 'chain')
BUDGETS = (20, None)                  # None = all anchors of the pair
EPSILONS = (0.0, 0.15)
SEEDS = (0, 1, 2)
FIXED_LAM = 10.0
TOL_MIN = 0.15
METHODS = ('Joint', 'Star', 'Composed pairwise', 'Composed PCHIP')


def world_seed(K: int, seed: int) -> int:
    return 1000 * K + seed


def cell_rng(K: int, seed: int, topology: str, budget, eps: float) -> np.random.Generator:
    """The thinning and corruption draws of one cell, independent of every other cell."""
    key = [K, seed, graphs.TOPOLOGIES.index(topology), 0 if budget is None else budget, round(eps * 1000)]
    return np.random.default_rng(key)


def run_cell(world, anchors, topology, budget, eps, rng, lams: dict[str, float] | None, lambdas):
    """Fit and score every method in one cell. lams None: choose λ per method here (seed 0)."""
    edges = graphs.edges_for(topology, world.K, world.sizes, rng)
    hub = graphs.hub_of(topology, world.sizes)
    kept = anchors.subset(graphs.thin(anchors, edges, budget, rng))
    kept, bad = graphs.corrupt(kept, eps, rng)
    fit = bl.fitters(world.K, edges, hub)
    out: dict = {'n_anchors': len(kept), 'n_edges': len(edges), 'lam': {}, 'cv_seconds': 0.0}
    if lams is None:
        lams = {}
        t0 = time.perf_counter()
        for name in METHODS:
            lams[name] = np.nan if name in bl.LAMBDA_FREE else metrics.select_lambda(fit[name], kept, lambdas)[0]
        out['cv_seconds'] = time.perf_counter() - t0
    for name in METHODS:
        variants = [('cv', lams[name])] + ([] if name in bl.LAMBDA_FREE else [('fixed', FIXED_LAM)])
        for tag, lam in variants:
            t0 = time.perf_counter()
            f = fit[name](kept, lam)
            secs = time.perf_counter() - t0
            errs = metrics.map_errors(world, f.map)
            row = {'lam': lam, 'seconds': secs, **metrics.summarise(errs, edges),
                   'max_all': float(max(e.max() for e in errs.values() if len(e)))}
            if f.model is not None:
                row['recall'], row['false_reject'] = metrics.rejection(f.model.weights, bad[f.used])
            if name.startswith('Composed') and tag == 'cv':
                row['cycle_median'], row['cycle_max'] = metrics.cycle_inconsistency(world, f.map)
            out[(name, tag)] = row
    out['lam'] = lams
    return out


def main(Ks=KS, topologies=TOPOLOGIES, budgets=BUDGETS, epsilons=EPSILONS, seeds=SEEDS,
         results: Path = RESULTS, lambdas=metrics.LAMBDAS, world_kw: dict | None = None) -> dict:
    t_start = time.perf_counter()
    cells: dict = {}
    timing: dict = {'world': {}, 'harvest': {}}
    chosen: dict = {}
    for K in Ks:
        for seed in sorted(seeds):
            t0 = time.perf_counter()
            world = worlds.make(K, world_seed(K, seed), **(world_kw or {}))
            t1 = time.perf_counter()
            anchors = harvest_anchors(world.chromas, world.tic_raws)
            t2 = time.perf_counter()
            timing['world'][(K, seed)], timing['harvest'][(K, seed)] = t1 - t0, t2 - t1
            print(f'K={K} seed={seed}: {len(anchors):,} anchors, world {t1 - t0:.1f} s, '
                  f'harvest {t2 - t1:.1f} s', flush=True)
            for topology, budget, eps in product(topologies, budgets, epsilons):
                cond = (K, topology, budget, eps)
                rng = cell_rng(K, seed, topology, budget, eps)
                t0 = time.perf_counter()
                res = run_cell(world, anchors, topology, budget, eps, rng,
                               chosen.get(cond), lambdas)
                res['seconds'] = time.perf_counter() - t0
                chosen.setdefault(cond, res['lam'])
                cells[(*cond, seed)] = res
    wall = time.perf_counter() - t_start
    lines = report(cells, timing, Ks, topologies, budgets, epsilons, seeds, wall)
    results.mkdir(parents=True, exist_ok=True)
    (results / 'pilot_sparsity.txt').write_text('\n'.join(lines) + '\n')
    np.savez(results / 'pilot_sparsity.npz', **flatten(cells))
    print('\n'.join(lines))
    return cells


def flatten(cells: dict) -> dict[str, np.ndarray]:
    """One array per (metric, method, λ variant): rows = cells in key order, plus the cell keys."""
    keys = sorted(cells, key=str)
    out: dict[str, np.ndarray] = {'cell_keys': np.array([str(k) for k in keys])}
    for name, tag in product(METHODS, ('cv', 'fixed')):
        rows = [cells[k].get((name, tag)) for k in keys]
        if all(r is None for r in rows):
            continue
        for metric in sorted({m for r in rows if r for m in r}):
            out[f'{name}|{tag}|{metric}'] = np.array([r.get(metric, np.nan) if r else np.nan for r in rows])
    return out


def fmt_b(budget) -> str:
    return 'all' if budget is None else str(budget)


def report(cells, timing, Ks, topologies, budgets, epsilons, seeds, wall) -> list[str]:
    L = [__doc__.strip().splitlines()[0], '', *header(), '',
         f'Grid: K {list(Ks)}; topologies {list(topologies)}; budgets {[fmt_b(b) for b in budgets]}; '
         f'ε {list(epsilons)}; seeds {list(seeds)}; drift family (i): shift ±{worlds.FAMILY_I["shift"]} min, '
         f'ramp ±{worlds.FAMILY_I["ramp"]} min, wiggle {worlds.FAMILY_I["wiggle"][0]}–'
         f'{worlds.FAMILY_I["wiggle"][1]} min; samples per batch {worlds.N_SAMPLES[0]}–{worlds.N_SAMPLES[1]}.',
         'Map error = |ĝ − g*| in minutes over every RT bin where compounds exist in both batches, for every '
         'ordered batch pair; "indirect" = pairs with no kept anchors between them.', '']
    L += ['=' * 118, '1. Map error vs truth (median over seeds of each statistic), λ chosen by RT-blocked CV',
          f'   {"K":>2} {"topology":<9} {"budget":>6} {"ε":>5} {"method":<18} {"λ":>7} {"median":>7} '
          f'{"p95":>7} {"med dir":>8} {"med ind":>8} {"fit s":>7}']
    for K, topo, budget, eps in product(Ks, topologies, budgets, epsilons):
        rows = [cells[(K, topo, budget, eps, s)] for s in seeds]
        for name in METHODS:
            r = [c[(name, 'cv')] for c in rows]
            med = lambda key, r=r: float(np.nanmedian([x[key] for x in r]))
            lam = rows[0]['lam'][name]
            L.append(f'   {K:>2} {topo:<9} {fmt_b(budget):>6} {eps:>5} {name:<18} '
                     f'{"—" if np.isnan(lam) else f"{lam:g}":>7} {med("median_all"):>7.3f} {med("p95_all"):>7.3f} '
                     f'{med("median_direct"):>8.3f} {med("median_indirect"):>8.3f} {med("seconds"):>7.2f}')
        L.append('')
    L += ['=' * 118, f'2. Sensitivity: fixed λ = {FIXED_LAM:g} (median map error, median over seeds)',
          f'   {"K":>2} {"topology":<9} {"budget":>6} {"ε":>5} ' + ' '.join(f'{m:>18}' for m in METHODS[:3])]
    for K, topo, budget, eps in product(Ks, topologies, budgets, epsilons):
        rows = [cells[(K, topo, budget, eps, s)] for s in seeds]
        vals = [float(np.median([c[(m, 'fixed')]['median_all'] for c in rows])) for m in METHODS[:3]]
        L.append(f'   {K:>2} {topo:<9} {fmt_b(budget):>6} {eps:>5} ' + ' '.join(f'{v:>18.3f}' for v in vals))
    L += ['', '=' * 118, f'3. Sanity check: complete graph, ε = 0, joint model, max |error| ≤ {TOL_MIN} min '
          'at every scored point?']
    ok_all = True
    for K, budget, s in product(Ks, budgets, seeds):
        if 'complete' not in topologies or 0.0 not in epsilons:
            continue
        r = cells[(K, 'complete', budget, 0.0, s)][('Joint', 'cv')]
        ok = r['max_all'] <= TOL_MIN
        ok_all &= ok
        L.append(f'   K={K:>2} budget={fmt_b(budget):>4} seed={s}: max {r["max_all"]:.3f} min, '
                 f'p95 {r["p95_all"]:.3f} min → {"PASS" if ok else "FAIL"}')
    L.append(f'   overall: {"PASS" if ok_all else "FAIL — see above"}')
    L += ['', '=' * 118, '4. Robust reweighting at ε > 0 (median over seeds): share of corrupted anchors given '
          'weight 0 (recall) and of clean anchors given weight 0',
          f'   {"K":>2} {"topology":<9} {"budget":>6} {"ε":>5} {"Joint recall":>13} {"Joint false":>12} '
          f'{"Star recall":>12} {"Star false":>11}']
    for K, topo, budget, eps in product(Ks, topologies, budgets, [e for e in epsilons if e > 0]):
        rows = [cells[(K, topo, budget, eps, s)] for s in seeds]
        v = [float(np.nanmedian([c[(m, 'cv')][k] for c in rows]))
             for m in ('Joint', 'Star') for k in ('recall', 'false_reject')]
        L.append(f'   {K:>2} {topo:<9} {fmt_b(budget):>6} {eps:>5} {v[0]:>13.3f} {v[1]:>12.3f} '
                 f'{v[2]:>12.3f} {v[3]:>11.3f}')
    L += ['', '=' * 118, '5. Cycle inconsistency of the composed maps (median over seeds of the median over '
          'ordered triplets, min); joint-type models are 0 by construction',
          f'   {"K":>2} {"topology":<9} {"budget":>6} {"ε":>5} {"pairwise":>9} {"PCHIP":>9}']
    for K, topo, budget, eps in product(Ks, topologies, budgets, epsilons):
        rows = [cells[(K, topo, budget, eps, s)] for s in seeds]
        v = [float(np.nanmedian([c[(m, 'cv')]['cycle_median'] for c in rows]))
             for m in ('Composed pairwise', 'Composed PCHIP')]
        L.append(f'   {K:>2} {topo:<9} {fmt_b(budget):>6} {eps:>5} {v[0]:>9.4f} {v[1]:>9.4f}')
    L += ['', '=' * 118, '6. Run time']
    for K in Ks:
        ks = [k for k in cells if k[0] == K]
        cv = [cells[k]['cv_seconds'] for k in ks if cells[k]['cv_seconds'] > 0]
        L.append(f'   K={K:>2}: world {np.mean([timing["world"][(K, s)] for s in seeds]):.1f} s, harvest '
                 f'{np.mean([timing["harvest"][(K, s)] for s in seeds]):.1f} s per world; '
                 f'per cell {np.mean([cells[k]["seconds"] for k in ks]):.1f} s on average '
                 f'(λ selection, seed 0 only: {np.mean(cv) if cv else 0:.1f} s per cell)')
    L += [f'   Wall time {wall:.0f} s', '']
    return L


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, default=RESULTS)
    ap.add_argument('--device', default='cpu', choices=['cpu'], help='CPU only (bit-reproducible)')
    a = ap.parse_args()
    main(results=a.results)

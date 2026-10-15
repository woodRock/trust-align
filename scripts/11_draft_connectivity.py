"""
11_draft_connectivity.py — DRAFT run: in which connectivity and noise regimes does joint alignment
beat independent (composed pairwise) and reference-based (star) alignment?

DRAFT, for preliminary paper content only; superseded by the Experiment 1 full grid (script 13).
Deviations from that protocol, chosen to run in about 20 minutes: fixed λ = 10 for every method
(no cross-validation), K = 10 only, one anchor budget (20 per batch pair), drift family (i).

Structures (which batch pairs keep anchors):
  star          every batch linked to batch 0 only
  chain         0–1–…–9, no loops
  chain+1/+2    the chain plus 1 or 2 random chords, each closing a loop
  ring          the chain closed into one loop
  density f     the complete graph with edges deleted in a random order (never disconnecting it)
                until a fraction f remains; f = 1.0 (complete), 0.6, 0.4, 0.2. Nested: a lower f
                removes a superset of the edges a higher f removes.
Noise on the kept anchors' t_b: clean; Gaussian timing jitter σ = 0.5 or 1 bin; gross mismatches
(the tutorial's corrupt()) for ε = 15 % or 30 % of anchors.

Within one world every structure and noise level uses the same anchor subsample (20 per pair, drawn
once) and the same noise draws, so the comparisons are paired. Star uses as hub the batch with the
most kept anchors and is scored only on the pairs it can reach (hub and its neighbours); its
coverage is reported. The truth is used only to score.

Output: results/draft_connectivity.txt, results/draft_connectivity.npz (not in git),
figs/draft/*.png (not in git) and figs/draft/captions.md (numbers computed here).
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')        # one BLAS thread per worker: the workers share the cores

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations, permutations, product
from pathlib import Path

import numpy as np
from jointalign import harvest_anchors
from scipy.stats import wilcoxon

from trustalign import baselines as bl
from trustalign import graphs, metrics, worlds
from trustalign.provenance import header

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FIGS = ROOT / 'figs' / 'draft'

K = 10
BUDGET = 20
LAM = 10.0
SEEDS = tuple(range(10))
STRUCTURES = ('star', 'chain', 'chain+1', 'chain+2', 'ring', 'density1.0', 'density0.6', 'density0.4',
              'density0.2')
NOISES = (('clean', 0.0, 0.0), ('jitter 0.5', 0.5, 0.0), ('jitter 1.0', 1.0, 0.0), ('ε 15%', 0.0, 0.15),
          ('ε 30%', 0.0, 0.30))
METHODS = ('Joint', 'Star', 'Composed pairwise', 'Composed PCHIP')


def structures_for(world_K: int, sizes: list[int], rng: np.random.Generator) -> dict[str, set]:
    """Every structure of one world; chords and the deletion order are drawn once, so chain+1 ⊂
    chain+2 and the densities are nested."""
    chain = graphs.edges_for('chain', world_K, sizes)
    cands = [(a, b) for a, b in combinations(range(world_K), 2) if b - a >= 2]
    chords = [cands[i] for i in rng.permutation(len(cands))[:2]]
    order = graphs.removal_order(world_K, rng)
    out = {'star': graphs.edges_for('star', world_K, sizes), 'chain': chain,
           'chain+1': chain | set(chords[:1]), 'chain+2': chain | set(chords),
           'ring': graphs.edges_for('ring', world_K, sizes)}
    for f in (1.0, 0.6, 0.4, 0.2):
        out[f'density{f}'] = graphs.thinned_complete(world_K, f, order)
    return out


def run_world(seed: int, lam: float = LAM, structures=STRUCTURES, noises=NOISES, k: int = K,
              budget: int = BUDGET, world_kw: dict | None = None) -> dict:
    t0 = time.perf_counter()
    world = worlds.make(k, seed=5000 + seed, **(world_kw or {}))
    anchors = harvest_anchors(world.chromas, world.tic_raws)
    rng = np.random.default_rng([seed, 11])
    complete = set(combinations(range(k), 2))
    pool = anchors.subset(graphs.thin(anchors, complete, budget, rng))
    edges_by = structures_for(k, world.sizes, rng)
    noisy = {}
    for i, (name, sigma, eps) in enumerate(noises):
        nrng = np.random.default_rng([seed, 12, i])
        a = graphs.jitter(pool, sigma, nrng)
        noisy[name] = graphs.corrupt(a, eps, nrng)
    out: dict = {'seed': seed, 'n_harvested': len(anchors)}
    for struct, (name, _, _) in product(structures, noises):
        edges = edges_by[struct]
        anc, bad = noisy[name]
        keep = np.isin(np.arange(len(pool)), graphs.thin(pool, edges, None, rng))
        kept, kbad = anc.subset(keep), bad[keep]
        counts = np.bincount(np.concatenate([kept.ba, kept.bb]), minlength=k)
        hub = int(np.argmax(counts))
        reach = {hub} | {b for a, b in edges if a == hub} | {a for a, b in edges if b == hub}
        star_pairs = [(a, b) for a, b in permutations(sorted(reach), 2)]
        cell: dict = {'n_anchors': len(kept), 'n_edges': len(edges), 'hub': hub,
                      'star_coverage': len(star_pairs) / (k * (k - 1))}
        fits = {'Joint': bl.joint(kept, k, lam), 'Star': bl.star(kept, k, lam, hub),
                'Composed pairwise': bl.composed_pairwise(kept, k, lam, edges),
                'Composed PCHIP': bl.composed_pchip(kept, k, edges)}
        for m, f in fits.items():
            errs = metrics.map_errors(world, f.map, star_pairs if m == 'Star' else None)
            row = metrics.summarise(errs, edges)
            if f.model is not None and f.used is not None:
                row['recall'], row['false_reject'] = metrics.rejection(f.model.weights, kbad[f.used])
            cell[m] = row
        cell['Joint on star pairs'] = metrics.summarise(metrics.map_errors(world, fits['Joint'].map, star_pairs),
                                                       edges)
        out[(struct, name)] = cell
    out['seconds'] = time.perf_counter() - t0
    return out


def main(seeds=SEEDS, workers: int = 3, results: Path = RESULTS, figs: Path | None = FIGS, **kw) -> dict:
    t0 = time.perf_counter()
    runs: dict[int, dict] = {}
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(run_world, s, **kw): s for s in seeds}
            for fut in as_completed(futs):
                r = fut.result()
                runs[r['seed']] = r
                print(f'seed {r["seed"]}: {r["n_harvested"]:,} anchors, {r["seconds"]:.0f} s '
                      f'({len(runs)}/{len(seeds)})', flush=True)
    else:
        for s in seeds:
            runs[s] = run_world(s, **kw)
            print(f'seed {s}: {runs[s]["seconds"]:.0f} s', flush=True)
    wall = time.perf_counter() - t0
    structures, noises = kw.get('structures', STRUCTURES), kw.get('noises', NOISES)
    lines = report(runs, sorted(seeds), structures, noises, wall, workers)
    results.mkdir(parents=True, exist_ok=True)
    (results / 'draft_connectivity.txt').write_text('\n'.join(lines) + '\n')
    np.savez(results / 'draft_connectivity.npz', **flatten(runs, structures, noises))
    if figs is not None:
        figures(runs, sorted(seeds), structures, noises, figs)
    print('\n'.join(lines))
    return runs


def stat(runs, seeds, struct, noise, method, key='median_all') -> np.ndarray:
    return np.array([runs[s][(struct, noise)][method][key] for s in seeds])


def flatten(runs, structures, noises) -> dict[str, np.ndarray]:
    seeds = sorted(runs)
    out: dict[str, np.ndarray] = {'seeds': np.array(seeds), 'structures': np.array(structures),
                                  'noises': np.array([n for n, _, _ in noises])}
    for (struct, (noise, _, _)), m in product(product(structures, noises), (*METHODS, 'Joint on star pairs')):
        for key in runs[seeds[0]][(struct, noise)][m]:
            out[f'{struct}|{noise}|{m}|{key}'] = stat(runs, seeds, struct, noise, m, key)
    for struct, (noise, _, _) in product(structures, noises):
        cov = [runs[s][(struct, noise)]['star_coverage'] for s in seeds]
        out[f'{struct}|{noise}|star_coverage'] = np.array(cov)
    return out


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, int, float]:
    """Median of a − b, how many seeds a < b, and the two-sided Wilcoxon p (nan if all equal)."""
    d = a - b
    ok = np.isfinite(d)
    d = d[ok]
    p = float(wilcoxon(d).pvalue) if len(d) and np.any(d != 0) else np.nan
    return float(np.median(d)) if len(d) else np.nan, int(np.sum(d < 0)), p


def report(runs, seeds, structures, noises, wall, workers) -> list[str]:
    n = len(seeds)
    L = [__doc__.strip().splitlines()[0], '',
         'DRAFT — fixed λ = 10, not the Experiment 1 full-grid protocol (script 13).', '', *header(), '',
         f'K = {K}; {BUDGET} anchors per kept batch pair; λ = {LAM:g} for every method; seeds {seeds}; drift '
         f'family (i) (shift ±{worlds.FAMILY_I["shift"]} min, ramp ±{worlds.FAMILY_I["ramp"]} min, wiggle '
         f'{worlds.FAMILY_I["wiggle"][0]}–{worlds.FAMILY_I["wiggle"][1]} min).',
         'Map error = |ĝ − g*| (min) over every RT bin where compounds exist in both batches; statistics are '
         f'medians over the {n} seeds of each seed\'s median over pairs and times.', '']
    L += ['=' * 120, '1. Median map error (min), all ordered batch pairs (Star: only the pairs it reaches; '
          'coverage in its own column)',
          f'   {"structure":<11} {"edges":>5} {"noise":<11} ' + ' '.join(f'{m:>18}' for m in METHODS)
          + f' {"star cover":>11}']
    for struct, (noise, _, _) in product(structures, noises):
        e = int(np.median([runs[s][(struct, noise)]['n_edges'] for s in seeds]))
        cov = float(np.median([runs[s][(struct, noise)]['star_coverage'] for s in seeds]))
        vals = [float(np.nanmedian(stat(runs, seeds, struct, noise, m))) for m in METHODS]
        L.append(f'   {struct:<11} {e:>5} {noise:<11} ' + ' '.join(f'{v:>18.3f}' for v in vals) + f' {cov:>11.2f}')
    L += ['', '=' * 120, '2. Pairs with no direct anchors only: median map error (min)',
          f'   {"structure":<11} {"noise":<11} ' + ' '.join(f'{m:>18}' for m in ('Joint', 'Composed pairwise',
                                                                                 'Composed PCHIP'))]
    for struct, (noise, _, _) in product(structures, noises):
        if struct == 'density1.0':
            continue
        vals = [float(np.nanmedian(stat(runs, seeds, struct, noise, m, 'median_indirect')))
                for m in ('Joint', 'Composed pairwise', 'Composed PCHIP')]
        L.append(f'   {struct:<11} {noise:<11} ' + ' '.join(f'{v:>18.3f}' for v in vals))
    L += ['', '=' * 120, f'3. Paired comparisons over the {n} seeds (indicative: draft protocol). Δ = median of '
          '(joint − competitor) in min; wins = seeds where joint is lower; p = two-sided Wilcoxon',
          f'   {"structure":<11} {"noise":<11} {"vs composed pairwise":>30} {"vs star (its pairs)":>30} '
          f'{"vs pairwise, indirect pairs":>32}']
    for struct, (noise, _, _) in product(structures, noises):
        j = stat(runs, seeds, struct, noise, 'Joint')
        c1 = paired(j, stat(runs, seeds, struct, noise, 'Composed pairwise'))
        c2 = paired(stat(runs, seeds, struct, noise, 'Joint on star pairs'),
                    stat(runs, seeds, struct, noise, 'Star'))
        c3 = paired(stat(runs, seeds, struct, noise, 'Joint', 'median_indirect'),
                    stat(runs, seeds, struct, noise, 'Composed pairwise', 'median_indirect'))
        cell = lambda c: f'{c[0]:+.4f} {c[1]:>2}/{n} p={c[2]:.3f}' if np.isfinite(c[0]) else '—'
        L.append(f'   {struct:<11} {noise:<11} {cell(c1):>30} {cell(c2):>30} {cell(c3):>32}')
    L += ['', '=' * 120, '4. Robust reweighting under gross mismatches: share of corrupted anchors given weight 0 '
          '(recall) / clean anchors given weight 0 (joint model)',
          f'   {"structure":<11} {"noise":<11} {"recall":>8} {"false":>8}']
    for struct, (noise, _, eps) in product(structures, noises):
        if eps > 0:
            r = float(np.nanmedian(stat(runs, seeds, struct, noise, 'Joint', 'recall')))
            f = float(np.nanmedian(stat(runs, seeds, struct, noise, 'Joint', 'false_reject')))
            L.append(f'   {struct:<11} {noise:<11} {r:>8.3f} {f:>8.3f}')
    L += ['', '=' * 120, f'5. Run time: {np.mean([runs[s]["seconds"] for s in seeds]):.0f} s per world '
          f'(all {len(structures) * len(noises)} cells), {workers} workers; wall time {wall:.0f} s', '']
    return L


def figures(runs, seeds, structures, noises, out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    noise_names = [n for n, _, _ in noises]
    med = lambda st, no, m, key='median_all': float(np.nanmedian(stat(runs, seeds, st, no, m, key)))
    colors = {'Joint': '#2a78d6', 'Star': '#eda100', 'Composed pairwise': '#eb6834', 'Composed PCHIP': '#1baf7a'}
    caps = ['# Draft figures (scripts/11_draft_connectivity.py)', '',
            'DRAFT: fixed λ = 10, K = 10, 20 anchors per kept pair, drift family (i), '
            f'{len(seeds)} seeds. Superseded by the Experiment 1 full grid (script 13). Numbers are computed.', '']

    # Regime map: joint − composed pairwise and joint − star, per structure × noise.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    panels = [('Joint − composed pairwise', 'Composed pairwise', 'Joint'),
              ('Joint − star (pairs star reaches)', 'Star', 'Joint on star pairs')]
    for ax, (title, comp, jkey) in zip(axes, panels):
        D = np.array([[med(st, no, jkey) - med(st, no, comp) for no in noise_names] for st in structures])
        lim = np.nanmax(np.abs(D)) or 1.0
        ax.imshow(D, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
        for i, j in product(range(D.shape[0]), range(D.shape[1])):
            ax.text(j, i, f'{D[i, j]:+.3f}', ha='center', va='center', fontsize=7.5)
        ax.set_xticks(range(len(noise_names)), noise_names, rotation=30, ha='right')
        ax.set_yticks(range(len(structures)), structures)
        ax.set_title(title, fontsize=10)
    fig.suptitle('Median map error difference (min); blue = joint better', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'regime_map.png', dpi=160)
    plt.close(fig)
    gap = lambda c: med(*c, 'Joint') - med(*c, 'Composed pairwise')
    best = min(((st, no) for st in structures for no in noise_names), key=gap)
    caps += ['**regime_map.png.** Median map error of the joint model minus that of composed pairwise alignment '
             '(left) and of a star fit through the best-connected batch, on the pairs the star reaches (right), '
             'for each anchor-graph structure and noise level. Blue: joint lower. The joint model\'s largest '
             f'advantage over composed pairwise is {gap(best):+.3f} min '
             f'({best[0]}, {best[1]}).', '']

    # Loop closure: chain → chain+1 → chain+2 → ring.
    loop = [s for s in ('chain', 'chain+1', 'chain+2', 'ring') if s in structures]
    pick = [n for n in ('clean', 'ε 30%') if n in noise_names]
    if len(loop) > 1 and pick:
        fig, axes = plt.subplots(1, len(pick), figsize=(5 * len(pick), 3.6), sharey=True, squeeze=False)
        for ax, no in zip(axes[0], pick):
            for m in ('Joint', 'Composed pairwise', 'Composed PCHIP'):
                ax.plot(range(len(loop)), [med(st, no, m, 'median_indirect') for st in loop], 'o-', lw=2,
                        color=colors[m], label=m)
            ax.set_xticks(range(len(loop)), loop)
            ax.set_title(no, fontsize=10)
            ax.set_yscale('log')
        axes[0][0].set_ylabel('median map error, pairs without\ndirect anchors (min)')
        axes[0][0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / 'loop_closure.png', dpi=160)
        plt.close(fig)
        no = pick[-1]
        caps += ['**loop_closure.png.** Median map error on batch pairs without direct anchors as loops are added '
                 f'to a chain of {K} batches. Under {no}: joint '
                 f'{med(loop[0], no, "Joint", "median_indirect"):.3f} min '
                 f'on the chain and {med(loop[-1], no, "Joint", "median_indirect"):.3f} min on the {loop[-1]}; '
                 f'composed pairwise {med(loop[0], no, "Composed pairwise", "median_indirect"):.3f} and '
                 f'{med(loop[-1], no, "Composed pairwise", "median_indirect"):.3f} min.', '']

    # Edge removal: density sweep.
    dens = [s for s in structures if s.startswith('density')]
    if len(dens) > 1 and pick:
        frac = [float(s[7:]) for s in dens]
        fig, axes = plt.subplots(1, len(pick), figsize=(5 * len(pick), 3.6), sharey=True, squeeze=False)
        for ax, no in zip(axes[0], pick):
            for m in ('Joint', 'Composed pairwise', 'Composed PCHIP'):
                ax.plot(frac, [med(st, no, m) for st in dens], 'o-', lw=2, color=colors[m], label=m)
            ax.invert_xaxis()
            ax.set_xlabel('fraction of batch pairs with anchors')
            ax.set_title(no, fontsize=10)
            ax.set_yscale('log')
        axes[0][0].set_ylabel('median map error, all pairs (min)')
        axes[0][0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / 'edge_removal.png', dpi=160)
        plt.close(fig)
        no = pick[-1]
        caps += ['**edge_removal.png.** Median map error over all batch pairs as direct anchor connections are '
                 f'removed from the complete graph (nested deletions, always connected). Under {no}, from '
                 f'{frac[0]:g} to {frac[-1]:g} of pairs kept: joint {med(dens[0], no, "Joint"):.3f} → '
                 f'{med(dens[-1], no, "Joint"):.3f} min, composed pairwise '
                 f'{med(dens[0], no, "Composed pairwise"):.3f} '
                 f'→ {med(dens[-1], no, "Composed pairwise"):.3f} min.', '']
    (out / 'captions.md').write_text('\n'.join(caps) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--device', default='cpu', choices=['cpu'], help='CPU only (bit-reproducible)')
    a = ap.parse_args()
    main(workers=a.workers)

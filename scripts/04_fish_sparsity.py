"""
04_fish_sparsity.py — Experiment 1, fish arm: do the synthetic findings hold on real anchors?

The full grid (scripts/02_full_sparsity.py) on paper one's fish oil data: the 25,011 spectral anchors
harvested from every cross-batch sample pair of the five batches (jointalign.harvest_anchors, as in
paper one) are thinned and corrupted like a synthetic world, and the same methods are compared.

There is no ground truth, so each method is scored on real anchors it never saw: every anchor that
the thinning removed (all anchors of dropped batch pairs, and those subsampled away in kept pairs).
Held-out anchors are never corrupted. Error = |ĝ_a→b(t_a) − t_b| in minutes; per draw, the median over
batch pairs of each pair's median (pairs weighted equally; jan2016 × sep2015 alone has 10,227 anchors).
The few spectral mismatches among held-out anchors inflate every method alike, and medians resist
them. TIC correlation (paper one's metric, bins ≥ EVAL_START_BIN) is reported as a secondary measure.

Settings (K = 5 batches):
  topology  complete; star through sep2015 (most anchors); weak_star through aug2015 (2 samples);
            chain and ring in acquisition order aug2015–sep2015–jan2016–apr2016–jul2016 (the ring adds
            jul2016–aug2015); density0.4 (4 of the 10 batch pairs, a random connected subset per draw)
  budget    10, 30, 100 anchors per kept batch pair (a pair with fewer keeps all it has)
  ε         0, 0.15, 0.30 of the kept anchors corrupted (graphs.corrupt, as in script 02)
  draws     20 random thinning and corruption draws per setting. One dataset: the spread over draws
            measures sensitivity to which anchors survive, not to other datasets.
λ: paper one's rule on each draw's own kept anchors (RT-blocked CV; script 03 found it slightly better
than a shared tuned λ). Methods, fitters and statistics are script 02's (solver='active_set').

Data stays private: read in place from data/fish_oil (a git-ignored symlink) or --data. The
harvested anchors and TICs are cached locally (results/fish_anchors.npz, never committed: *.npz is
git-ignored); tasks are checkpointed under results/fish_sparsity_cache/.

Output: results/fish_sparsity.txt
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
from itertools import combinations, pairwise, permutations, product
from pathlib import Path

import numpy as np
from alignbench.axis import FISH_OIL_AXIS
from alignbench.metrics.tic import pearson_rows
from alignbench.warps import warp_rows
from jointalign import Anchors, harvest_anchors

from trustalign import baselines as bl
from trustalign import graphs, metrics
from trustalign.provenance import header

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
BIN_MIN, EVAL_BINS, N_BINS = FISH_OIL_AXIS.bin_min, FISH_OIL_AXIS.eval_bins, FISH_OIL_AXIS.n_bins
DATA = ROOT / 'data' / 'fish_oil'   # a symlink to the private data

TOPOLOGIES = ('complete', 'star', 'weak_star', 'chain', 'ring', 'density0.4')
BUDGETS = (10, 30, 100)
EPSILONS = (0.0, 0.15, 0.30)
DRAWS = tuple(range(20))
ORDER = ('batch_aug2015', 'batch_sep2015', 'batch_jan2016', 'batch_apr2016', 'batch_jul2016')   # acquisition
DRIFTED_R = 0.5          # paper one: a pair is drifted if its unaligned mean TIC r is below this
GRID = np.arange(N_BINS, dtype=float)
THREE = ('Joint', 'Composed pairwise', 'Composed PCHIP')


def _script(name: str):
    spec = importlib.util.spec_from_file_location(Path(name).stem, ROOT / 'scripts' / name)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_S13 = None


def s13():
    global _S13
    if _S13 is None:
        _S13 = _script('02_full_sparsity.py')
    return _S13


# ── data ──────────────────────────────────────────────────────────────────────

def load_fish(data: Path, cache: Path) -> dict:
    """Batch names and sizes, normalised TICs and every cross-batch anchor (harvested once, cached)."""
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        names = [str(n) for n in z['names']]
        tics = [z[f'tic_{k}'] for k in range(len(names))]
        anc = Anchors(*(z[k] for k in ('ba', 'bb', 'sa', 'sb', 'ta', 'tb')))
        return {'names': names, 'sizes': [len(t) for t in tics], 'tics': tics, 'anchors': anc}
    from alignbench.datasets.fish_oil import load_batch
    names = sorted(p.name for p in Path(data).glob('batch_*') if p.is_dir())
    batches = [load_batch(b, data) for b in names]
    anc = harvest_anchors([b.chromas for b in batches], [b.tic_raw for b in batches])
    tics = [np.asarray(b.tic, dtype=np.float32) for b in batches]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, names=np.array(names), ba=anc.ba, bb=anc.bb, sa=anc.sa, sb=anc.sb, ta=anc.ta, tb=anc.tb,
             **{f'tic_{k}': t for k, t in enumerate(tics)})
    return {'names': names, 'sizes': [len(t) for t in tics], 'tics': tics, 'anchors': anc}


def pair_counts(anc: Anchors, K: int) -> dict[tuple[int, int], int]:
    return {(a, b): int(np.sum((anc.ba == a) & (anc.bb == b))) for a, b in combinations(range(K), 2)}


def hubs_of(fish: dict) -> dict[str, int]:
    """The strong hub (most anchors, paper one's reference) and the weak hub (fewest samples)."""
    K = len(fish['names'])
    n = pair_counts(fish['anchors'], K)
    total = [sum(v for p, v in n.items() if k in p) for k in range(K)]
    return {'Star': int(np.argmax(total)), 'Star (weak hub)': graphs.weak_hub(fish['sizes'])}


def edges_for(topology: str, fish: dict, rng: np.random.Generator) -> set:
    K, names = len(fish['names']), fish['names']
    hubs = hubs_of(fish)
    e = lambda a, b: (min(a, b), max(a, b))
    if topology == 'complete':
        return set(combinations(range(K), 2))
    if topology in ('star', 'weak_star'):
        h = hubs['Star' if topology == 'star' else 'Star (weak hub)']
        return {e(h, k) for k in range(K) if k != h}
    if topology in ('chain', 'ring'):
        path = [names.index(b) for b in ORDER]
        out = {e(a, b) for a, b in pairwise(path)}
        return out | ({e(path[0], path[-1])} if topology == 'ring' else set())
    if topology.startswith('density'):
        return graphs.thinned_complete(K, float(topology[7:]), graphs.removal_order(K, rng))
    raise ValueError(f'unknown topology {topology!r}')


# ── one cell ──────────────────────────────────────────────────────────────────

def cell_rng(topology: str, budget: int, eps: float, draw: int) -> np.random.Generator:
    return np.random.default_rng([15, TOPOLOGIES.index(topology), budget, round(eps * 1000), draw])


def heldout_errors(g, anc: Anchors, pairs) -> dict[tuple[int, int], np.ndarray]:
    """|ĝ_a→b(t_a) − t_b| (min) of the held-out anchors of every unordered pair (a < b) given."""
    out = {}
    for a, b in pairs:
        m = (anc.ba == a) & (anc.bb == b)
        if m.any():
            out[(a, b)] = np.abs(g(a, b, anc.ta[m]) - anc.tb[m]) * BIN_MIN
    return out


def summarise(errs: dict, edges: set) -> dict[str, float]:
    """Median over pairs of each pair's median held-out error: all, direct (kept) and indirect pairs."""
    med = {p: float(np.median(v)) for p, v in errs.items()}
    out = {}
    for name, sel in (('all', list(med)), ('direct', [p for p in med if p in edges]),
                      ('indirect', [p for p in med if p not in edges])):
        out[f'median_{name}'] = float(np.median([med[p] for p in sel])) if sel else np.nan
    return out


def tic_r(g, tics, pairs) -> dict[tuple[int, int], float]:
    """Mean TIC r of every pair (a < b) given, b's samples warped into a's frame (paper one's rule)."""
    out = {}
    for a, b in pairs:
        src = g(a, b, GRID)                                   # where each bin of a sits in b
        out[(a, b)] = float(np.nanmean(pearson_rows(warp_rows(tics[b], src), tics[a], EVAL_BINS)))
    return out


def run_cell(fish: dict, setting: tuple, draw: int, lambdas) -> dict:
    topology, budget, eps = setting
    t0 = time.perf_counter()
    K, anc = len(fish['names']), fish['anchors']
    rng = cell_rng(topology, budget, eps, draw)
    edges = edges_for(topology, fish, rng)
    train_idx = graphs.thin(anc, edges, budget, rng)
    test = np.ones(len(anc), bool)
    test[train_idx] = False
    kept, bad = graphs.corrupt(anc.subset(train_idx), eps, rng)
    held = anc.subset(test)
    hubs = hubs_of(fish)
    fit = s13().fitters(K, edges, hubs)
    reach = {}
    for m, h in hubs.items():
        r = {h} | {b for a, b in edges if a == h} | {a for a, b in edges if b == h}
        reach[m] = sorted({(min(p), max(p)) for p in permutations(sorted(r), 2)})
    all_pairs = list(combinations(range(K), 2))
    out: dict = {'n_kept': len(kept), 'n_held': len(held), 'n_edges': len(edges), 'lam': {}}
    for name in s13().METHODS:
        lam = np.nan if name in bl.LAMBDA_FREE else metrics.select_lambda(fit[name], kept, lambdas)[0]
        f = fit[name](kept, lam)
        pairs = reach.get(name, all_pairs)
        row = summarise(heldout_errors(f.map, held, pairs), edges)
        row['tic_r'] = tic_r(f.map, fish['tics'], pairs)
        if f.model is not None and f.used is not None:
            row['recall'], row['false_reject'] = metrics.rejection(f.model.weights, bad[f.used])
        out[name] = row
        out['lam'][name] = lam
        if name == 'Joint':
            jm = f.map
    for star, key in s13().STARS.items():
        row = summarise(heldout_errors(jm, held, reach[star]), edges)
        row['tic_r'] = tic_r(jm, fish['tics'], reach[star])
        out[key] = row
    out['seconds'] = time.perf_counter() - t0
    return out


def run_task(topology: str, draw: int, cells: list[tuple], lambdas, cache_npz: Path, data: Path):
    fish = load_fish(data, cache_npz)
    return {'topology': topology, 'draw': draw, 'cells': {s: run_cell(fish, s, draw, lambdas) for s in cells}}


def run_all(settings, draws, workers, cache: Path, lambdas, cache_npz: Path, data: Path) -> dict:
    by: dict = {}
    for s in settings:
        for d in draws:
            by.setdefault((s[0], d), []).append(s)
    done, todo = [], []
    for (topo, d), cells in sorted(by.items(), key=lambda x: (TOPOLOGIES.index(x[0][0]), x[0][1])):
        f = cache / f'fish_{topo}_d{d}.pkl'
        if f.exists():
            r = pickle.loads(f.read_bytes())
            if set(r['cells']) == set(cells):
                done.append(r)
                continue
        todo.append((topo, d, cells, f))
    t0 = time.perf_counter()

    def finish(r, f):
        f.write_bytes(pickle.dumps(r))
        done.append(r)
        print(f'{r["topology"]} draw {r["draw"]} ({len(done)}/{len(by)}, '
              f'{time.perf_counter() - t0:.0f} s elapsed)', flush=True)

    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            futs = {ex.submit(run_task, t, d, c, lambdas, cache_npz, data): f for t, d, c, f in todo}
            for fut in as_completed(futs):
                finish(fut.result(), futs[fut])
    else:
        for t, d, c, f in todo:
            finish(run_task(t, d, c, lambdas, cache_npz, data), f)
    return {(s, r['draw']): c for r in done for s, c in r['cells'].items()}


def main(settings: list[tuple] | None = None, draws=DRAWS, workers: int = 6, results: Path = RESULTS,
         data: Path = DATA, lambdas=metrics.LAMBDAS, cache: Path | None = None, n_boot: int = 2000) -> list[str]:
    settings = list(product(TOPOLOGIES, BUDGETS, EPSILONS)) if settings is None else settings
    cache = results / 'fish_sparsity_cache' if cache is None else cache
    cache.mkdir(parents=True, exist_ok=True)
    cache_npz = results / 'fish_anchors.npz'
    t0 = time.perf_counter()
    fish = load_fish(data, cache_npz)                         # harvest once, before the workers start
    cells = run_all(settings, draws, workers, cache, lambdas, cache_npz, data)
    wall = time.perf_counter() - t0
    lines = report(fish, settings, sorted(draws), cells, wall, workers, n_boot)
    results.mkdir(parents=True, exist_ok=True)
    (results / 'fish_sparsity.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return lines


# ── report ────────────────────────────────────────────────────────────────────

def stat(cells, setting, draws, method, key='median_all') -> np.ndarray:
    return np.array([cells[(setting, d)][method][key] for d in draws])


def comparisons(settings) -> list[tuple]:
    """(setting, label, competitor, joint key, error key): the RQ1b-style comparisons that can differ."""
    rows = []
    for s in settings:
        topo = s[0]
        if topo == 'complete':
            rows.append((s, 'vs Star (weak hub), its pairs', 'Star (weak hub)', 'Joint on weak-hub pairs',
                         'median_all'))
        if topo in ('chain', 'ring', 'density0.4'):
            rows.append((s, 'vs Star, indirect pairs', 'Star', 'Joint on star pairs', 'median_indirect'))
        if topo in ('weak_star', 'chain', 'ring', 'density0.4'):
            rows.append((s, 'vs composed pairwise, indirect', 'Composed pairwise', 'Joint', 'median_indirect'))
    return rows


def report(fish, settings, draws, cells, wall, workers, n_boot) -> list[str]:
    m = s13()
    n = len(draws)
    rng = np.random.default_rng(15)
    names = [b.replace('batch_', '') for b in fish['names']]
    K = len(names)
    hubs = hubs_of(fish)
    cnt = pair_counts(fish['anchors'], K)
    raw = {p: float(np.nanmean(pearson_rows(fish['tics'][p[1]], fish['tics'][p[0]], EVAL_BINS))) for p in cnt}
    drifted = {p for p, r in raw.items() if r < DRIFTED_R}
    meth = m.METHODS
    L = [__doc__.strip().splitlines()[0], '', *header(), '',
         '1. Data: batches ' + ', '.join(f'{b} (n={s})' for b, s in zip(names, fish['sizes']))
         + f'; {len(fish["anchors"]):,} anchors. Strong hub {names[hubs["Star"]]}, weak hub '
         f'{names[hubs["Star (weak hub)"]]}; chain order {" – ".join(b.replace("batch_", "") for b in ORDER)}.',
         '   Anchors per pair (unaligned TIC r; drifted if < 0.5): '
         + '  '.join(f'{names[a]}×{names[b]}={c} ({raw[(a, b)]:.2f}{"*" if (a, b) in drifted else ""})'
                     for (a, b), c in cnt.items()),
         f'   {n} draws per setting. Held-out error = median over batch pairs of the median |ĝ(t_a) − t_b| (min) '
         'over the real anchors the draw did not keep; tables give the median over draws. Stars: only the pairs '
         'they reach. A draw is "broken" if its error exceeds '
         f'{m.BREAK_MIN} min.', '']

    L += ['=' * 124, '2. Held-out anchor error (min), all scored pairs [median λ over draws]',
          f'   {"topology":<10} {"budget":>6} {"ε":>5} ' + ' '.join(f'{x:>19}' for x in meth)]
    for s in settings:
        vals = []
        for x in meth:
            v = float(np.nanmedian(stat(cells, s, draws, x)))
            lam = np.nan if x in bl.LAMBDA_FREE else np.nanmedian([cells[(s, d)]['lam'][x] for d in draws])
            vals.append(f'{v:.3f}' + ('' if np.isnan(lam) else f' [{lam:g}]'))
        L.append(f'   {s[0]:<10} {s[1]:>6} {s[2]:>5} ' + ' '.join(f'{v:>19}' for v in vals))

    L += ['', '=' * 124, '3. Pairs without direct anchors: held-out anchor error (min)',
          f'   {"topology":<10} {"budget":>6} {"ε":>5} ' + ' '.join(f'{x:>19}' for x in THREE)]
    for s in settings:
        if s[0] == 'complete':
            continue
        vals = [float(np.nanmedian(stat(cells, s, draws, x, 'median_indirect'))) for x in THREE]
        L.append(f'   {s[0]:<10} {s[1]:>6} {s[2]:>5} ' + ' '.join(f'{v:>19.3f}' for v in vals))

    L += ['', '=' * 124, '4. Well-connected data (RQ1a): complete graph, budget 100. Δ = joint − star through '
          f'{names[hubs["Star"]]} per draw (min); median [95 % CI], two-sided Wilcoxon p over draws',
          f'   {"ε":>5} {"median Δ [95% CI]":>28} {"[min Δ, max Δ]":>20} {"p":>7}']
    for s in settings:
        if s[0] == 'complete' and s[1] == 100:
            d = stat(cells, s, draws, m.STARS['Star']) - stat(cells, s, draws, 'Star')
            med, lo, hi = m.median_ci(d, n_boot, rng)
            L.append(f'   {s[2]:>5} {f"{med:+.4f} [{lo:+.4f}, {hi:+.4f}]":>28} '
                     f'{f"[{np.nanmin(d):+.4f}, {np.nanmax(d):+.4f}]":>20} {m.fmt_p(m.signed_rank(d)):>7}')

    rows = comparisons(settings)
    diffs = [stat(cells, s, draws, jk, key) - stat(cells, s, draws, comp, key) for s, _, comp, jk, key in rows]
    keep = [i for i, d in enumerate(diffs) if np.isfinite(d).any() and np.any(d[np.isfinite(d)] != 0)]
    rows, diffs = [rows[i] for i in keep], [diffs[i] for i in keep]
    ps = [m.signed_rank(d) for d in diffs]
    adj = m.holm(ps)
    L += ['', '=' * 124, '5. Weak hub, chain, ring, density0.4 (RQ1b). Δ = joint − competitor (min) per draw; '
          f'two-sided Wilcoxon over draws, Holm over all {len(rows)} comparisons; wins = draws where joint is '
          'lower',
          f'   {"topology":<10} {"budget":>6} {"ε":>5} {"comparison":<32} {"median Δ":>9} {"wins":>6} {"p":>7} '
          f'{"Holm p":>7}']
    tally: dict = {}
    for (s, label, _, _, _), d, p, pa in zip(rows, diffs, ps, adj):
        ok = d[np.isfinite(d)]
        med = float(np.median(ok)) if len(ok) else np.nan
        sig = np.isfinite(pa) and pa < m.ALPHA
        key = (s[0], label)
        tally.setdefault(key, [0, 0, 0])[0 if (sig and med < 0) else 1 if sig else 2] += 1
        L.append(f'   {s[0]:<10} {s[1]:>6} {s[2]:>5} {label:<32} {med:>+9.4f} '
                 f'{int(np.sum(ok < 0)):>3}/{len(ok):<2} {m.fmt_p(p):>7} {m.fmt_p(pa):>7}'
                 + ('  *' if sig else ''))
    L += ['', '   Summary (Holm-significant): joint better / joint worse / not significant']
    L += [f'   {t:<10} {lab:<32} {v[0]:>3} / {v[1]:>3} / {v[2]:>3}' for (t, lab), v in tally.items()]

    L += ['', '=' * 124, f'6. Broken draws (held-out error > {m.BREAK_MIN} min) [Wilson 95 % CI], ε > 0',
          f'   {"topology":<10} {"budget":>6} {"ε":>5} {"Joint":>18} {"Composed pairwise":>18}']
    for s in settings:
        if s[2] == 0:
            continue
        out = []
        for x in ('Joint', 'Composed pairwise'):
            v = stat(cells, s, draws, x)
            k = int(np.sum(v > m.BREAK_MIN))
            lo, hi = m.wilson(k, n)
            out.append(f'{k / n:.2f} [{lo:.2f}, {hi:.2f}]')
        L.append(f'   {s[0]:<10} {s[1]:>6} {s[2]:>5} {out[0]:>18} {out[1]:>18}')

    L += ['', '=' * 124, '7. TIC correlation (secondary; paper one\'s metric), mean over draws of the mean over '
          f'pairs, drifted pairs / near-aligned pairs. Unaligned: {np.mean([raw[p] for p in drifted]):.3f} / '
          f'{np.mean([raw[p] for p in raw if p not in drifted]):.3f}',
          f'   {"topology":<10} {"budget":>6} {"ε":>5} ' + ' '.join(f'{x:>19}' for x in THREE)]
    for s in settings:
        vals = []
        for x in THREE:
            dr = np.mean([np.mean([cells[(s, d)][x]['tic_r'][p] for p in drifted]) for d in draws])
            na = np.mean([np.mean([cells[(s, d)][x]['tic_r'][p] for p in raw if p not in drifted]) for d in draws])
            vals.append(f'{dr:.3f} / {na:.3f}')
        L.append(f'   {s[0]:<10} {s[1]:>6} {s[2]:>5} ' + ' '.join(f'{v:>19}' for v in vals))

    secs = [c['seconds'] for c in cells.values()]
    L += ['', '=' * 124, f'8. Run time: {np.mean(secs):.1f} s per cell (mean); {workers} workers; '
          f'wall time {wall:.0f} s (of this invocation; checkpointed tasks are reused)', '']
    return L


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--data', type=Path, default=DATA, help='batch_* folders (default: the fish oil data)')
    ap.add_argument('--results', type=Path, default=RESULTS)
    ap.add_argument('--device', default='cpu', choices=['cpu'], help='CPU only (bit-reproducible)')
    a = ap.parse_args()
    main(workers=a.workers, results=a.results, data=a.data)

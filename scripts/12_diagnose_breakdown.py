"""
12_diagnose_breakdown.py — Why does the joint model break down on corrupted rings and chains?

Diagnostic, not an experiment. Rebuilds the exact cells where the joint model failed:
  draft  scripts/11_draft_connectivity.py, K = 10, 20 anchors per pair, λ = 10, ε = 30 %, ring and chain
  pilot  scripts/10_pilot_sparsity.py, K = 10, chain, all anchors, ε = 15 %, λ = 100 (its seed-0 choice)
and, per seed, reports where the error sits and which refit removes it:
  - per batch: median |error| of the map batch k → batch 0 (which batches are misplaced);
  - per edge: share of its anchors that are corrupted, share of those the robust fit kept (weight > 0),
    share of clean anchors it rejected (weight 0), and the error of the map along that edge;
  - refits of the same anchors: without the corrupted ones (oracle), with 10 Huber rounds instead of
    3, and with λ × 10.
The truth (and the corruption mask) is used only to diagnose, never to fit.

Output: results/diagnose_breakdown.txt
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from jointalign import Anchors, fit_joint, harvest_anchors
from jointalign.synthetic import BIN_MIN

from trustalign import baselines as bl
from trustalign import graphs, metrics, worlds
from trustalign.provenance import header

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
BROKEN = 0.3          # min: a seed counts as broken if the joint model's median map error exceeds this


def _script(name: str):
    spec = importlib.util.spec_from_file_location(Path(name).stem, ROOT / 'scripts' / name)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def draft_cell(s11, seed: int, structure: str):
    """The anchors, corruption mask, edges and world of one draft cell, drawn exactly as script 11."""
    k = s11.K
    world = worlds.make(k, seed=5000 + seed)
    anchors = harvest_anchors(world.chromas, world.tic_raws)
    rng = np.random.default_rng([seed, 11])
    pool = anchors.subset(graphs.thin(anchors, set(combinations(range(k), 2)), s11.BUDGET, rng))
    edges = s11.structures_for(k, world.sizes, rng)[structure]
    i = [n for n, _, _ in s11.NOISES].index('ε 30%')
    nrng = np.random.default_rng([seed, 12, i])
    anc, bad = graphs.corrupt(graphs.jitter(pool, 0.0, nrng), 0.30, nrng)
    keep = np.isin(np.arange(len(pool)), graphs.thin(pool, edges, None, rng))
    return world, anc.subset(keep), bad[keep], edges, s11.LAM


def pilot_cell(s10, seed: int):
    """The anchors, corruption mask, edges and world of the pilot's K = 10 chain, all anchors, ε = 0.15."""
    world = worlds.make(10, s10.world_seed(10, seed))
    anchors = harvest_anchors(world.chromas, world.tic_raws)
    rng = s10.cell_rng(10, seed, 'chain', None, 0.15)
    edges = graphs.edges_for('chain', 10, world.sizes, rng)
    kept = anchors.subset(graphs.thin(anchors, edges, None, rng))
    kept, bad = graphs.corrupt(kept, 0.15, rng)
    return world, kept, bad, edges, 100.0


def med_err(world, g, pairs=None) -> float:
    e = metrics.map_errors(world, g, pairs)
    v = np.concatenate([x for x in e.values() if len(x)])
    return float(np.median(v))


def diagnose(world, kept: Anchors, bad: np.ndarray, edges, lam: float) -> dict:
    K = world.K
    m = fit_joint(kept, K, lam=lam, gauge=bl.GAUGE)
    out: dict = {'joint': med_err(world, m.map),
                 'pairwise': med_err(world, bl.composed_pairwise(kept, K, lam, edges).map),
                 'recall': metrics.rejection(m.weights, bad)[0], 'false': metrics.rejection(m.weights, bad)[1]}
    # Which batches are misplaced: the map into batch 0 (the gauge reference).
    out['batch'] = [med_err(world, m.map, [(k, 0)]) if k else 0.0 for k in range(K)]
    # Per edge: corruption, what the robust fit kept, and the direct map's error.
    rows = []
    for a, b in sorted(edges):
        sel = (kept.ba == a) & (kept.bb == b)
        nb, w = bad[sel], m.weights[sel]
        rows.append(dict(edge=(a, b), n=int(sel.sum()), bad=float(nb.mean()) if sel.any() else np.nan,
                         bad_kept=float(np.mean(w[nb] > 0)) if nb.any() else np.nan,
                         clean_rejected=float(np.mean(w[~nb] == 0)) if (~nb).any() else np.nan,
                         err=med_err(world, m.map, [(a, b), (b, a)])))
    out['edges'] = rows
    # Refits of the same anchors.
    out['oracle'] = med_err(world, fit_joint(kept.subset(~bad), K, lam=lam, gauge=bl.GAUGE).map)
    out['huber10'] = med_err(world, fit_joint(kept, K, lam=lam, gauge=bl.GAUGE, n_huber=10).map)
    out['lam10x'] = med_err(world, fit_joint(kept, K, lam=10 * lam, gauge=bl.GAUGE).map)
    # Do the corrupted anchors on an edge lean one way? (signed mean shift, bins)
    out['lean'] = []
    for a, b in sorted(edges):
        sel = (kept.ba == a) & (kept.bb == b) & bad
        if sel.any():
            t = world.true_map(a, b, kept.ta[sel] * BIN_MIN) / BIN_MIN
            out['lean'].append(((a, b), float(np.mean(kept.tb[sel] - t))))
    return out


def main(results: Path = RESULTS, draft_seeds=tuple(range(10)), pilot_seeds=(0, 1, 2)) -> list[str]:
    t0 = time.perf_counter()
    s10, s11 = _script('10_pilot_sparsity.py'), _script('11_draft_connectivity.py')
    cases = [(f'draft ring ε30% seed {s}', draft_cell(s11, s, 'ring')) for s in draft_seeds]
    cases += [(f'draft chain ε30% seed {s}', draft_cell(s11, s, 'chain')) for s in draft_seeds]
    cases += [(f'pilot chain all ε15% seed {s}', pilot_cell(s10, s)) for s in pilot_seeds]
    res = {name: diagnose(*cell) for name, cell in cases}
    L = [__doc__.strip().splitlines()[0], '', *header(), '',
         f'A seed is "broken" if the joint model\'s median map error exceeds {BROKEN} min. Errors in minutes.', '',
         '=' * 118,
         '1. Per seed: joint vs composed pairwise, robust-fit rejection, and refits of the same anchors',
         f'   {"case":<30} {"joint":>7} {"pairw":>7} {"recall":>7} {"false":>7} {"oracle":>7} {"huber10":>8} '
         f'{"λ×10":>7}  broken?']
    for name, r in res.items():
        L.append(f'   {name:<30} {r["joint"]:>7.3f} {r["pairwise"]:>7.3f} {r["recall"]:>7.3f} {r["false"]:>7.3f} '
                 f'{r["oracle"]:>7.3f} {r["huber10"]:>8.3f} {r["lam10x"]:>7.3f}  '
                 f'{"BROKEN" if r["joint"] > BROKEN else ""}')
    broken = [n for n, r in res.items() if r['joint'] > BROKEN]
    L += ['', f'   broken: {len(broken)} of {len(res)} — ' + (', '.join(broken) if broken else 'none'), '']
    L += ['=' * 118, '2. Broken seeds in detail', '']
    for name in broken:
        r = res[name]
        L.append(f'   {name}')
        L.append('     median |error| of the map batch k → batch 0, per batch: '
                 + ' '.join(f'{k}:{e:.2f}' for k, e in enumerate(r['batch'])))
        L.append(f'     {"edge":>8} {"n":>5} {"bad":>6} {"bad kept":>9} {"clean rej":>10} {"edge err":>9}')
        for e in r['edges']:
            L.append(f'     {e["edge"]!s:>8} {e["n"]:>5} {e["bad"]:>6.2f} {e["bad_kept"]:>9.2f} '
                     f'{e["clean_rejected"]:>10.2f} {e["err"]:>9.3f}')
        L.append('     signed mean offset of the corrupted anchors per edge (bins): '
                 + ' '.join(f'{ab}:{v:+.1f}' for ab, v in r['lean']))
        L.append('')
    L += ['=' * 118, '3. Summary over seeds (median)',
          f'   {"cases":<28} {"joint":>7} {"oracle":>7} {"huber10":>8} {"λ×10":>7} {"broken":>7}']
    for prefix in ('draft ring', 'draft chain', 'pilot chain'):
        rs = [r for n, r in res.items() if n.startswith(prefix)]
        L.append(f'   {prefix:<28} ' + ' '.join(f'{np.median([r[k] for r in rs]):>7.3f}'
                                              for k in ('joint', 'oracle')) +
                 f' {np.median([r["huber10"] for r in rs]):>8.3f} {np.median([r["lam10x"] for r in rs]):>7.3f} '
                 f'{sum(r["joint"] > BROKEN for r in rs):>4}/{len(rs)}')
    L += ['', f'Wall time {time.perf_counter() - t0:.0f} s', '']
    results.mkdir(parents=True, exist_ok=True)
    (results / 'diagnose_breakdown.txt').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))
    return L


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, default=RESULTS)
    a = ap.parse_args()
    main(a.results)

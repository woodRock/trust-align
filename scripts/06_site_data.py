"""
06_site_data.py — the project website's data (docs/data/*.json) from the caches behind the committed
reports, verified against those reports, plus the numbers in the page text (docs/index.html).

Runs locally after script 05 (the caches are git-ignored; CI never regenerates them). Every number the
site shows, in a chart or in prose, comes from these files; none is typed into the HTML.

Inputs
  results/full_sparsity_cache/, results/diagnose_lambda_cache/, results/fish_sparsity_cache/
      the per-world checkpoints of scripts 02–04
  results/fish_anchors.npz   the fish arm's harvested anchors (script 04's local cache; aggregates only
                             are exported)
  results/cache_plot_data.npz  the per-world arrays, fig 4 (a) curves and anchors written by script 05
  results/*.txt              the committed reports, which everything is checked against

Verification: scripts 02–04's report functions are rerun on the checkpoints and every line of the
regenerated reports (the run-time sections and the provenance header aside) must equal the committed
text; the script stops on the first difference. Every median and count exported for the site is then
recomputed from the per-world arrays actually written and compared with the report at its printed
precision. The site's tables show the report's printed strings, so they cannot drift from it.

Outputs (docs/data/, compact JSON with sorted keys; minutes rounded to 4 decimals)
  meta.json       provenance of the results, generation date, the grid, methods, colours, run times
  grid.json       per setting: per-world median map errors per method, tuned λ, broken counts, the
                  RQ1a/RQ1b tests of full_sparsity.txt §3 and §4 that concern the setting
  graphs.json     the anchor graph of every K × topology (test world 6, as fig 2), density 0.4 per cell
  breakdown.json  full_sparsity.txt §6: share of broken test worlds with Wilson CIs
  lambda.json     per-world errors and broken worlds under three λ rules (script 03; fig 4 b)
  scaling.json    fig 3 (b): per-world Joint − Pairwise against K, fits, CIs and bootstrap bands
  fig4a.json      the broken chain world of fig 4 (a): anchors and the fitted drift curves
  fish.json       the fish arm: per-draw held-out errors by setting, medians, tests, broken draws
  world.json      the synthetic world of fig 1 (b, c): TICs, true drifts, one anchor chain, kept anchors
  numbers.json    every number that appears in the page's prose, by key
docs/index.html   its <data data-key="…">…</data> elements are filled from numbers.json in place.

Usage: python3.11 scripts/06_site_data.py [--out docs/data] [--html docs/index.html]
"""
from __future__ import annotations

import argparse
import datetime as dt
import html as html_
import importlib.util
import json
import math
import re
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
DOCS = ROOT / 'docs'
OUT = DOCS / 'data'
HTML = DOCS / 'index.html'
WORLD_SEED = 6                    # the representative test world of fig 2 (script 05)
ND_MIN = 4                        # decimals for minutes
SIZE_BUDGET = 2 * 1024 * 1024     # bytes, docs/data/ in total
RULES = {'tuned': 'cv', 'λ = 10': 'fixed', 'per world': 'oracle'}
RULE_LABELS = {'cv': 'Cross-validated λ', 'fixed': 'Fixed λ = 10', 'oracle': 'Per-world oracle λ'}
PROVENANCE_PREFIXES = ('trustalign ', 'alignbench ', 'joint-align ', 'warpalign ', 'environment:')


def _script(name: str):
    spec = importlib.util.spec_from_file_location(Path(name).stem, ROOT / 'scripts' / name)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── JSON helpers ──────────────────────────────────────────────────────────────

def jsonable(x: Any, nd: int = ND_MIN) -> Any:
    """Plain Python for json.dumps: floats rounded to nd decimals (NaN → None), arrays → lists, tuple
    keys joined with '|'."""
    if isinstance(x, dict):
        return {('|'.join(str(k) for k in key) if isinstance(key, tuple) else str(key)): jsonable(v, nd)
                for key, v in x.items()}
    if isinstance(x, Precise):
        return [None if not math.isfinite(v) else v for v in x]
    if isinstance(x, (list, tuple)):
        return [jsonable(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return [jsonable(v, nd) for v in x.tolist()]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else round(float(x), nd)
    return x


class Precise(list):
    """A list of floats jsonable() leaves at their own precision (already rounded by the builder)."""



def dumps(obj: Any) -> str:
    return json.dumps(jsonable(obj), sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n'


def public(line: str) -> str:
    """A provenance line without the local checkout path (the site is public)."""
    return re.sub(r' from \S.*$', '', line)


def provenance_of(path: Path) -> list[str]:
    """The provenance header of a committed report."""
    return [public(ln) for ln in path.read_text().splitlines()[:12] if ln.startswith(PROVENANCE_PREFIXES)]


# ── parsers for the report sections (committed or regenerated lines) ──────────

def section(lines: list[str], n: int) -> list[str]:
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f'{n}. '))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith('=')), len(lines))
    return lines[start:end]


def parse_medians(lines: list[str], n: int, with_K: bool = True) -> dict:
    """§1 of full_sparsity.txt / §2 of fish_sparsity.txt: setting → [(median string, λ string or None)]."""
    out = {}
    pat = r'^\s+(\d+) (\S+)\s+(\d+)\s+([\d.]+)\s+(.*)$' if with_K else r'^\s+(\S+)\s+(\d+)\s+([\d.]+)\s+(.*)$'
    for ln in section(lines, n):
        m = re.match(pat, ln)
        if not m:
            continue
        g = m.groups()
        key = (int(g[0]), g[1], int(g[2]), float(g[3])) if with_K else (g[0], int(g[1]), float(g[2]))
        out[key] = [(v, lam) for v, lam in re.findall(r'(\d+\.\d{3})(?: \[([^\]]+)\])?', g[-1])]
    return out


def parse_rq1a(lines: list[str], n: int, with_K: bool = True) -> dict:
    """§3 of full_sparsity.txt (K, ε) / §4 of fish_sparsity.txt (ε): the reference-star comparison."""
    out = {}
    pat = (r'^\s+(\d+)\s+([\d.]+)\s+([+-]\d\.\d{4}) \[([+-]\d\.\d{4}), ([+-]\d\.\d{4})\]\s+'
           r'\[([+-]\d\.\d{4}), ([+-]\d\.\d{4})\]\s+(?:(\d+) / (\d+)\s+)?(\S+)$')
    if not with_K:
        pat = pat.replace(r'^\s+(\d+)\s+([\d.]+)', r'^\s+()([\d.]+)')
    for ln in section(lines, n):
        m = re.match(pat, ln)
        if m:
            g = m.groups()
            key = (int(g[0]), float(g[1])) if with_K else float(g[1])
            row = {'median_delta': g[2], 'ci': [g[3], g[4]], 'range': [g[5], g[6]], 'p': g[9]}
            if g[7] is not None:
                row['anchors_per_pair'] = [int(g[7]), int(g[8])]
            out[key] = row
    return out


def parse_rq1b(lines: list[str], n: int, with_K: bool = True) -> dict:
    """§4 of full_sparsity.txt / §5 of fish_sparsity.txt: setting → [comparison rows]."""
    out: dict = {}
    head = r'^\s+(\d+) (\S+)\s+(\d+)\s+([\d.]+) ' if with_K else r'^\s+(\S+)\s+(\d+)\s+([\d.]+) '
    pat = head + r'(vs [^+-]+?)\s+([+-]\d\.\d{4})\s+(\d+)/(\d+)\s+(\S+)\s+(\S+)(\s+\*)?$'
    for ln in section(lines, n):
        m = re.match(pat, ln)
        if not m:
            continue
        g = m.groups()
        key = (int(g[0]), g[1], int(g[2]), float(g[3])) if with_K else (g[0], int(g[1]), float(g[2]))
        label, med, wins, nw, p, holm, star = g[-7:]
        out.setdefault(key, []).append({'comparison': label.strip(), 'median_delta': med,
                                        'wins': f'{int(wins)}/{int(nw)}', 'p': p, 'holm': holm,
                                        'significant': star is not None})
    return out


def parse_rq1c(lines: list[str]) -> dict:
    """§5 of full_sparsity.txt: (topology, ε) → medians per K and the slope with its CI (strings)."""
    out = {}
    for ln in section(lines, 5):
        m = re.match(r'^\s+(\S+)\s+([\d.]+)\s+([+-]\d\.\d{4}|—)\s+([+-]\d\.\d{4}|—)\s+([+-]\d\.\d{4}|—)\s+'
                     r'([+-]\d\.\d{4}) \[([+-]\d\.\d{4}), ([+-]\d\.\d{4})\]', ln)
        if m:
            g = m.groups()
            out[(g[0], float(g[1]))] = {'medians': {5: g[2], 10: g[3], 20: g[4]}, 'slope': g[5],
                                        'ci': [g[6], g[7]]}
    return out


def parse_breakdown(lines: list[str], n: int, with_K: bool = True) -> dict:
    """§6 of either report: setting → {method: (share, lo, hi)} as printed."""
    out = {}
    head = r'^\s+(\d+) (\S+)\s+(\d+)\s+([\d.]+)\s+' if with_K else r'^\s+(\S+)\s+(\d+)\s+([\d.]+)\s+'
    pat = head + r'(\d\.\d{2}) \[(\d\.\d{2}), (\d\.\d{2})\]\s+(\d\.\d{2}) \[(\d\.\d{2}), (\d\.\d{2})\]$'
    for ln in section(lines, n):
        m = re.match(pat, ln)
        if m:
            g = m.groups()
            key = (int(g[0]), g[1], int(g[2]), float(g[3])) if with_K else (g[0], int(g[1]), float(g[2]))
            out[key] = {'Joint': {'share': g[-6], 'ci': [g[-5], g[-4]]},
                        'Composed pairwise': {'share': g[-3], 'ci': [g[-2], g[-1]]}}
    return out


def parse_lambda_totals(lines: list[str]) -> dict:
    """§3 of diagnose_lambda.txt: method → {rule: broken count}, and the number of worlds."""
    sec = section(lines, 3)
    n = int(re.search(r'\((\d+) test worlds\)', sec[0]).group(1))  # type: ignore[union-attr]
    out: dict = {'n': n}
    for ln in sec:
        m = re.match(r'^\s+(Joint|Composed pairwise)\s+(.*)$', ln)
        if m:
            out[m.group(1)] = dict(zip(('tuned', 'λ = 10', 'own CV', 'best λ', 'per world'), m.group(2).split()))
    return out


def parse_runtime(full: list[str], diag: list[str], fish: list[str]) -> dict:
    """The run-time sections: script 02's CPU hours, scripts 03 and 04's seconds per cell."""
    cpu = re.search(r'compute ([\d.]+) CPU h', '\n'.join(section(full, 11)))
    s03 = re.search(r'([\d.]+) s per test cell', '\n'.join(section(diag, 4)))
    s04 = re.search(r'([\d.]+) s per cell', '\n'.join(section(fish, 8)))
    assert cpu and s03 and s04
    return {'02_cpu_h': cpu.group(1), '03_s_per_cell': s03.group(1), '04_s_per_cell': s04.group(1)}


# ── the page's numbers ────────────────────────────────────────────────────────

DATA_RE = re.compile(r'(<data\b[^>]*\bdata-key="([^"]+)"[^>]*>)(.*?)(</data>)', re.S)


def fill_numbers(html: str, numbers: dict[str, str]) -> str:
    """Replace the text of every <data data-key="…">…</data> element with its number; unknown keys
    are an error (a number no script produced)."""
    missing = sorted({m.group(2) for m in DATA_RE.finditer(html)} - set(numbers))
    if missing:
        raise KeyError(f'index.html uses numbers no script produced: {missing}')
    return DATA_RE.sub(lambda m: f'{m.group(1)}{html_.escape(numbers[m.group(2)], quote=False)}{m.group(4)}', html)


def html_numbers(html: str) -> dict[str, str]:
    """key → text of every <data> element in the page (HTML entities decoded, e.g. &lt;0.001)."""
    return {m.group(2): html_.unescape(m.group(3)) for m in DATA_RE.finditer(html)}


# ── verification against the committed reports ───────────────────────────────

def comparable(lines: list[str], runtime_section: int) -> list[str]:
    """A report without its provenance header and run-time section (both differ between runs)."""
    end = next(i for i, ln in enumerate(lines) if ln.startswith(f'{runtime_section}. '))
    return [ln for ln in lines[:end] if not ln.startswith(PROVENANCE_PREFIXES)]


def check_same(name: str, regenerated: list[str], committed: list[str], runtime_section: int) -> None:
    a, b = comparable(regenerated, runtime_section), comparable(committed, runtime_section)
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            raise AssertionError(f'{name}: regenerated report differs from the committed one at line {i}:\n'
                                 f'  regenerated: {x!r}\n  committed:   {y!r}')
    if len(a) != len(b):
        raise AssertionError(f'{name}: regenerated report has {len(a)} comparable lines, committed {len(b)}')


def replay_report(d02, s02) -> tuple[list[str], dict]:
    """Script 02's report on its checkpoints, recording every bootstrap replicate of slope_ci (the same
    generator and order of draws, so the slopes and CIs are exactly the committed ones)."""
    rec: dict = {}

    def slope_ci(groups, n_boot, rng):
        def fit(xs, ys):
            X = np.column_stack([np.ones(len(ys)), np.vstack(xs)])
            ok = np.isfinite(ys)
            if np.linalg.matrix_rank(X[ok]) < X.shape[1]:
                return np.full(X.shape[1], np.nan)
            return np.linalg.lstsq(X[ok], ys[ok], rcond=None)[0]
        xs = [g[0] for g in groups.values()]
        ys = [g[1] for g in groups.values()]
        est = fit(xs, np.concatenate(ys))
        boots = []
        for _ in range(n_boot):
            picks = [rng.integers(0, len(y), len(y)) for y in ys]
            boots.append(fit([x[p] for x, p in zip(xs, picks)], np.concatenate([y[p] for y, p in zip(ys, picks)])))
        b = np.array(boots)
        rec[tuple(groups)] = (est, b)
        if np.all(np.isnan(b[:, 1:])):
            return est[1:], est[1:], est[1:]
        return est[1:], np.nanpercentile(b[:, 1:], 2.5, axis=0), np.nanpercentile(b[:, 1:], 97.5, axis=0)

    original = s02.slope_ci
    s02.slope_ci = slope_ci
    try:
        lines = s02.report(d02['cells'], d02['grid'], d02['seeds'], d02['lams'], d02['tune'], d02['test'], 0.0, 6,
                           s02.N_BOOT)
    finally:
        s02.slope_ci = original
    return lines, rec


def check_median(what: str, values: np.ndarray, printed: str) -> None:
    got = f'{float(np.nanmedian(values)):.3f}'
    if got != printed:
        raise AssertionError(f'{what}: median of the exported values is {got}, the report prints {printed}')


def check_count(what: str, got: int, n: int, printed_share: str) -> None:
    if f'{got / n:.2f}' != printed_share:
        raise AssertionError(f'{what}: {got}/{n} broken from the exported values, the report prints '
                             f'{printed_share}')


# ── builders ──────────────────────────────────────────────────────────────────

def build_grid(d02, s02, s05, lines: list[str]) -> tuple[dict, dict]:
    """grid.json and the §6 breakdown, from the checkpoints and the (verified) report lines."""
    meds = parse_medians(lines, 1)
    indirect = parse_medians(lines, 2)
    rq1a, rq1b, bd = parse_rq1a(lines, 3), parse_rq1b(lines, 4), parse_breakdown(lines, 6)
    n = len(d02['seeds'])
    pair_counts: dict = {}
    for r in d02['tune']:
        for s, c in r['cells'].items():
            pair_counts.setdefault(s, []).append(c['pair_counts'])
    grid, breakdown = {}, {}
    for s in d02['grid']:
        K, topo, budget, eps = s
        worlds = {m: s05.stat02(d02, s, m) for m in s02.METHODS}
        rounded = {m: np.round(v, ND_MIN) for m, v in worlds.items()}
        entry: dict = {'K': K, 'topology': topo, 'budget': budget, 'eps': eps, 'worlds': rounded,
                       'median': {}, 'lam': {}, 'broken': {}, 'tests': []}
        for m, (v, lam) in zip(s02.METHODS, meds[s]):
            check_median(f'{s} {m}', worlds[m], v)          # the raw values, as the report
            entry['median'][m] = v
            entry['lam'][m] = float(lam) if lam else None
            k_raw, k_round = int(np.sum(worlds[m] > s02.BREAK_MIN)), int(np.sum(rounded[m] > s02.BREAK_MIN))
            if k_raw != k_round:
                raise AssertionError(f'{s} {m}: rounding to {ND_MIN} decimals changes the broken count')
            entry['broken'][m] = k_raw
        if s in indirect:
            entry['median_indirect'] = dict(zip(('Joint', 'Composed pairwise', 'Composed PCHIP'),
                                                [v for v, _ in indirect[s]]))
        pc = np.concatenate(pair_counts[s])
        entry['anchors_per_pair'] = [int(pc.min()), int(np.median(pc))]
        cell = d02['cells'][(*s, d02['seeds'][0])]
        entry['n_edges'], entry['weak_hub'] = int(cell['n_edges']), int(cell['weak_hub'])
        if (K, eps) in rq1a and topo == 'complete' and budget == 100:
            entry['tests'].append({'comparison': 'vs Star (reference), all pairs', **rq1a[(K, eps)]})
        entry['tests'] += rq1b.get(s, [])
        if s in bd:
            for m in ('Joint', 'Composed pairwise'):
                check_count(f'{s} {m}', entry['broken'][m], n, bd[s][m]['share'])
            breakdown[s] = {m: {**bd[s][m], 'k': entry['broken'][m]} for m in bd[s]}
        grid[s] = entry
    return grid, breakdown


def build_graphs(d02, s02) -> dict:
    """graphs.json: every topology's edges for test world WORLD_SEED at each K (fig 2's world), density
    0.4 per cell, with the batch sizes and hubs; edge counts checked against the checkpoints."""
    from trustalign import graphs, worlds
    out = {}
    for K in sorted({s[0] for s in d02['grid']}):
        w = worlds.make(K, s02.world_seed(K, WORLD_SEED))
        entry: dict = {'sizes': w.sizes, 'hub': 0, 'weak_hub': int(graphs.weak_hub(w.sizes)), 'topologies': {},
                       'density0.4': {}}
        for s in (x for x in d02['grid'] if x[0] == K):
            _, topo, budget, eps = s
            rng = s02.cell_rng(K, WORLD_SEED, topo, budget, eps)
            edges = sorted(graphs.edges_for(topo, K, w.sizes, rng))
            cell = d02['cells'][(*s, WORLD_SEED)]
            if len(edges) != cell['n_edges'] or entry['weak_hub'] != cell['weak_hub']:
                raise AssertionError(f'{s}: {len(edges)} edges / weak hub {entry["weak_hub"]} rebuilt, checkpoint '
                                     f'has {cell["n_edges"]} / {cell["weak_hub"]}')
            if topo == 'density0.4':
                entry['density0.4'][(budget, eps)] = edges
            else:
                entry['topologies'][topo] = edges
        out[K] = entry
    return out


def build_lambda(d02, d03, s02, s03, s05, diag_lines: list[str]) -> dict:
    """lambda.json: per corrupted setting of script 03, the per-world errors and broken worlds under
    the cross-validated, fixed and per-world oracle λ; totals checked against diagnose_lambda.txt §3."""
    seeds = sorted(s02.TEST_SEEDS)
    totals = parse_lambda_totals(diag_lines)
    settings: dict = {}
    sums = dict.fromkeys(RULES.values(), 0)
    for s in d03['grid']:
        if s[3] == 0:
            continue
        e = s03.rule_errors(d03['sweep'], s, seeds, 'Joint', d03['lams'][s]['Joint'])
        entry: dict = {'K': s[0], 'topology': s[1], 'budget': s[2], 'eps': s[3], 'errors': {}, 'broken': {}}
        for rule, key in RULES.items():
            v = e[rule]
            # One more decimal than elsewhere: a world at 0.30004 min must stay on the broken side.
            rounded = np.round(v, ND_MIN + 1)
            entry['errors'][key] = Precise(rounded.tolist())
            entry['broken'][key] = [int(seeds[i]) for i in np.flatnonzero(v > s02.BREAK_MIN)]
            if int(np.sum(rounded > s02.BREAK_MIN)) != len(entry['broken'][key]):
                raise AssertionError(f'{s} {rule}: rounding to {ND_MIN + 1} decimals changes the broken count')
            sums[key] += len(entry['broken'][key])
        settings[s] = entry
    for rule, key in RULES.items():
        if str(sums[key]) != totals['Joint'][rule]:
            raise AssertionError(f'λ rule {rule}: {sums[key]} broken worlds exported, report says '
                                 f'{totals["Joint"][rule]}')
    by_topology = s05.lambda_counts(d03, s02, s03)
    return {'seeds': seeds, 'n_worlds': totals['n'], 'rules': RULE_LABELS, 'settings': settings,
            'totals': {RULES[r]: sums[RULES[r]] for r in RULES},
            'by_topology': {(t, RULES[r]): {'k': k, 'n': n} for (t, r), (k, n) in by_topology.items()}}


def build_scaling(d02, s02, s05, rec: dict, lines: list[str]) -> dict:
    """scaling.json: fig 3 (b) and full_sparsity.txt §5."""
    rq1c = parse_rq1c(lines)
    Ks = list(s05.F3B_KS)
    lsx = np.log2(np.array(Ks, dtype=float))
    xx = np.linspace(lsx[0] - 0.15, lsx[-1] + 0.15, 50)
    fig3b = {}
    for topo in s05.F3B_TOPOS:
        sets = tuple((K, topo, s05.F3B_BUDGET, s05.F3B_EPS) for K in Ks)
        points = {}
        for s in sets:
            d = s05.stat02(d02, s, 'Joint') - s05.stat02(d02, s, 'Composed pairwise')
            check = f'{np.median(d):+.4f}'
            if check != rq1c[(topo, s05.F3B_EPS)]['medians'][s[0]]:
                raise AssertionError(f'{s}: median Δ {check} differs from §5')
            points[s[0]] = np.round(d, ND_MIN)
        est, boots = rec[sets]
        lo, hi = np.nanpercentile(boots[:, 1], [2.5, 97.5])
        if f'{est[1]:+.4f}' != rq1c[(topo, s05.F3B_EPS)]['slope'] or \
                [f'{lo:+.4f}', f'{hi:+.4f}'] != rq1c[(topo, s05.F3B_EPS)]['ci']:
            raise AssertionError(f'{topo}: slope {est[1]:+.4f} [{lo:+.4f}, {hi:+.4f}] differs from §5')
        band = boots[:, :1] + boots[:, 1:2] * xx[None, :]
        blo, bhi = np.nanpercentile(band, [2.5, 97.5], axis=0)
        fig3b[topo] = {'points': points,
                       'fit': {'intercept': round(float(est[0]), 6), 'slope': round(float(est[1]), 6)},
                       'band': {'x_log2': np.round(xx, 4), 'lo': np.round(blo, ND_MIN),
                                'hi': np.round(bhi, ND_MIN)},
                       **rq1c[(topo, s05.F3B_EPS)]}
    return {'budget': s05.F3B_BUDGET, 'eps': s05.F3B_EPS, 'Ks': Ks, 'fig3b': fig3b, 'table': rq1c}


def build_fig4a(z, lines: list[str], s05) -> dict:
    """fig4a.json from the arrays script 05 saved for fig 4 (a)."""
    a, b, seed = (int(x) for x in z['fig4a|edge_seed'])
    lam_cv, lam_fixed, lam_pw = (float(x) for x in z['fig4a|lams'])
    e_cv, e_fixed, e_pw = (float(x) for x in z['fig4a|world_errors'])
    med, n_broken = float(z['fig4a|setting_median_nbroken'][0]), int(z['fig4a|setting_median_nbroken'][1])
    setting = s05.F4A
    printed = parse_medians(lines, 1)[setting][0]
    if f'{med:.3f}' != printed[0] or f'{lam_cv:g}' != printed[1]:
        raise AssertionError(f'fig 4a: setting median {med:.3f} [{lam_cv:g}] differs from §1 {printed}')
    return {'setting': {'K': setting[0], 'topology': setting[1], 'budget': setting[2], 'eps': setting[3]},
            'seed': seed, 'edge': [a, b], 'setting_median': printed[0], 'n_broken': n_broken,
            'lams': {'joint_cv': lam_cv, 'joint_fixed': lam_fixed, 'pairwise': lam_pw},
            'world_errors': {'joint_cv': f'{e_cv:.3f}', 'joint_fixed': f'{e_fixed:.3f}',
                             'pairwise': f'{e_pw:.3f}'},
            't': np.round(z['fig4a|t'], ND_MIN),
            'curves': {'truth': np.round(z['fig4a|truth'], ND_MIN),
                       'joint_cv': np.round(z['fig4a|Joint cv'], ND_MIN),
                       'joint_fixed': np.round(z['fig4a|Joint fixed'], ND_MIN),
                       'pairwise': np.round(z['fig4a|Pairwise cv'], ND_MIN)},
            'anchors': {'ta': np.round(z['fig4a|anchors_ta'], ND_MIN),
                        'dt': np.round(z['fig4a|anchors_dt'], ND_MIN),
                        'bad': [bool(x) for x in z['fig4a|anchors_bad']]}}


def build_fish(fish_cells, fish, s02, s04, lines: list[str]) -> dict:
    """fish.json: aggregates of the fish arm only (per-draw held-out error medians), with the report's
    printed medians, tests and broken shares."""
    meds, indirect = parse_medians(lines, 2, with_K=False), parse_medians(lines, 3, with_K=False)
    rq1a, rq1b, bd = (parse_rq1a(lines, 4, with_K=False), parse_rq1b(lines, 5, with_K=False),
                      parse_breakdown(lines, 6, with_K=False))
    draws = list(s04.DRAWS)
    settings = {}
    for s in product(s04.TOPOLOGIES, s04.BUDGETS, s04.EPSILONS):
        topo, budget, eps = s
        entry: dict = {'topology': topo, 'budget': budget, 'eps': eps, 'draws': {}, 'median': {}, 'lam': {},
                       'broken': {}, 'tests': []}
        for m, (v, lam) in zip(s02.METHODS, meds[s]):
            raw = np.array([fish_cells[(s, d)][m]['median_all'] for d in draws])
            entry['draws'][m] = np.round(raw, ND_MIN)
            check_median(f'fish {s} {m}', raw, v)
            entry['median'][m] = v
            entry['lam'][m] = float(lam) if lam else None
            k_raw, k_round = int(np.sum(raw > s02.BREAK_MIN)), int(np.sum(entry['draws'][m] > s02.BREAK_MIN))
            if k_raw != k_round:
                raise AssertionError(f'fish {s} {m}: rounding changes the broken count')
            entry['broken'][m] = k_raw
        if s in indirect:
            entry['median_indirect'] = dict(zip(('Joint', 'Composed pairwise', 'Composed PCHIP'),
                                                [v for v, _ in indirect[s]]))
        if topo == 'complete' and budget == 100:
            entry['tests'].append({'comparison': 'vs Star (sep2015), all pairs', **rq1a[eps]})
        entry['tests'] += rq1b.get(s, [])
        if s in bd:
            for m in ('Joint', 'Composed pairwise'):
                check_count(f'fish {s} {m}', entry['broken'][m], len(draws), bd[s][m]['share'])
            entry['breakdown'] = {m: {**bd[s][m], 'k': entry['broken'][m]} for m in bd[s]}
        settings[s] = entry
    names = [b.replace('batch_', '') for b in fish['names']]
    hubs = s04.hubs_of(fish)
    return {'n_draws': len(draws), 'n_anchors': len(fish['anchors']),
            'batches': [{'name': b, 'n_samples': int(n)} for b, n in zip(names, fish['sizes'])],
            'hub': names[hubs['Star']], 'weak_hub': names[hubs['Star (weak hub)']],
            'chain_order': [b.replace('batch_', '') for b in s04.ORDER],
            'settings': settings}


def build_world(s02, s05) -> dict:
    """world.json: the representative synthetic test world of fig 1 (b, c) for the animated introduction:
    first-sample TICs of every batch (normalised), the true drift of every batch on the latent axis, one
    compound's anchor chain, and the kept anchors (clean and corrupted) of fig 1's corruption cell between
    the batches fig 1 (b) shows. Synthetic data only; nothing here comes from a real sample."""
    from jointalign.synthetic import BIN_MIN, TIME_AXIS
    world, anchors, _ = s02.world_and_anchors(s05.WORLD_K, s05.WORLD_SEED)
    a, b, ta, tb_true, tb_bad = s05.corrupted_anchor(world, anchors, s02)
    _, kept, bad, _, _ = s02.prepare(world, anchors, s05.CORRUPT_CELL, s05.WORLD_SEED)
    shown = list(s05.SHOW)
    pairs = [(shown[i], shown[i + 1]) for i in range(len(shown) - 1)]
    sel = np.zeros(len(kept), bool)
    for pa, pb in pairs:
        sel |= (kept.ba == pa) & (kept.bb == pb)
    tau = np.asarray(world.truth['compound_tau_min'])
    tic0 = world.tic_raws[shown[0]][0] / world.tic_raws[shown[0]][0].max()
    h0 = np.interp(tau + world.specs[shown[0]].drift(tau), TIME_AXIS, tic0)
    inside = (tau > 15) & (tau < 25)
    tc = float(tau[inside][np.argmax(h0[inside])])          # fig 1 (b)'s chain compound
    return {
        'K': world.K, 'seed': s05.WORLD_SEED, 'world_seed': world.seed, 'bin_min': float(BIN_MIN),
        'time_min': np.round(TIME_AXIS, 4), 'shown': shown, 'sizes': world.sizes,
        'tic': [np.round(world.tic_raws[k][0] / world.tic_raws[k][0].max(), 4) for k in range(world.K)],
        'drift': [np.round(spec.drift(TIME_AXIS), 4) for spec in world.specs],
        'chain': {'latent_min': round(tc, 4),
                  'observed_min': [round(float(tc + world.specs[k].drift(tc)), 4) for k in range(world.K)]},
        'corruption_cell': dict(zip(('K', 'topology', 'budget', 'eps'), s05.CORRUPT_CELL)),
        'highlight': {'a': a, 'b': b, 'ta_min': round(ta * BIN_MIN, 4), 'tb_true_min': round(tb_true * BIN_MIN, 4),
                      'tb_bad_min': round(tb_bad * BIN_MIN, 4), 'moved_bins': round(tb_bad - tb_true, 1)},
        'anchors': {'a': [int(x) for x in kept.ba[sel]], 'b': [int(x) for x in kept.bb[sel]],
                    'ta_min': np.round(kept.ta[sel] * BIN_MIN, 4), 'tb_min': np.round(kept.tb[sel] * BIN_MIN, 4),
                    'bad': [bool(x) for x in bad[sel]]},
    }


def build_numbers(meta: dict, grid: dict, breakdown: dict, lam: dict, scaling: dict, fig4a: dict, fish: dict,
                  graphs_: dict, world: dict, full_lines: list[str]) -> dict[str, str]:
    """Every number the page's prose can cite, by key."""
    N: dict[str, str] = {}
    for key, line in meta['results_provenance'].items():
        N[f'prov|{key}'] = line
    N['generated'] = meta['generated']
    N['n_settings'] = str(len(grid))
    N['n_worlds'] = str(meta['n_test_worlds'])
    N['n_tuning_worlds'] = str(len(meta['tuning_seeds']))
    N['break_min'] = f'{meta["break_min"]:g}'
    N['n_methods'] = str(len(meta['methods']))
    for k, v in meta['runtime'].items():
        N[f'runtime|{k}'] = v
    n03 = sum(1 for _ in lam['all_settings']) * meta['n_test_worlds']
    N['runtime|03_cpu_min'] = f'{float(meta["runtime"]["03_s_per_cell"]) * n03 / 60:.0f}'
    n04 = len(fish['settings']) * fish['n_draws']
    N['runtime|04_cpu_min'] = f'{float(meta["runtime"]["04_s_per_cell"]) * n04 / 60:.0f}'
    for s, e in grid.items():
        key = '|'.join(str(x) for x in s)
        for m, v in e['median'].items():
            N[f't1|{key}|{m}'] = v
            N[f'broken|{key}|{m}'] = str(e['broken'][m])
            if e['lam'][m] is not None:
                N[f'lam|{key}|{m}'] = f'{e["lam"][m]:g}'
        for t in e['tests']:
            tag = t['comparison']
            N[f'test|{key}|{tag}|delta'] = t['median_delta']
            N[f'test|{key}|{tag}|p'] = t['p']
            if 'wins' in t:
                N[f'test|{key}|{tag}|wins'] = t['wins']
                N[f'test|{key}|{tag}|holm'] = t['holm']
            if 'ci' in t:
                N[f'test|{key}|{tag}|ci'] = f'[{t["ci"][0]}, {t["ci"][1]}]'
    for s, e in breakdown.items():
        key = '|'.join(str(x) for x in s)
        for m, v in e.items():
            N[f'bd|{key}|{m}'] = v['share']
            N[f'bd|{key}|{m}|ci'] = f'[{v["ci"][0]}, {v["ci"][1]}]'
    rq1a = parse_rq1a(full_lines, 3)
    deltas = [float(r['median_delta']) for r in rq1a.values()]
    N['rq1a|delta_min'] = f'{min(deltas):+.4f}'
    N['rq1a|delta_max'] = f'{max(deltas):+.4f}'
    for rule, k in lam['totals'].items():
        N[f'lamtot|{rule}'] = str(k)
    N['lamtot|n'] = str(lam['n_worlds'])
    N['lamtot|n_settings'] = str(len(lam['settings']))
    N['lamtopo|n'] = str(next(iter(lam['by_topology'].values()))['n'])
    N['rq1b|n_comparisons'] = str(sum(len(v) for v in parse_rq1b(full_lines, 4).values()))
    N['n_breakdown_settings'] = str(len(breakdown))
    for (t, rule), v in lam['by_topology'].items():
        N[f'lamtopo|{t}|{rule}'] = f'{v["k"]}/{v["n"]}'
    for (topo, eps), r in scaling['table'].items():
        for K, v in r['medians'].items():
            N[f'rq1c|{topo}|{eps}|K{K}'] = v
        N[f'rq1c|{topo}|{eps}|slope'] = r['slope']
        N[f'rq1c|{topo}|{eps}|ci'] = f'[{r["ci"][0]}, {r["ci"][1]}]'
    N['fig4a|seed'] = str(fig4a['seed'])
    N['fig4a|edge'] = f'{fig4a["edge"][0]}–{fig4a["edge"][1]}'
    N['fig4a|n_broken'] = str(fig4a['n_broken'])
    N['fig4a|setting_median'] = fig4a['setting_median']
    for k, v in fig4a['lams'].items():
        N[f'fig4a|lam|{k}'] = f'{v:g}'
    for k, v in fig4a['world_errors'].items():
        N[f'fig4a|err|{k}'] = v
    N['fig4a|n_clean'] = str(sum(not b for b in fig4a['anchors']['bad']))
    N['fig4a|n_bad'] = str(sum(fig4a['anchors']['bad']))
    N['fish|n_anchors'] = f'{fish["n_anchors"]:,}'
    N['fish|n_batches'] = str(len(fish['batches']))
    N['fish|n_draws'] = str(fish['n_draws'])
    N['fish|hub'] = fish['hub']
    N['fish|weak_hub'] = fish['weak_hub']
    for b in fish['batches']:
        N[f'fish|n|{b["name"]}'] = str(b['n_samples'])
    for s, e in fish['settings'].items():
        key = '|'.join(str(x) for x in s)
        for m, v in e['median'].items():
            N[f'fish|{key}|{m}'] = v
            N[f'fishbroken|{key}|{m}'] = str(e['broken'][m])
        for t in e['tests']:
            N[f'fishtest|{key}|{t["comparison"]}|delta'] = t['median_delta']
            N[f'fishtest|{key}|{t["comparison"]}|p'] = t['p']
    N['world|K'] = str(world['K'])
    N['world|seed'] = str(world['seed'])
    N['world|chain_latent_min'] = f'{world["chain"]["latent_min"]:.2f}'
    N['world|moved_bins'] = f'{world["highlight"]["moved_bins"]:+.1f}'
    N['world|moved_min'] = f'{world["highlight"]["moved_bins"] * world["bin_min"]:+.1f}'
    N['world|pair'] = f'{world["highlight"]["a"]}–{world["highlight"]["b"]}'
    N['world|n_anchors_shown'] = str(len(world['anchors']['a']))
    N['world|shown_list'] = ', '.join(str(k) for k in world['shown'])
    N['world|n_bad_shown'] = str(sum(world['anchors']['bad']))
    drifts = np.array(world['drift'])
    N['world|drift_min'] = f'{drifts.min():+.2f}'
    N['world|drift_max'] = f'{drifts.max():+.2f}'
    for K, g in graphs_.items():
        for topo, edges in g['topologies'].items():
            N[f'edges|{K}|{topo}'] = str(len(edges))
        for cell, edges in g['density0.4'].items():
            N[f'edges|{K}|density0.4|{cell[0]}|{cell[1]}'] = str(len(edges))
        N[f'weak_hub|{K}'] = str(g['weak_hub'])
        N[f'weak_hub_n|{K}'] = str(g['sizes'][g['weak_hub']])
        N[f'hub_n|{K}'] = str(g['sizes'][0])
    return N


# ── main ──────────────────────────────────────────────────────────────────────

def main(out: Path = OUT, results: Path = RESULTS, html: Path | None = HTML) -> dict:
    s02, s03, s04, s05 = (_script(n) for n in ('02_full_sparsity.py', '03_diagnose_lambda.py',
                                                '04_fish_sparsity.py', '05_paper_figures.py'))
    committed = {n: (results / f'{n}.txt').read_text().splitlines()
                 for n in ('full_sparsity', 'diagnose_lambda', 'fish_sparsity')}

    print('loading script 02 checkpoints and replaying its report …', flush=True)
    d02 = s05.load_s02(results / 'full_sparsity_cache', s02)
    full_lines, rec = replay_report(d02, s02)
    check_same('full_sparsity', full_lines, committed['full_sparsity'], 11)

    print('replaying script 03\'s report …', flush=True)
    d03 = s05.load_s03(results / 'diagnose_lambda_cache', s02, s03, d02)
    full03 = {k: c for k, c in d02['cells'].items() if k[:4] in set(d03['grid'])}
    tuned03 = [{**r, 'cells': {s: c for s, c in r['cells'].items() if s in set(d03['grid'])}} for r in d02['tune']
               if any(s in set(d03['grid']) for s in r['cells'])]
    diag_lines = s03.report(d03['grid'], d02['seeds'], tuned03, d03['lams'], full03, d03['sweep'],
                            s02.metrics.LAMBDAS, 0.0, 6)
    check_same('diagnose_lambda', diag_lines, committed['diagnose_lambda'], 4)

    print('replaying script 04\'s report …', flush=True)
    fish_cells = s05.load_s04(results / 'fish_sparsity_cache', s04)
    fish = s04.load_fish(s04.DATA, results / 'fish_anchors.npz')
    fish_settings = list(product(s04.TOPOLOGIES, s04.BUDGETS, s04.EPSILONS))
    fish_lines = s04.report(fish, fish_settings, list(s04.DRAWS), fish_cells, 0.0, 6, 2000)
    check_same('fish_sparsity', fish_lines, committed['fish_sparsity'], 8)

    print('building the site data …', flush=True)
    z = np.load(results / 'cache_plot_data.npz')
    grid, breakdown = build_grid(d02, s02, s05, full_lines)
    graphs_ = build_graphs(d02, s02)
    lam = build_lambda(d02, d03, s02, s03, s05, diag_lines)
    lam['all_settings'] = list(d03['grid'])
    scaling = build_scaling(d02, s02, s05, rec, full_lines)
    fig4a = build_fig4a(z, full_lines, s05)
    fish_json = build_fish(fish_cells, fish, s02, s04, fish_lines)
    world = build_world(s02, s05)
    meta = {
        'generated': dt.date.today().isoformat(),
        'results_provenance': dict(zip(('trustalign', 'alignbench', 'joint-align', 'warpalign', 'environment'),
                                       provenance_of(results / 'full_sparsity.txt'))),
        'site_data_provenance': [public(ln) for ln in s02.header()],
        'n_settings': len(grid), 'n_test_worlds': len(d02['seeds']), 'test_seeds': d02['seeds'],
        'tuning_seeds': list(s02.TUNE_SEEDS), 'break_min': s02.BREAK_MIN, 'fixed_lam': s02.FIXED_LAM,
        'lambdas': list(s02.metrics.LAMBDAS),
        'Ks': sorted({s[0] for s in grid}), 'topologies': list(s02.MAIN['topologies']),
        'budgets': list(s02.MAIN['budgets']), 'epsilons': list(s02.MAIN['epsilons']),
        'settings': [list(s) for s in d02['grid']],
        'methods': list(s02.METHODS), 'labels': s05.LABEL, 'colors': s05.C, 'topology_labels': s05.TOPO_LABEL,
        'runtime': parse_runtime(committed['full_sparsity'], committed['diagnose_lambda'],
                                 committed['fish_sparsity']),
        'sections': {'grid': 'full_sparsity.txt §1, §3, §4', 'breakdown': 'full_sparsity.txt §6',
                     'scaling': 'full_sparsity.txt §5', 'lambda': 'diagnose_lambda.txt §2, §3',
                     'fish': 'fish_sparsity.txt §2, §4, §5, §6'},
    }
    numbers = build_numbers(meta, grid, breakdown, lam, scaling, fig4a, fish_json, graphs_, world, full_lines)
    lam.pop('all_settings')

    out.mkdir(parents=True, exist_ok=True)
    files = {'meta.json': meta, 'grid.json': grid, 'graphs.json': graphs_, 'breakdown.json': breakdown,
             'lambda.json': lam, 'scaling.json': scaling, 'fig4a.json': fig4a, 'fish.json': fish_json,
             'world.json': world, 'numbers.json': numbers}
    total = 0
    for name, obj in files.items():
        text = dumps(obj)
        (out / name).write_text(text)
        total += len(text.encode())
        print(f'  {name:<16} {len(text.encode()) / 1024:7.1f} kB')
    print(f'  {"total":<16} {total / 1024:7.1f} kB (budget {SIZE_BUDGET / 1024:.0f} kB)')
    if total > SIZE_BUDGET:
        raise AssertionError(f'docs/data is {total} bytes, over the {SIZE_BUDGET}-byte budget')
    if html is not None and html.exists():
        filled = fill_numbers(html.read_text(), numbers)
        html.write_text(filled)
        print(f'filled {len(html_numbers(filled))} numbers in {html.relative_to(ROOT)}')
    return files


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT)
    ap.add_argument('--results', type=Path, default=RESULTS)
    ap.add_argument('--html', type=Path, default=HTML, help='the page whose <data> numbers are filled in')
    a = ap.parse_args()
    main(out=a.out, results=a.results, html=a.html)

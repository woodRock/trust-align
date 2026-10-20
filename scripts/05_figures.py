"""
05_figures.py — the five figures of Experiment 1 (IEEE TAI format), from the per-world checkpoints behind
the committed reports, with a verification table against the reports and the manuscript.

  fig1_data                 (a) one real fish oil chromatogram (sep2015, the strong hub): 2-D log
                            intensity (RT × m/z) under its TIC, with the mass spectrum at the tallest
                            apex; (b) four batch TICs of one Experiment 1 test world (K = 10, test seed
                            3) with one compound's anchor chain and one real corrupted anchor of that
                            world's K = 10, complete, budget 30, ε = 30 % cell; (c) the true drift of all
                            ten batches of that world
  fig2_graphs               the six topologies of the 114-setting grid, drawn for that same world
                            (density 0.4 and the weak hub exactly as script 02 drew them)
  fig3_synthetic_results    (a) per-world median map error, K = 5, budget 30, ε = 30 %, four topologies
                            × Joint / Star / Composed pairwise; (b) per-world Joint − Composed pairwise
                            vs K at budget 30, ε = 15 %, with script 02's log2 K regression and its
                            bootstrap band (script 02's own resampling, replayed exactly)
  fig4_breakdown_diagnosis  (a) one broken K = 5 chain world (budget 30, ε = 30 %), refitted: pairwise
                            drift of its worst chain edge under Joint (CV λ), Joint (λ = 10) and
                            Composed pairwise, with the clean and corrupted anchors; (b) broken test
                            worlds over script 03's 48 corrupted settings under three λ rules
  fig5_fish_results         held-out anchor error on the fish oil data, complete graph, budgets 10 and
                            100, ε ∈ {0, 15, 30} %: median and IQR over the 20 draws

Sources (never the superseded pilot or draft runs; the code that produced each report is recorded in
its first lines, currently trustalign 0.2.1 with alignbench 0.1.0.dev0 and joint-align 2.0.2):
  results/full_sparsity_cache/   script 02's checkpoints (results/full_sparsity.txt): tuning seeds 0–2,
                                 test seeds 3–22, drift family (i)
  results/diagnose_lambda_cache/ script 03's checkpoints (results/diagnose_lambda.txt): refits of script
                                 02's exact test cells; its Section 0 checks them against script 02
  results/fish_sparsity_cache/   script 04's checkpoints (results/fish_sparsity.txt), 20 draws
  results/fish_anchors.npz, and the fish oil data read in place from data/fish_oil (fig1 a)

Every plotted number is computed from those checkpoints (or from a refit that is checked against
them); the script stops if any recomputed median differs from the committed reports. The per-world
arrays it plots are written to results/cache_plot_data.npz (git-ignored), captions with computed
numbers to figs/paper/captions_tai.md.

With --site, panels (b) and (c) of fig 1 and fig 2 are also written as SVG, in a light and a dark
variant, to docs/assets/ for the project website (never panel (a): a figure of a real fish oil sample
is shown only after the collaborators agree). Site data itself is scripts/06_site_data.py's.

Usage: python3.11 scripts/05_figures.py [--out figs/paper] [--site [docs/assets]]
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import importlib.util
import itertools
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FIGS = ROOT / 'figs' / 'paper'

# ── IEEE TAI style ────────────────────────────────────────────────────────────
DOUBLE, SINGLE = 7.16, 3.5          # \textwidth and \columnwidth of IEEEtai.cls (in)
C = {
    'Joint': '#0072B2',
    'Star': '#E69F00',
    'Star (weak hub)': '#F0E442',
    'Weak-Star': '#F0E442',
    'Composed pairwise': '#009E73',
    'Composed PCHIP': '#CC79A7',
    'truth': '#000000',
    'bad': '#D55E00',
    'fixed': '#56B4E9',
}
LABEL = {
    'Joint': 'Joint',
    'Star': 'Star',
    'Star (weak hub)': 'Weak-star',
    'Composed pairwise': 'Pairwise',
    'Composed PCHIP': 'PCHIP',
}
MARK = {'Joint': 'o', 'Star': 's', 'Composed pairwise': '^'}
TOPO_LABEL = {
    'complete': 'Complete',
    'density0.4': 'Density 0.4',
    'ring': 'Ring',
    'chain': 'Chain',
    'star': 'Star',
    'weak_star': 'Weak-star',
}

# ── what each figure shows (choices, not results) ────────────────────────────
# Representative K = 10 test world (weak hub != 0) used for fig 1 (b, c) and fig 2
WORLD_K, WORLD_SEED = 10, 6
SHOW = (0, 3, 6, 9)                          # batches drawn in fig 1 (b)
CORRUPT_CELL = (10, 'complete', 30, 0.30)    # fig 1 (b): the cell whose corrupted anchor is shown
DENSITY_CELL = (10, 'density0.4', 30, 0.15)  # fig 2: the cell whose density-0.4 graph is drawn
FISH_BATCH = 'batch_sep2015'                 # fig 1 (a): the strong hub of the fish arm
FISH_WINDOW = (10.0, 30.0)                   # min
MZ_RANGE = (40, 400)                         # m/z shown in fig 1 (a)

F3A_K: int = 5
F3A_BUDGET: int = 30
F3A_EPS: float = 0.30
F3A_TOPOS: tuple[str, ...] = ('complete', 'density0.4', 'ring', 'chain')
F3A_METHODS: tuple[str, ...] = ('Joint', 'Star', 'Composed pairwise')
F3A = dict(K=F3A_K, budget=F3A_BUDGET, eps=F3A_EPS, topologies=F3A_TOPOS, methods=F3A_METHODS)

F3B_BUDGET: int = 30
F3B_EPS: float = 0.15
F3B_KS: tuple[int, ...] = (5, 10, 20)
F3B_TOPOS: tuple[str, ...] = ('density0.4', 'complete')
F3B = dict(budget=F3B_BUDGET, eps=F3B_EPS, Ks=F3B_KS, topologies=F3B_TOPOS)

F4A = (5, 'chain', 30, 0.30)
F4A_ERR_MIN = 2.5            # fig 4 (a) caption: where the CV joint fit's error on the edge exceeds this (min)
F4B_UNDIAGNOSED = ('complete', 'star')   # not in script 03's diagnosis: broken counts under CV λ from script 02

F5_BUDGETS: tuple[int, ...] = (10, 100)
F5_EPSILONS: tuple[float, ...] = (0.0, 0.15, 0.30)
F5_METHODS: tuple[str, ...] = ('Joint', 'Star', 'Composed pairwise')
F5 = dict(budgets=F5_BUDGETS, epsilons=F5_EPSILONS, methods=F5_METHODS)

F4B_RULES = (
    ('tuned', 'Cross-validated λ', C['Joint']),
    ('λ = 10', 'Fixed λ = 10', C['fixed']),
    ('per world', 'Per-world oracle λ', C['truth']),
)

# Manuscript numbers: checks only (asserted against the recomputed values; nothing is plotted from
# them). Keys name the cell each number reports.
MANUSCRIPT = {
    'Table I  K=5 complete B=10 ε=0    Joint': '0.033',
    'Table I  K=5 complete B=10 ε=0.3  Joint': '0.043',
    'Table I  K=5 complete B=30 ε=0.3  Joint': '0.028',
    'Table I  K=5 complete B=30 ε=0.3  Star': '0.048',
    'Table I  K=5 complete B=30 ε=0.3  Pairwise': '0.076',
    'Table I  K=5 chain    B=30 ε=0.3  Joint': '1.525',
    'Table I  K=5 chain    B=30 ε=0.3  Star': '0.052',
    'Table I  K=5 chain    B=30 ε=0.3  Pairwise': '0.140',
    'Table I  K=5 ring     B=30 ε=0.3  Joint': '0.045',
    'Table I  K=5 ring     B=30 ε=0.3  Star': '0.071',
    'Table I  K=5 ring     B=30 ε=0.3  Pairwise': '0.116',
    'RQ1c density0.4 ε=0.15 K=5  Δ': '+0.0034',
    'RQ1c density0.4 ε=0.15 K=10 Δ': '-0.0052',
    'RQ1c density0.4 ε=0.15 K=20 Δ': '-0.0118',
    'RQ1c complete   ε=0.15 K=5  Δ': '-0.0143',
    'RQ1c complete   ε=0.15 K=10 Δ': '-0.0105',
    'RQ1c complete   ε=0.15 K=20 Δ': '-0.0090',
    'RQ1c density0.4 ε=0.15 slope': '-0.0084',
    'RQ1c density0.4 ε=0.15 slope CI low': '-0.0107',
    'RQ1c density0.4 ε=0.15 slope CI high': '-0.0063',
    'λ diag broken, CV λ': '175/960',
    'λ diag broken, λ = 10': '160/960',
    'λ diag broken, per-world oracle λ': '80/960',
    # Table II: the 103-run fish data (warp-align step 03 --keep-blank-headers), script 04 of 2026-10-20
    'Table II B=10  ε=0.3 Joint': '0.118',
    'Table II B=10  ε=0.3 Star': '0.279',
    'Table II B=10  ε=0.3 Pairwise': '0.194',
    'Table II B=100 ε=0   Joint': '0.057',
    'Table II B=100 ε=0   Star': '0.075',
    'Table II B=100 ε=0   Pairwise': '0.055',
}


def _script(name: str):
    spec = importlib.util.spec_from_file_location(Path(name).stem.replace('-', '_'), ROOT / 'scripts' / name)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': 8, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 8, 'axes.titlesize': 9.5, 'axes.titlelocation': 'left',
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'lines.linewidth': 1.0, 'legend.frameon': False, 'savefig.dpi': 300,
    })
    return plt


def _title(ax, letter: str, text: str, **kw) -> None:
    kw.setdefault('fontsize', 9.5)
    kw.setdefault('loc', 'left')
    ax.set_title(rf'$\mathbf{{({letter})}}$ {text}', **kw)


def _despine(ax) -> None:
    ax.spines[['top', 'right']].set_visible(False)


def _save(fig, out: Path, name: str) -> Path:
    path = out / f'{name}.pdf'
    fig.savefig(path, bbox_inches='tight', pad_inches=0.04, dpi=300)
    return path


# The website's two colour themes: ink (text, axes, neutral lines) and the panel background that
# labels and legends sit on. The data colours (C, viridis) are the same in both.
SITE_THEMES = {'light': ('#1f1f1f', '#ffffff'), 'dark': ('#d9d9d4', '#161616')}


def _site_rc(ink: str) -> dict:
    return {'text.color': ink, 'axes.labelcolor': ink, 'axes.edgecolor': ink, 'xtick.color': ink,
            'ytick.color': ink, 'svg.fonttype': 'none', 'svg.hashsalt': 'trust-align', 'font.family': 'sans-serif',
            'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']}


def _save_svg(fig, out: Path, name: str) -> Path:
    path = out / f'{name}.svg'
    fig.savefig(path, format='svg', bbox_inches='tight', pad_inches=0.04, transparent=True,
                metadata={'Date': None})
    return path


def batch_colors(K: int):
    import matplotlib
    cmap = matplotlib.colormaps['viridis']
    return [cmap(x) for x in np.linspace(0.0, 0.88, K)]


# ── data: the checkpoints behind the reports ──────────────────────────────────

def load_s02(cache: Path, s02) -> dict:
    """Script 02's tuning and test checkpoints for its full grid (every one must exist)."""
    grid = s02.full_grid()
    out: dict[str, Any] = {}
    for phase, seeds in (('tune', s02.TUNE_SEEDS), ('test', s02.TEST_SEEDS)):
        runs = []
        for K, seed, topo, cells in s02.tasks_for(grid, seeds):
            f = cache / f'{phase}_K{K}_s{seed}_{topo}.pkl'
            if not f.exists():
                raise FileNotFoundError(f'{f} missing: run scripts/02_full_sparsity.py first')
            r = pickle.loads(f.read_bytes())
            if set(r['cells']) != set(cells):
                raise ValueError(f'{f} does not hold the full grid\'s cells')
            runs.append(r)
        out[phase] = runs
    out['cells'] = {(*s, r['seed']): c for r in out['test'] for s, c in r['cells'].items()}
    out['lams'], _ = s02.choose_lambdas(out['tune'], grid)
    out['grid'], out['seeds'] = grid, sorted(s02.TEST_SEEDS)
    return out


def stat02(d02, setting, method, key='median_all', tag='cv') -> np.ndarray:
    return np.array([d02['cells'][(*setting, s)][(method, tag)][key] for s in d02['seeds']])


def replay_rq1c(d02, s02) -> dict:
    """Script 02's report, rerun on its checkpoints with slope_ci recording every bootstrap replicate
    (intercept and slopes). The resampling is script 02's own: same generator, same order of draws,
    so the slopes and CIs are exactly those of results/full_sparsity.txt."""
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
        s02.report(d02['cells'], d02['grid'], d02['seeds'], d02['lams'], d02['tune'], d02['test'], 0.0, 6,
                   s02.N_BOOT)
    finally:
        s02.slope_ci = original
    return rec


def load_s03(cache: Path, s02, s03, d02) -> dict:
    grid = [s for s in s02.full_grid() if s in set(s02.settings(s03.SUBSET))]
    sweep: dict[tuple[tuple, int], Any] = {}
    for K, seed, topo, _cells in s02.tasks_for(grid, s02.TEST_SEEDS):
        f = cache / f'sweep_K{K}_s{seed}_{topo}.pkl'
        if not f.exists():
            raise FileNotFoundError(f'{f} missing: run scripts/03_diagnose_lambda.py first')
        r = pickle.loads(f.read_bytes())
        sweep.update({(s, r['seed']): c for s, c in r['cells'].items()})
    tuned = [r for r in d02['tune'] if any(s in grid for s in r['cells'])]
    tuned = [{**r, 'cells': {s: c for s, c in r['cells'].items() if s in grid}} for r in tuned]
    lams, _ = s02.choose_lambdas(tuned, grid)
    return {'grid': grid, 'sweep': sweep, 'lams': lams}


def load_s04(cache: Path, s04) -> dict:
    cells = {}
    for topo in s04.TOPOLOGIES:
        for d in s04.DRAWS:
            f = cache / f'fish_{topo}_d{d}.pkl'
            if not f.exists():
                raise FileNotFoundError(f'{f} missing: run scripts/04_fish_sparsity.py first')
            r = pickle.loads(f.read_bytes())
            cells.update({(s, r['draw']): c for s, c in r['cells'].items()})
    return cells


def stat04(cells, setting, method, s04) -> np.ndarray:
    return np.array([cells[(setting, d)][method]['median_all'] for d in s04.DRAWS])


# ── parsers for the committed reports (verification only) ────────────────────

def _section(lines: list[str], n: int) -> list[str]:
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f'{n}. '))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith('=')), len(lines))
    return lines[start:end]


def parse_full_table1(path: Path) -> dict:
    """Section 1 of full_sparsity.txt: (K, topology, budget, ε) → five medians as printed."""
    out = {}
    for ln in _section(path.read_text().splitlines(), 1):
        m = re.match(r'^\s+(\d+) (\S+)\s+(\d+)\s+([\d.]+)\s+(.*)$', ln)
        if m:
            vals = re.findall(r'(\d+\.\d{3})(?: \[[^\]]+\])?', m.group(5))
            out[(int(m.group(1)), m.group(2), int(m.group(3)), float(m.group(4)))] = vals
    return out


def parse_full_rq1c(path: Path) -> dict:
    """Section 5 of full_sparsity.txt: (topology, ε) → (Δ at K = 5, 10, 20, slope, CI low, CI high)."""
    out = {}
    for ln in _section(path.read_text().splitlines(), 5):
        m = re.match(r'^\s+(\S+)\s+([\d.]+)\s+([+-]\d\.\d{4})\s+([+-]\d\.\d{4})\s+([+-]\d\.\d{4})\s+'
                     r'([+-]\d\.\d{4}) \[([+-]\d\.\d{4}), ([+-]\d\.\d{4})\]', ln)
        if m:
            out[(m.group(1), float(m.group(2)))] = m.groups()[2:]
    return out


def parse_lambda_totals(path: Path) -> dict:
    """Section 3 of diagnose_lambda.txt: method → {rule: broken count}, and the number of worlds."""
    sec = _section(path.read_text().splitlines(), 3)
    match = re.search(r'\((\d+) test worlds\)', sec[0])
    if not match:
        raise ValueError('Could not parse test worlds count')
    n = int(match.group(1))
    rules = ('tuned', 'λ = 10', 'own CV', 'best λ', 'per world')
    out: dict[str, Any] = {}
    for ln in sec:
        m = re.match(r'^\s+(Joint|Composed pairwise)\s+(.*)$', ln)
        if m:
            out[m.group(1)] = dict(zip(rules, m.group(2).split()))
    return {'n': n, **out}


def parse_fish_table2(path: Path) -> dict:
    out = {}
    for ln in _section(path.read_text().splitlines(), 2):
        m = re.match(r'^\s+(\S+)\s+(\d+)\s+([\d.]+)\s+(.*)$', ln)
        if m:
            out[(m.group(1), int(m.group(2)), float(m.group(3)))] = re.findall(r'(\d+\.\d{3})(?: \[[^\]]+\])?',
                                                                              m.group(4))
    return out


# ── Figure 1 ──────────────────────────────────────────────────────────────────

def fish_sample(data: Path):
    """The FISH_BATCH sample with the largest TIC inside FISH_WINDOW: its chroma, TIC and name."""
    from alignbench.axis import FISH_OIL_AXIS
    from alignbench.datasets.fish_oil import load_batch
    TIME_AXIS = FISH_OIL_AXIS.times
    b = load_batch(FISH_BATCH, data)
    win = (TIME_AXIS >= FISH_WINDOW[0]) & (TIME_AXIS <= FISH_WINDOW[1])
    i = int(np.argmax(b.tic_raw[:, win].sum(axis=1)))
    return b.chromas[i], b.tic_raw[i], b.sample_ids[i]


def fig1(out: Path, world, cell_bad, fish) -> tuple[Path, str]:
    plt = _plt()
    from jointalign.synthetic import BIN_MIN
    from matplotlib.gridspec import GridSpec
    cols = batch_colors(world.K)
    fig = plt.figure(figsize=(DOUBLE, 2.4))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.25, 0.95], wspace=0.42)

    # (a) fish oil: TIC over the 2-D matrix, spectrum at the apex
    chroma, tic, sid = fish
    from alignbench.axis import FISH_OIL_AXIS
    FT = FISH_OIL_AXIS.times
    sub = gs[0].subgridspec(2, 1, height_ratios=[1, 2.3], hspace=0.06)
    ax_t, ax_h = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    win = (FT >= FISH_WINDOW[0]) & (FT <= FISH_WINDOW[1])
    inten = (chroma.astype(float) ** 2) * tic[:, None]         # chroma: sqrt-compressed, L2-normalised rows
    mz = np.arange(chroma.shape[1])
    msel = (mz >= MZ_RANGE[0]) & (mz <= MZ_RANGE[1])
    img = inten[np.ix_(win, msel)]
    logimg = np.log10(img + img.max() * 1e-4)
    ax_h.imshow(logimg.T, origin='lower', aspect='auto', cmap='Greys', interpolation='nearest',
                extent=[FT[win][0] - BIN_MIN / 2, FT[win][-1] + BIN_MIN / 2, MZ_RANGE[0] - 0.5, MZ_RANGE[1] + 0.5])
    apex = int(np.flatnonzero(win)[np.argmax(tic[win])])
    ax_t.plot(FT[win], tic[win] / tic[win].max(), color='k', lw=0.8)
    ax_t.plot(FT[apex], 1.0, 'v', color=C['bad'], ms=4, clip_on=False)
    ax_h.axvline(FT[apex], color=C['bad'], lw=0.6, ls='--')
    ax_t.set_xlim(*FISH_WINDOW)
    ax_h.set_xlim(*FISH_WINDOW)
    ax_t.set_xticklabels([])
    ax_t.set_yticks([0, 1])
    ax_t.set_ylabel('TIC', labelpad=1)
    ax_h.set_xlabel('retention time (min)')
    ax_h.set_ylabel('$m/z$')
    _title(ax_t, 'a', 'Fish oil GC–MS sample')
    spec = inten[apex] / inten[apex].max() * 100
    ins = ax_h.inset_axes([0.52, 0.5, 0.45, 0.46])
    sm = (mz >= MZ_RANGE[0]) & (mz <= MZ_RANGE[1])
    ins.vlines(mz[sm], 0, spec[sm], color='k', lw=0.5)
    top = mz[sm][np.argsort(spec[sm])[-3:]]
    offsets = {55: (-4.0, 3.0, 'right'), 74: (0.0, 5.0, 'center'), 87: (4.0, 3.0, 'left')}
    for m_ in top:
        dx, dy, ha = offsets.get(int(m_), (0.0, 4.0, 'center'))
        ins.annotate(f'{m_}', (m_, spec[m_]), xytext=(dx, dy), textcoords='offset points',
                     fontsize=5.5, ha=ha, va='bottom')
    ins.set_xlim(*MZ_RANGE)
    ins.set_ylim(0, 150)
    ins.set_yticks([])
    ins.tick_params(labelsize=5.5, length=1.5, pad=1)
    ins.set_xlabel(f'$m/z$ at {FT[apex]:.2f} min', fontsize=5.5, labelpad=0.5)
    ins.patch.set_facecolor('white')
    ins.patch.set_alpha(0.92)
    for ax in (ax_t, ax_h, ins):
        _despine(ax)

    # (b) four batch TICs of the test world, one anchor chain and one corrupted anchor
    ax = fig.add_subplot(gs[1])
    tc, a, b, tb_true, tb_bad = panel_1b(ax, world, cell_bad, cols)

    # (c) true drift of every batch
    ax = fig.add_subplot(gs[2])
    drifts = panel_1c(ax, world, cols)

    p = _save(fig, out, 'fig1_data')
    plt.close(fig)
    cap = (f'**fig1_data.** (a) Fish oil sample {sid} ({FISH_BATCH.replace("batch_", "")}), '
           f'{FISH_WINDOW[0]:g}–{FISH_WINDOW[1]:g} min, m/z {MZ_RANGE[0]}–{MZ_RANGE[1]}: TIC (top), log intensity '
           f'(bottom), spectrum at the tallest apex ({FT[apex]:.2f} min; largest ions m/z '
           f'{", ".join(str(int(x)) for x in sorted(top))}). (b) First-sample TICs of batches '
           f'{", ".join(map(str, SHOW))} of test world {WORLD_SEED} (K = {world.K}, world seed {world.seed}); '
           f'triangles: one compound (latent {tc:.2f} min) in each batch; red: a corrupted anchor of batch pair '
           f'{a}–{b} in the K = {CORRUPT_CELL[0]}, {CORRUPT_CELL[1]}, budget {CORRUPT_CELL[2]}, '
           f'ε = {CORRUPT_CELL[3]:.0%} cell, moved {tb_bad - tb_true:+.1f} bins. (c) True drifts span '
           f'{drifts.min():+.2f} to {drifts.max():+.2f} min.')
    return p, cap


def panel_1b(ax, world, cell_bad, cols, ink: str = 'k', bg: str = 'white') -> tuple:
    """Fig 1 (b): four batch TICs, one compound's anchor chain and one corrupted anchor. Returns the
    chain compound's latent time, the corrupted pair and its true and corrupted times (bins)."""
    from jointalign.synthetic import BIN_MIN, TIME_AXIS
    gap = 1.25
    off = {k: (len(SHOW) - 1 - r) * gap for r, k in enumerate(SHOW)}
    tics = {k: world.tic_raws[k][0] / world.tic_raws[k][0].max() for k in SHOW}
    lvl = lambda k, t: off[k] + np.interp(t, TIME_AXIS, tics[k])
    for k in SHOW:
        ax.plot(TIME_AXIS, tics[k] + off[k], color=cols[k], lw=0.7)
        ax.text(44.5, off[k] + 0.15, f'batch {k}', color=cols[k], fontsize=6.8, va='bottom', ha='right',
                bbox=dict(facecolor=bg, alpha=0.75, edgecolor='none', pad=0.5))
    tau = np.asarray(world.truth['compound_tau_min'])
    h0 = np.interp(tau + world.specs[SHOW[0]].drift(tau), TIME_AXIS, tics[SHOW[0]])
    sel = (tau > 15) & (tau < 25)
    tc = float(tau[sel][np.argmax(h0[sel])])                 # tallest batch-0 compound, 15–25 min latent
    xs = [tc + world.specs[k].drift(tc) for k in SHOW]
    ys = [lvl(k, x) + 0.1 for k, x in zip(SHOW, xs)]
    ax.plot(xs, ys, ls=':', color=ink, lw=0.9)
    ax.plot(xs, ys, 'v', color=ink, ms=3.5, label='valid chain')
    a, b, ta, tb_true, tb_bad = cell_bad
    ya = lvl(a, ta * BIN_MIN) + 0.1
    yb_true = lvl(b, tb_true * BIN_MIN) + 0.1
    yb_bad = lvl(b, tb_bad * BIN_MIN) + 0.1
    ax.plot([ta * BIN_MIN, tb_true * BIN_MIN], [ya, yb_true], ls=':', color='0.55', lw=0.8)
    ax.plot([ta * BIN_MIN, tb_bad * BIN_MIN], [ya, yb_bad], ls='--', color=C['bad'], lw=0.9)
    ax.plot(ta * BIN_MIN, ya, 'v', color=C['bad'], ms=3.5)
    ax.plot(tb_true * BIN_MIN, yb_true, 'v', color='0.55', ms=3.5)
    ax.plot(tb_bad * BIN_MIN, yb_bad, 'x', color=C['bad'], ms=4.5, mew=1.2,
            label=f'corrupted ({tb_bad - tb_true:+.0f} bins)')
    ax.set_xlim(0, 45)
    ax.set_ylim(-0.1, off[SHOW[0]] + 1.85)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('retention time in each batch (min)')
    ax.legend(loc='upper left', fontsize=7.0, handlelength=1.2, borderaxespad=0.2, ncol=1,
              frameon=True, facecolor=bg, framealpha=0.85, edgecolor='none')
    _title(ax, 'b', f'Multi-batch TICs (K = {world.K})')
    _despine(ax)
    return tc, a, b, tb_true, tb_bad


def panel_1c(ax, world, cols) -> np.ndarray:
    """Fig 1 (c): the true drift of every batch. Returns the drift curves (batches × time)."""
    from jointalign.synthetic import TIME_AXIS
    drifts = np.array([s.drift(TIME_AXIS) for s in world.specs])
    for k in range(world.K):
        ax.plot(TIME_AXIS, drifts[k], color=cols[k], lw=1.0)
    ax.axhline(0, color='0.6', lw=0.5, zorder=0)
    ax.set_xlim(0, 45)
    ax.set_ylim(-3.5, 3.5)
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xlabel('latent retention time (min)')
    ax.set_ylabel(r'true drift $d_k^*(t)$ (min)')
    _title(ax, 'c', f'True drift (K = {world.K})')
    _despine(ax)
    return drifts


def corrupted_anchor(world, anchors, s02) -> tuple:
    """One real corrupted anchor of CORRUPT_CELL between two batches drawn in fig 1 (b): the one at the
    tallest batch-a TIC among the corrupted anchors of consecutive shown pairs (first samples only
    when any exist)."""
    from jointalign.synthetic import BIN_MIN, TIME_AXIS
    _, kept, bad, _, _ = s02.prepare(world, anchors, CORRUPT_CELL, WORLD_SEED)
    shown = {(min(p), max(p)) for p in itertools.pairwise(SHOW)}
    best, score = None, -1.0
    for i in np.flatnonzero(bad):
        a, b = int(kept.ba[i]), int(kept.bb[i])
        if (a, b) not in shown:
            continue
        tic_a = world.tic_raws[a][0] / world.tic_raws[a][0].max()
        h = float(np.interp(kept.ta[i], np.arange(len(TIME_AXIS)), tic_a)) + (kept.sa[i] == 0) + (kept.sb[i] == 0)
        if h > score:
            best, score = i, h
    if best is None:
        raise ValueError('no corrupted anchor between the batches shown')
    tb_true = world.true_map(int(kept.ba[best]), int(kept.bb[best]),
                             np.array([kept.ta[best] * BIN_MIN]))[0] / BIN_MIN
    return int(kept.ba[best]), int(kept.bb[best]), float(kept.ta[best]), float(tb_true), float(kept.tb[best])


# ── Figure 2 ──────────────────────────────────────────────────────────────────

def graph_panels(world, anchors, s02) -> tuple:
    """Fig 2's six (name, edges, hub) panels for the representative world: density 0.4 and the weak hub
    exactly as script 02 drew them in DENSITY_CELL."""
    from trustalign import graphs
    K = world.K
    dens_edges = s02.prepare(world, anchors, DENSITY_CELL, WORLD_SEED)[0]
    weak = graphs.weak_hub(world.sizes)
    return (('complete', graphs.edges_for('complete', K, world.sizes), None),
            ('density 0.4', dens_edges, None),
            ('ring', graphs.edges_for('ring', K, world.sizes), None),
            ('chain', graphs.edges_for('chain', K, world.sizes), None),
            ('star', graphs.edges_for('star', K, world.sizes), 0),
            ('weak-star', graphs.edges_for('weak_star', K, world.sizes), weak))


def draw_graphs(fig, axes, world, panels, cols, ink: str = 'k') -> None:
    K = world.K
    ang = np.pi / 2 - 2 * np.pi * np.arange(K) / K               # batch 0 (the reference) at the top
    xy = np.c_[np.cos(ang), np.sin(ang)]
    for i, (ax, (name, edges, hub)) in enumerate(zip(axes, panels)):
        for a, b in sorted(edges):
            ax.plot(*xy[[a, b]].T, color='0.5', lw=0.55, zorder=1)
        ax.scatter(*xy.T, s=16, c=cols, zorder=3, edgecolor=ink, linewidth=0.3)
        if hub is not None:
            strong = name == 'star'
            halo_col = C['Star'] if strong else C['Weak-Star']
            ax.scatter(*xy[hub], s=95, facecolor='none', zorder=2, linewidth=1.3,
                       edgecolor=halo_col, linestyle='-' if strong else (0, (1.5, 1)))
            ax.scatter(*xy[hub], s=35, c=[cols[hub]], zorder=4, edgecolor=ink, linewidth=0.4)
            if strong:
                note = f'hub: batch {hub} ({world.sizes[hub]} samples)'
            else:
                note = f'weak hub: batch {hub} ({world.sizes[hub]} samples)'
            ax.text(0, -1.45, note, ha='center', va='top', fontsize=6)
        ax.set_title(rf'$\mathbf{{({"abcdef"[i]})}}$ {name} ({len(edges)})', fontsize=9.5, loc='center', pad=2)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')
    fig.subplots_adjust(wspace=0.05, left=0.0, right=1.0)


def fig2(out: Path, world, anchors, s02) -> tuple[Path, str]:
    plt = _plt()
    K = world.K
    cols = batch_colors(K)
    panels = graph_panels(world, anchors, s02)
    weak = panels[-1][2]
    fig, axes = plt.subplots(1, len(panels), figsize=(DOUBLE, 1.35))
    draw_graphs(fig, axes, world, panels, cols)
    p = _save(fig, out, 'fig2_graphs')
    plt.close(fig)
    cap = (
        f'**fig2_graphs.** The six topologies for K = {K} '
        f'(test world {WORLD_SEED}; node colours as in fig. 1c). Edges: '
        + ', '.join(f'{n} {len(e)}' for n, e, _ in panels)
        + f'. Star hub: batch 0 ({world.sizes[0]} samples); '
        + f'weak-star hub: batch {weak}, the batch with the fewest samples ({world.sizes[weak]} samples). '
        + f'Density 0.4 as drawn in the K = {DENSITY_CELL[0]}, budget {DENSITY_CELL[2]}, '
        + f'ε = {DENSITY_CELL[3]:.0%} cell.'
    )
    return p, cap


# ── Figure 3 ──────────────────────────────────────────────────────────────────

def fig3(out: Path, d02, rec, s02) -> tuple[Path, str, dict]:
    plt = _plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), gridspec_kw={'width_ratios': [1.35, 1]})
    rng = np.random.default_rng(0)                                 # strip-plot jitter only
    topos: tuple[str, ...] = F3A_TOPOS
    meths: tuple[str, ...] = F3A_METHODS
    w = 0.24
    arrays: dict[str, np.ndarray] = {}
    for gi, topo in enumerate(topos):
        s = (F3A_K, topo, F3A_BUDGET, F3A_EPS)
        for mi, m in enumerate(meths):
            v = stat02(d02, s, m)
            arrays[f'fig3a|{topo}|{m}'] = v
            x = gi + (mi - 1) * (w + 0.03)
            ax1.boxplot(v, positions=[x], widths=w, whis=(0, 100), patch_artist=True, showfliers=False,
                        medianprops=dict(color='k', lw=1.0), boxprops=dict(facecolor=C[m], alpha=0.35, lw=0.6,
                                                                           edgecolor=C[m]),
                        whiskerprops=dict(color=C[m], lw=0.6), capprops=dict(color=C[m], lw=0.6))
            ax1.scatter(x + rng.uniform(-w / 3, w / 3, len(v)), v, s=5, color=C[m], edgecolor='none', zorder=3,
                        alpha=0.9)
    ax1.axhline(s02.BREAK_MIN, color=C['bad'], ls='--', lw=0.8)
    ax1.text(-0.48, 0.33, f'Broken-world cutoff ({s02.BREAK_MIN:.2f} min)',
             color=C['bad'], fontsize=7, ha='left', va='bottom')
    ax1.set_yscale('log')
    ax1.set_ylim(0.01, 4)
    ax1.set_xticks(range(len(topos)), [TOPO_LABEL[t] for t in topos])
    ax1.set_xlim(-0.55, len(topos) - 0.45)
    ax1.set_ylabel('median map error (min)')
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=C[m], alpha=0.6, edgecolor=C[m]) for m in meths]
    ax1.legend(handles, [LABEL[m] for m in meths], loc='upper left', ncol=3, handlelength=1.0, columnspacing=1.0)
    _title(ax1, 'a', f'K = {F3A_K}, budget {F3A_BUDGET}, ε = {F3A_EPS:.0%} '
                     f'({len(d02["seeds"])} test worlds per box)')

    Ks: tuple[int, ...] = F3B_KS
    lsx = np.log2(np.array(Ks, dtype=float))
    marks = {'density0.4': ('o', C['Joint']), 'complete': ('D', '0.35')}
    trends: dict[str, tuple[np.ndarray, float, float, list[float]]] = {}
    for ti, topo in enumerate(F3B_TOPOS):
        sets = [(K, topo, F3B_BUDGET, F3B_EPS) for K in Ks]
        mk, col = marks[topo]
        dx = (ti - 0.5) * 0.14
        meds = []
        for s, x in zip(sets, lsx):
            d = stat02(d02, s, 'Joint') - stat02(d02, s, 'Composed pairwise')
            arrays[f'fig3b|{topo}|K{s[0]}'] = d
            meds.append(float(np.median(d)))
            ax2.scatter(x + dx + rng.uniform(-0.04, 0.04, len(d)), d, s=4, color=col, alpha=0.45, edgecolor='none')
        est, boots = rec[tuple(sets)]
        lo, hi = np.nanpercentile(boots[:, 1], [2.5, 97.5])
        xx = np.linspace(lsx[0] - 0.15, lsx[-1] + 0.15, 50)
        band = boots[:, :1] + boots[:, 1:2] * xx[None, :]
        ax2.fill_between(xx, *np.nanpercentile(band, [2.5, 97.5], axis=0), color=col, alpha=0.18, lw=0)
        ax2.plot(xx, est[0] + est[1] * xx, color=col, lw=1.1,
                 label=f'{TOPO_LABEL[topo]}: slope {est[1]:+.4f} [{lo:+.4f}, {hi:+.4f}]')
        ax2.plot(lsx + dx, meds, mk, color=col, ms=4.5, mec='k', mew=0.5, zorder=4)
        trends[topo] = (est, lo, hi, meds)
    ax2.axhline(0, color='k', lw=0.6, ls='--')
    ax2.text(lsx[-1] + 0.2, -0.0018, 'Pairwise parity', fontsize=7, ha='right', va='top')
    ax2.set_xticks(lsx, [str(K) for K in Ks])
    ax2.set_xlim(lsx[0] - 0.3, lsx[-1] + 0.3)
    arrays['fig3b|all'] = np.concatenate([arrays[f'fig3b|{t}|K{K}'] for t in F3B_TOPOS for K in Ks])
    ax2.set_ylim(-0.028, 0.036)
    ax2.set_yticks([-0.02, -0.01, 0.0, 0.01, 0.02, 0.03])
    ax2.set_xlabel('batches K (log$_2$ scale)')
    ax2.set_ylabel(r'Error$_{\mathrm{Joint}}$ − Error$_{\mathrm{Pairwise}}$ (min)')
    ax2.legend(loc='upper right', fontsize=6.8, handlelength=1.4, title='min per doubling of K [95% CI]',
               title_fontsize=6.8, frameon=True, facecolor='white', framealpha=0.85, edgecolor='none')
    _title(ax2, 'b', f'Scaling with K (budget {F3B_BUDGET}, ε = {F3B_EPS:.0%})')
    for ax in (ax1, ax2):
        _despine(ax)
    fig.tight_layout(w_pad=1.5)
    p = _save(fig, out, 'fig3_synthetic_results')
    plt.close(fig)
    dt = trends['density0.4']
    cap = (
        f'**fig3_synthetic_results.** (a) World-level median map error over the {len(d02["seeds"])} test worlds '
        f'(boxes: quartiles, whiskers: range). Stars are scored on the pairs they reach. (b) Paired per-world '
        f'difference; large markers: medians; lines: least-squares fit on log2 K with 95 % bands from '
        f'resampling test worlds within each K (script 02). Density 0.4 slope {dt[0][1]:+.4f} min per '
        f'doubling [{dt[1]:+.4f}, {dt[2]:+.4f}].'
    )
    return p, cap, arrays


# ── Figure 4 ──────────────────────────────────────────────────────────────────

def refit_broken_world(d02, s02) -> dict:
    """The broken F4A test world closest to the setting's median joint error, refitted and checked."""
    from jointalign.synthetic import BIN_MIN

    from trustalign import metrics
    v = stat02(d02, F4A, 'Joint')
    med = float(np.median(v))
    broken = [i for i in range(len(v)) if v[i] > s02.BREAK_MIN]
    i = min(broken, key=lambda j: (abs(v[j] - med), j))
    seed = d02['seeds'][i]
    world, anchors, _ = s02.world_and_anchors(F4A[0], seed)
    edges, kept, bad, hubs, _ = s02.prepare(world, anchors, F4A, seed)
    fit = s02.fitters(world.K, edges, hubs)
    cell = d02['cells'][(*F4A, seed)]
    runs = {
        'Joint cv': ('Joint', 'cv'),
        'Joint fixed': ('Joint', 'fixed'),
        'Pairwise cv': ('Composed pairwise', 'cv'),
    }
    fits, lam = {}, {}
    for key, (m, tag) in runs.items():
        lam[key] = cell[(m, tag)]['lam']
        fits[key] = fit[m](kept, lam[key])
        err = metrics.summarise(metrics.map_errors(world, fits[key].map), edges)['median_all']
        if abs(err - cell[(m, tag)]['median_all']) > 1e-8:
            raise AssertionError(f'refit of {key} gives {err}, script 02 has {cell[(m, tag)]["median_all"]}')
    errs = metrics.map_errors(world, fits['Joint cv'].map, sorted(edges))
    edge = max(sorted(edges), key=lambda e: float(np.median(errs[e])))
    return {'world': world, 'seed': seed, 'edges': edges, 'kept': kept, 'bad': bad, 'fits': fits, 'lam': lam,
            'edge': edge, 'errors': {k: cell[runs[k]]['median_all'] for k in runs}, 'median': med,
            'n_broken': len(broken), 'bin': BIN_MIN}


def cv_counts(d02, s02) -> dict:
    """Broken test worlds (joint model, CV λ) for the topologies script 03 does not diagnose, over the same
    K ∈ {5, 10} × budgets × ε > 0 settings: {topology: [broken, worlds]} (script 02's per-world errors)."""
    out: dict = {tp: [0, 0] for tp in F4B_UNDIAGNOSED}
    for s in d02['grid']:
        if s[1] in out and s[0] in (5, 10) and s[3] > 0:
            v = stat02(d02, s, 'Joint')
            out[s[1]][0] += int(np.sum(v > s02.BREAK_MIN))
            out[s[1]][1] += len(v)
    return out


def lambda_counts(d03, s02, s03) -> dict:
    """Broken test worlds per topology and λ rule over the corrupted settings of script 03."""
    out: dict = {}
    for s in d03['grid']:
        if s[3] == 0:
            continue
        e = s03.rule_errors(d03['sweep'], s, sorted(s02.TEST_SEEDS), 'Joint', d03['lams'][s]['Joint'])
        for rule, _, _ in F4B_RULES:
            out.setdefault((s[1], rule), [0, 0])
            out[(s[1], rule)][0] += int(np.sum(e[rule] > s02.BREAK_MIN))
            out[(s[1], rule)][1] += len(e[rule])
    return out


def fig4(out: Path, bw: dict, counts: dict, cv: dict, s02) -> tuple[Path, str, dict]:
    plt = _plt()
    from jointalign.synthetic import TIME_AXIS

    from trustalign import metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.6), gridspec_kw={'width_ratios': [1.25, 1]})
    world, (a, b), B = bw['world'], bw['edge'], bw['bin']
    t = TIME_AXIS[metrics.inside_mask(world, a, b, TIME_AXIS)]
    truth = world.true_map(a, b, t) - t
    curves = {'truth': truth}
    style = (('Pairwise cv', f'Pairwise (CV λ = {bw["lam"]["Pairwise cv"]:g})', C['Composed pairwise'], '-'),
             ('Joint fixed', 'Joint (fixed λ = 10)', C['Joint'], ':'),
             ('Joint cv', f'Joint (CV λ = {bw["lam"]["Joint cv"]:g})', C['Joint'], '-'))
    kept, bad = bw['kept'], bw['bad']
    m = (kept.ba == a) & (kept.bb == b)
    ta, tb = kept.ta[m] * B, kept.tb[m] * B
    ax1.scatter(ta[~bad[m]], tb[~bad[m]] - ta[~bad[m]], s=9, facecolor='none', edgecolor='0.3', lw=0.6,
                label=f'clean anchors ({int(np.sum(~bad[m]))})', zorder=2)
    ax1.scatter(ta[bad[m]], tb[bad[m]] - ta[bad[m]], s=14, marker='x', color=C['bad'], lw=0.9,
                label=f'corrupted anchors ({int(np.sum(bad[m]))})', zorder=2)
    for key, lab, col, ls in style:
        y = bw['fits'][key].map(a, b, t / B) * B - t
        curves[key] = y
        ax1.plot(t, y, color=col, ls=ls, lw=1.3 if key == 'Joint cv' else 1.1, label=lab, zorder=3)
    ax1.plot(t, truth, color=C['truth'], ls='--', lw=1.0, label='ground truth', zorder=4)
    ax1.set_xlim(0, 45)
    ax1.set_ylim(-12.0, 11.5)
    ax1.set_xlabel(f'retention time in batch {a} (min)')
    ax1.set_ylabel(rf'drift $g_{{{a}\to{b}}}(t) - t$ (min)')
    ax1.legend(loc='upper left', fontsize=6.8, ncol=2, handlelength=1.5, columnspacing=0.8,
               frameon=True, facecolor='white', framealpha=0.85, edgecolor='none')
    _title(ax1, 'a', f'Chain breakdown: test world {bw["seed"]}, edge {a}–{b}')

    topos = list(dict.fromkeys(tp for tp, _ in counts))
    order = [tp for tp in ('weak_star', 'chain', 'ring', 'density0.4') if tp in topos]
    width = 0.26
    tot = {}
    for ri, (rule, lab, col) in enumerate(F4B_RULES):
        ks = [counts[(tp, rule)][0] for tp in order]
        ns = [counts[(tp, rule)][1] for tp in order]
        tot[rule] = (sum(ks), sum(ns))
        x = np.arange(len(order)) + (ri - 1) * width
        ax2.bar(x, ks, width * 0.92, color=col, label=f'{lab}: {sum(ks)}/{sum(ns)} broken', edgecolor='k', lw=0.4)
        for xi, k in zip(x, ks):
            ax2.text(xi, k + 1.2, str(k), ha='center', va='bottom', fontsize=6.5)
    n_per = counts[(order[0], F4B_RULES[0][0])][1]
    ax2.set_xticks(range(len(order)), [TOPO_LABEL[tp] for tp in order])
    ax2.set_ylabel(f'broken test worlds (of {n_per})')
    ax2.set_ylim(0, max(counts[(tp, r)][0] for tp in order for r, _, _ in F4B_RULES) * 1.5)
    ax2.legend(loc='upper right', fontsize=6.8, handlelength=1.2)
    undiagnosed = ', '.join(f'{TOPO_LABEL[tp]} {k}/{n}' for tp, (k, n) in cv.items())
    ax2.text(0.97, 0.72, 'Not diagnosed (CV λ):\n' + undiagnosed,
             transform=ax2.transAxes,
             ha='right', va='top', fontsize=6.5, color='0.4', style='italic')
    _title(ax2, 'b', f'λ rule vs breakdown ({tot["tuned"][1]} corrupted worlds)')
    for ax in (ax1, ax2):
        _despine(ax)
    fig.tight_layout(w_pad=2.0)
    fig.subplots_adjust(wspace=0.28)
    p = _save(fig, out, 'fig4_breakdown_diagnosis')
    plt.close(fig)
    e = bw['errors']
    err = np.abs(curves['Joint cv'] - truth)
    above = t[err > F4A_ERR_MIN]
    half = t >= 22.5
    cap = (
        f'**fig4_breakdown_diagnosis.** (a) Test world {bw["seed"]} of K = {F4A[0]}, chain, budget {F4A[2]}, '
        f'ε = {F4A[3]:.0%} (the broken world closest to the setting median {bw["median"]:.3f} min; '
        f'{bw["n_broken"]}/{len(range(3, 23))} worlds broken), refitted on script 02\'s anchors. '
        f'Edge {a}–{b} is the chain edge with the largest joint error. World median map error: '
        f'Joint (CV) {e["Joint cv"]:.3f}, Joint (λ = 10) {e["Joint fixed"]:.3f}, '
        f'Pairwise {e["Pairwise cv"]:.3f} min. Edge {a}–{b}: {int(np.sum(~bad[m]))} clean and '
        f'{int(np.sum(bad[m]))} corrupted anchors; CV λ Joint {bw["lam"]["Joint cv"]:g}, '
        f'Pairwise {bw["lam"]["Pairwise cv"]:g}; Joint (CV) error above {F4A_ERR_MIN} min from '
        f'{above.min():.1f} to {above.max():.1f} min, at most {err.max():.2f} min '
        f'(at most {err[half].max():.2f} min from 22.5 min on). (b) Broken test worlds '
        f'(> {s02.BREAK_MIN} min) over script 03\'s corrupted settings ({len(order)} topologies × K ∈ {{5, 10}} × '
        f'3 budgets × 2 ε): ' + '; '.join(f'{lab} {tot[r][0]}/{tot[r][1]}' for r, lab, _ in F4B_RULES)
        + '. The oracle uses the truth: a bound, not a rule. '
        + 'Not diagnosed (script 02, joint, CV λ, same settings): '
        + '; '.join(f'{tp} {k}/{n}' for tp, (k, n) in cv.items()) + ' broken.'
    )
    arrays = {f'fig4a|{k}': v for k, v in curves.items()}
    arrays['fig4a|t'] = t
    arrays['fig4a|anchors_ta'] = ta                                  # min, batch a
    arrays['fig4a|anchors_dt'] = tb - ta                             # min, anchor drift t_b − t_a
    arrays['fig4a|anchors_bad'] = bad[m].astype(int)
    arrays['fig4a|edge_seed'] = np.array([a, b, bw['seed']])
    arrays['fig4a|lams'] = np.array([bw['lam'][k] for k in ('Joint cv', 'Joint fixed', 'Pairwise cv')])
    arrays['fig4a|world_errors'] = np.array([e[k] for k in ('Joint cv', 'Joint fixed', 'Pairwise cv')])
    arrays['fig4a|setting_median_nbroken'] = np.array([bw['median'], bw['n_broken']])
    arrays.update({f'fig4b|{tp}|{r}': np.array(counts[(tp, r)]) for tp in order for r, _, _ in F4B_RULES})
    return p, cap, {'arrays': arrays, 'totals': tot}


# ── Figure 5 ──────────────────────────────────────────────────────────────────

def fig5(out: Path, fish_cells, s04) -> tuple[Path, str, dict]:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(SINGLE, 2.5), sharey=True)
    eps: tuple[float, ...] = F5_EPSILONS
    budgets: tuple[int, ...] = F5_BUDGETS
    meths: tuple[str, ...] = F5_METHODS
    x = np.arange(len(eps), dtype=float)
    arrays: dict[str, np.ndarray] = {}
    for pi, (ax, budget) in enumerate(zip(axes, budgets)):
        for mi, m in enumerate(meths):
            q_list = []
            for e in eps:
                v = stat04(fish_cells, ('complete', budget, e), m, s04)
                arrays[f'fig5|B{budget}|eps{e}|{m}'] = v
                q_list.append(np.nanpercentile(v, [25, 50, 75]))
            q_arr = np.array(q_list)
            xx = x + (mi - 1) * 0.18
            ax.errorbar(xx, q_arr[:, 1], yerr=[q_arr[:, 1] - q_arr[:, 0], q_arr[:, 2] - q_arr[:, 1]],
                        color=C[m], marker=MARK[m], ms=3.8, lw=0.9, capsize=1.8, elinewidth=0.8,
                        mec='k', mew=0.3, label='Star (strong hub)' if m == 'Star' else LABEL[m])
        ax.set_xticks(x, [f'{e:.0%}' for e in eps])
        ax.set_xlim(-0.45, len(eps) - 0.55)
        ax.set_yscale('log')
        ax.set_xlabel('corruption ε')
        _title(ax, 'ab'[pi], f'budget {budget}', fontsize=9.5)
        _despine(ax)
    axes[0].set_ylabel('held-out anchor error (min)')
    from matplotlib.ticker import FuncFormatter, LogLocator
    axes[0].yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
    axes[0].yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc='upper center', ncol=3, bbox_to_anchor=(0.55, 1.07), fontsize=7, handlelength=1.4,
               columnspacing=0.8)
    fig.tight_layout(w_pad=0.6)
    p = _save(fig, out, 'fig5_fish_results')
    plt.close(fig)
    cap = (f'**fig5_fish_results.** Fish oil data (K = 5, complete graph): held-out spectral-anchor error, median '
           f'and interquartile range over {len(s04.DRAWS)} thinning/corruption draws per condition.')
    return p, cap, arrays


# ── website figures ───────────────────────────────────────────────────────────

def site_figures(site: Path, world, anchors, cell_bad, s02) -> list[Path]:
    """Fig 1 (b, c) and fig 2 as SVG for docs/assets/, one file per colour theme (fig1_bc.svg,
    fig1_bc_dark.svg, fig2_graphs.svg, fig2_graphs_dark.svg). Panel (a) of fig 1 is never exported."""
    plt = _plt()
    site.mkdir(parents=True, exist_ok=True)
    cols = batch_colors(world.K)
    panels = graph_panels(world, anchors, s02)
    paths = []
    for theme, (ink, bg) in SITE_THEMES.items():
        suffix = '' if theme == 'light' else '_dark'
        with plt.rc_context(_site_rc(ink)):
            fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(DOUBLE, 2.5), gridspec_kw={'width_ratios': [1.3, 1]})
            panel_1b(ax_b, world, cell_bad, cols, ink=ink, bg=bg)
            panel_1c(ax_c, world, cols)
            fig.tight_layout(w_pad=2.0)
            paths.append(_save_svg(fig, site, f'fig1_bc{suffix}'))
            plt.close(fig)
            fig, axes = plt.subplots(1, len(panels), figsize=(DOUBLE, 1.35))
            draw_graphs(fig, axes, world, panels, cols, ink=ink)
            paths.append(_save_svg(fig, site, f'fig2_graphs{suffix}'))
            plt.close(fig)
    return paths


# ── verification ──────────────────────────────────────────────────────────────

def verify(d02, rec, tot4, fish_cells, s02, s04) -> list[str]:
    """Recomputed values vs the committed reports and the manuscript; raises on any mismatch."""
    t1 = parse_full_table1(RESULTS / 'full_sparsity.txt')
    rq = parse_full_rq1c(RESULTS / 'full_sparsity.txt')
    lt = parse_lambda_totals(RESULTS / 'diagnose_lambda.txt')
    t2 = parse_fish_table2(RESULTS / 'fish_sparsity.txt')
    col02 = {m: i for i, m in enumerate(s02.METHODS)}
    rows = []

    def med13(s, m):
        return f'{float(np.nanmedian(stat02(d02, s, m))):.3f}'

    for key, (s, m) in {
        'Table I  K=5 complete B=10 ε=0    Joint': ((5, 'complete', 10, 0.0), 'Joint'),
        'Table I  K=5 complete B=10 ε=0.3  Joint': ((5, 'complete', 10, 0.3), 'Joint'),
        **{f'Table I  K=5 {t:<8} B=30 ε=0.3  {LABEL[m]}': ((5, t, 30, 0.3), m)
           for t in ('complete', 'chain', 'ring') for m in ('Joint', 'Star', 'Composed pairwise')},
    }.items():
        rows.append((key, med13(s, m), t1[s][col02[m]], MANUSCRIPT[key]))
    for topo in ('density0.4', 'complete'):
        sets = tuple((K, topo, 30, 0.15) for K in (5, 10, 20))
        rep = rq[(topo, 0.15)]
        for j, s in enumerate(sets):
            d = stat02(d02, s, 'Joint') - stat02(d02, s, 'Composed pairwise')
            key = f'RQ1c {topo:<10} ε=0.15 K={s[0]:<2} Δ'
            rows.append((key, f'{np.median(d):+.4f}', rep[j], MANUSCRIPT[key]))
        if topo == 'density0.4':
            est, boots = rec[sets]
            lo, hi = np.nanpercentile(boots[:, 1], [2.5, 97.5])
            for key, v, r in (('RQ1c density0.4 ε=0.15 slope', est[1], rep[3]),
                              ('RQ1c density0.4 ε=0.15 slope CI low', lo, rep[4]),
                              ('RQ1c density0.4 ε=0.15 slope CI high', hi, rep[5])):
                rows.append((key, f'{v:+.4f}', r, MANUSCRIPT[key]))
    for key, rule in (('λ diag broken, CV λ', 'tuned'), ('λ diag broken, λ = 10', 'λ = 10'),
                      ('λ diag broken, per-world oracle λ', 'per world')):
        k, n = tot4[rule]
        rows.append((key, f'{k}/{n}', f'{lt["Joint"][rule]}/{lt["n"]}', MANUSCRIPT[key]))
    col04 = {m: i for i, m in enumerate(s02.METHODS)}
    for budget, e in ((10, 0.3), (100, 0.0)):
        for m in F5_METHODS:
            key = f'Table II B={budget:<3} ε={e:<3g} {LABEL[m]}'
            v = f'{float(np.nanmedian(stat04(fish_cells, ("complete", budget, e), m, s04))):.3f}'
            rows.append((key, v, t2[('complete', budget, e)][col04[m]], MANUSCRIPT[key]))
    w = max(len(r[0]) for r in rows)
    L = [f'{"quantity":<{w}}  {"recomputed":>10}  {"report":>10}  {"manuscript":>10}  ok',
         '-' * (w + 42)]
    bad = []
    for key, v, r, ms in rows:
        ok = v == r == ms
        L.append(f'{key:<{w}}  {v:>10}  {r:>10}  {ms:>10}  {"✓" if ok else "✗"}')
        if not ok:
            bad.append(key)
    missing = set(MANUSCRIPT) - {r[0] for r in rows}
    if missing:
        raise AssertionError(f'manuscript checks not computed: {sorted(missing)}')
    print('\n'.join(L))
    if bad:
        raise AssertionError(f'{len(bad)} mismatches: {bad}')
    return L


# ── main ──────────────────────────────────────────────────────────────────────

def main(out: Path = FIGS, results: Path = RESULTS, site: Path | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    s02, s03, s04 = (_script(n) for n in ('02_full_sparsity.py', '03_diagnose_lambda.py', '04_fish_sparsity.py'))
    print('loading script 02 checkpoints …', flush=True)
    d02 = load_s02(results / 'full_sparsity_cache', s02)
    print('replaying script 02\'s report (exact bootstrap) …', flush=True)
    rec = replay_rq1c(d02, s02)
    d03 = load_s03(results / 'diagnose_lambda_cache', s02, s03, d02)
    fish_cells = load_s04(results / 'fish_sparsity_cache', s04)

    print('building test world and refitting fig 4 (a) …', flush=True)
    world, anchors, _ = s02.world_and_anchors(WORLD_K, WORLD_SEED)
    cell_bad = corrupted_anchor(world, anchors, s02)
    fish = fish_sample(s04.DATA)
    bw = refit_broken_world(d02, s02)
    counts = lambda_counts(d03, s02, s03)
    cv = cv_counts(d02, s02)

    paths, caps, arrays = [], [], {}
    p, c = fig1(out, world, cell_bad, fish)
    paths.append(p)
    caps.append(c)
    p, c = fig2(out, world, anchors, s02)
    paths.append(p)
    caps.append(c)
    p, c, a = fig3(out, d02, rec, s02)
    paths.append(p)
    caps.append(c)
    arrays.update(a)
    p, c, r4 = fig4(out, bw, counts, cv, s02)
    paths.append(p)
    caps.append(c)
    arrays.update(r4['arrays'])
    p, c, a = fig5(out, fish_cells, s04)
    paths.append(p)
    caps.append(c)
    arrays.update(a)

    if site is not None:
        paths += site_figures(site, world, anchors, cell_bad, s02)

    np.savez(results / 'cache_plot_data.npz', **arrays)
    (out / 'captions_tai.md').write_text('# Experiment 1 figures (scripts/05_figures.py)\n\n'
                                         + '\n\n'.join(caps) + '\n')
    print()
    table = verify(d02, rec, r4['totals'], fish_cells, s02, s04)
    print('\n' + '\n'.join(str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in paths))
    return {'paths': paths, 'captions': caps, 'table': table}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=FIGS)
    ap.add_argument('--site', type=Path, nargs='?', const=ROOT / 'docs' / 'assets', default=None,
                    help='also write the website\'s SVG figures (fig 1 b, c and fig 2) to this folder')
    a = ap.parse_args()
    main(out=a.out, site=a.site)

"""
20_paper_figures.py — the paper's figures that script 11 does not draw, and their captions.

  fig1_data       one draft-run world (scripts/11_draft_connectivity.py, seed 0 → world seed 5000):
                  (a) one TIC per batch for four batches, each on its own retention-time axis, with
                  three compounds marked; (b) the true drift d_k of all ten batches
  fig2_graphs     the anchor-graph structures of that same world, drawn exactly as script 11 drew
                  them (same anchor subsample, chords and deletion order)
  fig_regime      the regime map from results/draft_connectivity.txt, Section 3: paired median over
                  worlds of (joint − competitor) and the number of worlds the joint model wins, on a
                  symmetric-log colour scale
  fig_breakdown   per-world refits from results/diagnose_breakdown.txt, Section 1: joint, oracle (no
                  corrupted anchors), 10 Huber rounds, λ × 10 and composed pairwise, ring and chain
  fig_connectivity  results/draft_connectivity.txt, Sections 1 and 2, as one full-width figure:
                  (a, b) median map error over all pairs as edges are removed from the complete graph;
                  (c, d) median map error on indirect pairs as loops are added to the chain; clean and
                  ε = 30 %. Same numbers as script 11's edge_removal.png and loop_closure.png, redrawn
                  to be legible at journal size.

Every number in a caption is computed here from the world or the results files; nothing is typed.
fig_regime and fig_breakdown read the committed text reports only, so they need neither the data nor
a rerun. fig1_data and fig2_graphs rebuild one synthetic world and harvest its anchors (seconds).

Output: figs/paper/*.pdf and *.png (not in git), figs/paper/captions.md (tracked).
"""
from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path

import numpy as np
from jointalign import harvest_anchors
from jointalign.synthetic import TIME_AXIS

from trustalign import graphs, worlds

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FIGS = ROOT / 'figs' / 'paper'

DRAFT_SEED = 0                 # script 11's first world (worlds.make(K, seed=5000 + seed))
SHOW = (0, 3, 6, 9)            # batches drawn in fig1_data (a)
MARK_WINDOWS = ((10, 18), (20, 28), (30, 38))   # min: one abundant compound marked in each
NOISES = ('clean', 'jitter 0.5', 'jitter 1.0', 'ε 15%', 'ε 30%')
NOISE_LABELS = ('clean', 'jitter 0.5', 'jitter 1', 'ε 15%', 'ε 30%')
GRAPH_PANELS = (('star', 'star'), ('chain', 'chain'), ('chain+2', 'chain+2'), ('ring', 'ring'),
                ('density1.0', 'density 1.0'), ('density0.4', 'density 0.4'), ('density0.2', 'density 0.2'))
BROKEN = 0.3                   # min: scripts/12_diagnose_breakdown.py's threshold
COLORS = {'Joint': '#2a78d6', 'Composed pairwise': '#eb6834', 'Star': '#eda100'}   # script 11's


def _load_script(name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(Path(name).stem, ROOT / 'scripts' / name)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(out / f'{name}.png', dpi=200, bbox_inches='tight')


# ── Figures 1 and 2: one draft world ──────────────────────────────────────────────────────────

def draft_world(seed: int = DRAFT_SEED, **world_kw):
    """The world, its structures and its batch specs, drawn exactly as script 11's run_world."""
    s11 = _load_script('11_draft_connectivity.py')
    k = s11.K
    world = worlds.make(k, seed=5000 + seed, **world_kw)
    anchors = harvest_anchors(world.chromas, world.tic_raws)
    rng = np.random.default_rng([seed, 11])
    graphs.thin(anchors, set(combinations(range(k), 2)), s11.BUDGET, rng)    # advances rng as script 11
    return world, s11.structures_for(k, world.sizes, rng)


def marked_compounds(world) -> np.ndarray:
    """The latent time of the compound with the tallest batch-0 TIC peak in each window."""
    tau = np.asarray(world.truth['compound_tau_min'])
    tic0 = world.tic_raws[0][0] / world.tic_raws[0][0].max()
    height = np.interp(tau + world.specs[0].drift(tau), TIME_AXIS, tic0)
    out = []
    for lo, hi in MARK_WINDOWS:
        sel = (tau > lo) & (tau < hi)
        out.append(tau[sel][np.argmax(height[sel])])
    return np.array(out)


def fig_data(world, out: Path) -> str:
    plt = _plt()
    cols = plt.get_cmap('tab10').colors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6), gridspec_kw={'width_ratios': [1.35, 1]})
    marks = marked_compounds(world)
    rows = len(SHOW)
    tics = {k: world.tic_raws[k][0] / world.tic_raws[k][0].max() for k in SHOW}
    level = lambda r, k, t: (rows - 1 - r) * 1.15 + np.interp(t, TIME_AXIS, tics[k]) + 0.12
    for r, k in enumerate(SHOW):
        off = (rows - 1 - r) * 1.15
        ax1.plot(TIME_AXIS, tics[k] + off, color=cols[k], lw=0.8)
        ax1.text(44.5, off + 0.35, f'batch {k}', color=cols[k], ha='right', fontsize=7)
        rt = marks + world.specs[k].drift(marks)
        ax1.plot(rt, level(r, k, rt), 'v', color='k', ms=3.5)
    for c in range(len(marks)):
        xs = [marks[c] + world.specs[k].drift(marks[c]) for k in SHOW]
        ax1.plot(xs, [level(r, k, x) for r, (k, x) in enumerate(zip(SHOW, xs))], ls=':', color='0.4', lw=0.8)
    ax1.set_xlim(0, 45)
    ax1.set_yticks([])
    ax1.set_xlabel('retention time in each batch (min)')
    ax1.set_title('(a) One sample per batch, each on its own axis', fontsize=8, loc='left')
    drifts = np.array([s.drift(TIME_AXIS) for s in world.specs])
    for k in range(world.K):
        ax2.plot(TIME_AXIS, drifts[k], color=cols[k], lw=1)
    ax2.axhline(0, color='0.7', lw=0.6)
    ax2.set_xlabel('latent retention time (min)')
    ax2.set_ylabel('true drift $d_k$ (min)')
    ax2.set_title(f'(b) True drift of all {world.K} batches', fontsize=8, loc='left')
    fig.tight_layout()
    _save(fig, out, 'fig1_data')
    plt.close(fig)
    return (f'**fig1_data.** One draft-run world ({world.K} batches, world seed {world.seed}). (a) Normalised '
            f'TIC of the first sample of batches {", ".join(map(str, SHOW))}, each on its own retention-time '
            f'axis; triangles mark three compounds (latent {", ".join(f"{m:.2f}" for m in marks)} min). '
            f'(b) True drift of every batch, from {drifts.min():.2f} to {drifts.max():.2f} min.')


def fig_graphs(world, structures: dict, out: Path) -> str:
    plt = _plt()
    cols = plt.get_cmap('tab10').colors
    k = world.K
    ang = np.pi / 2 - 2 * np.pi * np.arange(k) / k
    xy = np.c_[np.cos(ang), np.sin(ang)]
    fig, axes = plt.subplots(1, len(GRAPH_PANELS), figsize=(7.16, 1.35))
    counts = []
    for ax, (key, name) in zip(axes, GRAPH_PANELS):
        edges = structures[key]
        counts.append(f'{name} {len(edges)}')
        for a, b in sorted(edges):
            ax.plot(*xy[[a, b]].T, color='0.35', lw=0.6, zorder=1)
        ax.scatter(*xy.T, s=14, c=[cols[i] for i in range(k)], zorder=2, edgecolor='k', linewidth=0.3)
        ax.set_title(f'{name} ({len(edges)})', fontsize=7)
        ax.set_aspect('equal')
        ax.axis('off')
    fig.tight_layout(pad=0.2)
    _save(fig, out, 'fig2_graphs')
    plt.close(fig)
    return (f'**fig2_graphs.** Anchor-graph structures of the same world, exactly as script 11 drew them; '
            f'edges per structure: {", ".join(counts)}.')


# ── Figure: regime map from results/draft_connectivity.txt ───────────────────────────────────

_PAIRED = re.compile(r'([+-]\d+\.\d+)\s+(\d+)/(\d+) p=(\S+)')


def parse_paired(path: Path) -> dict[tuple[str, str, int], tuple[float, int, int, float]]:
    """Section 3 of draft_connectivity.txt: (structure, noise, column) → (Δ, wins, n, p). Columns:
    0 vs composed pairwise, 1 vs star (its pairs), 2 vs composed pairwise on indirect pairs."""
    lines = path.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith('3. Paired comparisons'))
    out = {}
    for ln in lines[start + 2:]:
        if not ln.strip() or ln.startswith('='):
            break
        first = _PAIRED.search(ln)
        if first is None:
            continue
        structure, noise = ln[:first.start()].split(maxsplit=1)
        noise = noise.strip()
        if noise not in NOISES:
            raise ValueError(f'unknown noise level {noise!r}')
        for j, m in enumerate(_PAIRED.finditer(ln)):
            p = float('nan') if m.group(4) == 'nan' else float(m.group(4))
            out[(structure, noise, j)] = (float(m.group(1)), int(m.group(2)), int(m.group(3)), p)
    if not out:
        raise ValueError(f'no paired comparisons found in {path}')
    return out


def fig_regime(vals: dict, out: Path) -> str:
    plt = _plt()
    from matplotlib.colors import SymLogNorm
    structs = list(dict.fromkeys(s for s, _, _ in vals))
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.5), sharey=True)
    norm = SymLogNorm(linthresh=0.01, linscale=1.0, vmin=-1.2, vmax=1.2, base=10)
    panels = ((0, 'Joint − composed pairwise (all pairs)'), (1, 'Joint − star (pairs the star reaches)'))
    im = None
    for ax, (j, title) in zip(axes, panels):
        D = np.full((len(structs), len(NOISES)), np.nan)
        for r, st in enumerate(structs):
            for c, no in enumerate(NOISES):
                if (st, no, j) in vals:
                    D[r, c] = vals[(st, no, j)][0]
        im = ax.imshow(np.ma.masked_invalid(D), cmap='RdBu_r', norm=norm, aspect='auto')
        for r, st in enumerate(structs):
            for c, no in enumerate(NOISES):
                if (st, no, j) not in vals:
                    continue
                d, w, n, _ = vals[(st, no, j)]
                txt = 'same fit' if (j == 1 and st == 'star') else f'{d:+.3f}\n{w}/{n}'
                ax.text(c, r, txt, ha='center', va='center', fontsize=6.3, linespacing=1.1,
                        color='white' if abs(d) > 0.25 else '0.1')
        ax.set_xticks(range(len(NOISES)), NOISE_LABELS, rotation=30, ha='right')
        ax.set_yticks(range(len(structs)), [s.replace('density', 'density ') for s in structs])
        ax.set_title(title, fontsize=8)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, ticks=[-1, -0.1, -0.01, 0, 0.01, 0.1, 1])
    cb.ax.set_yticklabels(['−1', '−0.1', '−0.01', '0', '0.01', '0.1', '1'])
    cb.set_label('paired median difference (min); blue: joint lower')
    _save(fig, out, 'fig_regime')
    plt.close(fig)
    left = {key: v for key, v in vals.items() if key[2] == 0}
    best = min(left, key=lambda key: left[key][0])
    d, w, n, _ = left[best]
    return (f'**fig_regime.** Paired median over worlds of (joint − competitor) median map error (min), with '
            f'the worlds the joint model wins; symmetric-log colour, linear within ±0.01 min. Largest advantage '
            f'over composed pairwise: {d:+.3f} min ({best[0]}, {best[1]}, {w}/{n} worlds).')


# ── Figure: breakdown from results/diagnose_breakdown.txt ────────────────────────────────────

BREAKDOWN_COLS = ('joint', 'pairw', 'recall', 'false', 'oracle', 'huber10', 'lam10')
_ROW = re.compile(r'^\s+draft (ring|chain) ε30% seed (\d+)\s+' + r'\s+'.join([r'(\d+\.\d+)'] * 7))


def parse_breakdown(path: Path) -> dict[str, np.ndarray]:
    """Section 1 of diagnose_breakdown.txt for the draft cells: case → (seeds, 7) array in the column
    order of BREAKDOWN_COLS."""
    rows: dict[str, dict[int, list[float]]] = {}
    for ln in path.read_text().splitlines():
        m = _ROW.match(ln)
        if m:
            rows.setdefault(m.group(1), {})[int(m.group(2))] = [float(x) for x in m.groups()[2:]]
    if not rows:
        raise ValueError(f'no draft rows found in {path}')
    return {case: np.array([r[s] for s in sorted(r)]) for case, r in rows.items()}


def fig_breakdown(data: dict[str, np.ndarray], out: Path) -> str:
    plt = _plt()
    series = (('pairw', 'Composed pairwise', COLORS['Composed pairwise'], 's'),
              ('lam10', 'Joint, λ × 10', COLORS['Star'], 'v'),
              ('joint', 'Joint', COLORS['Joint'], 'o'),
              ('huber10', 'Joint, 10 Huber rounds', '#7a4fc9', 'D'),
              ('oracle', 'Joint, corrupted anchors removed', '0.45', 'x'))
    cases = [c for c in ('ring', 'chain') if c in data]
    fig, axes = plt.subplots(1, len(cases), figsize=(7.16, 2.5), sharey=True, squeeze=False)
    offsets = np.linspace(-0.24, 0.24, len(series))
    notes = []
    for ax, case in zip(axes[0], cases):
        D = data[case]
        n = len(D)
        for o, (key, name, color, mk) in zip(offsets, series):
            ax.plot(np.arange(n) + o, D[:, BREAKDOWN_COLS.index(key)], mk, color=color, ms=4.5, mew=1.2,
                    label=name, ls='none')
        ax.axhline(BROKEN, color='0.5', ls='--', lw=0.8)
        broken = np.flatnonzero(D[:, 0] > BROKEN)
        for s in broken:
            ax.axvspan(s - 0.4, s + 0.4, color='0.92', zorder=0)
        ax.set_yscale('log')
        ax.set_xticks(range(n))
        ax.set_xlabel('world (seed)')
        ax.set_title(f'{case}, ε = 30%', fontsize=8, loc='left')
        oracle = D[:, BREAKDOWN_COLS.index('oracle')]
        notes.append(f'{case}: {len(broken)}/{n} worlds broken, oracle at most {oracle.max():.3f} min')
    axes[0][0].set_ylabel('median map error (min)')
    axes[0][-1].legend(fontsize=6.5, frameon=False, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    _save(fig, out, 'fig_breakdown')
    plt.close(fig)
    return (f'**fig_breakdown.** Per-world median map error under ε = 30 %, same anchors refitted; shaded: '
            f'joint above {BROKEN} min. {"; ".join(notes)}.')


# ── Figure: connectivity sweeps from results/draft_connectivity.txt ──────────────────────────

METHODS3 = ('Joint', 'Composed pairwise', 'Composed PCHIP')
_NUM = r'(\d+\.\d+)'


def parse_medians(path: Path) -> tuple[dict, dict]:
    """Sections 1 and 2 of draft_connectivity.txt: (structure, noise) → {method: median map error}
    over all ordered pairs (Section 1: Joint, Star, Composed pairwise, Composed PCHIP) and over
    indirect pairs (Section 2: Joint, Composed pairwise, Composed PCHIP)."""
    lines = path.read_text().splitlines()
    noise_re = '|'.join(re.escape(n) for n in NOISES)
    s1 = re.compile(rf'^\s+(\S+)\s+\d+\s+({noise_re})\s+' + r'\s+'.join([_NUM] * 4) + r'\s+' + _NUM + r'$')
    s2 = re.compile(rf'^\s+(\S+)\s+({noise_re})\s+' + r'\s+'.join([_NUM] * 3) + r'$')
    sec = {k: next(i for i, ln in enumerate(lines) if ln.startswith(f'{k}. ')) for k in (1, 2, 3)}
    all_pairs, indirect = {}, {}
    for ln in lines[sec[1]:sec[2]]:
        m = s1.match(ln)
        if m:
            v = [float(x) for x in m.groups()[2:6]]
            all_pairs[(m.group(1), m.group(2))] = dict(zip(('Joint', 'Star', 'Composed pairwise',
                                                             'Composed PCHIP'), v))
    for ln in lines[sec[2]:sec[3]]:
        m = s2.match(ln)
        if m:
            indirect[(m.group(1), m.group(2))] = dict(zip(METHODS3, [float(x) for x in m.groups()[2:]]))
    if not all_pairs or not indirect:
        raise ValueError(f'no median tables found in {path}')
    return all_pairs, indirect


def fig_connectivity(all_pairs: dict, indirect: dict, out: Path) -> str:
    plt = _plt()
    colors = {'Joint': COLORS['Joint'], 'Composed pairwise': COLORS['Composed pairwise'],
              'Composed PCHIP': '#1baf7a'}
    markers = {'Joint': 'o', 'Composed pairwise': 's', 'Composed PCHIP': '^'}
    dens = ['density1.0', 'density0.6', 'density0.4', 'density0.2']
    loops = ['chain', 'chain+1', 'chain+2', 'ring']
    frac = [float(d[7:]) for d in dens]
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 2.15), sharey=True)
    panels = (('(a) Removing edges, clean', all_pairs, dens, 'clean'),
              ('(b) Removing edges, ε = 30%', all_pairs, dens, 'ε 30%'),
              ('(c) Adding loops, clean', indirect, loops, 'clean'),
              ('(d) Adding loops, ε = 30%', indirect, loops, 'ε 30%'))
    for i, (ax, (title, table, structs, noise)) in enumerate(zip(axes, panels)):
        x = frac if i < 2 else list(range(len(structs)))
        for m in METHODS3:
            ax.plot(x, [table[(s, noise)][m] for s in structs], marker=markers[m], color=colors[m], lw=1.5,
                    ms=4, label=m)
        if i < 2:
            ax.invert_xaxis()
            ax.set_xticks(frac, [f'{f:g}' for f in frac])
            ax.set_xlabel('fraction of pairs linked')
        else:
            ax.set_xticks(x, ['chain', '+1', '+2', 'ring'])
            ax.set_xlabel('structure')
        ax.set_yscale('log')
        ax.set_title(title, fontsize=7.5, loc='left')
    axes[0].set_ylabel('median map error (min)')
    axes[2].text(0.02, 0.02, 'indirect pairs only', transform=axes[2].transAxes, fontsize=6.5, color='0.4')
    axes[3].text(0.02, 0.02, 'indirect pairs only', transform=axes[3].transAxes, fontsize=6.5, color='0.4')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    _save(fig, out, 'fig_connectivity')
    plt.close(fig)
    j, p = 'Joint', 'Composed pairwise'
    a, b = all_pairs[('density1.0', 'ε 30%')], all_pairs[('density0.2', 'ε 30%')]
    c, d = indirect[('chain', 'ε 30%')], indirect[('ring', 'ε 30%')]
    return (f'**fig_connectivity.** Under ε 30 %, from all pairs to 0.2 linked: joint {a[j]:.3f} → '
            f'{b[j]:.3f} min, composed pairwise {a[p]:.3f} → {b[p]:.3f} min. Indirect pairs under ε 30 %: '
            f'joint {c[j]:.3f} min on '
            f'the chain and {d[j]:.3f} min on the ring; composed pairwise {c[p]:.3f} and {d[p]:.3f} min.')


# ── Main ──────────────────────────────────────────────────────────────────────────────────────

def main(results: Path = RESULTS, out: Path = FIGS, world: bool = True, **world_kw) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    caps = ['# Paper figures (scripts/20_paper_figures.py)', '',
            'Numbers are computed from the world or from the results files named in each caption.', '']
    if world:
        w, structures = draft_world(**world_kw)
        caps += [fig_data(w, out), '', fig_graphs(w, structures, out), '']
    caps += [fig_regime(parse_paired(results / 'draft_connectivity.txt'), out), '',
             fig_breakdown(parse_breakdown(results / 'diagnose_breakdown.txt'), out), '',
             fig_connectivity(*parse_medians(results / 'draft_connectivity.txt'), out), '']
    (out / 'captions.md').write_text('\n'.join(caps))
    print('\n'.join(caps))
    return caps


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-world', action='store_true',
                    help='skip fig1_data and fig2_graphs (no world rebuild); draw only the results figures')
    a = ap.parse_args()
    main(world=not a.no_world)

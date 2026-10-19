"""
export_ledger.py — the numbers behind the paper as taura ledger results (ledger/results/*.json).

Reads only the committed reports, so it runs anywhere (CI included) and re-running is a no-op:

  results/full_sparsity.txt    script 02: the synthetic grid (map error, RQ1a–c, breakdown, sanity
                               check, fixed λ)
  results/diagnose_lambda.txt  script 03: map error and broken test worlds under each λ rule
  results/fish_sparsity.txt    script 04: the fish arm (data, held-out error, RQ1a, broken draws)
  figs/paper/captions_tai.md   script 05: the numbers in the figure captions

Values carry the precision they were printed with. A setting is k<K>-<topology>-b<budget>-e<ε in %>
(k5-chain-b30-e30); methods are joint, star, star-weak, pairwise, pchip, with <method>-lambda for the
tuned λ in brackets. Keys for \\res{} in the paper: map-error/k5-chain-b30-e30/joint,
rq1a/k5-e0/delta, scaling/density0.4-e15/slope, lambda-broken/all/joint-per-world,
fish-error/complete-b10-e30/star, fish-data/anchors/all.

Output: ledger/results/*.json
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
FULL = RESULTS / 'full_sparsity.txt'
LAMBDA = RESULTS / 'diagnose_lambda.txt'
FISH = RESULTS / 'fish_sparsity.txt'
CAPTIONS = ROOT / 'figs' / 'paper' / 'captions_tai.md'
OUT = ROOT / 'ledger' / 'results'

METHODS = {'Joint': 'joint', 'Star': 'star', 'Star (weak hub)': 'star-weak',
           'Composed pairwise': 'pairwise', 'Composed PCHIP': 'pchip'}
RULES = {'tuned': 'tuned', 'λ = 10': 'lambda10', 'own CV': 'own-cv', 'best λ': 'best', 'per world': 'per-world'}
TOPOLOGIES = ('complete', 'density0.4', 'ring', 'chain', 'star', 'weak_star')
NUM = r'[-+]?(?:\d[\d,]*\.?\d*|\.\d+)'
CELL = re.compile(rf'(—|{NUM})(?:\s*\[\s*({NUM})\s*\])?')
# ε, median Δ [95 % CI] and [min Δ, max Δ], as in the RQ1a tables
DELTA_CI = rf'({NUM})\s+({NUM}) \[({NUM}), ({NUM})\]\s+\[({NUM}), ({NUM})\]'
SETTING = re.compile(rf'^\s*(?:(\d+)\s+)?({"|".join(map(re.escape, TOPOLOGIES))})\s+(\d+)\s+({NUM})\s+(.*)$')


def num(s: str) -> float | int:
    s = s.replace(',', '').replace('+', '')
    return int(s) if re.fullmatch(r'-?\d+', s) else float(s)


def eps(s: str) -> str:
    return f'e{round(float(s) * 100)}'


def setting(k: str | None, topo: str, budget: str, e: str) -> str:
    return '-'.join(([f'k{k}'] if k else []) + [topo.replace('_', '-'), f'b{budget}', eps(e)])


def provenance(text: str) -> list[str]:
    """The report's provenance header: package lines and the environment line."""
    return [line for line in text.splitlines()[2:8] if ' @ ' in line or line.startswith('environment:')]


def source(path: Path) -> dict:
    out = {'path': str(path.relative_to(ROOT)), 'hash': 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()}
    prov = provenance(path.read_text()) if path.suffix == '.txt' else []
    return out | ({'provenance': prov} if prov else {})


def sections(text: str) -> dict[str, list[str]]:
    """{'1': lines of section 1, ...}: a section starts at a line '<n>. ...' at the margin."""
    out: dict[str, list[str]] = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r'^(\d+)\. ', line)
        if m:
            cur = m.group(1)
            out[cur] = [line]
        elif cur and not line.startswith('====='):
            out[cur].append(line)
    return out


def columns(header: str, after: str) -> list[str]:
    """Column names of a fixed-width header, after the column named `after`."""
    return [c for c in re.split(r'\s{2,}', header.split(after, 1)[1].strip()) if c]


def grid(lines: list[str], after: str = 'ε') -> dict[str, dict]:
    """A table with one row per setting and 'value [λ]' cells: {setting: {method: v, method-lambda: λ}}."""
    cols = columns(lines[1], after)
    out: dict[str, dict] = {}
    for line in lines[2:]:
        m = SETTING.match(line)
        if not m:
            continue
        cells = CELL.findall(m.group(5))
        if len(cells) != len(cols):
            raise ValueError(f'{len(cells)} cells for {len(cols)} columns: {line!r}')
        row = out.setdefault(setting(*m.groups()[:4]), {})
        for col, (v, lam) in zip(cols, cells):
            name = METHODS.get(col, re.sub(r'[^a-z0-9.]+', '-', col.lower()).strip('-'))
            if v != '—':
                row[name] = num(v)
            if lam:
                row[f'{name}-lambda'] = num(lam)
    if not out:
        raise ValueError(f'no rows under {lines[0][:60]!r}')
    return out


def result(name: str, label: str, metrics: dict, path: Path) -> dict:
    return {'name': name, 'label': label, 'metrics': metrics, 'runs': [],
            'produced_by': 'scripts/export_ledger.py', 'source': source(path)}


# ── script 02: the synthetic grid ────────────────────────────────────────────────────────────────

def full_sparsity() -> dict[str, dict]:
    text = FULL.read_text()
    s = sections(text)
    head = re.search(r'(\d+) settings; tuning worlds \[([\d, ]+)\], test worlds (\d+)–(\d+) \((\d+)\)', text)
    assert head, 'full_sparsity.txt: no grid summary line'
    tuning = [int(x) for x in head.group(2).split(',')]
    design = {'settings': int(head.group(1)), 'tuning-worlds': len(tuning), 'test-worlds': int(head.group(5)),
              'tuning-seed-min': min(tuning), 'tuning-seed-max': max(tuning),
              'test-seed-min': int(head.group(3)), 'test-seed-max': int(head.group(4))}

    rq1a = {}
    for line in s['3'][2:]:
        m = re.match(rf'^\s*(\d+)\s+{DELTA_CI}\s+(\d+) / (\d+)\s+<?({NUM})', line)
        if m:
            k, e, d, lo, hi, dmin, dmax, amin, amed, p = m.groups()
            rq1a[f'k{k}-{eps(e)}'] = {'delta': num(d), 'ci-low': num(lo), 'ci-high': num(hi),
                                      'delta-min': num(dmin), 'delta-max': num(dmax),
                                      'anchors-min': num(amin), 'anchors-median': num(amed),
                                      'p': num(p)}  # p < 0.001 is printed as its bound

    scaling = {}
    for line in s['5'][2:]:
        m = re.match(rf'^\s*(\S+)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM}) \[({NUM}), ({NUM})\]', line)
        if m:
            topo, e, k5, k10, k20, slope, lo, hi = m.groups()
            scaling[f'{topo}-{eps(e)}'] = {'k5': num(k5), 'k10': num(k10), 'k20': num(k20), 'slope': num(slope),
                                           'slope-ci-low': num(lo), 'slope-ci-high': num(hi)}

    worlds = design['test-worlds']
    breakdown = {}
    for line in s['6'][2:]:
        m = SETTING.match(line)
        if m:
            shares = re.findall(rf'({NUM}) \[({NUM}), ({NUM})\]', m.group(5))
            row = {}
            for method, (share, lo, hi) in zip(('joint', 'pairwise'), shares):
                row |= {f'{method}-share': num(share), f'{method}-ci-low': num(lo), f'{method}-ci-high': num(hi),
                        f'{method}-broken': round(float(share) * worlds)}
            breakdown[setting(*m.groups()[:4])] = row

    sanity = {}
    for line in s['9'][1:]:
        m = re.match(rf'^\s*K=\s*(\d+) budget=\s*(\d+): (\d+)/(\d+) pass; '
                     rf'median max ({NUM}) min, worst ({NUM}) min', line)
        if m:
            k, b, ok, n, med, worst = m.groups()
            sanity[f'k{k}-b{b}'] = {'pass': int(ok), 'worlds': int(n), 'max-median': num(med),
                                    'max-worst': num(worst)}

    return {
        'design': result('design', 'Synthetic grid design (script 02)', design, FULL),
        'map-error': result('map-error', 'Median map error (min), all pairs, λ tuned on the tuning worlds',
                            grid(s['1']), FULL),
        'map-error-indirect': result('map-error-indirect', 'Median map error (min), pairs without direct anchors',
                                     grid(s['2']), FULL),
        'rq1a': result('rq1a', 'RQ1a: joint − star, complete graph, budget 100 (median Δ, CI, p)', rq1a, FULL),
        'scaling': result('scaling', 'RQ1c: joint − composed pairwise vs K at budget 30; slope per doubling of K',
                          scaling, FULL),
        'breakdown': result('breakdown', 'Share and count of broken test worlds (> 0.3 min), ε > 0',
                            breakdown, FULL),
        'sanity': result('sanity', 'Complete graph, ε = 0, joint: test worlds with max error ≤ 0.15 min',
                         sanity, FULL),
        'map-error-fixed': result('map-error-fixed', 'Median map error (min), all pairs, fixed λ = 10',
                                  grid(s['10']), FULL),
    }


# ── script 03: the λ diagnosis ───────────────────────────────────────────────────────────────────

def diagnose_lambda() -> dict[str, dict]:
    text = LAMBDA.read_text()
    s = sections(text)
    cols = [RULES[c] for c in columns(s['2'][1], 'method') if c in RULES]
    rules: dict[str, dict] = {}
    by_topology: dict[str, dict] = {}
    for line in s['2'][2:]:
        m = re.match(rf'^\s*(\d+) (\S+)\s+(\d+)\s+({NUM}) (Joint|Composed pairwise)\s+(.*)$', line)
        if not m:
            continue
        k, topo, b, e, method, rest = m.groups()
        cells = re.findall(rf'(?:({NUM}) \[\s*(\d+)\]|—)', rest)
        best = rest.split()[-1]
        row = rules.setdefault(setting(k, topo, b, e), {})
        name = METHODS[method]
        for rule, (v, broken) in zip(cols, cells):
            if v:
                row[f'{name}-{rule}'] = num(v)
                row[f'{name}-{rule}-broken'] = int(broken)
                t = by_topology.setdefault(topo.replace('_', '-'), {})
                t[f'{name}-{rule}'] = t.get(f'{name}-{rule}', 0) + int(broken)
        row[f'{name}-best-lambda'] = num(best)
    totals: dict = {}
    for line in s['3'][1:]:
        m = re.match(r'^\s*(Joint|Composed pairwise)\s+(.*)$', line)
        if m:
            for rule, v in zip(cols, re.findall(r'(\d+|—)', m.group(2))):
                if v != '—':
                    totals[f'{METHODS[m.group(1)]}-{rule}'] = int(v)
    head = re.search(r'summed over the (\d+) corrupted settings \((\d+) test worlds\)', text)
    assert head, 'diagnose_lambda.txt: no section 3 summary'
    totals |= {'settings': int(head.group(1)), 'worlds': int(head.group(2))}
    per_topology = int(head.group(2)) // len(by_topology)
    for t in by_topology.values():
        t['worlds'] = per_topology
    # the sums per topology must add up to the report's totals
    for key, total in totals.items():
        if key not in ('settings', 'worlds'):
            got = sum(t.get(key, 0) for t in by_topology.values())
            if got != total:
                raise ValueError(f'diagnose_lambda.txt: {key} sums to {got} over topologies, report says {total}')
    return {
        'lambda-rules': result('lambda-rules', 'Median map error and broken test worlds under each λ rule',
                               rules, LAMBDA),
        'lambda-broken': result('lambda-broken', 'Broken test worlds over the corrupted settings, by λ rule',
                                {'all': totals, **by_topology}, LAMBDA),
    }


# ── script 04: the fish arm ──────────────────────────────────────────────────────────────────────

def fish_sparsity() -> dict[str, dict]:
    text = FISH.read_text()
    s = sections(text)
    data: dict = {}
    batches = re.findall(r'([a-z]{3}\d{4}) \(n=(\d+)\)', s['1'][0])
    for b, n in batches:
        data.setdefault('samples', {})[b] = int(n)
    data['samples']['all'] = sum(int(n) for _, n in batches)
    data['batches'] = len(batches)
    m = re.search(rf'({NUM}) anchors\.', s['1'][0])
    assert m, 'fish_sparsity.txt: no anchor count'
    data['anchors'] = {'all': num(m.group(1))}
    for a, b, n in re.findall(rf'([a-z]{{3}}\d{{4}})×([a-z]{{3}}\d{{4}})=(\d+) \({NUM}\*?\)', text):
        data['anchors'][f'{a}-{b}'] = int(n)
    m = re.search(r'(\d+) draws per setting', text)
    assert m, 'fish_sparsity.txt: no draw count'
    data['draws'] = int(m.group(1))

    rq1a = {}
    for line in s['4'][2:]:
        m = re.match(rf'^\s*{DELTA_CI}\s+<?({NUM})', line)
        if m:
            e, d, lo, hi, dmin, dmax, p = m.groups()
            rq1a[eps(e)] = {'delta': num(d), 'ci-low': num(lo), 'ci-high': num(hi), 'delta-min': num(dmin),
                            'delta-max': num(dmax), 'p': num(p)}  # p < 0.001 is printed as its bound
    broken = {}
    draws = data['draws']
    for line in s['6'][2:]:
        m = SETTING.match(line)
        if m:
            row = {}
            shares = re.findall(rf'({NUM}) \[({NUM}), ({NUM})\]', m.group(5))
            for method, (share, lo, hi) in zip(('joint', 'pairwise'), shares):
                row |= {f'{method}-share': num(share), f'{method}-ci-low': num(lo), f'{method}-ci-high': num(hi),
                        f'{method}-broken': round(float(share) * draws)}
            broken[setting(*m.groups()[:4])] = row
    return {
        'fish-data': result('fish-data', 'Fish oil data: batches, samples and anchors (script 04)', data, FISH),
        'fish-error': result('fish-error', 'Fish oil: median held-out anchor error (min), all scored pairs',
                             grid(s['2']), FISH),
        'fish-rq1a': result('fish-rq1a', 'Fish oil RQ1a: joint − star through sep2015, complete graph, budget 100',
                            rq1a, FISH),
        'fish-broken': result('fish-broken', 'Fish oil: share and count of broken draws (> 0.3 min), ε > 0',
                              broken, FISH),
    }


# ── script 05: figure captions ───────────────────────────────────────────────────────────────────

def captions() -> dict[str, dict]:
    text = CAPTIONS.read_text()
    m = re.search(rf'Joint \(CV\) ({NUM}), Joint \(λ = 10\) ({NUM}), Pairwise ({NUM}) min', text)
    assert m, 'captions_tai.md: no fig 4 (a) world errors'
    fig4a: dict = {'world-error': {'joint-cv': num(m.group(1)), 'joint-lambda10': num(m.group(2)),
                                   'pairwise': num(m.group(3))}}
    m = re.search(r'Test world (\d+) of K = (\d+)', text)
    assert m, 'captions_tai.md: no fig 4 (a) world'
    fig4a |= {'test-world': int(m.group(1))}
    m = re.search(rf'True drifts span ({NUM}) to ({NUM}) min', text)
    assert m, 'captions_tai.md: no drift span'
    fig1 = {'drift-min': num(m.group(1)), 'drift-max': num(m.group(2))}
    m = re.search(rf'tallest apex \(({NUM}) min', text)
    assert m, 'captions_tai.md: no apex time'
    fig1['apex-minutes'] = num(m.group(1))
    m = re.search(rf'moved ({NUM}) bins', text)
    assert m, 'captions_tai.md: no corrupted-anchor shift'
    fig1['corrupt-shift-bins'] = num(m.group(1))
    return {'fig1': result('fig1', 'Figure 1 caption numbers (script 05)', fig1, CAPTIONS),
            'fig4a': result('fig4a', 'Figure 4 (a): the broken chain world (script 05)', fig4a, CAPTIONS)}


def main() -> None:
    out = full_sparsity() | diagnose_lambda() | fish_sparsity() | captions()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, r in out.items():
        path = OUT / f'{name}.json'
        path.write_text(json.dumps(r, indent=2, ensure_ascii=False) + '\n')
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == '__main__':
    main()

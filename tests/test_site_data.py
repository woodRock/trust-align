"""scripts/06_site_data.py and the committed website data (docs/data/*.json, docs/index.html).

In CI (no caches): the committed JSON is well formed, within budget, and every value that also appears in
results/*.txt matches it; the numbers in the page's prose are the ones numbers.json holds. Locally, with
the caches: the slow test regenerates everything into a temporary folder and diffs it against docs/data.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
DOCS = ROOT / 'docs'
DATA = DOCS / 'data'
FILES = ('meta.json', 'grid.json', 'graphs.json', 'breakdown.json', 'lambda.json', 'scaling.json', 'fig4a.json',
         'fish.json', 'world.json', 'numbers.json')


def load_script(filename: str):
    spec = importlib.util.spec_from_file_location(Path(filename).stem, ROOT / 'scripts' / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def s06():
    return load_script('06_site_data.py')


@pytest.fixture(scope='module')
def site() -> dict:
    return {f: json.loads((DATA / f).read_text()) for f in FILES}


@pytest.fixture(scope='module')
def reports() -> dict:
    return {n: (RESULTS / f'{n}.txt').read_text().splitlines() for n in ('full_sparsity', 'diagnose_lambda',
                                                                         'fish_sparsity')}


def close(values, printed: str, decimals: int = 3) -> bool:
    """The median of 4-decimal values is within half a printed unit of the report's 3-decimal string."""
    return abs(float(np.nanmedian(values)) - float(printed)) <= 0.5 * 10 ** -decimals + 1e-9


# ── pure helpers ──────────────────────────────────────────────────────────────

def test_jsonable_rounds_sorts_and_drops_nan(s06):
    obj = {(5, 'chain', 30, 0.3): {'b': np.array([0.123456, np.nan]), 'a': np.int64(3), 'c': (True, 1.0)}}
    assert s06.jsonable(obj) == {'5|chain|30|0.3': {'b': [0.1235, None], 'a': 3, 'c': [True, 1.0]}}
    text = s06.dumps({'z': 1, 'a': {'y': 2, 'x': 3}})
    assert text == '{"a":{"x":3,"y":2},"z":1}\n'
    assert s06.public('trustalign 0.2.1 @ 782257c (src clean) from ~/Desktop/trust-align') == \
        'trustalign 0.2.1 @ 782257c (src clean)'


def test_fill_numbers(s06):
    html = '<p><data data-key="a">old</data> and <data class="n" data-key="b"></data></p>'
    out = s06.fill_numbers(html, {'a': '1.525', 'b': '20/20', 'unused': 'x'})
    assert out == '<p><data data-key="a">1.525</data> and <data class="n" data-key="b">20/20</data></p>'
    assert s06.html_numbers(out) == {'a': '1.525', 'b': '20/20'}
    with pytest.raises(KeyError):
        s06.fill_numbers(html, {'a': '1'})


def test_report_parsers(s06, reports):
    full, diag, fish = reports['full_sparsity'], reports['diagnose_lambda'], reports['fish_sparsity']
    t1 = s06.parse_medians(full, 1)
    assert len(t1) == 114 and t1[(5, 'chain', 30, 0.3)][0] == ('1.525', '300')
    assert t1[(5, 'chain', 30, 0.3)][4] == ('2.590', '')                       # PCHIP has no λ
    assert len(s06.parse_medians(full, 2)) == 114 - 20                           # no complete graphs
    rq1a = s06.parse_rq1a(full, 3)
    assert rq1a[(5, 0.0)] == {'median_delta': '-0.0020', 'ci': ['-0.0057', '+0.0025'],
                              'range': ['-0.0143', '+0.0051'], 'anchors_per_pair': [57, 100], 'p': '0.123'}
    rq1b = s06.parse_rq1b(full, 4)
    assert sum(len(v) for v in rq1b.values()) == 94
    assert rq1b[(5, 'chain', 30, 0.3)] == [{'comparison': 'vs composed pairwise, indirect',
                                            'median_delta': '+1.0754', 'wins': '2/20', 'p': '<0.001',
                                            'holm': '0.003', 'significant': True}]
    rq1c = s06.parse_rq1c(full)
    assert rq1c[('density0.4', 0.15)] == {'medians': {5: '+0.0034', 10: '-0.0052', 20: '-0.0118'},
                                          'slope': '-0.0084', 'ci': ['-0.0107', '-0.0063']}
    bd = s06.parse_breakdown(full, 6)
    assert len(bd) == 72 + 3 and bd[(5, 'chain', 30, 0.3)]['Joint'] == {'share': '0.80', 'ci': ['0.58', '0.92']}
    lt = s06.parse_lambda_totals(diag)
    assert lt['n'] == 960 and lt['Joint']['tuned'] == '175' and lt['Joint']['per world'] == '80'
    f2 = s06.parse_medians(fish, 2, with_K=False)
    assert len(f2) == 54 and f2[('complete', 10, 0.3)][0] == ('0.133', '200')
    assert s06.parse_rq1a(fish, 4, with_K=False)[0.3]['median_delta'] == '-0.0272'
    assert sum(len(v) for v in s06.parse_rq1b(fish, 5, with_K=False).values()) == 72
    assert s06.parse_breakdown(fish, 6, with_K=False)[('complete', 10, 0.3)]['Joint']['share'] == '0.25'
    rt = s06.parse_runtime(full, diag, fish)
    assert rt == {'02_cpu_h': '3.59', '03_s_per_cell': '2.7', '04_s_per_cell': '4.6'}
    assert s06.provenance_of(RESULTS / 'full_sparsity.txt')[0].startswith('trustalign 0.2.1 @ 782257c')


def test_comparable_strips_header_and_runtime(s06, reports):
    full = reports['full_sparsity']
    c = s06.comparable(full, 11)
    assert not any(ln.startswith('trustalign') for ln in c) and not any(ln.startswith('11. Run') for ln in c)
    s06.check_same('x', full, full, 11)
    with pytest.raises(AssertionError):
        s06.check_same('x', full, [*full[:20], 'changed', *full[21:]], 11)


# ── the committed site data ───────────────────────────────────────────────────

def test_site_files_are_compact_sorted_and_within_budget(s06, site):
    total = 0
    for f in FILES:
        text = (DATA / f).read_text()
        total += len(text.encode())
        canonical = json.dumps(site[f], sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n'
        assert text == canonical, f'{f} is not in the canonical compact, sorted form'
    assert total < s06.SIZE_BUDGET
    assert (DOCS / '.nojekyll').exists()


def test_meta(site):
    meta = site['meta.json']
    assert meta['n_settings'] == 114 and meta['n_test_worlds'] == 20 and meta['break_min'] == 0.3
    assert meta['results_provenance']['trustalign'].startswith('trustalign ')
    assert not any(' from ' in v for v in meta['results_provenance'].values())        # no local paths
    assert set(meta['methods']) == set(meta['labels']) and all(m in meta['colors'] for m in meta['methods'])
    assert len(meta['settings']) == 114


def test_grid_matches_the_report(s06, site, reports):
    full = reports['full_sparsity']
    t1, bd = s06.parse_medians(full, 1), s06.parse_breakdown(full, 6)
    rq1a, rq1b = s06.parse_rq1a(full, 3), s06.parse_rq1b(full, 4)
    grid = site['grid.json']
    assert len(grid) == 114
    methods = site['meta.json']['methods']
    for key, e in grid.items():
        s = (e['K'], e['topology'], e['budget'], e['eps'])
        assert key == '|'.join(str(x) for x in s)
        for m, (v, lam) in zip(methods, t1[s]):
            assert e['median'][m] == v and len(e['worlds'][m]) == 20 and close(e['worlds'][m], v)
            assert e['lam'][m] == (float(lam) if lam else None)
            assert e['broken'][m] == sum(x > 0.3 for x in e['worlds'][m])
        if s in bd:
            for m in ('Joint', 'Composed pairwise'):
                assert f'{e["broken"][m] / 20:.2f}' == bd[s][m]['share']
                assert site['breakdown.json'][key][m]['share'] == bd[s][m]['share']
        expected = ([{'comparison': 'vs Star (reference), all pairs', **rq1a[(s[0], s[3])]}]
                    if s[1] == 'complete' and s[2] == 100 else []) + rq1b.get(s, [])
        assert e['tests'] == expected
    assert len(site['breakdown.json']) == len(bd)


def test_graphs(site):
    g = site['graphs.json']
    assert set(g) == {'5', '10', '20'}
    for K, e in g.items():
        k = int(K)
        assert len(e['sizes']) == k and e['hub'] == 0 and e['sizes'][e['weak_hub']] == min(e['sizes'])
        assert len(e['topologies']['complete']) == k * (k - 1) // 2
        assert len(e['topologies']['chain']) == k - 1
        if k < 20:
            assert len(e['topologies']['ring']) == k and len(e['topologies']['star']) == k - 1
            assert all(e['weak_hub'] in edge for edge in e['topologies']['weak_star'])
        for edges in e['density0.4'].values():
            assert len(edges) == max(round(0.4 * k * (k - 1) / 2), k - 1)
    assert set(g['10']['density0.4']) == {f'{b}|{eps}' for b in (10, 30, 100) for eps in (0.0, 0.15, 0.3)}


def test_lambda(s06, site, reports):
    lam = site['lambda.json']
    lt = s06.parse_lambda_totals(reports['diagnose_lambda'])
    assert lam['n_worlds'] == lt['n'] == 960 and len(lam['settings']) == 48 and lam['seeds'] == list(range(3, 23))
    for rule, key in s06.RULES.items():
        assert lam['totals'][key] == int(lt['Joint'][rule])
        assert sum(len(e['broken'][key]) for e in lam['settings'].values()) == lam['totals'][key]
        by_rule = [v['k'] for k, v in lam['by_topology'].items() if k.split('|')[1] == key]
        assert sum(by_rule) == lam['totals'][key]
    for e in lam['settings'].values():
        for key, errs in e['errors'].items():
            assert len(errs) == 20
            assert e['broken'][key] == [s for s, v in zip(lam['seeds'], errs) if v > 0.3]


def test_scaling(s06, site, reports):
    sc = site['scaling.json']
    rq1c = s06.parse_rq1c(reports['full_sparsity'])
    assert len(sc['table']) == 6
    for key, r in rq1c.items():
        t = sc['table']['|'.join(str(x) for x in key)]
        assert t['slope'] == r['slope'] and t['ci'] == r['ci']
        assert t['medians'] == {str(k): v for k, v in r['medians'].items()}
    for topo, f in sc['fig3b'].items():
        r = rq1c[(topo, sc['eps'])]
        assert f['slope'] == r['slope'] and f['ci'] == r['ci']
        assert f'{f["fit"]["slope"]:+.4f}' == r['slope']
        for K, pts in f['points'].items():
            assert len(pts) == 20 and abs(float(np.median(pts)) - float(r['medians'][int(K)])) <= 0.5e-4 + 1e-9
        assert len(f['band']['x_log2']) == len(f['band']['lo']) == len(f['band']['hi']) == 50
        assert all(lo <= hi for lo, hi in zip(f['band']['lo'], f['band']['hi']))


def test_world_is_synthetic_only(site):
    w = site['world.json']
    assert w['K'] == 10 and len(w['tic']) == 10 and all(len(t) == len(w['time_min']) for t in w['tic'])
    assert len(w['drift']) == 10 and all(abs(v) <= 4.5 for d in w['drift'] for v in d)
    assert len(w['anchors']['a']) == len(w['anchors']['bad']) <= 3 * w['corruption_cell']['budget']
    assert set(w['anchors']['a']) | set(w['anchors']['b']) <= set(w['shown'])
    assert w['world_seed'] == 130_000 + 1000 * w['K'] + w['seed']      # script 02's synthetic world seed
    assert 'fish' not in json.dumps(w).lower()


def test_fig4a(s06, site, reports):
    f = site['fig4a.json']
    t1 = s06.parse_medians(reports['full_sparsity'], 1)
    s = (f['setting']['K'], f['setting']['topology'], f['setting']['budget'], f['setting']['eps'])
    assert s == (5, 'chain', 30, 0.3) and f['setting_median'] == t1[s][0][0]
    assert f['lams']['joint_cv'] == float(t1[s][0][1])
    n = len(f['t'])
    assert all(len(c) == n for c in f['curves'].values()) and n > 100
    assert len(f['anchors']['ta']) == len(f['anchors']['dt']) == len(f['anchors']['bad']) == 30   # budget 30
    assert f['edge'][0] < f['edge'][1] < 5 and 3 <= f['seed'] <= 22
    assert f['n_broken'] == site['grid.json']['5|chain|30|0.3']['broken']['Joint']
    assert float(f['world_errors']['joint_cv']) > 0.3 > float(f['world_errors']['pairwise'])


def test_fish_matches_the_report(s06, site, reports):
    fish, lines = site['fish.json'], reports['fish_sparsity']
    t2, bd = s06.parse_medians(lines, 2, with_K=False), s06.parse_breakdown(lines, 6, with_K=False)
    rq1a, rq1b = s06.parse_rq1a(lines, 4, with_K=False), s06.parse_rq1b(lines, 5, with_K=False)
    assert len(fish['settings']) == 54 and fish['n_draws'] == 20 and fish['n_anchors'] == 25_011
    assert [b['name'] for b in fish['batches']] == ['apr2016', 'aug2015', 'jan2016', 'jul2016', 'sep2015']
    methods = site['meta.json']['methods']
    for key, e in fish['settings'].items():
        s = (e['topology'], e['budget'], e['eps'])
        assert key == '|'.join(str(x) for x in s)
        for m, (v, _lam) in zip(methods, t2[s]):
            assert e['median'][m] == v and len(e['draws'][m]) == 20 and close(e['draws'][m], v)
            assert e['broken'][m] == sum(x > 0.3 for x in e['draws'][m])
        if s in bd:
            for m in ('Joint', 'Composed pairwise'):
                assert e['breakdown'][m]['share'] == bd[s][m]['share'] == f'{e["broken"][m] / 20:.2f}'
        expected = ([{'comparison': 'vs Star (sep2015), all pairs', **rq1a[s[2]]}]
                    if s[0] == 'complete' and s[1] == 100 else []) + rq1b.get(s, [])
        assert e['tests'] == expected
    # Aggregates only: nothing that looks like a per-sample or per-anchor retention time.
    text = json.dumps(fish)
    assert 'tic' not in text.lower() and 'spectrum' not in text.lower()
    assert not any(len(v) > 20 for e in fish['settings'].values() for v in e['draws'].values())


def test_page_numbers_come_from_numbers_json(s06, site):
    html = (DOCS / 'index.html').read_text()
    shown = s06.html_numbers(html)
    numbers = site['numbers.json']
    assert len(shown) > 40
    for key, text in shown.items():
        assert key in numbers, f'<data data-key="{key}"> has no number'
        assert text == numbers[key], f'{key}: page shows {text!r}, numbers.json has {numbers[key]!r}'
    assert s06.fill_numbers(html, numbers) == html


def test_page_has_text_links_and_data_links_without_javascript():
    html = (DOCS / 'index.html').read_text()
    for f in FILES:
        if f != 'numbers.json':
            assert f'data/{f}' in html, f'no "Data" link to {f}'
    assert '<noscript>' in html
    assert 'integrity="sha384-' in html and 'cdn.jsdelivr.net/npm/d3@7.9.0' in html
    assert 'arxiv' not in html.lower()                  # no paper links until there is a preprint
    for asset in ('fig1_bc.svg', 'fig1_bc_dark.svg', 'fig2_graphs.svg', 'fig2_graphs_dark.svg'):
        assert (DOCS / 'assets' / asset).exists() and f'assets/{asset}' in html
    assert 'fig1_data' not in html                      # never the raw fish oil sample (fig 1 a)


# ── locally, with the caches ──────────────────────────────────────────────────

def _has_caches() -> bool:
    return all((RESULTS / d).is_dir() for d in ('full_sparsity_cache', 'diagnose_lambda_cache',
                                                 'fish_sparsity_cache')) \
        and (RESULTS / 'cache_plot_data.npz').exists() and (RESULTS / 'fish_anchors.npz').exists()


@pytest.mark.slow
@pytest.mark.skipif(not _has_caches(), reason='needs the experiment caches (local only)')
def test_regenerated_site_data_matches_the_committed_files(s06, tmp_path):
    s06.main(out=tmp_path, html=None)
    for f in FILES:
        new, old = json.loads((tmp_path / f).read_text()), json.loads((DATA / f).read_text())
        if f == 'meta.json':
            for k in ('generated', 'site_data_provenance'):
                new.pop(k), old.pop(k)
        if f == 'numbers.json':
            new.pop('generated'), old.pop('generated')
        assert new == old, f'{f} differs from the committed file: rerun scripts/06_site_data.py'

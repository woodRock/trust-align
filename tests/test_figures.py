"""scripts/05_figures.py: the Experiment 1 figures from the caches (parsers; end to end offline)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
SMALL = dict(n_compounds=30, mz_bins=300)


def load_script(filename: str):
    spec = importlib.util.spec_from_file_location(Path(filename).stem, ROOT / 'scripts' / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def s_tai():
    return load_script('05_figures.py')


def test_figures_parsers(s_tai):
    t1 = s_tai.parse_full_table1(RESULTS / 'full_sparsity.txt')
    assert (5, 'complete', 10, 0.0) in t1
    assert t1[(5, 'complete', 10, 0.0)][0] == '0.033'
    rq = s_tai.parse_full_rq1c(RESULTS / 'full_sparsity.txt')
    assert ('density0.4', 0.15) in rq
    assert rq[('density0.4', 0.15)][3] == '-0.0084'
    lt = s_tai.parse_lambda_totals(RESULTS / 'diagnose_lambda.txt')
    assert lt['n'] == 960
    assert lt['Joint']['tuned'] == '175'
    assert lt['Joint']['λ = 10'] == '160'
    assert lt['Joint']['per world'] == '80'
    t2 = s_tai.parse_fish_table2(RESULTS / 'fish_sparsity.txt')
    assert ('complete', 10, 0.3) in t2
    assert t2[('complete', 10, 0.3)][0] == '0.133'


def _has_paper_figure_inputs() -> bool:
    if not (
        (RESULTS / 'full_sparsity_cache').is_dir()
        and (RESULTS / 'diagnose_lambda_cache').is_dir()
        and (RESULTS / 'fish_sparsity_cache').is_dir()
    ):
        return False
    return (ROOT / 'data' / 'fish_oil').is_dir()


@pytest.mark.skipif(
    not _has_paper_figure_inputs(),
    reason='Requires experiment caches and private fish oil data (offline figure generation only)',
)
def test_figures_end_to_end(s_tai, tmp_path):
    res = s_tai.main(out=tmp_path, site=tmp_path / 'site')
    for name in ('fig1_data', 'fig2_graphs', 'fig3_synthetic_results',
                 'fig4_breakdown_diagnosis', 'fig5_fish_results'):
        pdf = tmp_path / f'{name}.pdf'
        assert pdf.exists()
        assert pdf.stat().st_size > 1000
        assert pdf.read_bytes()[:5] == b'%PDF-'
    assert (tmp_path / 'captions_tai.md').exists()
    assert len(res['paths']) == 5 + 4
    for name in ('fig1_bc', 'fig1_bc_dark', 'fig2_graphs', 'fig2_graphs_dark'):
        svg = (tmp_path / 'site' / f'{name}.svg').read_text()
        assert svg.startswith('<?xml') and '<svg' in svg
    assert 'Fish oil' not in (tmp_path / 'site' / 'fig1_bc.svg').read_text()      # never panel (a)
    assert len(res['captions']) == 5
    assert len(res['table']) > 20

"""scripts/20_paper_figures.py: parsers against the committed reports, and every figure drawn."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
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
def s20():
    return load_script('20_paper_figures.py')


def test_parse_paired_reads_every_cell(s20):
    vals = s20.parse_paired(RESULTS / 'draft_connectivity.txt')
    assert len({k for k in vals if k[2] == 0}) == 45            # 9 structures × 5 noise levels
    d, wins, n, p = vals[('chain', 'ε 30%', 0)]
    assert (d, wins, n, p) == (-1.1132, 8, 10, 0.010)
    assert np.isnan(vals[('star', 'clean', 1)][3])               # star vs itself: no test
    assert ('density1.0', 'clean', 2) not in vals                # no indirect pairs on the complete graph


def test_parse_breakdown_reads_draft_rows(s20):
    data = s20.parse_breakdown(RESULTS / 'diagnose_breakdown.txt')
    assert set(data) == {'ring', 'chain'}
    assert data['ring'].shape == (10, len(s20.BREAKDOWN_COLS))
    joint = data['ring'][:, s20.BREAKDOWN_COLS.index('joint')]
    assert np.sum(joint > s20.BROKEN) == 6 and np.median(joint) == pytest.approx(0.834)


def test_parsers_refuse_empty_input(s20, tmp_path):
    empty = tmp_path / 'empty.txt'
    empty.write_text('3. Paired comparisons\nheader\n\n')
    with pytest.raises(ValueError):
        s20.parse_paired(empty)
    with pytest.raises(ValueError):
        s20.parse_breakdown(empty)


def test_parse_paired_refuses_unknown_noise(s20, tmp_path):
    bad = tmp_path / 'bad.txt'
    bad.write_text('3. Paired comparisons\nheader\n   chain       odd noise   -0.0100  7/10 p=0.131\n')
    with pytest.raises(ValueError):
        s20.parse_paired(bad)


def test_parse_medians_reads_both_tables(s20):
    all_pairs, indirect = s20.parse_medians(RESULTS / 'draft_connectivity.txt')
    assert len(all_pairs) == 45 and len(indirect) == 40          # no indirect pairs on the complete graph
    assert all_pairs[('ring', 'ε 30%')] == {'Joint': 0.834, 'Star': 0.074, 'Composed pairwise': 1.047,
                                            'Composed PCHIP': 3.033}
    assert indirect[('chain', 'ε 30%')]['Composed pairwise'] == 1.938


def test_parse_medians_refuses_empty_tables(s20, tmp_path):
    empty = tmp_path / 'empty.txt'
    empty.write_text('1. Median\n\n2. Pairs\n\n3. Paired\n')
    with pytest.raises(ValueError):
        s20.parse_medians(empty)


def test_results_figures_and_captions(s20, tmp_path):
    caps = s20.main(results=RESULTS, out=tmp_path, world=False)
    for name in ('fig_regime', 'fig_breakdown', 'fig_connectivity'):
        assert (tmp_path / f'{name}.pdf').stat().st_size > 0
    text = (tmp_path / 'captions.md').read_text()
    assert '-1.113 min (chain, ε 30%, 8/10 worlds)' in text
    assert 'ring: 6/10 worlds broken, oracle at most 0.076 min' in text
    assert 'joint 0.385 min on the chain and 1.140 min on the ring' in text
    assert text == '\n'.join(caps)


@pytest.mark.slow
def test_world_figures_on_a_small_world(s20, tmp_path):
    caps = s20.main(results=RESULTS, out=tmp_path, world=True, **SMALL)
    for name in ('fig1_data', 'fig2_graphs'):
        assert (tmp_path / f'{name}.png').stat().st_size > 0
    assert any(c.startswith('**fig2_graphs.**') and 'star 9' in c and 'density 1.0 45' in c for c in caps)


@pytest.fixture(scope='module')
def s_tai():
    return load_script('make_paper_figures.py')


def test_make_paper_figures_parsers(s_tai):
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
    try:
        from warpalign.config import Paths
        return Paths().fish_oil.is_dir()
    except Exception:
        return False


@pytest.mark.skipif(
    not _has_paper_figure_inputs(),
    reason='Requires experiment caches and private fish oil data (offline figure generation only)',
)
def test_make_paper_figures_end_to_end(s_tai, tmp_path):
    res = s_tai.main(out=tmp_path)
    for name in ('fig1_data', 'fig2_graphs', 'fig3_synthetic_results',
                 'fig4_breakdown_diagnosis', 'fig5_fish_results'):
        pdf = tmp_path / f'{name}.pdf'
        assert pdf.exists()
        assert pdf.stat().st_size > 1000
        assert pdf.read_bytes()[:5] == b'%PDF-'
    assert (tmp_path / 'captions_tai.md').exists()
    assert len(res['paths']) == 5
    assert len(res['captions']) == 5
    assert len(res['table']) > 20

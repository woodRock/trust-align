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


def test_results_figures_and_captions(s20, tmp_path):
    caps = s20.main(results=RESULTS, out=tmp_path, world=False)
    for name in ('fig_regime', 'fig_breakdown'):
        assert (tmp_path / f'{name}.pdf').stat().st_size > 0
    text = (tmp_path / 'captions.md').read_text()
    assert '-1.113 min (chain, ε 30%, 8/10 worlds)' in text
    assert 'ring: 6/10 worlds broken, oracle at most 0.076 min' in text
    assert text == '\n'.join(caps)


@pytest.mark.slow
def test_world_figures_on_a_small_world(s20, tmp_path):
    caps = s20.main(results=RESULTS, out=tmp_path, world=True, **SMALL)
    for name in ('fig1_data', 'fig2_graphs'):
        assert (tmp_path / f'{name}.png').stat().st_size > 0
    assert any(c.startswith('**fig2_graphs.**') and 'star 9' in c and 'density 1.0 45' in c for c in caps)

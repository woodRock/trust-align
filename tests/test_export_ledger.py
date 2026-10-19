"""scripts/export_ledger.py: the paper's numbers as taura ledger results, from the committed reports."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _script(filename: str):
    spec = importlib.util.spec_from_file_location(Path(filename).stem, ROOT / 'scripts' / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def ex():
    return _script('export_ledger.py')


@pytest.fixture(scope='module')
def out(ex):
    return ex.full_sparsity() | ex.diagnose_lambda() | ex.fish_sparsity() | ex.captions()


def m(out, key: str):
    result, *path = key.split('/', 2)
    node = out[result]['metrics']
    for part in path:
        node = node[part]
    return node


def test_the_numbers_script_05_checks_against_the_manuscript(out):
    """The values script 05 asserts against the manuscript (MANUSCRIPT there), read from the ledger."""
    keys = {
        'map-error/k5-complete-b10-e0/joint': 0.033, 'map-error/k5-complete-b10-e30/joint': 0.043,
        'map-error/k5-chain-b30-e30/joint': 1.525, 'map-error/k5-chain-b30-e30/star': 0.052,
        'map-error/k5-chain-b30-e30/pairwise': 0.140, 'map-error/k5-ring-b30-e30/pairwise': 0.116,
        'scaling/density0.4-e15/k5': 0.0034, 'scaling/density0.4-e15/k20': -0.0118,
        'scaling/density0.4-e15/slope': -0.0084, 'scaling/density0.4-e15/slope-ci-low': -0.0107,
        'scaling/complete-e15/k10': -0.0105,
        'lambda-broken/all/joint-tuned': 175, 'lambda-broken/all/joint-lambda10': 160,
        'lambda-broken/all/joint-per-world': 80, 'lambda-broken/all/worlds': 960,
        'fish-error/complete-b10-e30/joint': 0.133, 'fish-error/complete-b10-e30/star': 0.410,
        'fish-error/complete-b100-e0/pairwise': 0.056,
    }
    for key, value in keys.items():
        assert m(out, key) == value, key


def test_sections(out):
    assert m(out, 'design/settings') == 114 and m(out, 'design/test-worlds') == 20
    assert len(out['map-error']['metrics']) == 114
    assert m(out, 'map-error/k5-chain-b30-e30/joint-lambda') == 300
    assert m(out, 'rq1a/k10-e30/delta') == 0.0006 and m(out, 'rq1a/k5-e0/p') == 0.123
    assert m(out, 'breakdown/k5-chain-b30-e30/joint-broken') == 16
    assert m(out, 'sanity/k20-b30/pass') == 4
    assert m(out, 'map-error-fixed/k5-ring-b10-e30/joint') == 0.572
    rules = m(out, 'lambda-rules/k5-ring-b10-e30')
    assert (rules['joint-tuned-broken'], rules['joint-lambda10-broken']) == (14, 17)
    assert [m(out, f'lambda-broken/{t}/joint-per-world') for t in ('chain', 'ring', 'density0.4', 'weak-star')] \
        == [32, 29, 12, 7]
    assert m(out, 'fish-data/anchors/all') == 25011 and m(out, 'fish-data/samples/all') == 90
    assert m(out, 'fish-data/samples/aug2015') == 2 and m(out, 'fish-data/anchors/jul2016-sep2015') == 4396
    assert m(out, 'fish-rq1a/e30/delta') == -0.0272 and m(out, 'fish-rq1a/e30/p') == 0.001
    assert m(out, 'fish-broken/complete-b10-e30/joint-broken') == 5
    assert m(out, 'fig4a/world-error/joint-cv') == 1.538 and m(out, 'fig1/apex-minutes') == 14.4
    assert (m(out, 'fig4a/anchors/clean'), m(out, 'fig4a/anchors/corrupted')) == (21, 9)
    assert m(out, 'fig4a/lambda/joint-cv') == 300 and m(out, 'fig4a/edge-error/max') == 7.63
    assert (m(out, 'fig4b/complete/broken'), m(out, 'fig4b/star/broken')) == (5, 11)
    assert m(out, 'fig4b/star/worlds') == 240
    src = out['map-error']['source']
    assert src['path'] == 'results/full_sparsity.txt' and src['hash'].startswith('sha256:')
    assert any(line.startswith('trustalign ') for line in src['provenance'])
    assert 'provenance' not in out['fig1']['source']        # captions have no provenance header


def test_main_writes_every_result(ex, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ex, 'OUT', tmp_path / 'ledger' / 'results')
    ex.main()
    files = sorted(p.stem for p in (tmp_path / 'ledger' / 'results').glob('*.json'))
    assert len(files) == 17 and 'map-error' in files
    assert json.loads((tmp_path / 'ledger' / 'results' / 'rq1a.json').read_text())['name'] == 'rq1a'


def test_grid_rejects_a_short_row(ex):
    lines = ['1. x', '    K topology   budget     ε     Joint     Star', '    5 chain  10  0.0  0.1']
    with pytest.raises(ValueError, match='1 cells for 2 columns'):
        ex.grid(lines)
    with pytest.raises(ValueError, match='no rows'):
        ex.grid(lines[:2])


def test_broken_counts_must_add_up(ex, tmp_path, monkeypatch):
    text = (ROOT / 'results' / 'diagnose_lambda.txt').read_text()
    bad = tmp_path / 'diagnose_lambda.txt'
    bad.write_text(text.replace('   Joint                     175', '   Joint                     176'))
    monkeypatch.setattr(ex, 'LAMBDA', bad)
    monkeypatch.setattr(ex, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='joint-tuned sums to 175'):
        ex.diagnose_lambda()

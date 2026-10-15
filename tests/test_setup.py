"""The environment check and the provenance lines every result file starts with."""
from __future__ import annotations

import importlib.util
import re
import shutil
from pathlib import Path

import jointalign
import numpy as np
import pytest

import trustalign
from trustalign import provenance

ROOT = Path(__file__).resolve().parent.parent


def load_script(filename: str):
    spec = importlib.util.spec_from_file_location(Path(filename).stem, ROOT / 'scripts' / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_header_names_all_three_packages():
    lines = provenance.header()
    assert [ln.split()[0] for ln in lines] == ['trustalign', 'joint-align', 'warpalign']
    assert re.fullmatch(r'trustalign \S+ @ \w+ \(src (clean|dirty)\)', lines[0])
    assert jointalign.__version__ in lines[1]


def test_trustalign_revision_outside_git(tmp_path, monkeypatch):
    fake = tmp_path / 'site-packages' / 'trustalign' / '__init__.py'
    fake.parent.mkdir(parents=True)
    shutil.copy(trustalign.__file__, fake)
    monkeypatch.setattr(trustalign, '__file__', str(fake))
    assert provenance.trustalign_revision().endswith('(installed, no git)')


@pytest.mark.slow
def test_check_setup_on_synthetic_data(tmp_path):
    s00 = load_script('00_check_setup.py')
    lines = s00.main(data=tmp_path / 'no-data', results=tmp_path / 'results')
    txt = (tmp_path / 'results' / 'setup_check.txt').read_text()
    assert txt.splitlines()[2].startswith('trustalign')
    assert 'Data: not found' in txt
    err = float(re.search(r'median \|map error\| ([\d.]+) min', lines[-1]).group(1))
    assert err < 0.1                         # the pinned joint model still recovers the true drifts


def test_data_summary_counts_samples_without_reading_them(tmp_path):
    s00 = load_script('00_check_setup.py')
    for name, n in (('batch_a', 2), ('batch_b', 3)):
        (tmp_path / name / 'chroma').mkdir(parents=True)
        for i in range(n):
            (tmp_path / name / 'chroma' / f'{i}.npz').touch()
    lines = s00.data_summary(tmp_path)
    assert lines[1].split() == ['batch_a', '2', 'samples'] and lines[2].split() == ['batch_b', '3', 'samples']


@pytest.mark.slow
def test_pilot_end_to_end_on_a_tiny_grid(tmp_path):
    s10 = load_script('10_pilot_sparsity.py')
    cells = s10.main(Ks=(3,), topologies=('complete', 'chain'), budgets=(10, None), epsilons=(0.0, 0.15),
                     seeds=(0, 1), results=tmp_path, lambdas=(1.0, 100.0),
                     world_kw=dict(n_compounds=30, mz_bins=300))
    txt = (tmp_path / 'pilot_sparsity.txt').read_text()
    assert txt.splitlines()[2].startswith('trustalign')
    for section in ('1. Map error vs truth', '2. Sensitivity', '3. Sanity check', '4. Robust reweighting',
                    '5. Cycle inconsistency', '6. Run time', 'Wall time'):
        assert section in txt
    assert len(cells) == 16
    first = cells[(3, 'chain', 10, 0.15, 0)]
    assert cells[(3, 'chain', 10, 0.15, 1)]['lam'] == first['lam']        # λ chosen once, reused
    assert first['cv_seconds'] > 0 and cells[(3, 'chain', 10, 0.15, 1)]['cv_seconds'] == 0
    z = np.load(tmp_path / 'pilot_sparsity.npz')
    assert len(z['cell_keys']) == 16 and 'Joint|cv|median_all' in z.files


@pytest.mark.slow
def test_draft_connectivity_on_a_tiny_grid(tmp_path):
    s11 = load_script('11_draft_connectivity.py')
    noises = (('clean', 0.0, 0.0), ('jitter 1.0', 1.0, 0.0), ('ε 30%', 0.0, 0.30))
    runs = s11.main(seeds=(0, 1), workers=1, results=tmp_path, figs=tmp_path / 'figs', k=5, budget=10,
                    structures=('star', 'chain', 'chain+1', 'ring', 'density1.0', 'density0.4'), noises=noises,
                    world_kw=dict(n_compounds=30, mz_bins=300))
    txt = (tmp_path / 'draft_connectivity.txt').read_text()
    assert 'DRAFT' in txt and txt.splitlines()[4].startswith('trustalign')
    for section in ('1. Median map error', '2. Pairs with no direct anchors', '3. Paired comparisons',
                    '4. Robust reweighting', '5. Run time'):
        assert section in txt
    cell = runs[0][('chain', 'clean')]
    assert cell['n_edges'] == 4 and 0 < cell['star_coverage'] < 1
    assert runs[0][('star', 'clean')]['star_coverage'] == 1.0
    for f in ('regime_map.png', 'loop_closure.png', 'edge_removal.png', 'captions.md'):
        assert (tmp_path / 'figs' / f).stat().st_size > 0
    z = np.load(tmp_path / 'draft_connectivity.npz')
    assert list(z['seeds']) == [0, 1] and 'chain|clean|Joint|median_all' in z.files


@pytest.mark.slow
def test_diagnose_breakdown_on_one_seed(tmp_path):
    s12 = load_script('12_diagnose_breakdown.py')
    lines = s12.main(results=tmp_path, draft_seeds=(1,), pilot_seeds=(0,))
    txt = (tmp_path / 'diagnose_breakdown.txt').read_text()
    assert txt.splitlines()[2].startswith('trustalign')
    for section in ('1. Per seed', '2. Broken seeds in detail', '3. Summary over seeds', 'Wall time'):
        assert section in txt
    # The draft's ring seed 1 is the worst broken case; refitting without its corrupted anchors repairs it.
    row = next(ln for ln in lines if 'draft ring ε30% seed 1' in ln and 'BROKEN' in ln)
    joint, oracle = float(row.split()[5]), float(row.split()[9])
    assert joint > s12.BROKEN > oracle


TINY_GRID = [(4, 'complete', 100, 0.0), (4, 'complete', 10, 0.15), (4, 'weak_star', 10, 0.15),
             (4, 'chain', 10, 0.0), (4, 'chain', 30, 0.15), (4, 'density0.4', 30, 0.0), (5, 'density0.4', 30, 0.0)]


@pytest.mark.slow
def test_full_sparsity_on_a_tiny_grid_and_resumes(tmp_path):
    s13 = load_script('13_full_sparsity.py')
    kw = dict(grid=TINY_GRID, tune_seeds=(0,), test_seeds=(3, 4, 5), workers=1, results=tmp_path,
              lambdas=(1.0, 100.0), world_kw=dict(n_compounds=30, mz_bins=300), n_boot=50)
    out = s13.main(**kw)
    txt = (tmp_path / 'full_sparsity.txt').read_text()
    assert txt.splitlines()[2].startswith('trustalign')
    for section in ('1. Median map error', '2. Pairs without direct anchors', '3. RQ1a', '4. RQ1b',
                    'RQ1b trend', '5. RQ1c', '6. Breakdown', '7. Robust reweighting', '8. Cycle inconsistency',
                    '9. Sanity check', '10. Sensitivity', '11. Run time', 'wall time'):
        assert section in txt
    cells = out['cells']
    assert len(cells) == len(TINY_GRID) * 3
    for setting in TINY_GRID:                                  # one λ per setting and method, every test world
        lams = {s: cells[(*setting, s)][('Joint', 'cv')]['lam'] for s in (3, 4, 5)}
        assert len(set(lams.values())) == 1 and lams[3] == out['lams'][setting]['Joint']
    c = cells[(4, 'complete', 10, 0.15, 3)]
    assert c['coverage'] == {'Star': 1.0, 'Star (weak hub)': 1.0}
    assert 0 < cells[(4, 'chain', 10, 0.0, 3)]['coverage']['Star'] < 1
    # On a weak-hub star, the star through that hub uses every anchor: at one λ it is the joint model.
    w = cells[(4, 'weak_star', 10, 0.15, 3)]
    star, joint = w[('Star (weak hub)', 'fixed')], w[('Joint on weak-hub pairs', 'fixed')]
    assert star['median_all'] == pytest.approx(joint['median_all'])
    z = np.load(tmp_path / 'full_sparsity.npz')
    assert len(z['cell_keys']) == len(cells) and 'Joint|cv|median_all' in z.files
    # A rerun reuses every checkpointed task and gives the same numbers.
    again = s13.main(**kw)
    key = (4, 'chain', 30, 0.15, 4)
    assert again['cells'][key]['seconds'] == cells[key]['seconds']          # reused, not refitted
    assert again['cells'][key][('Joint', 'cv')]['median_all'] == cells[key][('Joint', 'cv')]['median_all']


def test_full_sparsity_statistics():
    s13 = load_script('13_full_sparsity.py')
    assert s13.holm([0.01, 0.04, np.nan, 0.03]) == pytest.approx([0.03, 0.06, np.nan, 0.06], nan_ok=True)
    lo, hi = s13.wilson(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2
    rng = np.random.default_rng(0)
    med, lo, hi = s13.median_ci(rng.normal(0.5, 0.1, 200), 500, rng)
    assert lo < 0.5 < hi and lo < med < hi and hi - lo < 0.1
    assert np.isnan(s13.median_ci(np.array([np.nan]), 10, rng)[0])
    assert s13.signed_rank(np.linspace(0.1, 1.0, 20)) < 0.001 and np.isnan(s13.signed_rank(np.zeros(5)))
    groups = {k: (np.full((30, 1), float(k)), 2.0 * k + rng.normal(0, 0.1, 30)) for k in (1, 2, 3)}
    est, lo, hi = s13.slope_ci(groups, 200, rng)
    assert lo[0] < 2.0 < hi[0] and abs(est[0] - 2.0) < 0.1
    assert len(s13.full_grid()) == 108 + 6 and len(s13.tasks_for(s13.full_grid(), s13.TEST_SEEDS)) == 20 * (12 + 3)
    assert s13.n_edges_of(10, 'density0.4') == 18 and s13.n_edges_of(5, 'density0.4') == 4

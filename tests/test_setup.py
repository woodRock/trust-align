"""The environment check and the provenance lines every result file starts with."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import jointalign
import numpy as np
import pytest

from trustalign import provenance

ROOT = Path(__file__).resolve().parent.parent


def load_script(filename: str):
    spec = importlib.util.spec_from_file_location(Path(filename).stem, ROOT / 'scripts' / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_header_names_the_packages_and_environment():
    lines = provenance.header()
    names = [ln.split()[0] for ln in lines]
    assert names[:3] == ['trustalign', 'alignbench', 'joint-align'] and names[-1] == 'environment:'
    assert re.fullmatch(r'trustalign \S+ @ \w+ \(src (clean|dirty)\) from .+', lines[0])
    assert jointalign.__version__ in lines[2]


def test_provenance_is_alignbench_s():
    # One implementation for every package (alignbench.provenance tests pinned, editable and plain
    # installs); trustalign keeps its names.
    from alignbench import provenance as ab

    assert provenance.trustalign_revision() == ab.revision('trustalign')


@pytest.mark.slow
def test_check_setup_on_synthetic_data(tmp_path):
    s01 = load_script('01_check_setup.py')
    lines = s01.main(data=tmp_path / 'no-data', results=tmp_path / 'results')
    txt = (tmp_path / 'results' / 'setup_check.txt').read_text()
    assert txt.splitlines()[2].startswith('trustalign')
    assert 'Data: not found' in txt
    err = float(re.search(r'median \|map error\| ([\d.]+) min', lines[-1]).group(1))
    assert err < 0.1                         # the pinned joint model still recovers the true drifts


def test_data_summary_counts_samples_without_reading_them(tmp_path):
    s01 = load_script('01_check_setup.py')
    for name, n in (('batch_a', 2), ('batch_b', 3)):
        (tmp_path / name / 'chroma').mkdir(parents=True)
        for i in range(n):
            (tmp_path / name / 'chroma' / f'{i}.npz').touch()
    lines = s01.data_summary(tmp_path)
    assert lines[1].split() == ['batch_a', '2', 'samples'] and lines[2].split() == ['batch_b', '3', 'samples']


TINY_GRID = [(4, 'complete', 100, 0.0), (4, 'complete', 10, 0.15), (4, 'weak_star', 10, 0.15),
             (4, 'chain', 10, 0.0), (4, 'chain', 30, 0.15), (4, 'density0.4', 30, 0.0), (5, 'density0.4', 30, 0.0)]


@pytest.mark.slow
def test_full_sparsity_on_a_tiny_grid_and_resumes(tmp_path):
    s02 = load_script('02_full_sparsity.py')
    kw = dict(grid=TINY_GRID, tune_seeds=(0,), test_seeds=(3, 4, 5), workers=1, results=tmp_path,
              lambdas=(1.0, 100.0), world_kw=dict(n_compounds=30, mz_bins=300), n_boot=50)
    out = s02.main(**kw)
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
    again = s02.main(**kw)
    key = (4, 'chain', 30, 0.15, 4)
    assert again['cells'][key]['seconds'] == cells[key]['seconds']          # reused, not refitted
    assert again['cells'][key][('Joint', 'cv')]['median_all'] == cells[key][('Joint', 'cv')]['median_all']


def test_full_sparsity_statistics():
    s02 = load_script('02_full_sparsity.py')
    assert s02.holm([0.01, 0.04, np.nan, 0.03]) == pytest.approx([0.03, 0.06, np.nan, 0.06], nan_ok=True)
    lo, hi = s02.wilson(0, 20)
    assert lo == 0.0 and 0.1 < hi < 0.2
    rng = np.random.default_rng(0)
    med, lo, hi = s02.median_ci(rng.normal(0.5, 0.1, 200), 500, rng)
    assert lo < 0.5 < hi and lo < med < hi and hi - lo < 0.1
    assert np.isnan(s02.median_ci(np.array([np.nan]), 10, rng)[0])
    assert s02.signed_rank(np.linspace(0.1, 1.0, 20)) < 0.001 and np.isnan(s02.signed_rank(np.zeros(5)))
    groups = {k: (np.full((30, 1), float(k)), 2.0 * k + rng.normal(0, 0.1, 30)) for k in (1, 2, 3)}
    est, lo, hi = s02.slope_ci(groups, 200, rng)
    assert lo[0] < 2.0 < hi[0] and abs(est[0] - 2.0) < 0.1
    assert len(s02.full_grid()) == 108 + 6 and len(s02.tasks_for(s02.full_grid(), s02.TEST_SEEDS)) == 20 * (12 + 3)
    assert s02.n_edges_of(10, 'density0.4') == 18 and s02.n_edges_of(5, 'density0.4') == 4
    labels = {(s[1], label) for s, label, _ in s02.h2_rows(s02.full_grid())}
    assert ('weak_star', 'vs Star (weak hub), indirect pairs') not in labels       # identical to joint
    assert not any(t == 'chain' and 'Star' in label for t, label in labels)       # no indirect pairs
    assert ('ring', 'vs Star, indirect pairs') in labels
    est, lo, hi = s02.slope_ci({0: (np.zeros((5, 1)), np.full(5, np.nan))}, 10, rng)
    assert np.isnan(est[0]) and np.isnan(lo[0])


@pytest.mark.slow
def test_diagnose_lambda_on_a_tiny_grid(tmp_path):
    s02 = load_script('02_full_sparsity.py')
    s03 = load_script('03_diagnose_lambda.py')
    grid = [(4, 'chain', 10, 0.30), (4, 'ring', 10, 0.0)]
    kw = dict(tune_seeds=(0,), test_seeds=(3, 4, 5), results=tmp_path, lambdas=(1.0, 10.0, 100.0),
              world_kw=dict(n_compounds=30, mz_bins=300))
    with pytest.raises(FileNotFoundError):                      # needs script 02's checkpoints first
        s03.main(grid=grid, workers=1, **kw)
    s02.main(grid=grid, workers=1, n_boot=20, **kw)
    lines = s03.main(grid=grid, workers=1, **kw)
    txt = (tmp_path / 'diagnose_lambda.txt').read_text()
    assert txt == '\n'.join(lines) + '\n' and txt.splitlines()[2].startswith('trustalign')
    for section in ('0. Check', '1. Joint model: tuned λ', '2. Median map error', '3. Broken test worlds',
                    '4. Run time'):
        assert section in txt
    check = next(line for line in lines if line.startswith('0. Check'))
    assert float(check.split('max |difference| ')[1].split()[0]) < 1e-9    # reproduces script 02 exactly
    sweep = s03.run_sweep(grid, (3, 4, 5), 1, tmp_path / 'diagnose_lambda_cache', (1.0, 10.0, 100.0),
                          (('mz_bins', 300), ('n_compounds', 30)))       # reused from the checkpoints
    e = s03.rule_errors(sweep, grid[0], (3, 4, 5), 'Joint', 10.0)
    assert np.all(e['per world'] <= e['best λ'] + 1e-12) and np.all(e['per world'] <= e['own CV'] + 1e-12)
    np.testing.assert_array_equal(e['tuned'], e['λ = 10'])


@pytest.mark.slow
def test_fish_sparsity_on_the_synthetic_stand_in(tmp_path):
    from jointalign.synthetic import make_dataset
    s04 = load_script('04_fish_sparsity.py')
    data = tmp_path / 'data'
    make_dataset(data, n_compounds=30, mz_bins=300, seed=1)          # same five batch names as the fish data
    settings = [('complete', 100, 0.0), ('weak_star', 10, 0.0), ('chain', 10, 0.15), ('ring', 10, 0.30)]
    kw = dict(settings=settings, draws=(0, 1, 2), workers=1, results=tmp_path / 'results', data=data,
              lambdas=(1.0, 10.0, 100.0), n_boot=20)
    lines = s04.main(**kw)
    txt = (tmp_path / 'results' / 'fish_sparsity.txt').read_text()
    assert txt.splitlines()[2].startswith('trustalign')
    for section in ('1. Data', '2. Held-out anchor error', '3. Pairs without direct anchors', '4. Well-connected',
                    '5. Weak hub', 'Summary (Holm-significant)', '6. Broken draws', '7. TIC correlation',
                    '8. Run time'):
        assert section in txt
    fish = s04.load_fish(data, tmp_path / 'results' / 'fish_anchors.npz')          # from the cache
    hubs = s04.hubs_of(fish)
    assert fish['names'][hubs['Star (weak hub)']] == 'batch_aug2015'
    assert fish['names'][hubs['Star']] == 'batch_sep2015'                       # most samples, most anchors
    rng = np.random.default_rng(0)
    ring = s04.edges_for('ring', fish, rng)
    chain = s04.edges_for('chain', fish, rng)
    assert len(ring) == 5 and len(chain) == 4 and ring > chain
    assert len(s04.edges_for('density0.4', fish, rng)) == 4
    # Held-out anchors are exactly those the draw did not keep, and none of them is corrupted.
    cell = s04.run_cell(fish, ('chain', 10, 0.30), 0, (10.0,))
    assert cell['n_kept'] + cell['n_held'] == len(fish['anchors']) and cell['n_kept'] <= 10 * 4
    assert np.isfinite(cell['Joint']['median_indirect'])
    # A rerun reuses every checkpointed task: identical report apart from the run-time line.
    again = s04.main(**kw)
    assert again[:-3] == lines[:-3]

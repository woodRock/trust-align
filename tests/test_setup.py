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
    assert [ln.split()[0] for ln in lines] == ['trustalign', 'joint-align', 'gcmsalign']
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

"""The environment check and the provenance lines every result file starts with."""
from __future__ import annotations

import importlib.util
import re
import shutil
from pathlib import Path

import jointalign
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

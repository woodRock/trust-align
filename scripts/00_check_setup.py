"""
00_check_setup.py — Check the environment before any experiment: versions, data access, and that
the pinned joint model still reproduces itself.

1. Provenance: trustalign, jointalign and gcmsalign revisions (as every result file records them).
2. Data: which fish oil batches are readable and how many samples each has. Nothing is copied: the
   data stays private in the gcmsalign checkout (gcmsalign.config.Paths().fish_oil).
3. Joint model: fit_joint at its defaults on jointalign's synthetic data, against the true drift.

Output: results/setup_check.txt (text only).
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
from gcmsalign.config import BIN_MIN, Paths
from gcmsalign.data import load_batch
from jointalign import fit_joint, harvest_anchors
from jointalign import synthetic as syn

from trustalign.provenance import display_path, header

ROOT = Path(__file__).resolve().parent.parent
DATA = Paths().fish_oil
RESULTS = ROOT / 'results'


def data_summary(data: Path) -> list[str]:
    batches = sorted(p for p in data.glob('batch_*') if p.is_dir()) if data.exists() else []
    if not batches:
        return [f'Data: not found at {display_path(data)} (only needed for the real-data experiments)']
    return [f'Data: {display_path(data)}'] + [
        f'  {b.name:<16} {len(list((b / "chroma").glob("*.npz")))} samples' for b in batches]


def joint_check(seed: int = 0) -> list[str]:
    """Fit the joint model to jointalign's synthetic batches, whose true drifts are known, and
    report the median error of every batch-to-batch map over 10–35 min (inside every run)."""
    with tempfile.TemporaryDirectory() as tmp:
        truth = syn.make_dataset(Path(tmp), seed=seed)
        names = sorted(spec.name for spec in syn.DEFAULT_BATCHES)
        batches = [load_batch(n, Path(tmp)) for n in names]
    anchors = harvest_anchors([b.chromas for b in batches], [b.tic_raw for b in batches])
    model = fit_joint(anchors, len(names), lam=10.0, gauge=0)
    t = np.linspace(10.0, 35.0, 26)                                     # minutes
    err = [np.abs(model.map(a, b, t / BIN_MIN) * BIN_MIN - syn.true_map(truth, names[a], names[b], t))
           for a in range(len(names)) for b in range(len(names)) if a != b]
    return [f'Joint model on synthetic data: {len(anchors.ta):,} anchors, median |map error| '
            f'{np.median(err):.3f} min over all {len(err)} ordered batch pairs (10–35 min)']


def main(data: Path = DATA, results: Path = RESULTS) -> list[str]:
    lines = ['00_check_setup.py — environment check', '', *header(), '', *data_summary(data), '',
             *joint_check()]
    results.mkdir(parents=True, exist_ok=True)
    (results / 'setup_check.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    return lines


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=DATA, help='batch_* folders (default: the fish oil data)')
    ap.add_argument('--results', type=Path, default=RESULTS)
    a = ap.parse_args()
    main(a.data, a.results)

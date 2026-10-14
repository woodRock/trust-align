"""
trustalign: how far can joint GC-MS retention-time alignment be trusted?

Paper-two experiments built on the joint model (jointalign, github.com/woodRock/joint-align):
sparse anchors, adaptive smoothness and calibrated uncertainty. The joint model itself is imported
at a pinned commit, never copied; new fitting options belong in jointalign, defaulting to off.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('trustalign')
except PackageNotFoundError:  # pragma: no cover - imported from a source tree that was never installed
    __version__ = 'unknown'

__all__ = ['__version__']

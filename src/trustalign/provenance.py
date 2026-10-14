"""Which code produced a result: the revision lines every result file in this repo starts with.

Three packages can change a number here: this one, the joint model (jointalign) and gcmsalign
(data loading, anchors, metrics). Locally the last two are usually editable checkouts, so each line
records the version, the git commit and whether that checkout's src/ was dirty.
"""
from __future__ import annotations

import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from jointalign.provenance import display_path, gcmsalign_revision, jointalign_revision

__all__ = ['display_path', 'header', 'trustalign_revision']


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def trustalign_revision() -> str:
    """e.g. 'trustalign 0.1.0 @ 1a2b3c4 (src clean)'; '(installed, no git)' outside a checkout."""
    import trustalign
    try:
        ver = version('trustalign')
    except PackageNotFoundError:
        ver = '?'
    root = Path(trustalign.__file__).resolve().parents[2]    # <root>/src/trustalign/__init__.py
    top = _git(root, 'rev-parse', '--show-toplevel')
    if not top or Path(top).resolve() != root:
        return f'trustalign {ver} (installed, no git)'
    sha = _git(root, 'rev-parse', '--short', 'HEAD') or 'unknown commit'
    state = 'src dirty' if _git(root, 'status', '--porcelain', '--', 'src') else 'src clean'
    return f'trustalign {ver} @ {sha} ({state})'


def header() -> list[str]:
    """The provenance lines for the top of a result file: this repo, the joint model, gcmsalign."""
    return [trustalign_revision(), jointalign_revision(), gcmsalign_revision()]

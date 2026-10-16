"""Which code produced a result: the revision lines every result file in this repo starts with.

Since the alignbench migration these come from alignbench.provenance (one implementation for every
package): this package, then alignbench, joint-align (an alignbench plugin) and the numeric
environment. Each line records the version, the commit (pinned git+ install or editable checkout)
and whether an editable checkout's src/ was dirty.
"""
from __future__ import annotations

from alignbench.provenance import display_path, revision
from alignbench.provenance import header as _header

__all__ = ['display_path', 'header', 'trustalign_revision']


def trustalign_revision() -> str:
    """e.g. 'trustalign 0.2.0 @ 1a2b3c4 (src clean) from ~/Desktop/trust-align'."""
    return revision('trustalign')


def header() -> list[str]:
    """The provenance lines for the top of a result file: this repo, alignbench, the joint model,
    the numeric environment."""
    return [trustalign_revision(), *_header(extra=('joint-align',))]

"""Table II: fish oil, complete graph: median held-out anchor error (min) over 20 draws.

Run by `taura build`; reads $TAURA_RESULTS and writes a taura table spec to $TAURA_OUTPUT. The rows
are a choice of settings (all of them: results/fish_sparsity.txt §2); B and ε come from the keys.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROWS = ['complete-b10-e0', 'complete-b10-e15', 'complete-b10-e30', 'complete-b30-e30',
        'complete-b100-e0', 'complete-b100-e30']
METHODS = [('Joint', 'joint'), ('Star', 'star'), ('Pairwise', 'pairwise')]


def main() -> None:
    R = json.loads(Path(os.environ['TAURA_RESULTS']).read_text())['fish-error']['metrics']
    cols = [{'label': '$B$ (Anchors/edge)'}, {'label': r'$\varepsilon$ (Corrupt.)'}]
    cols += [{'label': lab, 'metric': 'fish-error'} for lab, _ in METHODS]
    rows = []
    for key in ROWS:
        b, e = re.fullmatch(r'complete-b(\d+)-e(\d+)', key).groups()
        rows.append([b, rf'{e}\%', *[R[key][m] for _, m in METHODS]])
    Path(os.environ['TAURA_OUTPUT']).write_text(json.dumps({'columns': cols, 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()

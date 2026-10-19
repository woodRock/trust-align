"""Table I: median map error (min) over the 20 test worlds for selected settings of the synthetic grid.

Run by `taura build`; reads $TAURA_RESULTS and writes a taura table spec to $TAURA_OUTPUT. The rows
are a choice of settings (the full grid is results/full_sparsity.txt §1); K, B and ε are read back
from each setting's key, the errors from the ledger.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROWS = ['k5-complete-b10-e0', 'k5-complete-b10-e30', 'k5-complete-b30-e30', 'k5-chain-b30-e30',
        'k5-ring-b30-e30', 'k10-complete-b10-e30', 'k10-ring-b10-e30', 'k10-density0.4-b30-e15',
        'k20-complete-b30-e15']
TOPOLOGY = {'complete': r'Complete (\texttt{density 1.0})', 'chain': 'Chain', 'ring': 'Ring',
            'density0.4': r'\texttt{density 0.4}', 'star': 'Star', 'weak-star': 'Weak star'}
METHODS = [('Joint', 'joint'), ('Star', 'star'), ('Pairwise', 'pairwise'), ('PCHIP', 'pchip')]


def main() -> None:
    R = json.loads(Path(os.environ['TAURA_RESULTS']).read_text())['map-error']['metrics']
    cols = [{'label': '$K$'}, {'label': 'Anchors/edge ($B$)'}, {'label': r'Corruption ($\varepsilon$)'},
            {'label': 'Graph Topology'}]
    cols += [{'label': lab, 'metric': 'map-error'} for lab, _ in METHODS]
    rows = []
    for key in ROWS:
        k, topo, b, e = re.fullmatch(r'k(\d+)-(.+)-b(\d+)-e(\d+)', key).groups()
        rows.append([k, b, rf'{e}\%', TOPOLOGY[topo], *[R[key][m] for _, m in METHODS]])
    Path(os.environ['TAURA_OUTPUT']).write_text(json.dumps({'columns': cols, 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()

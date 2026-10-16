# trust-align

[![CI](https://github.com/woodRock/trust-align/actions/workflows/ci.yml/badge.svg)](https://github.com/woodRock/trust-align/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
[![Built on joint-align](https://img.shields.io/badge/built%20on-joint--align-6E56CF)](https://github.com/woodRock/joint-align)

**When can a joint alignment of many GC-MS batches be trusted, and how would a user know?**

Paper one ([joint-align](https://github.com/woodRock/joint-align)) fits one monotone warp per batch
into a shared retention-time frame, jointly over spectral anchors from every cross-batch sample pair,
so every batch-to-batch map is consistent by construction. Its fish oil results left three open
problems (numbers in joint-align's `results/multibatch_align.txt`):

1. **The joint model's advantage is small on well-connected data.** A star design through one
   dominant reference batch comes close to it, so that data cannot show *when* joint fitting matters.
2. **Uniform smoothing costs local accuracy.** Dense PCHIP, which interpolates every anchor, has lower
   held-out anchor error where anchors are dense.
3. **Uncertainty under-covers.** The bootstrap intervals contain fewer held-out anchors than their
   nominal 90 %, and barely widen where anchors are sparse.

Working title **[to decide]**: *When Does Joint Alignment Help? Sparse, Noisy Correspondences and
Calibrated Uncertainty in Many-Batch GC–MS Alignment.*

## Experiments

| | Question | Where |
|---|---|---|
| **1. Sparsity and corruption** | When does the joint model beat a star fit through one hub, and pairwise fits composed along paths? Synthetic worlds with exact truth, over topology × anchor budget × corruption × K, plus a thinned fish arm. | `worlds.py`, `graphs.py`, `baselines.py`, scripts 10 (pilot), 11 (draft connectivity run), 12 (breakdown diagnosis), 13 (full grid) |
| **2. Adaptive smoothness** | Can smoothness weights that follow anchor density close the local-accuracy gap without losing the joint model's gains? | `adaptive.py` (+ `smooth_weights` upstream), script 30 |
| **3. Calibrated uncertainty** | Can conformal intervals, using local evidence such as disagreement between pairwise paths, reach 90 % coverage in every stratum? | `calibrate.py`, script 40 |

As in paper one, the experiments are framed as questions rather than formal hypotheses. Each
experiment's grid, metrics and analyses are committed before its full run, and every outcome is
reported, including failures. Negative results count. (`HYPOTHESES.md` holds the hypotheses written
before the experiments' feasibility was known; they are superseded.)

## Status

| Step | State |
|---|---|
| Repository skeleton, CI, provenance, environment check | done |
| `make_world` in jointalign (in-memory synthetic worlds) | done (joint-align `20d81c8`) |
| Questions and analysis plan (hypotheses in `HYPOTHESES.md` superseded 2026-10-15) | done |
| Experiment 1 pilot (`worlds`, `graphs`, `baselines`, `metrics`, script 10) | pilot run, awaiting review |
| Experiment 1 draft connectivity run and breakdown diagnosis (scripts 11, 12) | done (draft protocol) |
| `solver='active_set'` in jointalign (joint-align `v1.1.0`) | done |
| Experiment 1 full grid (script 13) | running |
| Experiment 1 fish arm | not started |
| `smooth_weights` in jointalign, release → Experiment 2 | not started |
| Experiment 3 pilot → report → full run | not started |
| Paper figures for the draft (script 20; captions in `figs/paper/captions.md`) | done |
| Final figures and captions (numbers computed) | not started |

Each pilot stops for review before its full grid runs.

## How the repositories fit together

| Repository | Package | Role |
|---|---|---|
| [align-bench](https://github.com/woodRock/align-bench) | `alignbench` | data loading, anchors, synthetic worlds, graphs, baselines, metrics, protocols, provenance |
| [joint-align](https://github.com/woodRock/joint-align) | `jointalign` | the joint model (paper one) |
| trust-align (this) | `trustalign` | paper two's experiments |

Both upstreams are imported at pinned commits (`pyproject.toml`, mirrored in `requirements.txt`),
never copied. The worlds, anchor graphs, shortest-path composition, held-out folds and truth metrics
of Experiment 1 now live in alignbench (bit-identical; `trustalign.worlds`, `.graphs`, `.metrics`
keep their names), so they can be benchmarked with `alignbench run`. A new fitting option, such as per-knot smoothness weights, is added **to jointalign** on
its own branch: it defaults to today's behaviour, comes with a test that the default output is
bit-identical, and is released under a new tag that this repository then pins.

## Setup

```bash
git clone https://github.com/woodRock/trust-align.git
cd trust-align
pip install -e ".[dev]"            # pulls the pinned alignbench and joint-align
python scripts/00_check_setup.py   # versions, data access, joint model sanity check
make check                         # ruff, mypy, tests + coverage
```

To develop against editable checkouts of all three repositories, install the upstreams first and
this package with `--no-deps` (`make install`); otherwise pip replaces the editable checkouts with
the pinned commits:

```bash
pip install -e path/to/align-bench
pip install --no-deps -e path/to/joint-align
pip install --no-deps -e .
```

## Rules this repository follows

- **Data stays private.** The fish oil data is read in place from `data/fish_oil` (a git-ignored
  symlink to the private data) or `--data`. It is never copied, committed or uploaded, and real-data
  tests skip without it. Git holds text reports (`results/*.txt`) only, never data, caches or binaries.
- **No typed numbers.** Every number in a report, caption or this README is computed by a script and
  written to `results/*.txt`.
- **Provenance.** Every result file starts with `trustalign.provenance.header()`: version, commit and
  source state of trustalign, alignbench and jointalign, and the numeric environment.
- **Reproducible.** CPU only, and every random draw comes from an explicitly passed
  `np.random.default_rng(seed)`.
- **The evaluation metric never chooses anything.** λ, α and calibration quantiles are chosen by
  held-out anchor error on retention-time-blocked folds, never by TIC r or the synthetic truth.

## License

MIT (see `LICENSE`).

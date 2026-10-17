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
| **1. Sparsity and corruption** | When does the joint model beat a star fit through one hub, and pairwise fits composed along paths? Synthetic worlds with exact truth, over topology × anchor budget × corruption × K, plus a thinned fish arm. | `worlds.py`, `graphs.py`, `baselines.py`, scripts 02 (full grid), 03 (λ diagnosis), 04 (fish arm), 05 (figures) |
| **2. Adaptive smoothness** | Can smoothness weights that follow anchor density close the local-accuracy gap without losing the joint model's gains? | `adaptive.py` (+ `smooth_weights` upstream), scripts to come |
| **3. Calibrated uncertainty** | Can conformal intervals, using local evidence such as disagreement between pairwise paths, reach 90 % coverage in every stratum? | `calibrate.py`, scripts to come |

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
| Experiment 1 pilot, draft connectivity run and breakdown diagnosis | done; superseded by the full grid, scripts removed (in git history before v0.2.1) |
| `solver='active_set'` in jointalign (joint-align `v1.1.0`) | done |
| Experiment 1 full grid (script 02) and λ diagnosis (script 03) | done |
| Experiment 1 fish arm (script 04) | done |
| `smooth_weights` in jointalign, release → Experiment 2 | not started |
| Experiment 3 pilot → report → full run | not started |
| Experiment 1 figures and captions, numbers computed (script 05) | done |

Each pilot stops for review before its full grid runs.

## Reproduce Experiment 1

The scripts are numbered in the order that reproduces the paper's results (run from the repository
root; each step reads only what earlier steps wrote to `results/`):

```bash
python scripts/01_check_setup.py       # versions, data access, joint model sanity check (seconds)
python scripts/02_full_sparsity.py     # Experiment 1 full grid (3.6 CPU h; --workers 6 → ~40 min)
python scripts/03_diagnose_lambda.py   # λ diagnosis on script 02's cells (~12 min)
python scripts/04_fish_sparsity.py     # the fish oil arm (private data in data/fish_oil; ~15 min)
python scripts/05_paper_figures.py     # paper figures and captions from the caches above (~1 min)
```

Scripts 02–04 checkpoint every task under `results/*_cache/` and reuse it, so an interrupted run
resumes; delete (or move) those caches to recompute from scratch. A full rerun on 2026-10-17
reproduced every committed report (map errors to within 2e-10 min).

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
python scripts/01_check_setup.py   # versions, data access, joint model sanity check
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

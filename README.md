# trust-align

[![CI](https://github.com/woodRock/trust-align/actions/workflows/ci.yml/badge.svg)](https://github.com/woodRock/trust-align/actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)
[![Built on joint-align](https://img.shields.io/badge/built%20on-joint--align-6E56CF)](https://github.com/woodRock/joint-align)
[![Project site](https://img.shields.io/badge/site-woodrock.github.io%2Ftrust--align-0A7BBB)](https://woodrock.github.io/trust-align/)

**When can a joint retention-time alignment of many GC–MS batches be trusted?** Experiments on anchor-graph
connectivity and corrupted correspondences, on synthetic worlds with exact truth and on real fish oil anchors.

[Website](https://woodrock.github.io/trust-align/) · [Cite](#cite) ·
[joint-align](https://github.com/woodRock/joint-align) (the joint model) ·
[align-bench](https://github.com/woodRock/align-bench) (shared core)

## The question

A joint alignment ([joint-align](https://github.com/woodRock/joint-align)) fits one monotone
retention-time warp per GC–MS batch into a shared frame, from spectral anchors between batches, so
every batch-to-batch map is cycle-consistent by construction. Consistency is algebra, not evidence
of accuracy: on a sparse anchor graph with some wrong anchors, a joint fit can be perfectly
consistent and wrong everywhere. This repository asks when a joint fit of many batches can be
trusted, over anchor-graph topology, anchors per batch pair (*B*), the share of corrupted anchors
(ε) and the number of batches *K*, on synthetic worlds with exact truth and on joint-align's fish oil
anchors.

## Key findings

Every number below is copied from `results/*.txt`; the section it comes from is in brackets.

1. **On well-connected data, a star through a good reference is nearly as good as the joint fit.**
   Complete graph, *B* = 100: the median joint − star difference lies between −0.0053 and +0.0006 min
   over *K* and ε (`full_sparsity.txt` §3). Through the weak hub instead, the joint fit is better in
   20/20 test worlds at *B* = 10 (§4).
2. **Sparse, corrupted chains break the joint fit, and the broken fit is still cycle-consistent.**
   *K* = 5, chain, *B* = 30, ε = 0.3: median map error 1.525 min for the joint fit against 0.140 min
   for composed pairwise fits (§1); 0.80 of the test worlds broken (> 0.3 min) against 0.45 (§6).
3. **Choosing λ is only part of the story.** Over the 960 corrupted test worlds of script 03, the
   cross-validated λ leaves 175 broken, a fixed λ = 10 leaves 160, and an oracle λ chosen per world
   with the truth still leaves 80 (`diagnose_lambda.txt` §3).
4. **The joint advantage over composed pairwise fits grows with *K* at fixed edge density.** Density
   0.4, *B* = 30, ε = 0.15: slope −0.0084 min per doubling of *K* [−0.0107, −0.0063]
   (`full_sparsity.txt` §5).
5. **The fish oil anchors agree.** Complete graph, *B* = 10, ε = 0.3: held-out anchor error
   0.133 min (joint) against 0.410 (star) and 0.284 (composed pairwise); at *B* = 100, ε = 0:
   0.059, 0.069 and 0.056 min (`fish_sparsity.txt` §2).

## Scope and future work

The repository was planned around three experiments: sparsity and corruption, adaptive smoothness,
and calibrated uncertainty. Only the first was run, in full; it is a complete study on its own. The
other two remain future work (spatially adaptive or edge-grouped regularisation, and calibrated
uncertainty); no code for them exists here. `HYPOTHESES.md` is the superseded pre-registration written before the experiments'
feasibility was known; results are reported against research questions, and every outcome is
reported, including failures.

## Reproduce

The scripts are numbered in the order that reproduces every result in `results/` (run from the repository
root; each step reads only what earlier steps wrote to `results/`):

```bash
python scripts/01_check_setup.py       # versions, data access, joint model sanity check (seconds)
python scripts/02_full_sparsity.py     # Experiment 1 full grid (3.6 CPU h; --workers 6 → ~40 min)
python scripts/03_diagnose_lambda.py   # λ diagnosis on script 02's cells (~12 min)
python scripts/04_fish_sparsity.py     # the fish oil arm (private data in data/fish_oil; ~15 min)
python scripts/05_paper_figures.py     # figures and captions from the caches above (~1 min)
python scripts/06_site_data.py         # the website's data files (docs/data/*.json) from the same caches
```

Scripts 02–04 checkpoint every task under `results/*_cache/` and reuse it, so an interrupted run
resumes; delete (or move) those caches to recompute from scratch. The caches and `results/*.npz`
are git-ignored and exist only locally, so CI never regenerates results or site data: both are
generated locally and committed as text (`results/*.txt`, `docs/data/*.json`). A full rerun on
2026-10-17 after the alignbench refactor reproduced every committed report (map errors to within
2e-10 min); the reports record the code that produced them in their first lines.

`results/full_sparsity_timing.txt` is historical: the `--timing` sample run before the full grid, at
trustalign 0.1.0 under the scripts' old numbering. It projected 3.5 CPU h; the grid took 3.59 CPU h
(`full_sparsity.txt` §11). Rerun `scripts/02_full_sparsity.py --timing` (about 3 min) to refresh it.

## How the repositories fit together

| Repository | Package | Role |
|---|---|---|
| [align-bench](https://github.com/woodRock/align-bench) | `alignbench` | data loading, anchors, synthetic worlds, graphs, baselines, metrics, protocols, provenance |
| [joint-align](https://github.com/woodRock/joint-align) | `jointalign` | the joint model |
| trust-align (this) | `trustalign` | these experiments and the website |

Both upstreams are imported at pinned commits (`pyproject.toml`, mirrored in `requirements.txt`),
never copied. The worlds, anchor graphs, shortest-path composition, held-out folds and truth metrics
of Experiment 1 live in alignbench (bit-identical; `trustalign.worlds`, `.graphs`, `.metrics` keep
their names), so they can be benchmarked with `alignbench run`. A new fitting option, such as per-knot
smoothness weights, is added **to jointalign** on its own branch: it defaults to today's behaviour,
comes with a test that the default output is bit-identical, and is released under a new tag that this
repository then pins.

`warpalign` ([warp-align](https://github.com/woodRock/warp-align), the fish oil preprocessing and a
warp transformer) appears in every result header but is not a dependency: nothing
here imports it. alignbench's provenance header lists every installed `alignbench.methods` plugin,
and warpalign is installed as one on the machine that produced the results. The fish oil data is read
by `alignbench.datasets.fish_oil`, from the on-disk format warpalign's preprocessing wrote.

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

## Website

[woodrock.github.io/trust-align](https://woodrock.github.io/trust-align/) is a static site in `docs/`
(plain HTML, CSS and ES-module JavaScript; D3 from a pinned CDN URL), served by GitHub Pages from
`main`. Every number on it, in charts and in prose, comes from `docs/data/*.json`, written by
`scripts/06_site_data.py` from the same caches as the figures and checked against
`results/*.txt`; the numbers in the page text are inserted by that script, never typed. The static
figures in `docs/assets/` are written by `scripts/05_paper_figures.py --site`. Preview locally with
`python -m http.server -d docs`.

## Rules this repository follows

- **Data stays private.** The fish oil data is read in place from `data/fish_oil` (a git-ignored
  symlink to the private data) or `--data`. It is never copied, committed or uploaded, and real-data
  tests skip without it. Git holds text reports (`results/*.txt`) only, never data, caches or binaries.
  The website shows aggregate error statistics of the fish oil arm, plus one real run (fig. 1a: one sep2015 sample's TIC, display-quantised intensity matrix and spectra, agreed with the
  collaborators' data owner); no other spectra, TICs or retention times of real samples.
- **No typed numbers.** Every number in a report, caption, this README or the website is computed by
  a script and written to `results/*.txt` or `docs/data/*.json`.
- **Provenance.** Every result file starts with `trustalign.provenance.header()`: version, commit and
  source state of trustalign, alignbench and jointalign, and the numeric environment.
- **Reproducible.** CPU only, and every random draw comes from an explicitly passed
  `np.random.default_rng(seed)`.
- **The evaluation metric never chooses anything.** λ is chosen by held-out anchor error on
  retention-time-blocked folds, never by TIC r or the synthetic truth.

## Cite

If you use this code or its results, please cite the repository (`CITATION.cff` has the same entry; the
DOI is `TODO` until there is a release):

```bibtex
@software{wood_trustalign,
  author  = {Wood, Jesse},
  title   = {trust-align: when can joint {GC--MS} retention-time alignment be trusted?},
  version = {0.2.1},
  url     = {https://github.com/woodRock/trust-align},
  license = {MIT}
}
```

## License

MIT (see `LICENSE`).

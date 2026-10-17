# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository.

`trustalign` backs one paper: *When Can Joint Retention-Time Alignment Be Trusted? Graph Connectivity
and Corrupted Correspondences in GC–MS* (Wood et al., IEEE TAI, in preparation). It is paper two of
the GC–MS alignment work: [align-bench](https://github.com/woodRock/align-bench) (`alignbench`) is the
shared core, [joint-align](https://github.com/woodRock/joint-align) (`jointalign`) is the joint model
of paper one. Both are imported at the commits pinned in `pyproject.toml`, never copied.

## Commands

```bash
make check            # ruff + mypy + tests with coverage (≥ 90 %); CI runs the same
make test-fast        # pytest -m "not slow": seconds
python3.11 -m pytest tests/test_x.py::test_name
python3.11 scripts/0N_… # the numbered scripts, in order (README, "Reproduce")
python3.11 -m http.server -d docs   # preview the website
```

Call `python3.11` explicitly. The global `python3.11` holds editable checkouts of alignbench,
joint-align and trustalign (`make install`); `pip install -e ".[dev]"` would replace the upstream
checkouts with the pinned commits.

## Rules (README, "Rules this repository follows")

- **The private data stays private.** `data/fish_oil` is a git-ignored symlink. Never copy, commit or
  upload it; never put spectra, TICs or retention times of real samples on the website. Aggregate
  error statistics are fine. Tests that need the data skip without it.
- **No typed numbers.** Every number in `results/*.txt`, the figure captions, the README and the
  website is computed by a script. The website's prose numbers are `<data data-key="…">` elements
  filled by `scripts/06_site_data.py`.
- **Provenance headers** (`trustalign.provenance.header()`) start every result file. `warpalign`
  appears in them because it is an installed alignbench plugin on Jesse's machine, not a dependency.
- **CPU only, explicit seeds** (`np.random.default_rng(seed)` passed in, never global state).
- **The evaluation metric never chooses anything.** λ is chosen by RT-blocked held-out anchor error.
- Experiments 2 and 3 of the original plan (adaptive smoothness, calibrated uncertainty) are future
  work in the paper's Discussion. Do not add or remove code on their account.

## Layout

- `src/trustalign/`: thin layer over alignbench and jointalign (`worlds`, `graphs`, `metrics`,
  `baselines`, `provenance`).
- `scripts/01–05`: the paper's pipeline (setup check, full grid, λ diagnosis, fish arm, figures).
  Scripts 02–04 checkpoint under `results/*_cache/` (git-ignored, local only; script 02 is 3.6 CPU h,
  so CI can never regenerate them). Script 05 also writes `results/cache_plot_data.npz` and, with
  `--site`, the SVG figures in `docs/assets/`.
- `scripts/06_site_data.py`: reads the caches, `cache_plot_data.npz` and `results/*.txt`, regenerates
  the reports from the caches and **stops** if any line differs from the committed text, then writes
  `docs/data/*.json` (sorted keys, 4-decimal minutes) and fills the numbers in `docs/index.html`.
- `results/*.txt`: the committed reports. `results/full_sparsity_timing.txt` is historical (README).
- `docs/`: the website, served by GitHub Pages from `main`. `index.html`, `assets/site.css`,
  `assets/site.js` (ES module; D3 v7 from jsDelivr, pinned with an SRI hash), `assets/*.svg`,
  `data/*.json`, `.nojekyll`. No build step and no npm toolchain: the page's text works without
  JavaScript, charts are progressive enhancement, every chart links to its JSON.
- `tests/`: `test_site_data.py` validates the committed JSON against `results/*.txt` in CI (no
  caches) and regenerates and diffs locally (slow-marked, needs the caches).
- `HYPOTHESES.md`: the superseded pre-registration, kept as a record.

## Stop and ask before

enabling GitHub Pages, bumping a pin in `pyproject.toml`, regenerating a committed report, showing
any figure derived from real fish oil samples (fig. 1a), or tagging a release.

# Hypotheses (superseded 2026-10-15)

**Superseded.** These hypotheses were written on 2026-10-14, before it was known which experiments
could feasibly be run. Running the Experiment 1 grid showed that the planned design needed a much
faster solver (now joint-align `v1.1.0`) and changed details of the grid; for example, a budget of 100
anchors per batch pair cannot guarantee H1's "≥ 100 anchors", because some pairs have fewer. Following
paper one (joint-align), which reports results against research questions rather than formal
hypotheses, paper two does the same: each experiment's grid, metrics and analyses are committed before
its full run, and every outcome is reported (README, "Experiments"). The original text is kept below
as a record and is not used to judge results.

---

# Pre-registered hypotheses

Committed before any full experimental grid is run. The commit that adds this file is its
pre-registration date (`git log --follow HYPOTHESES.md`). Every hypothesis's outcome is reported,
including failures; any later change to this file is listed under "Amendments" with its reason.

Definitions (map error, held-out anchor error, λ selection, topologies, budgets, corruption) are those
of Handover 2, §5, as implemented in `src/trustalign/`.

## Experiment 1: sparsity and corruption

- **H1.** On complete graphs where the hub connects to all batches with ≥ 100 anchors, joint and star
  errors differ by less than 0.01 min. This is consistent with paper one's small TIC r gap between the
  joint model and the star design.
- **H2.** With a weak hub (fewest samples), or on chain and ring graphs, the joint model has lower
  error than star and composed pairwise for pairs with no direct anchors. The gap grows as the budget
  falls and as ε rises.
- **H3.** The joint model's advantage over composed pairwise grows with K at a fixed edge density.

## Experiment 2: adaptive smoothness

- **H4.** Adaptive smoothness (α > 0) reduces fish held-out anchor error without reducing drifted-pair
  TIC r by more than 0.005.

## Experiment 3: calibrated uncertainty

- **H5.** Global conformal (B) reaches 90 % marginal coverage but under-covers in the lowest density
  tertile. Local conformal (C) reaches 90 % ± 2 % in every stratum.
- **H6.** Path disagreement D_kl(t) predicts held-out |error| beyond what band width and density
  explain (a positive coefficient, significant on held-out data).

## Open decisions to settle before the grids run

- Drift-family ranges for the synthetic worlds (paper-one family and out-of-family).
- Experiment 2's success thresholds (held-out error, drifted-pair TIC r, near-aligned TIC r).

## Amendments

- 2026-10-15: superseded as a whole (see the note at the top).

"""Experiment 1 building blocks: worlds, anchor graphs, methods and metrics."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest
from jointalign import Anchors
from jointalign.synthetic import BIN_MIN, RUN_MIN, TIME_AXIS

from trustalign import baselines as bl
from trustalign import graphs, metrics, worlds

SMALL = dict(n_compounds=30, mz_bins=300)


@pytest.fixture(scope='module')
def world():
    return worlds.make(4, seed=3, **SMALL)


def shift_anchors(shifts: np.ndarray, pairs, n_per_pair: int = 40, seed: int = 0) -> Anchors:
    """Exact anchors for batches that differ by constant shifts (bins): t_b = t_a + s_b − s_a."""
    rng = np.random.default_rng(seed)
    cols: list[list] = [[] for _ in range(6)]
    for a, b in pairs:
        ta = np.sort(rng.uniform(30, 170, n_per_pair))
        for col, v in zip(cols, (np.full(n_per_pair, a), np.full(n_per_pair, b), np.zeros(n_per_pair, int),
                                 np.zeros(n_per_pair, int), ta, ta + shifts[b] - shifts[a])):
            col.append(v)
    return Anchors(*(np.concatenate(c) for c in cols))


# ── worlds ────────────────────────────────────────────────────────────────────

def test_world_is_reproducible_and_in_range(world):
    again = worlds.make(4, seed=3, **SMALL)
    assert again.specs == world.specs
    for a, b in zip(world.chromas, again.chromas):
        np.testing.assert_array_equal(a, b)
    for s in world.specs:
        assert abs(s.shift) <= 3 and abs(s.ramp) <= 3 and 0 <= s.wiggle <= 0.3
        assert worlds.N_SAMPLES[0] <= s.n <= worlds.N_SAMPLES[1] and s.start in worlds.STARTS
    assert [c.shape[0] for c in world.chromas] == world.sizes and world.K == 4
    assert worlds.make(4, seed=4, **SMALL).specs != world.specs


def test_world_truth_and_spans(world):
    t = np.array([10.0, 20.0, 30.0])
    np.testing.assert_allclose(world.true_map(1, 1, t), t, atol=1e-9)
    back = world.true_map(2, 0, world.true_map(0, 2, t))
    np.testing.assert_allclose(back, t, atol=1e-3)
    for k in range(world.K):
        lo, hi = world.observed_span(k)
        assert world.specs[k].start <= lo < hi < RUN_MIN


def test_unknown_family_is_refused():
    with pytest.raises(NotImplementedError):
        worlds.draw_specs(3, np.random.default_rng(0), family='ii')


# ── graphs ────────────────────────────────────────────────────────────────────

def test_topologies():
    sizes = [5, 3, 2, 6, 2, 4]
    K = len(sizes)
    assert len(graphs.edges_for('complete', K, sizes)) == 15
    assert graphs.edges_for('star', K, sizes) == {(0, k) for k in range(1, K)}
    assert graphs.weak_hub(sizes) == 2 and graphs.hub_of('weak_star', sizes) == 2
    assert graphs.hub_of('chain', sizes) == 0
    assert graphs.edges_for('weak_star', K, sizes) == {(min(2, k), max(2, k)) for k in range(K) if k != 2}
    assert graphs.edges_for('chain', K, sizes) == {(k, k + 1) for k in range(K - 1)}
    assert graphs.edges_for('ring', K, sizes) == graphs.edges_for('chain', K, sizes) | {(0, K - 1)}
    e1 = graphs.edges_for('er0.4', K, sizes, np.random.default_rng(1))
    assert e1 == graphs.edges_for('er0.4', K, sizes, np.random.default_rng(1))
    assert graphs.reachable(0, K, e1) == set(range(K))
    with pytest.raises(ValueError):
        graphs.edges_for('er0.2', K, sizes)
    with pytest.raises(ValueError):
        graphs.edges_for('lattice', K, sizes)


def test_thin_and_corrupt():
    anc = shift_anchors(np.zeros(3), [(0, 1), (0, 2), (1, 2)], n_per_pair=50)
    rng = np.random.default_rng(0)
    kept = anc.subset(graphs.thin(anc, {(0, 1), (1, 2)}, 20, rng))
    assert len(kept) == 40 and not np.any((kept.ba == 0) & (kept.bb == 2))
    assert len(graphs.thin(anc, {(0, 1)}, None, rng)) == 50
    bad_anc, bad = graphs.corrupt(kept, 0.25, rng)
    assert bad.sum() == 10
    d = np.abs(bad_anc.tb - kept.tb)
    assert np.all((d[bad] >= 15) & (d[bad] <= 35)) and np.all(d[~bad] == 0)
    np.testing.assert_array_equal(bad_anc.ta, kept.ta)


# ── methods ───────────────────────────────────────────────────────────────────

def test_shortest_paths_prefer_more_anchors():
    edges = {(0, 1), (1, 3), (0, 2), (2, 3)}
    paths = bl.shortest_paths(4, edges, {(0, 1): 5, (1, 3): 5, (0, 2): 9, (2, 3): 9})
    assert paths[(0, 3)] == [0, 2, 3] and paths[(3, 0)] == [3, 2, 0] and paths[(0, 1)] == [0, 1]


def test_every_method_recovers_constant_shifts():
    shifts = np.array([0.0, 6.0, -4.0, 3.0])
    edges = {(0, 1), (1, 2), (2, 3)}                                  # a chain
    anc = shift_anchors(shifts, sorted(edges))
    t = np.linspace(40, 150, 9)          # inside the anchors' range (PCHIP's pinned ends take over beyond it)
    for name, fit in bl.fitters(4, edges, hub=0).items():
        g = fit(anc, 10.0).map
        np.testing.assert_allclose(g(2, 2, t), t)
        for k, l in [(0, 1), (1, 2), (0, 3), (3, 1)]:
            if name == 'Star' and {k, l} - {0, 1}:
                continue                                              # the hub's star only reaches batch 1
            np.testing.assert_allclose(g(k, l, t), t + shifts[l] - shifts[k], atol=0.05, err_msg=f'{name} {k}→{l}')
    with pytest.raises(ValueError):
        bl.composed_pairwise(anc, 4, 10.0, {(0, 1)}).map(0, 3, t)


# ── metrics ───────────────────────────────────────────────────────────────────

def test_rt_folds_follow_paper_one():
    anc = shift_anchors(np.array([0.0, 2.0]), [(0, 1)], n_per_pair=200)
    folds = metrics.rt_folds(anc)
    assert len(folds) == metrics.N_FOLDS
    blk = lambda t: np.floor(t / 12).astype(int) % 5
    for f, (tr, te) in enumerate(folds):
        assert not np.any(tr & te)
        np.testing.assert_array_equal(te, (blk(anc.ta) == f) | (blk(anc.tb) == f))
    assert np.all(np.sum([te for _, te in folds], axis=0) >= 1)


def test_anchor_errors_and_lambda_selection():
    shifts = np.array([0.0, 5.0, -3.0])
    anc = shift_anchors(shifts, list(combinations(range(3), 2)))
    fit = lambda a, lam: bl.joint(a, 3, lam)
    np.testing.assert_allclose(metrics.anchor_errors(fit(anc, 10.0).map, anc), 0.0, atol=0.05)
    lam, scores = metrics.select_lambda(fit, anc, lambdas=(1.0, 100.0))
    assert set(scores) == {1.0, 100.0} and scores[lam] == min(scores.values())


def test_map_errors_summary_and_cycles(world):
    exact = lambda k, l, t: world.true_map(k, l, np.asarray(t) * BIN_MIN) / BIN_MIN
    errs = metrics.map_errors(world, exact)
    assert len(errs) == 12 and max(e.max() for e in errs.values() if len(e)) < 1e-9
    s = metrics.summarise({(0, 1): np.array([1.0, 3.0]), (1, 0): np.array([2.0]), (0, 2): np.array([5.0])},
                          direct={(0, 1)})
    assert s['median_direct'] == 2.0 and s['median_indirect'] == 5.0 and s['median_all'] == 2.5
    assert np.isnan(metrics.summarise({(0, 1): np.array([1.0])}, direct={(0, 1)})['median_indirect'])
    assert metrics.cycle_inconsistency(world, exact)[1] < 1e-6
    assert metrics.inside_mask(world, 0, 1, TIME_AXIS).any()


def test_rejection_rates():
    w = np.array([0.0, 0.0, 1.0, 0.5, 0.0])
    bad = np.array([True, True, False, False, False])
    assert metrics.rejection(w, bad) == (1.0, 1 / 3)
    assert np.isnan(metrics.rejection(w, np.zeros(5, bool))[0])


def test_loop_closure_and_edge_removal():
    K, rng = 8, np.random.default_rng(4)
    chain = graphs.edges_for('chain', K, [3] * K)
    plus2 = graphs.chain_with_chords(K, 2, rng)
    assert chain < plus2 and len(plus2) == K + 1 and all(b - a >= 2 for a, b in plus2 - chain)
    assert len(graphs.edges_for('chain+3', K, [3] * K, np.random.default_rng(0))) == K + 2
    order = graphs.removal_order(K, np.random.default_rng(1))
    dense = {f: graphs.thinned_complete(K, f, order) for f in (1.0, 0.6, 0.3, 0.0)}
    assert len(dense[1.0]) == 28 and len(dense[0.6]) == round(0.6 * 28) and len(dense[0.0]) == K - 1
    assert dense[0.0] <= dense[0.3] <= dense[0.6] <= dense[1.0]                     # nested deletions
    assert all(graphs.reachable(0, K, e) == set(range(K)) for e in dense.values())  # always connected
    assert len(graphs.edges_for('density0.5', K, [3] * K, np.random.default_rng(2))) == 14
    for topo in ('chain+1', 'density0.5'):
        with pytest.raises(ValueError):
            graphs.edges_for(topo, K, [3] * K)


def test_jitter_is_gaussian_on_tb_only():
    anc = shift_anchors(np.zeros(2), [(0, 1)], n_per_pair=4000)
    assert graphs.jitter(anc, 0.0, np.random.default_rng(0)) is anc
    j = graphs.jitter(anc, 1.0, np.random.default_rng(0))
    d = j.tb - anc.tb
    assert abs(d.mean()) < 0.05 and abs(d.std() - 1.0) < 0.05
    np.testing.assert_array_equal(j.ta, anc.ta)

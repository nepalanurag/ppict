"""Tests for permutation-assisted group lasso selection."""

from __future__ import annotations

import numpy as np
import pytest

from ppict.group_select import (
    _group_soft_threshold,
    _lam_max,
    calibrate_lambda,
    define_groups,
    empirical_fdp,
    group_lasso_fista,
    group_lasso_path,
    group_pen_from_z,
    jaccard,
    pairwise_jaccard,
    perm_null_group_counts,
    screen_features,
    select_groups,
    selected_features,
    stability_select,
)


def _orthonormal_design(m: int, p: int, seed: int = 0):
    """X with X^T X / m = I, so the group lasso solution is a closed form."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, p))
    Q, _ = np.linalg.qr(A)
    return Q * np.sqrt(m)


def test_group_lasso_matches_closed_form_on_orthonormal_design():
    """With X^T X / m = I the solution is exactly the group soft-thresholding of X^T y / m."""
    m, p = 60, 8
    X = _orthonormal_design(m, p, seed=1)
    rng = np.random.default_rng(2)
    y = rng.standard_normal(m)
    groups = [np.array([0, 1]), np.array([2]), np.array([3, 4, 5]), np.array([6, 7])]
    weights = np.array([1.0, 1.5, 0.5, 2.0])

    for lam in (0.01, 0.05, 0.2):
        got = group_lasso_fista(X, y, groups, lam, weights, max_iter=5000, tol=1e-14)
        z = X.T @ y / m
        expected = _group_soft_threshold(z, lam * weights, groups)
        assert np.allclose(got, expected, atol=1e-6), f"lam={lam} got={got} expected={expected}"


def test_group_lasso_is_zero_above_lam_max():
    X = _orthonormal_design(40, 6, seed=3)
    rng = np.random.default_rng(4)
    y = rng.standard_normal(40)
    groups = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    weights = np.ones(3)
    hi = _lam_max(X, y, groups, weights)
    coef = group_lasso_fista(X, y, groups, hi * 1.0001, weights, max_iter=2000, tol=1e-12)
    assert np.allclose(coef, 0.0, atol=1e-8)


def test_penalty_reduces_selected_groups_monotonically():
    rng = np.random.default_rng(5)
    n, p = 200, 20
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:4] = 1.0
    y = X @ beta + 0.5 * rng.standard_normal(n)
    groups = [np.arange(k, k + 4) for k in range(0, p, 4)]
    weights = np.ones(len(groups))

    counts = []
    for lam in np.geomspace(0.5, 0.005, 12):
        coef = group_lasso_fista(X, y, groups, float(lam), weights, max_iter=2000, tol=1e-12)
        counts.append(sum(float(np.linalg.norm(coef[g])) > 1e-12 for g in groups))
    assert counts[0] <= counts[-1]
    assert counts[-1] >= 1, "a strong signal should survive a moderate penalty"


def test_group_lasso_path_agrees_with_individual_fits():
    rng = np.random.default_rng(6)
    n, p = 120, 12
    X = rng.standard_normal((n, p))
    y = X[:, 0] * 2.0 + rng.standard_normal(n)
    groups = [np.arange(k, k + 3) for k in range(0, p, 3)]
    grid = np.array([0.4, 0.2, 0.1, 0.05])
    path = group_lasso_path(X, y, groups, grid, max_iter=3000, tol=1e-13)
    for k, lam in enumerate(grid):
        single = group_lasso_fista(X, y, groups, float(lam), max_iter=3000, tol=1e-13)
        assert np.allclose(path[k], single, atol=1e-5)


def test_define_groups_recovers_block_structure():
    rng = np.random.default_rng(7)
    n = 400
    rho = 0.85
    blocks = [(0, 4), (4, 7), (7, 10)]
    cols = []
    for lo, hi in blocks:
        s = hi - lo
        u = rng.standard_normal((n, 1))
        e = rng.standard_normal((n, s))
        cols.append(np.sqrt(rho) * u + np.sqrt(1 - rho) * e)
    X = np.column_stack(cols)

    groups = define_groups(X, threshold=0.6)
    got = sorted(tuple(int(i) for i in g) for g in groups)
    assert got == [(0, 1, 2, 3), (4, 5, 6), (7, 8, 9)]


def test_define_groups_guards_against_wide_matrices():
    X = np.zeros((5, 10))
    with pytest.raises(ValueError, match="max_features"):
        define_groups(X, max_features=4)


def test_group_pen_from_z_penalises_strong_groups_less():
    z = np.array([5.0, 4.0, -4.0, 0.1, -0.2, 0.05])
    groups = [np.array([0, 1, 2]), np.array([3, 4, 5])]
    w = group_pen_from_z(z, groups)
    assert w[0] < w[1]
    assert w.mean() == pytest.approx(1.0)


def test_screen_features_keeps_the_signal():
    rng = np.random.default_rng(8)
    n, p = 300, 40
    X = rng.standard_normal((n, p))
    y = 3.0 * X[:, 7] + rng.standard_normal(n)
    idx, r = screen_features(X, y, top_k=5)
    assert 7 in set(int(i) for i in idx)
    assert idx.size == 5
    assert r[7] == pytest.approx(r.max())


def test_calibration_respects_the_null_budget():
    """Under a global null the calibrated penalty keeps the null count in budget."""
    rng = np.random.default_rng(9)
    n, p = 150, 24
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)  # independent of X by construction
    groups = [np.arange(k, k + 4) for k in range(0, p, 4)]
    weights = np.ones(len(groups))
    q = 0.05

    lam, grid, mean_counts = calibrate_lambda(
        X, y, groups, weights, q=q, n_perm=30, n_grid=8, rng=11
    )
    chosen = int(np.argmin(np.abs(grid - lam)))
    assert mean_counts[chosen] <= q * len(groups) + 1e-12

    sel = select_groups(X, y, groups=groups, weights=weights, q=q, n_perm=30, rng=11)
    assert sel.n_selected <= 2, f"global null selected {sel.n_selected} groups"


def test_permutation_null_is_larger_for_smaller_penalty():
    rng = np.random.default_rng(12)
    X = rng.standard_normal((120, 16))
    y = rng.standard_normal(120)
    groups = [np.arange(k, k + 4) for k in range(0, 16, 4)]
    grid = np.array([0.01, 0.5])
    counts = perm_null_group_counts(X, y, groups, grid, np.ones(4), n_perm=15, rng=13)
    assert counts[:, 0].mean() >= counts[:, 1].mean()


def test_selection_finds_real_signal_groups():
    rng = np.random.default_rng(14)
    n, p = 400, 32
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[8:12] = 1.2
    y = X @ beta + rng.standard_normal(n)
    groups = [np.arange(k, k + 4) for k in range(0, p, 4)]
    sel = select_groups(X, y, groups=groups, q=0.10, n_perm=30, rng=15)
    truth = {2}
    assert truth & set(int(g) for g in sel.selected), "the signal group should be selected"
    fdp = empirical_fdp([int(g) for g in sel.selected], list(truth))
    assert fdp <= 0.5


def test_selected_features_are_sorted_and_unique():
    groups = [np.array([5, 1]), np.array([3, 9])]
    assert list(selected_features(groups, [0, 1])) == [1, 3, 5, 9]
    assert selected_features(groups, []).size == 0


def test_stability_selection_frequencies_are_bounded():
    rng = np.random.default_rng(16)
    n, p = 200, 16
    X = rng.standard_normal((n, p))
    y = 2.0 * X[:, 0] + rng.standard_normal(n)
    groups = [np.arange(k, k + 4) for k in range(0, p, 4)]
    stab = stability_select(X, y, groups=groups, B=8, subsample_frac=0.6, pi_thr=0.5,
                            q=0.10, n_perm=15, rng=17)
    assert np.all(stab.freq >= 0) and np.all(stab.freq <= 1)
    assert stab.selected.size <= len(groups)


def test_jaccard_and_fdp_edge_cases():
    assert jaccard([], []) == 1.0
    assert jaccard([1, 2], [2, 3]) == pytest.approx(1 / 3)
    assert empirical_fdp([], [1, 2]) == 0.0
    assert empirical_fdp([1, 2], [1]) == pytest.approx(0.5)
    assert pairwise_jaccard([[1], [1], [2]]).size == 3

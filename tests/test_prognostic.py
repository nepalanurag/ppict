"""Tests for prognostic score construction, including the leakage check."""

from __future__ import annotations

import numpy as np
import pytest

from ppict.prognostic import (
    cross_fitted_scores,
    external_score,
    fit_prognostic,
    r2_score,
)


def _toy(seed: int = 0, n: int = 150, p: int = 12):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[0:3] = 1.0
    y = X @ beta + 0.7 * rng.standard_normal(n)
    return X, y


def test_r2_score_is_one_for_a_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y, y) == pytest.approx(1.0)
    assert r2_score(y, np.full(4, y.mean())) == pytest.approx(0.0)
    assert np.isnan(r2_score(np.ones(4), y))


def test_cross_fitting_never_uses_a_folds_own_outcomes():
    """Changing the outcomes inside one fold must not move that fold's prediction.

    Each fold's prediction comes from a model trained on the other folds, so the
    only correct result is exact equality. The second half of the test confirms
    the other folds do move, which shows the test can detect a leak.
    """
    X, y = _toy(seed=1, n=120, p=10)
    kw = dict(n_folds=3, q=0.20, n_perm=5)

    a = cross_fitted_scores(X, y, rng=7, **kw)
    folded0 = np.flatnonzero(a.fold_assignment == 0)
    assert folded0.size > 0

    y2 = y.copy()
    y2[folded0] += 100.0
    b = cross_fitted_scores(X, y2, rng=7, **kw)

    assert np.array_equal(a.fold_assignment, b.fold_assignment)
    assert np.array_equal(
        a.oof[folded0], b.oof[folded0]
    ), "a fold prediction moved when only that fold's outcomes changed"

    others = np.flatnonzero(a.fold_assignment != 0)
    assert not np.allclose(a.oof[others], b.oof[others]), "test has no power to detect a leak"


def test_cross_fitted_scores_cover_every_row_once():
    X, y = _toy(seed=2, n=95, p=8)
    res = cross_fitted_scores(X, y, n_folds=4, q=0.2, n_perm=5, rng=3)
    assert np.all(np.isfinite(res.oof)), "every row should receive an out-of-fold score"
    assert res.fold_assignment.min() >= 0
    assert len(res.models) == 4
    assert np.isfinite(res.oof_r2)


def test_cross_fitting_is_deterministic_for_a_fixed_seed():
    X, y = _toy(seed=4)
    a = cross_fitted_scores(X, y, n_folds=3, q=0.2, n_perm=5, rng=11)
    b = cross_fitted_scores(X, y, n_folds=3, q=0.2, n_perm=5, rng=11)
    assert np.array_equal(a.oof, b.oof)
    assert np.array_equal(a.fold_assignment, b.fold_assignment)


def test_external_score_beats_the_mean_on_held_out_data():
    X, y = _toy(seed=5, n=200, p=12)
    X_tr, y_tr = X[:120], y[:120]
    X_te, y_te = X[120:], y[120:]
    preds, score = external_score(X_tr, y_tr, X_te, q=0.2, n_perm=10, rng=6)
    r2 = r2_score(y_te, preds)
    assert r2 > 0.25, f"out-of-sample R^2 was {r2}"
    assert score.n_features >= 1
    assert score.summary()["n_features"] == score.n_features


def test_external_score_is_frozen_after_training():
    """Predictions on new rows must not depend on later data."""
    X, y = _toy(seed=7, n=160, p=10)
    score = fit_prognostic(X[:100], y[:100], q=0.2, n_perm=10, rng=8)
    p1 = score.predict(X[100:])
    p2 = score.predict(X[100:])
    assert np.array_equal(p1, p2)
    assert p1.shape == (60,)


def test_score_handles_a_selection_that_finds_nothing():
    """A score with no selected features must still predict, as a constant."""
    X = np.zeros((60, 5))
    y = np.random.default_rng(0).standard_normal(60)
    score = fit_prognostic(X, y, q=0.1, n_perm=3, rng=0)
    assert score.n_features == 0
    preds = score.predict(np.ones((4, 5)))
    assert np.allclose(preds, score.intercept)


def test_screening_limits_which_features_can_be_selected():
    X, y = _toy(seed=9, n=200, p=30)
    score = fit_prognostic(X, y, top_k=6, q=0.2, n_perm=10, rng=10)
    assert np.all(score.feature_idx < 30)
    assert score.info["top_k"] == 6
    assert score.info["n_features_considered"] <= 6


def test_stability_frequency_is_recorded_when_requested():
    X, y = _toy(seed=11, n=200, p=12)
    score = fit_prognostic(X, y, q=0.2, n_perm=10, stability_B=6, rng=12)
    assert score.stability_freq is not None
    assert score.stability_freq.shape[0] == score.info["n_groups"]
    assert "n_groups_stable" in score.info

"""Prognostic score construction, with the two independence regimes PPI allows.

PLAN.md discrepancy D9 records the constraint this module exists to respect.
PPI requires the prediction rule to be independent of the data used for
inference, and Cross-PPI relaxes that only under a prediction-stability
assumption. Two paths are therefore offered and measured:

* ``fit_prognostic`` / ``external_score`` train the score on a cohort that is
  disjoint from the trial data, which is the assumption PPI states outright and
  the framing the EMA qualification of prognostic covariate adjustment uses.
* ``cross_fitted_scores`` produces out-of-fold predictions within one dataset,
  which is the Cross-PPI construction. Each fold's prediction comes from a model
  that never saw that fold. The stability of the selected feature sets across
  folds is measured and returned, because that is the condition Cross-PPI needs
  and this project does not assume it holds.

The score is deliberately refit by least squares on the selected features after
group selection. Group-lasso coefficients are shrunk, and a shrunk score is not
on the outcome scale, which matters because the estimator takes differences
between outcomes and scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .group_select import (
    define_groups,
    group_pen_from_z,
    pairwise_jaccard,
    screen_features,
    select_groups,
    selected_features,
)

__all__ = [
    "PrognosticScore",
    "CrossFitResult",
    "fit_prognostic",
    "external_score",
    "cross_fitted_scores",
    "r2_score",
]


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination. Returns nan when ``y_true`` has no variance."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot <= 0:
        return float("nan")
    ss_res = float(((y_true - y_pred) ** 2).sum())
    return 1.0 - ss_res / ss_tot


@dataclass
class PrognosticScore:
    """A frozen mapping from covariates to a predicted control outcome."""

    coef: np.ndarray
    intercept: float
    feature_idx: np.ndarray
    selected_groups: np.ndarray
    lam: float
    train_r2: float = float("nan")
    oos_r2: float = float("nan")
    stability_freq: Optional[np.ndarray] = None
    fold_jaccard: Optional[np.ndarray] = None
    info: Dict[str, object] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return int(self.feature_idx.size)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply the frozen score. Uses only features selected during training."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.feature_idx.size == 0:
            return np.full(X.shape[0], self.intercept, dtype=float)
        return X[:, self.feature_idx] @ self.coef + self.intercept

    def summary(self) -> Dict[str, object]:
        return {
            "n_features": self.n_features,
            "n_groups_selected": int(self.selected_groups.size),
            "lam": float(self.lam),
            "train_r2": float(self.train_r2),
            "oos_r2": float(self.oos_r2),
        }


def _refit_ols(
    X_sel: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Least squares refit with an intercept, via minimum-norm lstsq.

    Minimum-norm least squares keeps the refit defined when the selected feature
    set is wider than the training sample, which happens with wide omics blocks.
    """
    if X_sel.shape[1] == 0:
        return np.empty(0, dtype=float), float(np.mean(y))
    A = np.column_stack([np.ones(X_sel.shape[0]), X_sel])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return sol[1:], float(sol[0])


def fit_prognostic(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[Sequence[np.ndarray]] = None,
    weights: Optional[np.ndarray] = None,
    q: float = 0.10,
    n_perm: int = 25,
    lam: Optional[float] = None,
    group_threshold: float = 0.6,
    top_k: Optional[int] = None,
    stability_B: int = 0,
    stability_pi_thr: float = 0.6,
    rng: Optional[np.random.Generator | int] = None,
    max_iter: int = 300,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> PrognosticScore:
    """Fit a prognostic score on ``(X, y)``.

    ``y`` should be the *control* outcome, which is what makes the score
    prognostic in the sense of the prognostic covariate adjustment literature:
    the target is ``E[Y_0 | X]``.

    ``top_k`` screens features by marginal correlation before grouping, which is
    required on genome-wide matrices and harmless elsewhere. ``stability_B``
    above zero additionally measures selection frequency across subsamples.
    ``X_val``/``y_val`` measure out-of-sample R^2 and are also used to estimate
    the honest scale of the score for the calculator.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    original_idx = np.arange(X.shape[1], dtype=int)
    Xw = X
    if top_k is not None and X.shape[1] > top_k:
        keep, _ = screen_features(X, y, top_k=top_k)
        original_idx = keep
        Xw = X[:, keep]

    if groups is None:
        groups = define_groups(Xw, threshold=group_threshold)
    groups = [np.asarray(g, dtype=int) for g in groups]

    if weights is None:
        # Marginal z-scores inside the adaptive-weights function are computed
        # from the screened block, so the weights match the columns in play.
        weights = _z_from_block(Xw, y, groups)

    sel = select_groups(
        Xw, y, groups=groups, weights=weights, q=q, n_perm=n_perm, lam=lam,
        rng=rng, max_iter=max_iter,
    )

    local_idx = sel.selected_idx
    glob_idx = original_idx[local_idx] if local_idx.size else np.empty(0, dtype=int)
    X_sel = X[:, glob_idx] if glob_idx.size else np.empty((X.shape[0], 0))
    coef, intercept = _refit_ols(X_sel, y)

    train_pred = X_sel @ coef + intercept if glob_idx.size else np.full(X.shape[0], intercept)
    score = PrognosticScore(
        coef=coef,
        intercept=intercept,
        feature_idx=glob_idx,
        selected_groups=sel.selected,
        lam=sel.lam,
        train_r2=r2_score(y, train_pred),
        info={
            "q": float(q),
            "n_perm": int(n_perm),
            "n_groups": int(len(groups)),
            "group_threshold": float(group_threshold),
            "top_k": None if top_k is None else int(top_k),
            "n_features_considered": int(Xw.shape[1]),
        },
    )

    if stability_B and stability_B > 0:
        from .group_select import stability_select

        stab = stability_select(
            Xw, y, groups=groups, weights=weights, lam=sel.lam, B=stability_B,
            pi_thr=stability_pi_thr, rng=rng, max_iter=max_iter,
        )
        score.stability_freq = stab.freq
        score.info["stability_B"] = int(stability_B)
        score.info["stability_pi_thr"] = float(stability_pi_thr)
        score.info["n_groups_stable"] = int(stab.selected.size)

    if X_val is not None and y_val is not None:
        score.oos_r2 = r2_score(np.asarray(y_val, dtype=float), score.predict(X_val))

    return score


def _z_from_block(X: np.ndarray, y: np.ndarray, groups: Sequence[np.ndarray]) -> np.ndarray:
    """Adaptive group weights computed on the supplied (already screened) block."""
    from .group_select import _univariate_z

    z = _univariate_z(X, y)
    return group_pen_from_z(z, groups)


def external_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_apply: np.ndarray,
    **fit_kwargs,
) -> Tuple[np.ndarray, PrognosticScore]:
    """Train on a disjoint cohort and apply the frozen score to ``X_apply``.

    This is the independence regime PPI states in its own setup: the prediction
    rule is fit on data that is not used to form the estimator.
    """
    score = fit_prognostic(X_train, y_train, **fit_kwargs)
    return score.predict(X_apply), score


@dataclass
class CrossFitResult:
    """Out-of-fold scores plus the diagnostics the stability assumption needs."""

    oof: np.ndarray
    models: List[PrognosticScore]
    fold_assignment: np.ndarray
    oof_r2: float
    fold_jaccard: np.ndarray
    mean_fold_jaccard: float

    def summary(self) -> Dict[str, object]:
        return {
            "oof_r2": float(self.oof_r2),
            "mean_fold_jaccard": float(self.mean_fold_jaccard),
            "n_folds": int(len(self.models)),
            "mean_features_per_fold": float(np.mean([m.n_features for m in self.models])),
        }


def cross_fitted_scores(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    groups: Optional[Sequence[np.ndarray]] = None,
    rng: Optional[np.random.Generator | int] = None,
    **fit_kwargs,
) -> CrossFitResult:
    """Out-of-fold prognostic scores, each from a model that excluded its fold.

    Following Cross-PPI (arXiv:2309.16598), which trains one model per fold on all
    data except that fold. When ``groups`` is not supplied, grouping is re-derived
    inside each fold, so the fold model is a complete pipeline that never saw the
    held-out rows.

    The mean pairwise Jaccard index across folds is returned as a measured proxy
    for the prediction-stability condition. A low value means the fold models
    disagree about which features matter, which is exactly the regime the
    assumption excludes.

    Determinism: the fold assignment and one seed per fold are drawn from the
    outer generator up front, before any model is fitted. A fixed outer seed
    therefore fixes the whole cross-fit, and changing the outcomes in one fold
    cannot change that fold's own prediction.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[0]
    n_folds = int(min(max(2, n_folds), max(2, n)))
    rng = np.random.default_rng(rng)

    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    fold_seeds = [int(rng.integers(2**31)) for _ in range(n_folds)]

    oof = np.full(n, np.nan, dtype=float)
    assignment = np.full(n, -1, dtype=int)
    models: List[PrognosticScore] = []
    selected_sets: List[np.ndarray] = []

    for k, test_rows in enumerate(folds):
        train_rows = np.setdiff1d(idx, test_rows, assume_unique=False)
        if train_rows.size < 2 or test_rows.size == 0:
            continue
        score = fit_prognostic(
            X[train_rows], y[train_rows], groups=groups, rng=fold_seeds[k], **fit_kwargs
        )
        oof[test_rows] = score.predict(X[test_rows])
        assignment[test_rows] = k
        models.append(score)
        selected_sets.append(score.feature_idx)

    mask = assignment >= 0
    oof_r2 = r2_score(y[mask], oof[mask]) if mask.any() else float("nan")
    jac = pairwise_jaccard(selected_sets) if len(selected_sets) > 1 else np.array([])

    return CrossFitResult(
        oof=oof,
        models=models,
        fold_assignment=assignment,
        oof_r2=float(oof_r2),
        fold_jaccard=jac,
        mean_fold_jaccard=float(jac.mean()) if jac.size else float("nan"),
    )

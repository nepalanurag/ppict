"""Permutation-assisted group lasso for grouped covariates.

Python counterpart of the ``permgrouplasso`` R package that implements the
thesis method (see PLAN.md). Two things are stated plainly here because they are
easy to overclaim:

1. The permutation step *calibrates* the group-lasso penalty against a null
   distribution. It is not a model-X knockoff construction and it carries no
   formal false-discovery-rate guarantee. PLAN.md discrepancy D3 records why the
   project reports measured selection frequencies and measured false-discovery
   proportions instead of an asserted FDR target.
2. Stability is reported as an empirical selection frequency across subsamples,
   which is what the simulation harness measures and reports.

The group lasso solver is a proximal gradient method (FISTA) with group
soft-thresholding. It is written here rather than imported because the
permutation calibration refits the model at many fixed penalties on permuted
data, and because the orthonormal design has an exact closed-form solution that
turns the solver into something testable against an analytic answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

__all__ = [
    "GroupSelection",
    "StabilitySelection",
    "define_groups",
    "screen_features",
    "group_pen_from_z",
    "group_lasso_fista",
    "group_lasso_path",
    "perm_null_group_counts",
    "calibrate_lambda",
    "select_groups",
    "stability_select",
    "selected_features",
    "empirical_fdp",
    "jaccard",
    "pairwise_jaccard",
]


# --------------------------------------------------------------------------
# Group definition and adaptive weights
# --------------------------------------------------------------------------


def define_groups(
    X: np.ndarray,
    threshold: float = 0.6,
    max_features: int = 3000,
) -> List[np.ndarray]:
    """Group features by absolute correlation (connected components).

    Two features belong to the same group when their absolute Pearson
    correlation is at least ``threshold``. Groups are the connected components of
    that graph, so a group can hold more than two features.

    ``max_features`` guards against an accidental O(p^2) correlation matrix on a
    genome-wide matrix. Screen first (``screen_features``) and pass the reduced
    matrix.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    p = X.shape[1]
    if p == 0:
        return []
    if p > max_features:
        raise ValueError(
            f"define_groups received {p} features, above max_features={max_features}. "
            "Screen features first with screen_features() and group the reduced matrix."
        )
    if p == 1:
        return [np.array([0], dtype=int)]

    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    adj = np.abs(C) >= threshold
    np.fill_diagonal(adj, True)

    n_comp, labels = connected_components(csr_matrix(adj), directed=False)
    groups = [np.flatnonzero(labels == k).astype(int) for k in range(n_comp)]
    # Deterministic ordering: by smallest member index.
    groups.sort(key=lambda g: int(g[0]))
    return groups


def screen_features(
    X: np.ndarray, y: np.ndarray, top_k: int = 2000
) -> Tuple[np.ndarray, np.ndarray]:
    """Keep the ``top_k`` features by absolute Pearson correlation with ``y``.

    Returns ``(sorted_index_of_kept_features, all_absolute_correlations)``.
    The returned indices are sorted so downstream column order stays stable.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    denom = np.sqrt((Xc**2).sum(axis=0) * float(yc @ yc))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.abs(Xc.T @ yc) / denom
    r = np.nan_to_num(r, nan=0.0)
    k = int(min(top_k, X.shape[1]))
    idx = np.sort(np.argsort(-r, kind="stable")[:k])
    return idx, r


def group_pen_from_z(
    z: np.ndarray,
    groups: Sequence[np.ndarray],
    gamma: float = 1.0,
    eps: float = 1e-8,
    normalize: bool = True,
) -> np.ndarray:
    """Adaptive group penalties from univariate z-scores.

    A group whose members carry strong marginal signal receives a *smaller*
    penalty, which is the adaptive-lasso idea applied at group level. The group
    statistic is the root-mean-square z within the group, so group size does not
    by itself change the weight.

    Weights are normalised to mean 1 so that a given ``lam`` is interpretable as
    the average penalty across groups.
    """
    z = np.asarray(z, dtype=float).ravel()
    raw = np.empty(len(groups), dtype=float)
    for j, g in enumerate(groups):
        g = np.asarray(g, dtype=int)
        raw[j] = np.linalg.norm(z[g]) / np.sqrt(max(len(g), 1))
    w = 1.0 / (np.power(raw, gamma) + eps)
    if normalize and w.size and w.mean() > 0:
        w = w / w.mean()
    return w


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------


def _spectral_norm_sq(X: np.ndarray, n_iter: int = 100, tol: float = 1e-9) -> float:
    """Largest squared singular value of X, by power iteration."""
    p = X.shape[1]
    if p == 0:
        return 0.0
    v = np.full(p, 1.0 / np.sqrt(p))
    prev = None
    for _ in range(n_iter):
        u = X @ v
        v_new = X.T @ u
        nrm = float(np.linalg.norm(v_new))
        if nrm == 0.0:
            return 0.0
        v_new = v_new / nrm
        if prev is not None and abs(float(v_new @ prev) - 1.0) < tol:
            v = v_new
            break
        prev = v
        v = v_new
    return float(np.linalg.norm(X @ v) ** 2)


def _group_soft_threshold(v: np.ndarray, thr: np.ndarray, groups: Sequence[np.ndarray]) -> np.ndarray:
    out = np.zeros_like(v)
    for j, g in enumerate(groups):
        vg = v[g]
        nrm = float(np.linalg.norm(vg))
        if nrm <= thr[j] or nrm == 0.0:
            continue
        out[g] = vg * (1.0 - thr[j] / nrm)
    return out


def _group_objective(
    X: np.ndarray,
    y: np.ndarray,
    b: np.ndarray,
    lam: float,
    weights: np.ndarray,
    groups: Sequence[np.ndarray],
    m: float,
) -> float:
    r = y - X @ b
    quad = float(r @ r) / (2.0 * m)
    pen = 0.0
    for j, g in enumerate(groups):
        pen += weights[j] * float(np.linalg.norm(b[g]))
    return quad + lam * pen


def group_lasso_fista(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[np.ndarray],
    lam: float,
    weights: Optional[np.ndarray] = None,
    max_iter: int = 500,
    tol: float = 1e-9,
    b_init: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Minimise ``||y - Xb||^2 / (2m) + lam * sum_g w_g ||b_g||_2``.

    The objective is

        1/(2m) * ||y - Xb||^2  +  lam * sum_g weights_g * ||b_g||_2

    solved by FISTA with the exact group soft-thresholding prox operator. The
    Lipschitz constant of the smooth part is ``||X||_2^2 / m``.

    ``b_init`` allows a warm start, which the calibration path uses heavily.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m, p = X.shape
    groups = [np.asarray(g, dtype=int) for g in groups]
    if weights is None:
        weights = np.ones(len(groups), dtype=float)
    weights = np.asarray(weights, dtype=float).ravel()
    if weights.size != len(groups):
        raise ValueError("weights must have one entry per group")

    L = _spectral_norm_sq(X) / m
    if L <= 0:
        # X is all zeros: the minimiser is the zero vector.
        return np.zeros(p)
    step = 1.0 / L

    b = np.zeros(p) if b_init is None else np.array(b_init, dtype=float)
    z = b.copy()
    t = 1.0
    prev_obj = _group_objective(X, y, b, lam, weights, groups, float(m))

    for _ in range(max_iter):
        grad = -(X.T @ (y - X @ z)) / m
        v = z - step * grad
        b_new = _group_soft_threshold(v, step * lam * weights, groups)

        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = b_new + ((t - 1.0) / t_new) * (b_new - b)
        t = t_new

        obj = _group_objective(X, y, b_new, lam, weights, groups, float(m))
        if abs(prev_obj - obj) <= tol * max(1.0, abs(prev_obj)):
            b = b_new
            break
        b = b_new
        prev_obj = obj

    return b


def group_lasso_path(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[np.ndarray],
    lam_grid: Sequence[float],
    weights: Optional[np.ndarray] = None,
    max_iter: int = 500,
    tol: float = 1e-9,
) -> np.ndarray:
    """Solve for every penalty in ``lam_grid`` with warm starts.

    ``lam_grid`` is processed from largest to smallest so each solve starts from
    the previous solution. Returns an array of shape ``(len(lam_grid), p)`` in
    the order the penalties were supplied.
    """
    lam_grid = np.asarray(lam_grid, dtype=float).ravel()
    order = np.argsort(-lam_grid)
    out = np.zeros((lam_grid.size, X.shape[1]), dtype=float)
    b = None
    for pos in order:
        b = group_lasso_fista(
            X, y, groups, float(lam_grid[pos]), weights,
            max_iter=max_iter, tol=tol, b_init=b,
        )
        out[pos] = b
    return out


def _nonzero_groups(b: np.ndarray, groups: Sequence[np.ndarray], tol: float = 1e-12) -> np.ndarray:
    flags = [float(np.linalg.norm(b[np.asarray(g, dtype=int)])) > tol for g in groups]
    return np.flatnonzero(np.array(flags, dtype=bool))


def _lam_max(X: np.ndarray, y: np.ndarray, groups: Sequence[np.ndarray], weights: np.ndarray) -> float:
    """Smallest penalty whose solution is exactly zero, for this particular y.

    At b = 0 the proximal step zeroes group g unless
    ``lam >= ||X_g^T y|| / (m * w_g)``.
    """
    m = X.shape[0]
    gx = np.array([float(np.linalg.norm(X[:, np.asarray(g, dtype=int)].T @ y)) for g in groups])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = gx / (float(m) * weights)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0 or not np.any(ratios > 0):
        return 1.0
    return float(ratios.max())


def _lam_upper_bound(
    X: np.ndarray, y: np.ndarray, groups: Sequence[np.ndarray], weights: np.ndarray
) -> float:
    """Penalty above which no row-permutation of ``y`` can select any group.

    The permutation null refits the model on permuted outcomes, so a grid whose
    top end only zeroes the *observed* outcome can still select groups under
    permutation. That made the calibration budget unreachable and the search
    silently return an infeasible penalty.

    A permutation only reorders the entries of ``y``, so
    ``||X_g^T y_perm||_2 <= ||X_g||_F * ||y||_2`` for every permutation, by
    Cauchy-Schwarz. Taking the maximum over groups gives a penalty that zeroes
    every group under every permutation, so the budget is always achievable.
    """
    m = X.shape[0]
    y_norm = float(np.linalg.norm(y))
    vals = []
    for j, g in enumerate(groups):
        Xg = X[:, np.asarray(g, dtype=int)]
        vals.append(
            float(np.linalg.norm(Xg, "fro")) * y_norm / (float(m) * max(float(weights[j]), 1e-12))
        )
    vals = [v for v in vals if np.isfinite(v)]
    return float(max(vals)) if vals else 1.0


# --------------------------------------------------------------------------
# Permutation calibration
# --------------------------------------------------------------------------


def perm_null_group_counts(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[np.ndarray],
    lam_grid: Sequence[float],
    weights: Optional[np.ndarray] = None,
    n_perm: int = 25,
    rng: Optional[np.random.Generator | int] = None,
    max_iter: int = 300,
    tol: float = 1e-8,
) -> np.ndarray:
    """Selected-group counts under a row-permuted outcome, for each penalty.

    Permuting ``y`` (equivalently permuting the rows of ``X``) destroys any
    association between features and outcome while leaving the marginal
    distributions untouched. The return value has shape
    ``(n_perm, len(lam_grid))``.
    """
    rng = np.random.default_rng(rng)
    y = np.asarray(y, dtype=float).ravel()
    lam_grid = np.asarray(lam_grid, dtype=float).ravel()
    counts = np.zeros((n_perm, lam_grid.size), dtype=int)
    for b in range(n_perm):
        y_perm = rng.permutation(y)
        path = group_lasso_path(
            X, y_perm, groups, lam_grid, weights, max_iter=max_iter, tol=tol
        )
        for k in range(lam_grid.size):
            counts[b, k] = _nonzero_groups(path[k], groups).size
    return counts


def calibrate_lambda(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence[np.ndarray],
    weights: Optional[np.ndarray] = None,
    q: float = 0.10,
    n_perm: int = 25,
    lam_grid: Optional[Sequence[float]] = None,
    n_grid: int = 12,
    rng: Optional[np.random.Generator | int] = None,
    max_iter: int = 300,
    tol: float = 1e-8,
    allow_all_zero: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Smallest penalty whose permutation-null group count stays within budget.

    The per-family budget is    ``q * G`` selected groups in expectation under the
    global null, with ``G`` the number of groups. This is a calibration rule, not
    an FDR guarantee (PLAN.md D3).

    A larger penalty selects fewer groups, so the null count falls as the penalty
    rises. The chosen value is the *smallest* penalty on the grid whose null count
    is within budget, which is the most powerful choice that still respects it.
    Selection is by penalty value rather than grid position, so the rule does not
    depend on how the grid is ordered.

    Returns ``(lam, lam_grid, mean_null_count_per_grid_point)``. If no penalty on
    the grid satisfies the budget, the largest grid value is returned, which
    selects nothing.
    """
    if weights is None:
        weights = np.ones(len(groups), dtype=float)
    weights = np.asarray(weights, dtype=float).ravel()
    if lam_grid is None:
        hi = _lam_upper_bound(X, y, groups, weights)
        lo = max(hi * 1e-5, 1e-15)
        lam_grid = np.geomspace(max(hi, 1e-12), lo, int(n_grid))
    lam_grid = np.asarray(lam_grid, dtype=float).ravel()

    counts = perm_null_group_counts(
        X, y, groups, lam_grid, weights, n_perm=n_perm, rng=rng,
        max_iter=max_iter, tol=tol,
    )
    mean_counts = counts.mean(axis=0)
    budget = float(q) * len(groups)

    ok = np.flatnonzero(mean_counts <= budget)
    if ok.size == 0:
        if not allow_all_zero:
            raise RuntimeError(
                "no penalty on the grid met the permutation budget; "
                f"smallest mean null count was {mean_counts.min():.3f} against budget {budget:.3f}"
            )
        return float(lam_grid[int(np.argmax(lam_grid))]), lam_grid, mean_counts
    chosen = ok[int(np.argmin(lam_grid[ok]))]
    return float(lam_grid[chosen]), lam_grid, mean_counts


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------


@dataclass
class GroupSelection:
    """Result of one calibration-and-fit pass."""

    groups: List[np.ndarray]
    weights: np.ndarray
    lam: float
    coef: np.ndarray
    selected: np.ndarray  # indices into ``groups``
    lam_grid: np.ndarray
    null_mean_counts: np.ndarray
    n_perm: int = 0

    @property
    def n_selected(self) -> int:
        return int(self.selected.size)

    @property
    def selected_idx(self) -> np.ndarray:
        """Sorted feature indices belonging to selected groups."""
        if self.selected.size == 0:
            return np.empty(0, dtype=int)
        return selected_features(self.groups, self.selected)

    @property
    def selected_groups(self) -> List[np.ndarray]:
        return [self.groups[int(j)] for j in self.selected]


def selected_features(groups: Sequence[np.ndarray], selected: Sequence[int]) -> np.ndarray:
    """Sorted feature indices covered by the selected groups."""
    if len(selected) == 0:
        return np.empty(0, dtype=int)
    idx = np.concatenate([np.asarray(groups[int(j)], dtype=int) for j in selected])
    return np.sort(np.unique(idx))


def select_groups(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[Sequence[np.ndarray]] = None,
    weights: Optional[np.ndarray] = None,
    q: float = 0.10,
    n_perm: int = 25,
    lam: Optional[float] = None,
    lam_grid: Optional[Sequence[float]] = None,
    group_threshold: float = 0.6,
    rng: Optional[np.random.Generator | int] = None,
    max_iter: int = 300,
    tol: float = 1e-8,
) -> GroupSelection:
    """Calibrate the penalty (unless ``lam`` is given) and return what it selects."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if groups is None:
        groups = define_groups(X, threshold=group_threshold)
    groups = [np.asarray(g, dtype=int) for g in groups]

    if weights is None:
        z = _univariate_z(X, y)
        weights = group_pen_from_z(z, groups)
    weights = np.asarray(weights, dtype=float).ravel()

    if lam is None:
        lam, grid, mean_counts = calibrate_lambda(
            X, y, groups, weights, q=q, n_perm=n_perm, lam_grid=lam_grid,
            rng=rng, max_iter=max_iter, tol=tol,
        )
    else:
        grid = np.asarray([lam], dtype=float)
        mean_counts = np.zeros(1, dtype=float)

    coef = group_lasso_fista(X, y, groups, float(lam), weights, max_iter=max_iter, tol=tol)
    return GroupSelection(
        groups=list(groups),
        weights=weights,
        lam=float(lam),
        coef=coef,
        selected=_nonzero_groups(coef, groups),
        lam_grid=np.asarray(grid, dtype=float),
        null_mean_counts=np.asarray(mean_counts, dtype=float),
        n_perm=int(n_perm),
    )


def _univariate_z(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-feature z statistic for the slope in a simple linear regression."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    sxx = (Xc**2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = (Xc.T @ yc) / sxx
        resid = yc[:, None] - Xc * beta[None, :]
        dof = max(n - 2, 1)
        s2 = (resid**2).sum(axis=0) / dof
        se = np.sqrt(s2 / sxx)
        z = beta / se
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class StabilitySelection:
    """Selection frequency across subsamples at a fixed penalty."""

    groups: List[np.ndarray]
    lam: float
    freq: np.ndarray  # per group, in [0, 1]
    selected: np.ndarray
    B: int
    subsample_frac: float
    pi_thr: float
    per_fold_selected: List[np.ndarray] = field(default_factory=list)

    @property
    def selected_idx(self) -> np.ndarray:
        return selected_features(self.groups, self.selected)

    @property
    def mean_pairwise_jaccard(self) -> float:
        return pairwise_jaccard(self.per_fold_selected).mean() if self.B > 1 else float("nan")


def stability_select(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[Sequence[np.ndarray]] = None,
    weights: Optional[np.ndarray] = None,
    lam: Optional[float] = None,
    B: int = 30,
    subsample_frac: float = 0.5,
    pi_thr: float = 0.6,
    q: float = 0.10,
    n_perm: int = 25,
    group_threshold: float = 0.6,
    rng: Optional[np.random.Generator | int] = None,
    max_iter: int = 300,
    tol: float = 1e-8,
) -> StabilitySelection:
    """Refit on ``B`` subsamples at a fixed penalty and count group selections.

    The penalty is calibrated once on the full sample when ``lam`` is not given,
    then held fixed across subsamples, which is what makes the selection
    frequencies comparable. A group is reported as selected when its frequency is
    at least ``pi_thr``.
    """
    rng = np.random.default_rng(rng)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[0]

    base = select_groups(
        X, y, groups=groups, weights=weights, q=q, n_perm=n_perm, lam=lam,
        group_threshold=group_threshold, rng=rng, max_iter=max_iter, tol=tol,
    )
    groups = base.groups
    weights = base.weights
    lam = base.lam

    m = int(max(2, np.floor(subsample_frac * n)))
    freq = np.zeros(len(groups), dtype=float)
    per_fold: List[np.ndarray] = []
    if B > 0 and m >= 2:
        for _ in range(B):
            rows = rng.choice(n, size=m, replace=False)
            coef = group_lasso_fista(
                X[rows], y[rows], groups, lam, weights, max_iter=max_iter, tol=tol
            )
            sel = _nonzero_groups(coef, groups)
            per_fold.append(sel)
            if sel.size:
                freq[sel] += 1.0
        freq /= float(B)

    selected = np.flatnonzero(freq >= pi_thr)
    return StabilitySelection(
        groups=list(groups), lam=float(lam), freq=freq, selected=selected,
        B=int(B), subsample_frac=float(subsample_frac), pi_thr=float(pi_thr),
        per_fold_selected=per_fold,
    )


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def empirical_fdp(selected: Sequence[int], active: Sequence[int]) -> float:
    """Measured proportion of selected groups that are not in ``active``.

    Defined on feature indices or group indices; the caller keeps them consistent.
    Returns 0.0 when nothing was selected, and reports ``nan`` only if the caller
    passes an empty ``active`` set, which makes the quantity undefined.
    """
    sel = set(int(i) for i in selected)
    if not sel:
        return 0.0
    act = set(int(i) for i in active)
    return len(sel - act) / float(len(sel))


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    """Jaccard index of two index collections. Two empty sets score 1.0."""
    sa, sb = set(int(i) for i in a), set(int(i) for i in b)
    union = sa | sb
    if not union:
        return 1.0
    return len(sa & sb) / float(len(union))


def pairwise_jaccard(sets: Sequence[Sequence[int]]) -> np.ndarray:
    """All pairwise Jaccard indices, flattened."""
    out = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            out.append(jaccard(sets[i], sets[j]))
    return np.array(out, dtype=float)

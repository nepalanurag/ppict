"""Prediction-powered estimation and prognostic covariate adjustment.

Sources for every formula here, with the location in the primary text:

* Point estimate and tuning. Angelopoulos, Duchi, Zrnic, "PPI++: Efficient
  Prediction-Powered Inference", arXiv:2311.01453. Example 6.1 (mean estimator
  and closed-form tuning), Eq. (8) (plug-in tuning), Theorem 1 Eq. (7) (asymptotic
  variance), Corollary 4 (the plug-in tuning value carries no first-order variance
  penalty). The variance reduction for a mean is ``rho^2 * N / (n + N)``, which
  tends to ``rho^2`` as the unlabeled sample grows.
* Two-arm treatment effect. Poulet, Tran, Tezenas du Montcel, Dubois, Durrleman,
  Jedynak, "Prediction-powered inference for clinical trials: application to
  linear covariate adjustment", BMC Medical Research Methodology 25:204 (2025),
  doi:10.1186/s12874-025-02647-6, who establish that the tuned prediction-powered
  estimator of the ATE is asymptotically equivalent to ANCOVA with the prognostic
  score as the single covariate.
* Prognostic covariate adjustment and allocation. Schuler, Walsh, Hall, Walsh,
  Fisher, arXiv:2012.09935, Theorem 4 (adjusted variance bound as a function of
  the allocation), and the EMA CHMP qualification opinion for PROCOVA
  (20 September 2022), which states the approximate total sample size ratio
  ``1 - R^2``.

Two deliberate conventions, both documented because they are choices:

1. ``ppi_mean_*`` reproduces the authors' reference implementation
   (``ppi-python``) exactly, including its choice to clip the plug-in tuning
   value to ``[0, 1]`` and its use of ``ddof=0`` sample variances inside the
   confidence interval. The test suite asserts this agreement numerically.
   Pass ``lam_clip=None`` for the unconstrained optimum of the paper, which can
   exceed 1 or be negative when the score is anti-correlated with the outcome.
2. The two-arm estimators below are derived from the PPI++ variance expression
   rather than copied from a reference, so they use ``ddof=1`` sample
   moments and normal quantiles throughout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize, stats

__all__ = [
    "z_quantile",
    "tuned_lambda_mean",
    "ppi_mean_pointestimate",
    "ppi_mean_se",
    "ppi_mean_ci",
    "ppi_mean_pvalue",
    "ATEEstimate",
    "ate_unadjusted",
    "ate_ancova",
    "ate_ppi_single_lambda",
    "ate_ppi_per_arm",
    "ate_lasso_adjusted",
    "procova_variance",
    "unadjusted_min_variance",
    "optimal_allocation",
    "AllocationResult",
    "required_n_total",
    "sample_size_reduction",
    "control_arm_size",
]

# --------------------------------------------------------------------------
# Mean estimation (PPI++ Example 6.1)
# --------------------------------------------------------------------------


def z_quantile(p: float) -> float:
    """Standard normal quantile."""
    return float(stats.norm.ppf(p))


def tuned_lambda_mean(
    y: np.ndarray,
    f_labeled: np.ndarray,
    f_unlabeled: np.ndarray,
    lam_clip: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> float:
    """Plug-in tuning value for the PPI++ mean estimator.

    ``lambda_hat = Cov(Y, f) / ((1 + n/N) * Var(f))``, where the covariance uses
    the labeled sample and the variance uses the labeled and unlabeled
    predictions pooled. This is Eq. (8) of arXiv:2311.01453 specialized to
    Example 6.1, and it matches the reference implementation's
    ``_calc_lam_glm`` for the mean case.

    ``lam_clip`` defaults to ``(0, 1)`` to match the reference implementation.
    Pass ``None`` for the unconstrained value, which is the variance minimizer
    when the sample is large.
    """
    y = np.asarray(y, dtype=float).ravel()
    f_labeled = np.asarray(f_labeled, dtype=float).ravel()
    f_unlabeled = np.asarray(f_unlabeled, dtype=float).ravel()
    n = y.size
    N = f_unlabeled.size
    if n < 2:
        return 0.0

    yc = y - y.mean()
    fc = f_labeled - f_labeled.mean()
    cov = float(yc @ fc) / n  # biased covariance, as in the reference

    pooled = np.concatenate([f_labeled, f_unlabeled])
    if pooled.size < 2:
        return 0.0
    var_pooled = float(np.var(pooled, ddof=1))
    if var_pooled <= 0:
        return 0.0

    lam = cov / ((1.0 + n / float(max(N, 1))) * var_pooled)
    if lam_clip is not None:
        lam = float(np.clip(lam, lam_clip[0], lam_clip[1]))
    return float(lam)


def ppi_mean_pointestimate(
    y: np.ndarray,
    f_labeled: np.ndarray,
    f_unlabeled: np.ndarray,
    lam: Optional[float] = None,
    lam_clip: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> float:
    """PPI++ point estimate of a mean: ``mean(Y) + lam * (mean(f_unl) - mean(f_lab))``."""
    y = np.asarray(y, dtype=float).ravel()
    f_labeled = np.asarray(f_labeled, dtype=float).ravel()
    f_unlabeled = np.asarray(f_unlabeled, dtype=float).ravel()
    if lam is None:
        lam = tuned_lambda_mean(y, f_labeled, f_unlabeled, lam_clip=lam_clip)
    return float(
        lam * f_unlabeled.mean() + (y - lam * f_labeled).mean()
    )


def ppi_mean_se(
    y: np.ndarray,
    f_labeled: np.ndarray,
    f_unlabeled: np.ndarray,
    lam: float,
) -> float:
    """Standard error of the PPI++ mean, treating ``lam`` as fixed.

    ``Var(Y - lam f)/n + lam^2 Var(f)/N``. Corollary 4 of arXiv:2311.01453 is the
    justification for not adding a term for the estimation of ``lam``; that
    justification is asymptotic, and the simulation harness measures the
    finite-sample consequence instead of assuming it.
    """
    y = np.asarray(y, dtype=float).ravel()
    f_labeled = np.asarray(f_labeled, dtype=float).ravel()
    f_unlabeled = np.asarray(f_unlabeled, dtype=float).ravel()
    n = y.size
    N = f_unlabeled.size
    rect = float(np.std(y - lam * f_labeled, ddof=0)) / np.sqrt(n) if n else np.nan
    imp = float(np.std(lam * f_unlabeled, ddof=0)) / np.sqrt(N) if N else 0.0
    return float(np.sqrt(rect**2 + imp**2))


def ppi_mean_ci(
    y: np.ndarray,
    f_labeled: np.ndarray,
    f_unlabeled: np.ndarray,
    lam: Optional[float] = None,
    alpha: float = 0.05,
    lam_clip: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> Tuple[float, float, float]:
    """Two-sided normal-approximation PPI++ interval. Returns ``(low, high, lam)``."""
    if lam is None:
        lam = tuned_lambda_mean(y, f_labeled, f_unlabeled, lam_clip=lam_clip)
    point = ppi_mean_pointestimate(y, f_labeled, f_unlabeled, lam=lam)
    se = ppi_mean_se(y, f_labeled, f_unlabeled, lam=lam)
    half = z_quantile(1.0 - alpha / 2.0) * se
    return float(point - half), float(point + half), float(lam)


def ppi_mean_pvalue(
    y: np.ndarray,
    f_labeled: np.ndarray,
    f_unlabeled: np.ndarray,
    null: float = 0.0,
    lam: Optional[float] = None,
    lam_clip: Optional[Tuple[float, float]] = (0.0, 1.0),
) -> float:
    """Two-sided normal p-value for ``H0: mean = null``."""
    if lam is None:
        lam = tuned_lambda_mean(y, f_labeled, f_unlabeled, lam_clip=lam_clip)
    point = ppi_mean_pointestimate(y, f_labeled, f_unlabeled, lam=lam)
    se = ppi_mean_se(y, f_labeled, f_unlabeled, lam=lam)
    if se <= 0 or not np.isfinite(se):
        return 1.0
    z = (point - null) / se
    return float(2.0 * stats.norm.sf(abs(z)))


# --------------------------------------------------------------------------
# Treatment effect estimators
# --------------------------------------------------------------------------


@dataclass
class ATEEstimate:
    """One treatment effect estimate and everything needed to audit it."""

    method: str
    estimate: float
    se: float
    alpha: float = 0.05
    lam: float = float("nan")
    lambda_control: float = float("nan")
    lambda_treated: float = float("nan")
    n_treated: int = 0
    n_control: int = 0
    n_parameters: int = 0
    extra: Dict[str, float] = field(default_factory=dict)

    @property
    def ci_low(self) -> float:
        return self.estimate - z_quantile(1.0 - self.alpha / 2.0) * self.se

    @property
    def ci_high(self) -> float:
        return self.estimate + z_quantile(1.0 - self.alpha / 2.0) * self.se

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low

    @property
    def p_value(self) -> float:
        if self.se <= 0 or not np.isfinite(self.se):
            return 1.0
        z = self.estimate / self.se
        return float(2.0 * stats.norm.sf(abs(z)))

    @property
    def rejects(self) -> bool:
        return self.p_value < self.alpha

    def as_row(self) -> Dict[str, float]:
        return {
            "method": self.method,
            "estimate": self.estimate,
            "se": self.se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_width": self.ci_width,
            "p_value": self.p_value,
            "lam": self.lam,
            "lambda_control": self.lambda_control,
            "lambda_treated": self.lambda_treated,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "n_parameters": self.n_parameters,
        }


def _arm_stats(y: np.ndarray, t: np.ndarray, w: int) -> Tuple[int, float, float]:
    m = t == w
    n = int(m.sum())
    yy = y[m]
    if n < 2:
        return n, float(yy.mean()) if n else np.nan, np.nan
    return n, float(yy.mean()), float(np.var(yy, ddof=1))


def _ols_hc0(A: np.ndarray, y: np.ndarray, idx: int) -> Tuple[np.ndarray, float]:
    """Least squares coefficient ``idx`` with an HC0 heteroskedasticity-robust SE.

    HC0 is used because the variance of the outcome depends on the prognostic
    score by construction, so homoskedastic standard errors are not defensible.
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, k = A.shape
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    bread = np.linalg.pinv(A.T @ A)
    meat = A.T @ (A * (resid**2)[:, None])
    cov = bread @ meat @ bread
    var = float(cov[idx, idx])
    se = float(np.sqrt(var)) if var > 0 and np.isfinite(var) else np.nan
    return beta, se


def _cov_biased(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(((a - a.mean()) * (b - b.mean())).mean())


def ate_unadjusted(
    y: np.ndarray, t: np.ndarray, alpha: float = 0.05
) -> ATEEstimate:
    """Difference in means, the reference estimator.

    Uses the normal approximation with ``ddof=1`` arm variances for every method
    in this module, so that comparisons across methods are not confounded by a
    difference between t and normal quantiles. At the sample sizes in this
    project the two are numerically indistinguishable.
    """
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=int).ravel()
    n1, m1, v1 = _arm_stats(y, t, 1)
    n0, m0, v0 = _arm_stats(y, t, 0)
    se = float(np.sqrt(v1 / n1 + v0 / n0))
    return ATEEstimate(
        method="unadjusted", estimate=float(m1 - m0), se=se, alpha=alpha,
        n_treated=n1, n_control=n0, n_parameters=2,
    )


def ate_ancova(
    y: np.ndarray,
    t: np.ndarray,
    covariates: np.ndarray,
    alpha: float = 0.05,
    method: str = "ancova",
) -> ATEEstimate:
    """Linear adjustment for supplied covariates, treatment coefficient reported."""
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=int).ravel()
    Z = np.atleast_2d(np.asarray(covariates, dtype=float))
    if Z.shape[0] != y.size:
        Z = Z.T
    A = np.column_stack([np.ones(y.size), t.astype(float), Z])
    beta, se = _ols_hc0(A, y, idx=1)
    return ATEEstimate(
        method=method, estimate=float(beta[1]), se=se, alpha=alpha,
        n_treated=int((t == 1).sum()), n_control=int((t == 0).sum()),
        n_parameters=int(A.shape[1]),
    )


def ate_ppi_single_lambda(
    y: np.ndarray,
    t: np.ndarray,
    f: np.ndarray,
    alpha: float = 0.05,
    lam: Optional[float] = None,
    lam_clip: Optional[Tuple[float, float]] = None,
) -> ATEEstimate:
    """Tuned prediction-powered ATE with one shared tuning value.

    Derivation, which PLAN.md D7 records. With a single ``lam`` the unlabeled
    cohort's mean prediction cancels between arms:

        tau_hat(lam) = mean(Y|1) - mean(Y|0) - lam * (mean(f|1) - mean(f|0))

    so no external-cohort quantity survives in the estimator. Minimising

        Var = Var(Y1 - lam f)/n1 + Var(Y0 - lam f)/n0

    gives

        lam* = (Cov1(Y,f)/n1 + Cov0(Y,f)/n0) / (Var1(f)/n1 + Var0(f)/n0)

    which is a precision-weighted pooled within-arm slope.

    Convention, and why it matters: the covariance and variance in ``lam*`` here
    use biased (``ddof=0``) moments. With that convention, and because a
    regression on an intercept, a treatment dummy and one covariate reduces by
    Frisch-Waugh to the pooled within-arm slope, this estimator is *exactly* the
    ANCOVA treatment coefficient when the two arms have equal size, not merely
    asymptotically equivalent to it. The test suite asserts that identity to
    numerical precision. At unequal allocation the two differ at order 1/n, in
    the same direction as the allocation imbalance.
    """
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=int).ravel()
    f = np.asarray(f, dtype=float).ravel()
    m1, m0 = (t == 1), (t == 0)
    n1, n0 = int(m1.sum()), int(m0.sum())

    if lam is None:
        s1 = _cov_biased(y[m1], f[m1]) * n1
        s0 = _cov_biased(y[m0], f[m0]) * n0
        q1 = float(np.var(f[m1], ddof=0)) * n1
        q0 = float(np.var(f[m0], ddof=0)) * n0
        num = s1 / max(n1, 1) ** 2 + s0 / max(n0, 1) ** 2
        den = q1 / max(n1, 1) ** 2 + q0 / max(n0, 1) ** 2
        lam = float(num / den) if den > 0 else 0.0
        if lam_clip is not None:
            lam = float(np.clip(lam, lam_clip[0], lam_clip[1]))

    est = float(y[m1].mean() - y[m0].mean() - lam * (f[m1].mean() - f[m0].mean()))
    var = float(
        np.var(y[m1] - lam * f[m1], ddof=1) / max(n1, 1)
        + np.var(y[m0] - lam * f[m0], ddof=1) / max(n0, 1)
    )
    return ATEEstimate(
        method="ppi_single", estimate=est, se=float(np.sqrt(var)), alpha=alpha,
        lam=float(lam), n_treated=n1, n_control=n0, n_parameters=3,
    )


def ate_ppi_per_arm(
    y: np.ndarray,
    t: np.ndarray,
    f: np.ndarray,
    f_unlabeled: np.ndarray,
    alpha: float = 0.05,
    lam: Optional[Tuple[float, float]] = None,
    lam_clip: Optional[Tuple[float, float]] = (0.0, 1.0),
    method: str = "ppi_per_arm",
) -> ATEEstimate:
    """Per-arm PPI++ means with arm-specific tuning, then differenced.

        mu_hat_w = mean(Y|w) + lam_w * (mean(f_unlabeled) - mean(f|w))

    Unlike the shared-tuning form, this does retain a dependence on the external
    cohort through ``mean(f_unlabeled)``, so it is unbiased only when that cohort
    is drawn from the trial's covariate distribution. That assumption is stated
    on screen in the calculator and in the SAP appendix.

    Variance, with each ``lam_w`` treated as fixed:

        Var = Var(Y - lam1 f)/n1 + Var(Y - lam0 f)/n0
              + (lam1 - lam0)^2 * Var(f_unlabeled)/N
    """
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=int).ravel()
    f = np.asarray(f, dtype=float).ravel()
    f_unl = np.asarray(f_unlabeled, dtype=float).ravel()
    m1, m0 = (t == 1), (t == 0)
    n1, n0 = int(m1.sum()), int(m0.sum())

    if lam is None:
        lam1 = tuned_lambda_mean(y[m1], f[m1], f_unl, lam_clip=lam_clip)
        lam0 = tuned_lambda_mean(y[m0], f[m0], f_unl, lam_clip=lam_clip)
    else:
        lam1, lam0 = float(lam[0]), float(lam[1])

    mu1 = float(y[m1].mean() + lam1 * (f_unl.mean() - f[m1].mean()))
    mu0 = float(y[m0].mean() + lam0 * (f_unl.mean() - f[m0].mean()))
    est = mu1 - mu0

    var = float(
        np.var(y[m1] - lam1 * f[m1], ddof=1) / max(n1, 1)
        + np.var(y[m0] - lam0 * f[m0], ddof=1) / max(n0, 1)
    )
    if f_unl.size > 1:
        var += float((lam1 - lam0) ** 2 * np.var(f_unl, ddof=1) / f_unl.size)

    return ATEEstimate(
        method=method, estimate=est, se=float(np.sqrt(max(var, 0.0))), alpha=alpha,
        lam=float(lam1), lambda_control=float(lam0), lambda_treated=float(lam1),
        n_treated=n1, n_control=n0, n_parameters=4,
    )


def ate_lasso_adjusted(
    y: np.ndarray,
    t: np.ndarray,
    X: np.ndarray,
    alpha: float = 0.05,
    seed: int = 0,
    cv: int = 5,
) -> ATEEstimate:
    """Standard L1 lasso adjustment, used as a comparison baseline.

    The lasso is fitted to ``Y`` against the treatment indicator and all
    covariates at once, then the selected covariates plus treatment are refit by
    least squares with an HC0 SE. This is the naive post-selection procedure: the
    selection step uses the same data as the estimate, and no correction is made
    for that. It is included because the prompt asks for an L1 lasso comparison,
    not because it is recommended.
    """
    from sklearn.linear_model import LassoCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=int).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    design = np.column_stack([t.astype(float), X])
    pipe = make_pipeline(
        StandardScaler(with_mean=True),
        LassoCV(cv=cv, random_state=seed, max_iter=5000, n_alphas=50),
    )
    with np.errstate(all="ignore"):
        pipe.fit(design, y)
    coefs = pipe[-1].coef_
    selected = np.flatnonzero(np.abs(coefs[1:]) > 0)

    A = np.column_stack([np.ones(y.size), t.astype(float), X[:, selected]]) if selected.size else np.column_stack([np.ones(y.size), t.astype(float)])
    beta, se = _ols_hc0(A, y, idx=1)
    return ATEEstimate(
        method="lasso_l1", estimate=float(beta[1]), se=se, alpha=alpha,
        n_treated=int((t == 1).sum()), n_control=int((t == 0).sum()),
        n_parameters=int(A.shape[1]),
        extra={"n_selected": float(selected.size)},
    )


# --------------------------------------------------------------------------
# PROCOVA allocation and sample size
# --------------------------------------------------------------------------


def procova_variance(
    sigma0: float,
    sigma1: float,
    rho0: float,
    rho1: float,
    pi0: float,
) -> float:
    """Schuler et al. Theorem 4 adjusted-variance bound, as a per-patient constant.

    Returned ``v`` satisfies ``Var(tau_hat) = v / n_total``, so it can be fed
    directly to :func:`required_n_total`. ``rho_w`` is the correlation between the
    prognostic score and the outcome in arm ``w``.

        v = sigma0^2/pi0 + sigma1^2/pi1
            - pi0 * pi1 * (rho1 * sigma1 / pi1 + rho0 * sigma0 / pi0)^2

    In the symmetric case ``sigma0 = sigma1 = sigma``, ``rho0 = rho1 = rho``,
    ``pi0 = 1/2`` this returns ``4 sigma^2 (1 - rho^2)``, which is the ``1 - R^2``
    factor the EMA qualification opinion states for PROCOVA. The test suite
    asserts that identity.
    """
    pi1 = 1.0 - pi0
    if pi0 <= 0 or pi1 <= 0:
        return float("inf")
    base = sigma0**2 / pi0 + sigma1**2 / pi1
    gain = pi0 * pi1 * (rho1 * sigma1 / pi1 + rho0 * sigma0 / pi0) ** 2
    return float(max(base - gain, 0.0))


def unadjusted_min_variance(sigma0: float, sigma1: float) -> float:
    """Minimum of ``sigma0^2/pi0 + sigma1^2/pi1`` over the allocation.

    The closed form is ``(sigma0 + sigma1)^2`` at ``pi0 = sigma0/(sigma0+sigma1)``.
    Used as the no-adjustment reference and to check the numeric optimiser.
    """
    return float((sigma0 + sigma1) ** 2)


@dataclass
class AllocationResult:
    """Optimal allocation and the variance achieved at it."""

    pi0: float
    pi1: float
    variance: float
    unadjusted_variance: float
    unadjusted_pi0: float
    optimization_success: bool

    @property
    def variance_ratio(self) -> float:
        return float(self.variance / self.unadjusted_variance) if self.unadjusted_variance > 0 else float("nan")

    @property
    def variance_reduction(self) -> float:
        return float(1.0 - self.variance_ratio)

    def summary(self) -> Dict[str, float]:
        return {
            "pi0": self.pi0,
            "pi1": self.pi1,
            "variance": self.variance,
            "unadjusted_variance": self.unadjusted_variance,
            "variance_ratio": self.variance_ratio,
            "variance_reduction": self.variance_reduction,
            "unadjusted_pi0": self.unadjusted_pi0,
        }


def optimal_allocation(
    sigma0: float,
    sigma1: float,
    rho0: float,
    rho1: float,
) -> AllocationResult:
    """Minimise the adjusted variance bound over the control allocation.

    This is the ``PROCOVA optimal control-to-treated allocation`` the prompt asks
    for. It optimises the bound as written. As Schuler et al. note, when arm
    variances or score correlations differ, the gain does not reduce to a fixed
    factor on the control arm alone, and the treatment-arm score correlation is
    not identified by control-arm data alone. The calculator therefore takes both
    correlations as explicit user inputs rather than assuming a value.
    """
    def objective(pi0: float) -> float:
        return procova_variance(sigma0, sigma1, rho0, rho1, float(pi0))

    res = optimize.minimize_scalar(
        objective, bounds=(1e-6, 1.0 - 1e-6), method="bounded",
        options={"xatol": 1e-12},
    )
    pi0 = float(res.x)
    unadj_pi0 = float(sigma0 / (sigma0 + sigma1)) if (sigma0 + sigma1) > 0 else 0.5
    return AllocationResult(
        pi0=pi0,
        pi1=1.0 - pi0,
        variance=float(res.fun),
        unadjusted_variance=unadjusted_min_variance(sigma0, sigma1),
        unadjusted_pi0=unadj_pi0,
        optimization_success=bool(res.success),
    )


def required_n_total(
    variance_constant: float,
    effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Total patients needed for a two-sided normal test of ``effect``.

    ``n = (z_{1-alpha/2} + z_power)^2 * v / effect^2`` with ``v`` the per-patient
    variance constant from :func:`procova_variance`. Sanity check in the test
    suite: for an unadjusted balanced design ``v = 4 sigma^2`` and this returns
    the textbook ``4 (z + z)^2 sigma^2 / effect^2``.
    """
    if effect == 0 or not np.isfinite(effect):
        return float("inf")
    z_a = z_quantile(1.0 - alpha / 2.0)
    z_b = z_quantile(power)
    return float((z_a + z_b) ** 2 * variance_constant / effect**2)


def sample_size_reduction(variance_unadjusted: float, variance_adjusted: float) -> float:
    """Fractional reduction in required sample size, by the variance ratio.

    Because required ``n`` is proportional to the variance constant, this equals
    the fractional variance reduction. Reported only where measured.
    """
    if variance_unadjusted <= 0:
        return float("nan")
    return float(1.0 - variance_adjusted / variance_unadjusted)


def control_arm_size(n_total: float, pi0: float) -> float:
    """Control arm size implied by a total size and an allocation."""
    return float(np.ceil(pi0 * n_total))

"""Tests for the PPI++ estimator, the ATE estimators, and PROCOVA allocation.

The central tests here are the cross-check against the PPI authors' own reference
implementation, the exact-reduction checks (a powerless score must give back the
plain randomized-trial estimator), and the corruption invariances.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from ppict.ppi import (
    ate_ancova,
    ate_ppi_per_arm,
    ate_ppi_single_lambda,
    ate_unadjusted,
    control_arm_size,
    optimal_allocation,
    ppi_mean_ci,
    ppi_mean_pointestimate,
    ppi_mean_pvalue,
    ppi_mean_se,
    procova_variance,
    required_n_total,
    sample_size_reduction,
    tuned_lambda_mean,
    unadjusted_min_variance,
)
from ppict.simulate import TrialConfig, corrupt_score, make_trial_data

ppi_py = pytest.importorskip("ppi_py")


def _mean_data(seed: int = 0, n: int = 120, N: int = 400, rho: float = 0.7):
    rng = np.random.default_rng(seed)
    f_unl = rng.standard_normal(N)
    f_lab = rng.standard_normal(n)
    y = rho * f_lab + np.sqrt(1 - rho**2) * rng.standard_normal(n) + 2.0
    return y, f_lab, f_unl


# ---------------------------------------------------------------- mean


def test_mean_point_estimate_matches_reference_implementation():
    y, f_lab, f_unl = _mean_data(seed=1)
    for lam in (0.0, 0.5, 1.0, None):
        mine = ppi_mean_pointestimate(y, f_lab, f_unl, lam=lam)
        theirs = ppi_py.ppi_mean_pointestimate(y, f_lab, f_unl, lam=lam)
        assert mine == pytest.approx(float(theirs), rel=1e-10, abs=1e-12), f"lam={lam}"


def test_mean_ci_matches_reference_implementation():
    y, f_lab, f_unl = _mean_data(seed=2, rho=0.6)
    for lam in (0.0, 0.3, 1.0, None):
        lo, hi, used = ppi_mean_ci(y, f_lab, f_unl, lam=lam, alpha=0.05)
        lo_r, hi_r = ppi_py.ppi_mean_ci(y, f_lab, f_unl, alpha=0.05, lam=lam)
        assert lo == pytest.approx(float(lo_r), rel=1e-9, abs=1e-12), f"lam={lam}"
        assert hi == pytest.approx(float(hi_r), rel=1e-9, abs=1e-12), f"lam={lam}"


def test_tuned_lambda_matches_reference_tuning_rule():
    """The plug-in tuning value must equal the reference _calc_lam_glm for the mean."""
    from ppi_py.ppi import _calc_lam_glm

    y, f_lab, f_unl = _mean_data(seed=3, rho=0.5)
    point = ppi_py.ppi_mean_pointestimate(y, f_lab, f_unl, lam=1)
    lam_ref = _calc_lam_glm(
        y - point, f_lab - point, f_unl - point, np.eye(1),
        coord=None, clip=True, optim_mode="overall",
    )
    mine = tuned_lambda_mean(y, f_lab, f_unl)
    assert mine == pytest.approx(float(lam_ref), rel=1e-9, abs=1e-12)


def test_unclipped_tuning_can_be_negative_for_an_anticorrelated_score():
    rng = np.random.default_rng(4)
    f_unl = rng.standard_normal(300)
    f_lab = rng.standard_normal(150)
    y = -0.8 * f_lab + rng.standard_normal(150)
    clipped = tuned_lambda_mean(y, f_lab, f_unl, lam_clip=(0.0, 1.0))
    free = tuned_lambda_mean(y, f_lab, f_unl, lam_clip=None)
    assert clipped == 0.0
    assert free < 0.0


def test_lambda_zero_reduces_to_the_classical_mean_and_interval():
    y, f_lab, f_unl = _mean_data(seed=5)
    point = ppi_mean_pointestimate(y, f_lab, f_unl, lam=0.0)
    assert point == pytest.approx(float(y.mean()), rel=1e-15)

    lo, hi, _ = ppi_mean_ci(y, f_lab, f_unl, lam=0.0, alpha=0.05)
    se = float(np.std(y, ddof=0)) / np.sqrt(y.size)
    z = stats.norm.ppf(0.975)
    assert lo == pytest.approx(y.mean() - z * se, rel=1e-12)
    assert hi == pytest.approx(y.mean() + z * se, rel=1e-12)


def test_a_score_with_no_signal_carries_no_tuning_and_no_change():
    """With zero covariance between outcome and score the estimate is unchanged.

    The outcome is constant, so the centered outcome and therefore the covariance
    are exactly zero. The tuning value must then be exactly zero and the estimate
    must be exactly the sample mean.
    """
    y = np.ones(50)
    f_lab = np.linspace(-1, 1, 50)
    f_unl = np.linspace(-2, 2, 80)
    assert tuned_lambda_mean(y, f_lab, f_unl) == 0.0
    assert ppi_mean_pointestimate(y, f_lab, f_unl) == pytest.approx(1.0)


def test_ppi_se_reduces_to_the_plain_standard_error_at_zero_tuning():
    y, f_lab, f_unl = _mean_data(seed=6)
    mine = ppi_mean_se(y, f_lab, f_unl, lam=0.0)
    assert mine == pytest.approx(float(np.std(y, ddof=0)) / np.sqrt(y.size), rel=1e-12)


def test_variance_reduction_matches_the_paper_formula_in_expectation():
    """Measured variance reduction should track ``rho^2 * N/(n+N)``.

    This checks the corrected formula recorded in PLAN.md D2. The plug-in standard
    error is used rather than the empirical spread of the point estimates, because
    the plug-in is the quantity the paper's expression describes and it converges
    across replicates much faster. Since the outcome is scaled to unit variance,
    the unadjusted value of ``n * Var`` is exactly 1, so the reduction is directly
    ``1 - n * se^2``.
    """
    n, N, rho, reps = 300, 300, 0.8, 300
    rng = np.random.default_rng(7)
    scaled = np.empty(reps)
    for i in range(reps):
        f_unl = rng.standard_normal(N)
        f_lab = rng.standard_normal(n)
        y = rho * f_lab + np.sqrt(1 - rho**2) * rng.standard_normal(n)
        lam = tuned_lambda_mean(y, f_lab, f_unl)
        scaled[i] = n * ppi_mean_se(y, f_lab, f_unl, lam) ** 2

    measured_reduction = 1.0 - float(scaled.mean())
    expected_reduction = rho**2 * N / (n + N)
    assert measured_reduction == pytest.approx(expected_reduction, abs=0.02), (
        f"measured {measured_reduction:.4f} against expected {expected_reduction:.4f}"
    )


def test_variance_reduction_approaches_rho_squared_as_the_unlabeled_set_grows():
    """The limit N -> infinity is rho^2, which is the EMA 1 - R^2 factor."""
    n, rho = 100, 0.6
    rng = np.random.default_rng(21)
    reductions = []
    for N in (100, 1000, 100000):
        scaled = []
        for _ in range(40):
            f_unl = rng.standard_normal(N)
            f_lab = rng.standard_normal(n)
            y = rho * f_lab + np.sqrt(1 - rho**2) * rng.standard_normal(n)
            lam = tuned_lambda_mean(y, f_lab, f_unl)
            scaled.append(n * ppi_mean_se(y, f_lab, f_unl, lam) ** 2)
        reductions.append(1.0 - float(np.mean(scaled)))
    assert reductions[0] < reductions[1] < reductions[2]
    assert reductions[2] == pytest.approx(rho**2, abs=0.05)


def test_mean_pvalue_is_two_sided_and_sane():
    y, f_lab, f_unl = _mean_data(seed=8)
    p = ppi_mean_pvalue(y, f_lab, f_unl, null=float(y.mean()))
    assert 0.0 <= p <= 1.0


# ---------------------------------------------------------------- ATE


def _trial(seed: int = 0, **kw):
    base = dict(name="t", seed=seed, p=30, n_groups=6, n_active_groups=2,
                n_hist=250, n_unlabeled=250, n_treat=120, n_control=120,
                select_n_perm=10, signal_r2=0.35)
    base.update(kw)
    return make_trial_data(TrialConfig(**base))


def test_unadjusted_matches_a_manual_difference_in_means():
    data = _trial(seed=1)
    est = ate_unadjusted(data.y, data.t)
    m1 = data.y[data.t == 1].mean()
    m0 = data.y[data.t == 0].mean()
    assert est.estimate == pytest.approx(float(m1 - m0), rel=1e-12)
    se = np.sqrt(
        np.var(data.y[data.t == 1], ddof=1) / (data.t == 1).sum()
        + np.var(data.y[data.t == 0], ddof=1) / (data.t == 0).sum()
    )
    assert est.se == pytest.approx(float(se), rel=1e-12)


def test_both_estimators_reduce_exactly_to_the_unadjusted_estimator_at_zero_tuning():
    """The prompt's requirement: no prognostic signal means the plain RCT analysis."""
    data = _trial(seed=2)
    base = ate_unadjusted(data.y, data.t)

    single = ate_ppi_single_lambda(data.y, data.t, data.f, lam=0.0)
    assert single.estimate == pytest.approx(base.estimate, rel=1e-14)
    assert single.se == pytest.approx(base.se, rel=1e-14)

    per_arm = ate_ppi_per_arm(data.y, data.t, data.f, data.f_unlabeled, lam=(0.0, 0.0))
    assert per_arm.estimate == pytest.approx(base.estimate, rel=1e-14)
    assert per_arm.se == pytest.approx(base.se, rel=1e-14)


def test_single_tuning_estimator_is_exactly_ancova_when_arms_are_balanced():
    """Poulet et al. prove asymptotic equivalence; at balance this convention is exact."""
    data = _trial(seed=3, n_treat=130, n_control=130)
    est = ate_ppi_single_lambda(data.y, data.t, data.f)
    A = np.column_stack([np.ones(data.y.size), data.t.astype(float), data.f])
    beta = np.linalg.lstsq(A, data.y, rcond=None)[0]
    assert est.estimate == pytest.approx(float(beta[1]), rel=1e-9)
    # The reported standard error should agree closely with a robust ANCOVA error.
    anc = ate_ancova(data.y, data.t, data.f)
    assert est.se == pytest.approx(anc.se, rel=0.10)


def test_constant_bias_in_the_score_changes_nothing():
    """A shifted score must leave both the tuning value and the estimate untouched."""
    data = _trial(seed=4)
    base = ate_ppi_single_lambda(data.y, data.t, data.f)
    shifted = corrupt_score(data.f, "bias", 4.0, np.random.default_rng(0))
    after = ate_ppi_single_lambda(data.y, data.t, shifted)
    assert after.lam == pytest.approx(base.lam, rel=1e-12, abs=1e-15)
    assert after.estimate == pytest.approx(base.estimate, rel=1e-10, abs=1e-12)


def test_sign_flip_in_the_score_changes_nothing():
    """The tuning value absorbs a sign reversal, so the estimate is unchanged."""
    data = _trial(seed=5)
    base = ate_ppi_single_lambda(data.y, data.t, data.f)
    flipped = corrupt_score(data.f, "flip", 1.0, np.random.default_rng(0))
    after = ate_ppi_single_lambda(data.y, data.t, flipped)
    assert after.lam == pytest.approx(-base.lam, rel=1e-12, abs=1e-15)
    assert after.estimate == pytest.approx(base.estimate, rel=1e-10, abs=1e-12)


def test_shuffled_score_loses_its_tuning_weight():
    """A score carrying no information must be given little or no weight."""
    data = _trial(seed=6)
    good = ate_ppi_single_lambda(data.y, data.t, data.f)
    shuffled = corrupt_score(data.f, "shuffle", 0.0, np.random.default_rng(1))
    bad = ate_ppi_single_lambda(data.y, data.t, shuffled)
    assert abs(good.lam) > abs(bad.lam)
    assert abs(bad.lam) < 0.2


def test_noisy_score_keeps_the_estimate_close_to_unadjusted():
    """Destroying most of the score's signal must not move the point estimate far."""
    data = _trial(seed=7)
    noisy = corrupt_score(data.f, "noise", 3.0, np.random.default_rng(2))
    est = ate_ppi_single_lambda(data.y, data.t, noisy)
    plain = ate_unadjusted(data.y, data.t)
    assert abs(est.estimate - plain.estimate) <= 2.5 * plain.se


def test_external_cohort_enters_the_per_arm_estimator_and_not_the_single_tuning_one():
    """How the unlabeled cohort does and does not matter, stated as two checks.

    First, shifting only the unlabeled cohort's predictions moves the per-arm
    estimate, because that cohort's mean prediction enters each arm directly. That
    is exactly why the per-arm form needs the cohort to be exchangeable with the
    trial population, and why its assumption is spelled out in the calculator.

    Second, shifting the entire score mapping consistently leaves both estimators
    unchanged: the tuning value is shift invariant, and inside each arm's rectifier
    the shift cancels. PLAN.md D7 records this.
    """
    data = _trial(seed=8)

    per_arm_a = ate_ppi_per_arm(data.y, data.t, data.f, data.f_unlabeled)
    per_arm_b = ate_ppi_per_arm(data.y, data.t, data.f, data.f_unlabeled + 10.0)
    assert abs(per_arm_b.estimate - per_arm_a.estimate) > 1e-6, (
        "the per-arm estimator should respond to the unlabeled cohort"
    )

    single = ate_ppi_single_lambda(data.y, data.t, data.f)
    shifted = corrupt_score(data.f, "bias", 5.0, np.random.default_rng(0))
    single_shifted = ate_ppi_single_lambda(data.y, data.t, shifted)
    assert single_shifted.estimate == pytest.approx(single.estimate, rel=1e-10, abs=1e-12)

    # The structural version of the same claim: the single-tuning estimator has no
    # unlabeled input to depend on.
    import inspect

    assert "f_unlabeled" not in inspect.signature(ate_ppi_single_lambda).parameters


def test_a_good_score_narrows_the_interval():
    data = _trial(seed=9, signal_r2=0.5)
    plain = ate_unadjusted(data.y, data.t)
    adj = ate_ppi_single_lambda(data.y, data.t, data.f)
    assert adj.ci_width < plain.ci_width


# ---------------------------------------------------------------- PROCOVA


def test_symmetric_case_reproduces_the_one_minus_r_squared_rule():
    """The EMA qualification opinion's factor, recovered from the variance bound."""
    for s in (0.5, 1.0, 3.0):
        for r in (0.0, 0.3, 0.6, 0.9):
            v = procova_variance(s, s, r, r, 0.5)
            assert v == pytest.approx(4 * s**2 * (1 - r**2), rel=1e-12)
            ratio = v / (4 * s**2)
            assert ratio == pytest.approx(1 - r**2, rel=1e-12)


def test_no_correlation_gives_no_reduction():
    assert procova_variance(1.0, 1.0, 0.0, 0.0, 0.5) == pytest.approx(4.0)


def test_unadjusted_minimum_variance_closed_form():
    for s0, s1 in ((1.0, 1.0), (1.0, 2.0), (0.7, 1.9)):
        assert unadjusted_min_variance(s0, s1) == pytest.approx((s0 + s1) ** 2, rel=1e-12)
        pi0 = s0 / (s0 + s1)
        base = procova_variance(s0, s1, 0.0, 0.0, pi0)
        assert base == pytest.approx((s0 + s1) ** 2, rel=1e-12)


def test_optimal_allocation_is_balanced_when_arms_are_symmetric():
    res = optimal_allocation(1.0, 1.0, 0.4, 0.4)
    assert res.pi0 == pytest.approx(0.5, abs=1e-4)
    assert res.variance_ratio == pytest.approx(1 - 0.4**2, rel=1e-6)
    assert res.variance_reduction == pytest.approx(0.16, rel=1e-5)
    assert res.unadjusted_pi0 == pytest.approx(0.5)


def test_optimal_allocation_favours_the_noisier_arm_without_adjustment():
    res = optimal_allocation(2.0, 1.0, 0.0, 0.0)
    assert res.pi0 == pytest.approx(2.0 / 3.0, abs=1e-3)
    assert res.variance_ratio == pytest.approx(1.0, rel=1e-6)


def test_allocation_never_returns_a_worse_variance_than_unadjusted_minimum():
    for s0, s1, r0, r1 in ((1, 1, .3, .3), (1.5, 1.0, .5, .2), (1.0, 1.0, .8, .1), (0.5, 2.0, .4, .4)):
        res = optimal_allocation(s0, s1, r0, r1)
        assert res.variance <= res.unadjusted_variance + 1e-9


def test_required_n_total_matches_the_textbook_two_sample_formula():
    sigma, effect = 1.3, 0.4
    z_a = stats.norm.ppf(0.975)
    z_b = stats.norm.ppf(0.80)
    textbook = 4 * (z_a + z_b) ** 2 * sigma**2 / effect**2
    assert required_n_total(4 * sigma**2, effect, 0.05, 0.80) == pytest.approx(textbook, rel=1e-12)


def test_required_n_grows_with_a_smaller_effect_and_with_power():
    v = 4.0
    assert required_n_total(v, 0.2) > required_n_total(v, 0.5)
    assert required_n_total(v, 0.4, power=0.90) > required_n_total(v, 0.4, power=0.80)
    assert required_n_total(v, 0.0) == float("inf")


def test_sample_size_reduction_is_the_variance_ratio():
    assert sample_size_reduction(4.0, 3.0) == pytest.approx(0.25)
    assert control_arm_size(100.0, 0.5) == pytest.approx(50.0)


def test_procova_variance_is_infinite_on_a_degenerate_allocation():
    assert procova_variance(1.0, 1.0, 0.3, 0.3, 0.0) == float("inf")
    assert procova_variance(1.0, 1.0, 0.3, 0.3, 1.0) == float("inf")

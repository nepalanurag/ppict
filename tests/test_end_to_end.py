"""End-to-end tests on simulated trial data.

These exercise the whole path: generate a trial, build a prognostic score with
permutation-assisted group lasso, run every estimator, and summarize. The
Type I error question is answered properly in ``validation/``; what is checked
here is that the machinery is connected and behaves in the direction it claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from ppict.cli import preset_configs
from ppict.simulate import (
    TrialConfig,
    make_trial_data,
    run_monte_carlo,
    run_replicate,
    summarize,
)

SMALL = dict(
    p=24, n_groups=6, n_active_groups=2, n_hist=150, n_unlabeled=150,
    n_treat=70, n_control=70, select_n_perm=8, signal_r2=0.35,
)


def _cfg(name: str = "e2e", **kw) -> TrialConfig:
    # The dict spread keeps seed overridable: passing seed=0 positionally as well
    # would raise a duplicate keyword error.
    return TrialConfig(**{"name": name, "seed": 0, **SMALL, **kw})


def test_one_replicate_produces_a_row_for_every_method():
    rows = run_replicate(_cfg())
    methods = {r["method"] for r in rows}
    assert methods == {
        "unadjusted", "ancova", "ppi_single", "ppi_per_arm",
        "ppi_lambda1", "lasso_l1", "oracle_ancova",
    }
    for r in rows:
        assert np.isfinite(r["estimate"])
        assert np.isfinite(r["se"]) and r["se"] > 0
        assert r["ci_low"] < r["estimate"] < r["ci_high"]
        assert r["ci_width"] > 0


def test_the_same_seed_reproduces_the_same_numbers():
    a = run_replicate(_cfg(seed=12345))
    b = run_replicate(_cfg(seed=12345))
    for ra, rb in zip(a, b):
        assert ra["method"] == rb["method"]
        assert ra["estimate"] == pytest.approx(rb["estimate"], rel=0, abs=0)
        assert ra["se"] == pytest.approx(rb["se"], rel=0, abs=0)


def test_different_seeds_produce_different_data():
    a = make_trial_data(_cfg(seed=1))
    b = make_trial_data(_cfg(seed=2))
    assert not np.allclose(a.y, b.y)
    assert not np.allclose(a.X_trial, b.X_trial)


def test_generated_trial_matches_its_declared_signal_strength():
    cfg = _cfg(signal_r2=0.4, n_treat=2000, n_control=2000, n_hist=100)
    data = make_trial_data(cfg)
    r2 = 1.0 - np.var(data.y0 - (data.f_oracle), ddof=1) / np.var(data.y0, ddof=1)
    assert r2 == pytest.approx(0.4, abs=0.05)
    assert data.active_groups.size == 2


def test_treatment_effect_is_actually_present_when_configured():
    data = make_trial_data(_cfg(true_effect=0.75, n_treat=1500, n_control=1500))
    est = data.y[data.t == 1].mean() - data.y[data.t == 0].mean()
    assert est == pytest.approx(0.75, abs=0.15)


def test_estimators_are_unbiased_for_the_effect_they_declare():
    """Across replicates the mean estimate should sit at the true effect."""
    cfg = _cfg(name="bias_check", true_effect=0.5, signal_r2=0.4)
    df = run_monte_carlo([cfg], n_rep=60, seed_base=500, verbose=False)
    summ = summarize(df)
    for _, row in summ.iterrows():
        assert abs(row["bias"]) < 4.0 * row["empirical_sd"] / np.sqrt(row["n_rep"]) + 0.02, row["method"]


def test_the_score_narrows_intervals_when_it_carries_signal():
    cfg = _cfg(name="width", signal_r2=0.55)
    df = run_monte_carlo([cfg], n_rep=40, seed_base=900, verbose=False)
    summ = summarize(df)
    plain = summ[summ["method"] == "unadjusted"].iloc[0]
    good = summ[summ["method"] == "ppi_single"].iloc[0]
    oracle = summ[summ["method"] == "oracle_ancova"].iloc[0]
    assert good["mean_ci_width"] < plain["mean_ci_width"]
    assert oracle["mean_ci_width"] < good["mean_ci_width"]


def test_a_destroyed_score_does_not_inflate_the_interval_beyond_the_unadjusted_one():
    """The prompt's robustness requirement, checked coarsely here.

    A shuffled score carries no information, so the estimator should fall back to
    roughly the unadjusted analysis rather than losing power. The full
    measurement, with Monte Carlo error bars, lives in validation/.
    """
    cfg = _cfg(name="shuffled", corruption="shuffle")
    df = run_monte_carlo([cfg], n_rep=40, seed_base=1300, verbose=False)
    summ = summarize(df)
    plain = summ[summ["method"] == "unadjusted"].iloc[0]
    shuffled = summ[summ["method"] == "ppi_single"].iloc[0]
    assert shuffled["mean_ci_width"] <= 1.25 * plain["mean_ci_width"]
    assert shuffled["reject_rate"] <= 0.25


def test_summary_has_one_row_per_config_and_method():
    configs = [_cfg(name="a"), _cfg(name="b", corruption="noise", corruption_param=2.0)]
    df = run_monte_carlo(configs, n_rep=5, seed_base=77, verbose=False)
    summ = summarize(df)
    assert len(summ) == 2 * 7
    for col in ("reject_rate", "coverage", "mean_ci_width", "mean_lam", "mc_se_reject_rate"):
        assert col in summ.columns
    assert (summ["n_rep"] == 5).all()


def test_null_preset_configs_are_wired_up():
    configs = preset_configs("null", base_seed=0)
    names = {c.name for c in configs}
    assert "good_score" in names
    assert "corrupt_shuffle" in names
    assert all(c.true_effect == 0.0 for c in configs), "the null preset must have a zero effect"

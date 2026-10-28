"""Tests for the data-in, estimate-and-interval-out surface."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ppict.analysis import (
    DatasetError,
    PlanInputs,
    TrialDataset,
    analyze_trial,
    load_trial_dataset,
    plan_trial,
    pooled_sd,
    squared_correlation,
)
from ppict.ppi import required_n_total


def _trial_csv(path: Path, n_per_arm: int = 150, seed: int = 3, external: int = 400) -> Path:
    """A small balanced trial with a real prognostic score, written to disk."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_arm
    t = np.repeat([0, 1], n_per_arm)
    f = rng.normal(size=n)
    y = 0.4 * t + 0.8 * f + rng.normal(size=n)
    f_ext = rng.normal(size=external)

    lines = ["# synthetic fixture", "cohort,y,t,f"]
    for i in range(n):
        lines.append(f"trial,{y[i]:.6f},{int(t[i])},{f[i]:.6f}")
    for value in f_ext:
        lines.append(f"external,,,{value:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_splits_cohorts_and_records_provenance(tmp_path: Path) -> None:
    path = _trial_csv(tmp_path / "trial.csv")
    ds = load_trial_dataset(path)
    assert ds.n_trial == 300
    assert ds.n_treated == 150
    assert ds.n_control == 150
    assert ds.n_external == 400
    assert ds.n_dropped == 0
    assert len(ds.sha256) == 64
    assert ds.notes[0] == "synthetic fixture"


def test_load_reports_missing_column_with_what_it_found(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("y,t\n1,0\n2,1\n", encoding="utf-8")
    with pytest.raises(DatasetError) as err:
        load_trial_dataset(path)
    message = str(err.value)
    assert "f" in message
    assert "Columns found: y, t" in message


def test_load_accepts_common_column_names(tmp_path: Path) -> None:
    path = tmp_path / "aliases.csv"
    rows = ["outcome,treatment,score"]
    rng = np.random.default_rng(0)
    for i in range(60):
        rows.append(f"{rng.normal():.4f},{i % 2},{rng.normal():.4f}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    ds = load_trial_dataset(path)
    assert ds.n_trial == 60
    assert ds.f_external is None


def test_load_rejects_values_other_than_zero_one(tmp_path: Path) -> None:
    path = tmp_path / "bad_t.csv"
    rows = ["y,t,f"] + [f"{i * 0.1:.3f},2,0.5" for i in range(40)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="must be 0 or 1"):
        load_trial_dataset(path)


def test_load_requires_enough_per_arm(tmp_path: Path) -> None:
    path = tmp_path / "small.csv"
    # 14 treated and 6 control: enough rows in total, too few in one arm.
    arms = [1] * 14 + [0] * 6
    rows = ["y,t,f"] + [f"{i * 0.1:.3f},{arm},0.1" for i, arm in enumerate(arms)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="at least 8"):
        load_trial_dataset(path)


def test_load_counts_incomplete_trial_rows(tmp_path: Path) -> None:
    path = tmp_path / "gaps.csv"
    rows = ["y,t,f"]
    for i in range(40):
        rows.append(f"{i * 0.1:.3f},{i % 2},0.2")
    rows.append(",1,0.2")
    rows.append("0.5,,")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    ds = load_trial_dataset(path)
    assert ds.n_dropped == 2
    assert ds.n_trial == 40


def test_analyze_is_exact_ancova_at_equal_allocation(tmp_path: Path) -> None:
    ds = load_trial_dataset(_trial_csv(tmp_path / "trial.csv"))
    result = analyze_trial(ds)
    assert set(result.estimates) == {"unadjusted", "ancova", "ppi_single", "ppi_per_arm"}
    assert result.primary.method == "ppi_single"
    # The identity the package claims: at equal arm sizes the shared-tuning
    # estimator is the least squares treatment coefficient.
    assert abs(result.primary.estimate - result.estimates["ancova"].estimate) < 1e-10
    assert result.primary_width_ratio < 1.0
    assert result.estimates["ancova"].ci_width > 0


def test_analyze_without_external_cohort_says_why(tmp_path: Path) -> None:
    path = tmp_path / "no_ext.csv"
    rng = np.random.default_rng(1)
    rows = ["y,t,f"]
    for i in range(80):
        rows.append(f"{rng.normal():.4f},{i % 2},{rng.normal():.4f}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    result = analyze_trial(load_trial_dataset(path))
    payload = result.as_dict()
    assert "ppi_per_arm" not in result.estimates
    reasons = {row["method"]: row["reason"] for row in payload["unavailable"]}
    assert "ppi_per_arm" in reasons
    assert "external" in reasons["ppi_per_arm"]
    assert payload["primary_width_ratio"] is not None


def test_json_payload_has_no_nan(tmp_path: Path) -> None:
    import json

    ds = load_trial_dataset(_trial_csv(tmp_path / "trial.csv"))
    payload = analyze_trial(ds).as_dict()
    # allow_nan=False raises if any float is nan or infinite.
    json.dumps(payload, allow_nan=False)
    unadjusted = next(m for m in payload["methods"] if m["method"] == "unadjusted")
    assert unadjusted["lam"] is None
    assert unadjusted["lambda_control"] is None


def test_squared_correlation_ignores_offset() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=200)
    b = 2.0 * a + rng.normal(size=200)
    assert squared_correlation(a, b) == pytest.approx(squared_correlation(a, b + 10.0), rel=1e-12)
    assert squared_correlation(a, np.ones(200)) != squared_correlation(a, np.ones(200))  # nan


def test_pooled_sd_matches_direct_computation() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 4.0, 6.0, 8.0])
    dof = a.size + b.size - 2
    expected = math.sqrt(
        ((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)) / dof
    )
    assert pooled_sd(a, b) == pytest.approx(expected)


def test_plan_uses_the_estimator_variance_model() -> None:
    inputs = PlanInputs(
        cost_per_patient=5000.0,
        power=0.80,
        r2=0.40,
        n_total=600,
        effect=0.30,
        outcome_sd=1.1,
        patients_per_month=20.0,
        alpha=0.05,
    )
    plan = plan_trial(inputs)
    sigma, rho = 1.1, math.sqrt(0.40)
    v0 = 4 * sigma**2
    v1 = 4 * sigma**2 * (1 - rho**2)
    assert plan.variance_unadjusted == pytest.approx(v0)
    assert plan.variance_adjusted == pytest.approx(v1)

    n0 = math.ceil(required_n_total(v0, 0.30, alpha=0.05, power=0.80))
    n1 = math.ceil(required_n_total(v1, 0.30, alpha=0.05, power=0.80))
    assert plan.n_required_unadjusted == pytest.approx(n0)
    assert plan.n_required_adjusted == pytest.approx(n1)
    assert plan.patients_saved == pytest.approx(n0 - n1)
    assert plan.cost_saved == pytest.approx((n0 - n1) * 5000.0)
    assert plan.months_saved == pytest.approx((n0 - n1) / 20.0)

    # The two headline percentages are different quantities and must not be equal.
    assert plan.width_ratio == pytest.approx(math.sqrt(1 - 0.40))
    assert plan.ci_narrowing == pytest.approx(1 - math.sqrt(1 - 0.40))
    assert plan.ci_narrowing < 0.40


def test_plan_at_zero_r2_saves_nothing() -> None:
    plan = plan_trial(PlanInputs(r2=0.0, effect=0.3, outcome_sd=1.0, n_total=400))
    assert plan.width_ratio == pytest.approx(1.0)
    assert plan.patients_saved == pytest.approx(0.0)
    assert plan.cost_saved == pytest.approx(0.0)
    assert any("identical" in note for note in plan.notes)


def test_plan_flags_an_implausibly_small_adjusted_requirement() -> None:
    # A near-perfect score with a large effect reduces the model to one patient.
    plan = plan_trial(PlanInputs(r2=0.98, effect=0.5, outcome_sd=0.6, n_total=120))
    assert plan.n_required_adjusted < 2
    assert any("not a usable design" in note for note in plan.notes)


def test_plan_requires_more_patients_at_higher_power() -> None:
    base = dict(r2=0.3, effect=0.3, outcome_sd=1.0, n_total=500)
    low = plan_trial(PlanInputs(power=0.80, **base))
    high = plan_trial(PlanInputs(power=0.90, **base))
    assert high.n_required_adjusted > low.n_required_adjusted


@pytest.mark.parametrize(
    "kwargs",
    [
        {"r2": 1.0},
        {"r2": -0.1},
        {"power": 1.0},
        {"effect": 0.0},
        {"outcome_sd": 0.0},
        {"cost_per_patient": -1.0},
        {"patients_per_month": 0.0},
        {"n_total": 1},
    ],
)
def test_plan_rejects_impossible_inputs(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        PlanInputs(**kwargs)


def test_dataset_describe_is_json_safe(tmp_path: Path) -> None:
    ds = load_trial_dataset(_trial_csv(tmp_path / "trial.csv"))
    described = ds.describe()
    assert described["n_external"] == 400
    assert described["notes"] == ["synthetic fixture"]

"""Tests for the JSON report contract shared by the CLI and the calculator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ppict.analysis import analyze_trial, plan_trial, PlanInputs
from ppict.report import CONTRACT_VERSION, build_report


def _fixture_csv(path: Path, effect: float = 0.4, seed: int = 11, external: int = 200) -> Path:
    rng = np.random.default_rng(seed)
    n_per_arm = 120
    n = 2 * n_per_arm
    t = np.repeat([0, 1], n_per_arm)
    f = rng.normal(size=n)
    y = effect * t + 0.9 * f + rng.normal(size=n)
    f_ext = rng.normal(size=external)
    lines = ["# fixture for report tests", "cohort,y,t,f"]
    for i in range(n):
        lines.append(f"trial,{y[i]:.6f},{int(t[i])},{f[i]:.6f}")
    for value in f_ext:
        lines.append(f"external,,,{value:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_report_is_json_safe_and_matches_the_estimators(tmp_path: Path) -> None:
    path = _fixture_csv(tmp_path / "trial.csv")
    report = build_report(path, date="2026-03-01", command="python -m ppict.cli analyze --csv trial.csv")
    payload = report.as_dict()

    assert payload["contract"] == CONTRACT_VERSION
    # allow_nan=False is the actual assertion: the calculator parses plain JSON.
    text = json.dumps(payload, allow_nan=False)
    assert json.loads(text)["analysis"]["dataset"]["n_trial"] == 240

    direct = analyze_trial(report.analysis.dataset)
    methods = {row["method"]: row for row in payload["analysis"]["methods"]}
    assert methods["ppi_single"]["estimate"] == pytest.approx(direct.primary.estimate, abs=0)
    assert methods["unadjusted"]["estimate"] == pytest.approx(direct.reference.estimate, abs=0)
    assert payload["command"] == "python -m ppict.cli analyze --csv trial.csv"
    assert payload["sap_tex"].count("\\begin{document}") == 1
    assert payload["sap_filename"].startswith("ppict-sap-appendix-")


def test_measured_values_are_defaulted_and_labelled(tmp_path: Path) -> None:
    report = build_report(_fixture_csv(tmp_path / "trial.csv"), date="2026-03-01")
    sources = report.plan.sources
    assert sources["r2"] == "measured"
    assert sources["effect"] == "measured"
    assert sources["outcome_sd"] == "measured"
    assert sources["cost_per_patient"] == "assumed"
    assert sources["patients_per_month"] == "assumed"
    # The default r2 is the measured squared correlation in the control arm.
    assert report.plan.inputs.r2 == pytest.approx(
        report.analysis.score_r2_control, abs=1e-12
    )
    # The default effect is the unadjusted estimate, not the adjusted one, so the
    # sizing input does not depend on the method under test.
    assert report.plan.inputs.effect == pytest.approx(
        report.analysis.reference.estimate, abs=1e-12
    )


def test_overrides_are_recorded_as_assumptions(tmp_path: Path) -> None:
    path = _fixture_csv(tmp_path / "trial.csv")
    measured = build_report(path, date="2026-03-01")
    assumed = build_report(
        path,
        date="2026-03-01",
        r2=0.60,
        effect=0.25,
        outcome_sd=1.5,
        cost_per_patient=9000.0,
        power=0.90,
        trial_size=800,
        patients_per_month=40.0,
    )
    assert assumed.plan.sources["r2"] == "assumed"
    assert assumed.plan.sources["effect"] == "assumed"
    assert assumed.plan.sources["outcome_sd"] == "assumed"
    assert assumed.plan.inputs.r2 == pytest.approx(0.60)
    assert assumed.plan.inputs.n_total == 800
    assert assumed.plan.inputs.power == pytest.approx(0.90)
    assert assumed.plan.cost_saved == pytest.approx(
        assumed.plan.patients_saved * 9000.0
    )
    # More prognostic value and a higher power target do not reduce the saving.
    assert assumed.plan.patients_saved > measured.plan.patients_saved


def test_plan_in_report_equals_plan_built_directly(tmp_path: Path) -> None:
    path = _fixture_csv(tmp_path / "trial.csv")
    report = build_report(path, date="2026-03-01")
    direct = plan_trial(PlanInputs(**{
        "cost_per_patient": report.plan.inputs.cost_per_patient,
        "power": report.plan.inputs.power,
        "r2": report.plan.inputs.r2,
        "n_total": report.plan.inputs.n_total,
        "effect": report.plan.inputs.effect,
        "outcome_sd": report.plan.inputs.outcome_sd,
        "patients_per_month": report.plan.inputs.patients_per_month,
        "alpha": report.plan.inputs.alpha,
    }))
    assert report.plan.patients_saved == pytest.approx(direct.patients_saved)
    assert report.plan.cost_saved == pytest.approx(direct.cost_saved)
    assert report.plan.as_dict()["notes"] == direct.as_dict()["notes"]


def test_non_positive_measured_effect_is_refused_with_a_way_forward(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    n = 240
    t = np.repeat([0, 1], n // 2)
    f = rng.normal(size=n)
    y = -1.0 * t + 0.2 * f + 0.1 * rng.normal(size=n)
    rows = ["y,t,f"] + [f"{y[i]:.6f},{int(t[i])},{f[i]:.6f}" for i in range(n)]
    path = tmp_path / "negative.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Pass an effect size explicitly"):
        build_report(path)
    # Supplying an effect makes the same dataset usable for planning.
    report = build_report(path, effect=0.3)
    assert report.plan.sources["effect"] == "assumed"


def test_r2_at_the_model_limit_is_capped_and_explained(tmp_path: Path) -> None:
    rng = np.random.default_rng(8)
    n = 240
    t = np.repeat([0, 1], n // 2)
    f = rng.normal(size=n)
    y = 0.5 * t + 10.0 * f + 0.01 * rng.normal(size=n)
    rows = ["y,t,f"] + [f"{y[i]:.6f},{int(t[i])},{f[i]:.6f}" for i in range(n)]
    path = tmp_path / "near_perfect.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    report = build_report(path)
    assert report.analysis.score_r2_control >= 0.995
    assert report.plan.inputs.r2 == 0.99
    assert report.plan.sources["r2"] == "measured"
    assert any("outside the range" in note for note in report.plan.notes)


def test_r2_measured_nan_falls_back_to_zero(tmp_path: Path) -> None:
    rng = np.random.default_rng(6)
    n = 240
    t = np.repeat([0, 1], n // 2)
    y = 0.5 * t + rng.normal(size=n)
    rows = ["y,t,f"] + [f"{y[i]:.6f},{int(t[i])},1.0" for i in range(n)]
    path = tmp_path / "constant_score.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    report = build_report(path)
    assert report.plan.inputs.r2 == 0.0
    assert report.plan.sources["r2"] == "assumed"
    assert report.plan.patients_saved == 0.0

"""Tests for the statistical analysis plan appendix."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppict.analysis import analyze_trial, load_trial_dataset, PlanInputs, plan_trial
from ppict.sap import render_sap, tex_escape


def _fixture_csv(path: Path, effect: float = 0.4, seed: int = 5) -> Path:
    rng = np.random.default_rng(seed)
    n_per_arm = 120
    n = 2 * n_per_arm
    t = np.repeat([0, 1], n_per_arm)
    f = rng.normal(size=n)
    y = effect * t + 0.9 * f + rng.normal(size=n)
    lines = ["y,t,f"]
    for i in range(n):
        lines.append(f"{y[i]:.6f},{int(t[i])},{f[i]:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def rendered(tmp_path: Path) -> str:
    analysis = analyze_trial(load_trial_dataset(_fixture_csv(tmp_path / "t.csv")))
    plan = plan_trial(PlanInputs(r2=0.35, effect=0.4, outcome_sd=1.2, n_total=480))
    return render_sap(
        analysis,
        plan,
        command="python -m ppict.cli analyze --csv trial.csv",
        date="2026-01-31",
        version="0.1.0",
    )


def test_document_is_standalone(rendered: str) -> None:
    assert rendered.startswith("\\documentclass")
    assert "\\begin{document}" in rendered
    assert rendered.rstrip().endswith("\\end{document}")
    assert rendered.count("\\begin{") == rendered.count("\\end{")


def test_states_the_prespecified_model_and_variance(rendered: str) -> None:
    for phrase in (
        "Pre-specified Prognostic Covariate Adjustment",
        "average treatment effect",
        "HC0",
        "intention to treat",
        "complete case",
    ):
        assert phrase in rendered
    # The model, tuning rule and variance expression are all present as math.
    assert "\\hat{\\tau}" in rendered
    assert "\\hat{\\lambda}" in rendered
    assert "\\widehat{\\mathrm{Var}}(\\hat{\\tau})" in rendered
    assert "4\\sigma^{2}\\left(1 - \\rho^{2}\\right)" in rendered


def test_records_provenance_and_does_not_report_the_result(rendered: str) -> None:
    # The command is split one flag per line so long paths stay inside the margin.
    assert "command: python -m ppict.cli analyze" in rendered
    assert "--csv trial.csv" in rendered
    assert "sha256" in rendered
    assert "2026-01-31" in rendered
    assert "0.1.0" in rendered
    # An appendix fixed before unblinding must not contain the observed effect.
    analysis_estimate = "0.4"
    assert f"observed effect of {analysis_estimate}" not in rendered


def test_no_placeholder_leaked(rendered: str) -> None:
    import re

    assert not re.search(r"\bnan\b", rendered), "rendered appendix contains a bare nan"
    assert not re.search(r"\bNone\b", rendered), "rendered appendix contains None"
    for bad in ("not available", "_fmt"):
        assert bad not in rendered, f"rendered appendix contains {bad!r}"


def test_no_em_or_en_dashes(rendered: str) -> None:
    assert "\u2014" not in rendered
    assert "\u2013" not in rendered


def test_tex_escape_covers_special_characters() -> None:
    escaped = tex_escape("a_b & 50% #1 {x}")
    assert "\\_" in escaped
    assert "\\&" in escaped
    assert "\\%" in escaped
    assert "\\#" in escaped
    assert "\\{" in escaped and "\\}" in escaped


def test_design_warning_reaches_the_appendix(tmp_path: Path) -> None:
    analysis = analyze_trial(load_trial_dataset(_fixture_csv(tmp_path / "t.csv")))
    plan = plan_trial(PlanInputs(r2=0.98, effect=0.5, outcome_sd=0.6, n_total=120))
    text = render_sap(analysis, plan, date="2026-02-01", version="0.1.0")
    assert "Design note." in text
    assert "not a usable design" in text


def test_design_table_numbers_match_the_plan(tmp_path: Path) -> None:
    analysis = analyze_trial(load_trial_dataset(_fixture_csv(tmp_path / "t.csv")))
    plan = plan_trial(PlanInputs(r2=0.42, effect=0.33, outcome_sd=1.05, n_total=500))
    text = render_sap(analysis, plan, date="2026-02-01", version="0.1.0")
    assert f"{plan.n_required_adjusted:.0f}" in text
    assert f"{plan.n_required_unadjusted:.0f}" in text
    assert f"{plan.width_ratio:.4f}" in text
    assert "0.4200" in text
    assert "500" in text

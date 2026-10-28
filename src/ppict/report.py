"""The report contract shared by the command line and the calculator.

One function, :func:`build_report`, loads a dataset, runs the estimators, builds
the design plan and renders the LaTeX appendix. The command line prints its JSON
form and the calculator app parses exactly that output, so there is one
implementation of the numbers and no chance of the two drifting apart.

The JSON is versioned by ``contract``. If the shape changes, bump it and update
``calculator/lib/types.ts`` in the same commit.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

from . import __version__
from .analysis import (
    AnalysisResult,
    PlanInputs,
    TrialDataset,
    TrialPlan,
    analyze_trial,
    load_trial_dataset,
    plan_trial,
)
from .sap import render_sap

__all__ = ["CONTRACT_VERSION", "Report", "build_report"]

CONTRACT_VERSION = 1

# Where each planning input came from. The calculator shows this next to every
# number, because the difference between a measured quantity and an assumption is
# the whole argument about whether a saving is real.
_SOURCE_MEASURED = "measured"
_SOURCE_ASSUMED = "assumed"


@dataclass
class Report:
    analysis: AnalysisResult
    plan: TrialPlan
    sap_tex: str
    command: str
    generated_at: str = field(default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds"))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "contract": CONTRACT_VERSION,
            "generated_at": self.generated_at,
            "ppict_version": __version__,
            "command": self.command,
            "analysis": self.analysis.as_dict(),
            "plan": self.plan.as_dict(),
            "sap_tex": self.sap_tex,
            "sap_filename": f"ppict-sap-appendix-{self.generated_at[:10].replace('-', '')}.tex",
        }


def _clamp_r2(value: float, source: str) -> tuple[float, str]:
    """Handle a measured squared correlation that the design model cannot take."""
    if value != value:  # nan
        return 0.0, _SOURCE_ASSUMED
    if value < 0.0:
        return 0.0, source
    return float(value), source


def build_report(
    dataset_path: str | Path,
    *,
    alpha: float = 0.05,
    cost_per_patient: float = 4200.0,
    power: float = 0.80,
    trial_size: Optional[int] = None,
    r2: Optional[float] = None,
    effect: Optional[float] = None,
    outcome_sd: Optional[float] = None,
    patients_per_month: float = 25.0,
    command: str = "python -m ppict.cli analyze --csv <dataset>",
    date: Optional[str] = None,
    dataset: Optional[TrialDataset] = None,
) -> Report:
    """Build the full report for one dataset.

    ``r2``, ``effect`` and ``outcome_sd`` default to values measured on the loaded
    data. Passing any of them overrides the measurement and is recorded as an
    assumption in ``plan.sources``.
    """
    ds = dataset if dataset is not None else load_trial_dataset(dataset_path)
    analysis = analyze_trial(ds, alpha=alpha)

    sources: Dict[str, str] = {}

    if effect is None:
        measured_effect = float(analysis.reference.estimate)
        if not np.isfinite(measured_effect) or measured_effect <= 0:
            raise ValueError(
                "The measured treatment effect is not positive, so it cannot be used "
                "for a sample size calculation. Pass an effect size explicitly."
            )
        effect = measured_effect
        sources["effect"] = _SOURCE_MEASURED
    else:
        sources["effect"] = _SOURCE_ASSUMED

    if outcome_sd is None:
        sd = float(analysis.outcome_sd)
        if not np.isfinite(sd) or sd <= 0:
            raise ValueError(
                "The measured outcome standard deviation is not positive. Pass one explicitly."
            )
        outcome_sd = sd
        sources["outcome_sd"] = _SOURCE_MEASURED
    else:
        sources["outcome_sd"] = _SOURCE_ASSUMED

    capped_r2 = False
    if r2 is None:
        measured_r2 = float(analysis.score_r2_control)
        if np.isfinite(measured_r2) and measured_r2 >= 0.995:
            # The model accepts values below 0.995. Capping silently would make the
            # projection look like it used the measured value, so record a note.
            r2 = 0.99
            capped_r2 = True
            sources["r2"] = _SOURCE_MEASURED
        else:
            r2, source = _clamp_r2(measured_r2, _SOURCE_MEASURED)
            sources["r2"] = source
    else:
        r2 = float(r2)
        sources["r2"] = _SOURCE_ASSUMED

    n_total = int(trial_size) if trial_size is not None else int(ds.n_trial)
    for key in ("cost_per_patient", "power", "n_total", "patients_per_month"):
        sources[key] = _SOURCE_ASSUMED

    inputs = PlanInputs(
        cost_per_patient=float(cost_per_patient),
        power=float(power),
        r2=float(r2),
        n_total=n_total,
        effect=float(effect),
        outcome_sd=float(outcome_sd),
        patients_per_month=float(patients_per_month),
        alpha=float(alpha),
    )
    plan = plan_trial(inputs, sources=sources)
    if capped_r2:
        plan.notes.append(
            "The measured squared correlation is at or above 0.995, outside the range "
            "the design model accepts, so 0.99 was used for the projection."
        )
    sap = render_sap(
        analysis,
        plan,
        command=command,
        date=date,
        version=__version__,
    )
    return Report(analysis=analysis, plan=plan, sap_tex=sap, command=command)

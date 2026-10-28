"""Data in, estimate and interval out.

This module is the entry point the calculator uses. It exists so that a caller
with a trial dataset does not have to know which estimator to call, with which
columns, or in which order. Everything here delegates to :mod:`ppict.ppi`; no
formula is repeated.

Two kinds of numbers come out, and they answer different questions:

* :func:`analyze_trial` reports what the loaded data says. The point estimate,
  standard error and interval for each method are computed from the data alone.
* :func:`plan_trial` reports what the variance model implies for a future trial.
  It takes the prognostic correlation as an input, because the whole point of
  prognostic adjustment is that the trial has not been run yet. Its outputs are
  model-based quantities, not measurements, and the report labels them that way.

Keeping those apart is the reason this module is worth having. A calculator that
blends a measured interval width into a dollars-saved figure without saying so is
the kind of tool this project is arguing against.

Dataset format
--------------

A CSV with columns ``y``, ``t`` and ``f``:

* ``y`` the outcome, numeric
* ``t`` the arm, 0 or 1
* ``f`` the prognostic score, numeric, on the same scale as ``y``

and one optional column, ``cohort``, taking the values ``trial`` (default) and
``external``. External rows carry only ``f``: they are the unlabeled cohort that
the per-arm estimator uses, and they need no outcome. Files without a ``cohort``
column are treated as all trial rows, in which case the per-arm estimator is not
available and the report says why.

Numeric columns ignore rows with a missing ``y`` or ``t`` in trial rows, and the
report records how many were dropped. External rows may leave both blank.

Lines beginning with ``#`` are provenance comments and are skipped. Lines are not
otherwise restricted, so a file written by :mod:`csv` reads back unchanged.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from .ppi import (
    ATEEstimate,
    ate_ancova,
    ate_ppi_per_arm,
    ate_ppi_single_lambda,
    ate_unadjusted,
    procova_variance,
    required_n_total,
)

__all__ = [
    "COLUMN_ALIASES",
    "TrialDataset",
    "AnalysisResult",
    "PlanInputs",
    "TrialPlan",
    "load_trial_dataset",
    "analyze_trial",
    "plan_trial",
    "pooled_sd",
    "squared_correlation",
]

# Accepted spellings for the three required columns. Kept small on purpose: a
# clear error naming the columns that were found is more useful than a long
# synonym list that silently mis-binds a column.
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "y": ("y", "outcome", "y_i"),
    "t": ("t", "treat", "treatment", "arm", "a"),
    "f": ("f", "score", "f_hat", "fhat", "prognostic_score"),
}
COHORT_ALIASES: Tuple[str, ...] = ("cohort", "set")


class DatasetError(ValueError):
    """Raised when a dataset cannot be interpreted. Message is user-facing."""


def _normalise(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _resolve(columns: Mapping[str, str], key: str) -> Optional[str]:
    """Return the actual header for a logical column, or None if absent."""
    for alias in COLUMN_ALIASES[key]:
        if alias in columns:
            return columns[alias]
    return None


def _num(x: Any) -> Optional[float]:
    """A float for JSON, or None where the quantity does not exist.

    JSON has no NaN, and several fields here are genuinely undefined for some
    methods: the tuning value of an unadjusted analysis, for instance. Sending
    null is honest; sending 0 would invent a value.
    """
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def squared_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Squared Pearson correlation, or nan when either side has no variance.

    This is the quantity the variance formulas want. It is deliberately a
    correlation rather than a coefficient of determination: a score with a
    constant offset explains nothing in the R squared sense, yet the offset
    cancels in every estimator here, so the correlation is what matters.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 3 or b.size != a.size:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled within-group standard deviation with ``ddof=1`` in each group."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2 or b.size < 2:
        return float("nan")
    dof = a.size + b.size - 2
    var = ((a.size - 1) * np.var(a, ddof=1) + (b.size - 1) * np.var(b, ddof=1)) / dof
    return float(np.sqrt(var))


@dataclass(frozen=True)
class TrialDataset:
    """One trial dataset, already validated and split into cohorts."""

    y: np.ndarray
    t: np.ndarray
    f: np.ndarray
    f_external: Optional[np.ndarray]
    source: str
    notes: Tuple[str, ...] = ()
    sha256: str = ""
    n_dropped: int = 0

    @property
    def n_trial(self) -> int:
        return int(self.y.size)

    @property
    def n_treated(self) -> int:
        return int((self.t == 1).sum())

    @property
    def n_control(self) -> int:
        return int((self.t == 0).sum())

    @property
    def n_external(self) -> int:
        return 0 if self.f_external is None else int(self.f_external.size)

    @property
    def has_external(self) -> bool:
        return self.f_external is not None and self.f_external.size >= 2

    def describe(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "sha256": self.sha256,
            "n_trial": self.n_trial,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "n_external": self.n_external,
            "n_dropped": self.n_dropped,
            "notes": list(self.notes),
        }


def load_trial_dataset(path: str | Path) -> TrialDataset:
    """Read and validate a trial CSV. Raises :class:`DatasetError` with a plain message."""
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"no file at {p}")
    raw = p.read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")

    notes: List[str] = []
    body: List[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            notes.append(line.lstrip().lstrip("#").strip())
        elif line.strip():
            body.append(line)
    if not body:
        raise DatasetError("the file has no data rows (only comments or blank lines)")

    reader = csv.DictReader(body)
    header = [(h or "") for h in (reader.fieldnames or [])]
    columns = {_normalise(h): h for h in header if h}
    if not columns:
        raise DatasetError("the file has no header row with column names")

    col_y = _resolve(columns, "y")
    col_t = _resolve(columns, "t")
    col_f = _resolve(columns, "f")
    missing = [k for k, c in (("y", col_y), ("t", col_t), ("f", col_f)) if c is None]
    if missing:
        found = ", ".join(header) if header else "(none)"
        raise DatasetError(
            f"missing required column{'s' if len(missing) > 1 else ''} "
            f"{', '.join(missing)}. Columns found: {found}. "
            "The file needs y (outcome), t (arm, 0 or 1) and f (prognostic score)."
        )

    col_cohort = next((columns[a] for a in COHORT_ALIASES if a in columns), None)

    ys: List[float] = []
    ts: List[int] = []
    fs: List[float] = []
    ext: List[float] = []
    dropped = 0

    def _num(v: Any) -> Optional[float]:
        if v is None:
            return None
        s = str(v).strip()
        if s == "" or s.lower() in {"na", "nan", "null", "none"}:
            return None
        try:
            out = float(s)
        except ValueError:
            return None
        return out if np.isfinite(out) else None

    for row in reader:
        cohort = "trial"
        if col_cohort is not None:
            cohort = str(row.get(col_cohort) or "trial").strip().lower()
        if cohort not in {"trial", "external"}:
            raise DatasetError(
                f"unknown cohort value {cohort!r}; use 'trial' or 'external'"
            )

        f_val = _num(row.get(col_f))
        if f_val is None:
            if cohort == "external":
                continue
            dropped += 1
            continue

        if cohort == "external":
            ext.append(f_val)
            continue

        y_val = _num(row.get(col_y))
        t_val = _num(row.get(col_t))
        if y_val is None or t_val is None:
            dropped += 1
            continue
        if t_val not in (0.0, 1.0):
            raise DatasetError(
                f"column t must be 0 or 1; found {t_val:g}"
            )
        ys.append(y_val)
        ts.append(int(t_val))
        fs.append(f_val)

    if len(ys) < 16:
        raise DatasetError(
            f"only {len(ys)} complete trial rows; at least 16 are needed so each arm has 8"
        )
    y = np.asarray(ys, dtype=float)
    t = np.asarray(ts, dtype=int)
    f = np.asarray(fs, dtype=float)
    if t.sum() < 8 or (t == 0).sum() < 8:
        raise DatasetError(
            f"each arm needs at least 8 patients; found {(t == 1).sum()} treated "
            f"and {(t == 0).sum()} control"
        )

    f_ext = np.asarray(ext, dtype=float) if ext else None
    if f_ext is not None and f_ext.size < 2:
        f_ext = None

    return TrialDataset(
        y=y,
        t=t,
        f=f,
        f_external=f_ext,
        source=str(p),
        notes=tuple(notes),
        sha256=hashlib.sha256(raw).hexdigest(),
        n_dropped=dropped,
    )


@dataclass
class AnalysisResult:
    """What the loaded dataset says, method by method."""

    dataset: TrialDataset
    estimates: Dict[str, ATEEstimate]
    alpha: float = 0.05
    score_r2_control: float = float("nan")
    score_r2_treated: float = float("nan")
    outcome_sd: float = float("nan")

    @property
    def primary(self) -> ATEEstimate:
        """The pre-specified estimator: tuned, single tuning value."""
        return self.estimates["ppi_single"]

    @property
    def reference(self) -> ATEEstimate:
        return self.estimates["unadjusted"]

    def width_ratio(self, method: str) -> float:
        """Interval width for ``method`` relative to the unadjusted analysis."""
        base = self.reference.ci_width
        if not np.isfinite(base) or base <= 0:
            return float("nan")
        return float(self.estimates[method].ci_width / base)

    @property
    def primary_width_ratio(self) -> float:
        return self.width_ratio("ppi_single")

    def as_dict(self) -> Dict[str, Any]:
        methods = []
        for name, est in self.estimates.items():
            row = {k: (_num(v) if isinstance(v, float) else v) for k, v in est.as_row().items()}
            row["width_ratio"] = _num(self.width_ratio(name))
            methods.append(row)
        return {
            "dataset": self.dataset.describe(),
            "alpha": self.alpha,
            "methods": methods,
            "primary_method": self.primary.method,
            "primary_width_ratio": _num(self.primary_width_ratio),
            "measured": {
                "score_r2_control": _num(self.score_r2_control),
                "score_r2_treated": _num(self.score_r2_treated),
                "outcome_sd": _num(self.outcome_sd),
                "effect_unadjusted": _num(self.reference.estimate),
            },
            "unavailable": self._unavailable(),
        }

    def _unavailable(self) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if not self.dataset.has_external:
            out.append(
                {
                    "method": "ppi_per_arm",
                    "reason": "needs an external cohort; add rows with cohort=external and f values",
                }
            )
        return out


def analyze_trial(dataset: TrialDataset, alpha: float = 0.05) -> AnalysisResult:
    """Estimate the treatment effect by each supported method.

    The methods are all computed on the same rows, so the interval widths are
    directly comparable. ``ppi_single`` is the pre-specified one; ``ancova`` is
    included because the two agree up to the heterogeneity convention in the
    standard error, which is easier to trust when it is visible.
    """
    y, t, f = dataset.y, dataset.t, dataset.f
    m1, m0 = (t == 1), (t == 0)

    estimates: Dict[str, ATEEstimate] = {}
    estimates["unadjusted"] = ate_unadjusted(y, t, alpha=alpha)
    estimates["ancova"] = ate_ancova(y, t, f.reshape(-1, 1), alpha=alpha)
    estimates["ppi_single"] = ate_ppi_single_lambda(y, t, f, alpha=alpha)
    if dataset.has_external:
        assert dataset.f_external is not None
        estimates["ppi_per_arm"] = ate_ppi_per_arm(
            y, t, f, dataset.f_external, alpha=alpha
        )

    return AnalysisResult(
        dataset=dataset,
        estimates=estimates,
        alpha=alpha,
        score_r2_control=squared_correlation(y[m0], f[m0]),
        score_r2_treated=squared_correlation(y[m1], f[m1]),
        outcome_sd=pooled_sd(y[m0], y[m1]),
    )


@dataclass(frozen=True)
class PlanInputs:
    """Design assumptions for a future trial under the PROCOVA variance model.

    Every field here is an input, not a measurement. The sources mapping records
    where each came from, either ``measured`` (defaulted from the loaded dataset)
    or ``assumption`` (supplied by the caller), so that the report can show which
    numbers are doing the work.
    """

    cost_per_patient: float = 4200.0
    power: float = 0.80
    r2: float = 0.30
    n_total: int = 600
    effect: float = 0.35
    outcome_sd: float = 1.0
    patients_per_month: float = 25.0
    alpha: float = 0.05

    def __post_init__(self) -> None:
        if not (0.0 <= self.r2 < 0.995):
            raise ValueError("r2 must be in [0, 0.995)")
        if not (0.5 < self.power < 1.0):
            raise ValueError("power must be in (0.5, 1)")
        if self.effect <= 0:
            raise ValueError("effect must be positive")
        if self.outcome_sd <= 0:
            raise ValueError("outcome_sd must be positive")
        if self.cost_per_patient < 0:
            raise ValueError("cost_per_patient cannot be negative")
        if self.patients_per_month <= 0:
            raise ValueError("patients_per_month must be positive")
        if self.n_total < 2:
            raise ValueError("n_total must be at least 2")


@dataclass
class TrialPlan:
    """What the variance model implies, with the arithmetic kept auditable."""

    inputs: PlanInputs
    variance_unadjusted: float
    variance_adjusted: float
    n_required_unadjusted: float
    n_required_adjusted: float
    width_ratio: float
    equivalent_unadjusted_n: float
    sources: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def ci_narrowing(self) -> float:
        return float(1.0 - self.width_ratio)

    @property
    def patients_saved(self) -> float:
        return float(self.n_required_unadjusted - self.n_required_adjusted)

    @property
    def cost_saved(self) -> float:
        return float(self.patients_saved * self.inputs.cost_per_patient)

    @property
    def months_saved(self) -> float:
        return float(self.patients_saved / self.inputs.patients_per_month)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "inputs": {
                "cost_per_patient": self.inputs.cost_per_patient,
                "power": self.inputs.power,
                "r2": self.inputs.r2,
                "n_total": self.inputs.n_total,
                "effect": self.inputs.effect,
                "outcome_sd": self.inputs.outcome_sd,
                "patients_per_month": self.inputs.patients_per_month,
                "alpha": self.inputs.alpha,
            },
            "sources": dict(self.sources),
            "variance_unadjusted": _num(self.variance_unadjusted),
            "variance_adjusted": _num(self.variance_adjusted),
            "n_required_unadjusted": _num(self.n_required_unadjusted),
            "n_required_adjusted": _num(self.n_required_adjusted),
            "patients_saved": _num(self.patients_saved),
            "cost_saved": _num(self.cost_saved),
            "months_saved": _num(self.months_saved),
            "width_ratio": _num(self.width_ratio),
            "ci_narrowing": _num(self.ci_narrowing),
            "equivalent_unadjusted_n": _num(self.equivalent_unadjusted_n),
            "notes": list(self.notes),
        }


def plan_trial(inputs: PlanInputs, sources: Optional[Mapping[str, str]] = None) -> TrialPlan:
    """Design arithmetic under the prognostic covariate adjustment model.

    The model is the one the EMA qualification opinion states: with a balanced
    two-arm design, equal arm variances ``sigma`` and prognostic correlation
    ``rho`` in both arms, the per-patient variance constant of the treatment
    effect is ``4 sigma^2 (1 - rho^2)`` rather than ``4 sigma^2``. Both constants
    are produced by :func:`ppict.ppi.procova_variance`, the same function the
    allocation code uses, so this is not a second implementation of the formula.

    Because required sample size is proportional to that constant at fixed power,
    the fractional reduction in required patients equals the fractional variance
    reduction equals ``rho^2``. The confidence interval width shrinks by
    ``1 - sqrt(1 - rho^2)``, which is smaller than ``rho^2``, and the two should
    not be quoted interchangeably. The report keeps them in separate fields.
    Required sizes are rounded up to whole patients before the savings are
    computed, so the patient count, the cost and the recruitment time agree.
    """
    sigma = float(inputs.outcome_sd)
    rho = float(np.sqrt(inputs.r2))

    # Balanced allocation, equal arm variances. rho = 0 is the unadjusted design.
    v0 = procova_variance(sigma, sigma, 0.0, 0.0, 0.5)
    v1 = procova_variance(sigma, sigma, rho, rho, 0.5)

    # Rounded up to whole patients, because a design table that says 171.4
    # patients and a cost that follows from it cannot both be right.
    n0 = float(np.ceil(required_n_total(v0, inputs.effect, alpha=inputs.alpha, power=inputs.power)))
    n1 = float(np.ceil(required_n_total(v1, inputs.effect, alpha=inputs.alpha, power=inputs.power)))

    width_ratio = float(np.sqrt(v1 / v0)) if v0 > 0 else float("nan")
    equivalent = float(inputs.n_total / (1.0 - inputs.r2))

    notes: List[str] = []
    if n1 > inputs.n_total:
        notes.append(
            "The planned trial size is below the adjusted requirement for the stated power."
        )
    if inputs.r2 == 0:
        notes.append("At zero prognostic value the adjusted and unadjusted designs are identical.")
    if n1 < 2:
        # A near-perfect prognostic score plus a large effect can drive the model
        # below a single patient. The arithmetic is right and the design is not,
        # so say so rather than printing a one-patient trial.
        notes.append(
            "The adjusted requirement is fewer than two patients, so this projection is not a "
            "usable design. Check the effect, the standard deviation and the prognostic R squared."
        )

    return TrialPlan(
        inputs=inputs,
        variance_unadjusted=float(v0),
        variance_adjusted=float(v1),
        n_required_unadjusted=float(n0),
        n_required_adjusted=float(n1),
        width_ratio=width_ratio,
        equivalent_unadjusted_n=equivalent,
        sources=dict(sources or {}),
        notes=notes,
    )

"""LaTeX appendix for a statistical analysis plan.

The output is a standalone document, not a fragment, so that a reviewer can
compile it without the repository. It states the pre-specified model, the tuning
rule, the design arithmetic and the software provenance, and it deliberately does
not report the observed treatment effect: an appendix that fixes an analysis
before unblinding should not contain the answer.

Compilation is checked by ``validation/run_calculator_checks.py``, which runs
``latexmk`` and asserts that a PDF appears.
"""

from __future__ import annotations

import datetime as _dt
import textwrap
from typing import Optional

from .analysis import AnalysisResult, TrialPlan

__all__ = ["render_sap", "tex_escape"]


_TEX_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex_escape(text: str) -> str:
    """Escape plain text for use in LaTeX text mode."""
    return "".join(_TEX_REPLACEMENTS.get(ch, ch) for ch in str(text))


def _fmt(x: float, digits: int = 3) -> str:
    if x != x:  # nan
        return "not available"
    return f"{x:.{digits}f}"


def _pct(x: float, digits: int = 1) -> str:
    if x != x:
        return "not available"
    return f"{100.0 * x:.{digits}f}\\%"


def _provenance_block(text: str) -> str:
    """Typeset provenance so long paths do not run into the margin.

    A command is split at its flags, one flag per line, and every line is wrapped
    at a width that fits in a typewriter face at footnote size. Wrapping is the
    only change made to the text; nothing is abbreviated, so the block still
    identifies exactly what was run.
    """
    lines: list[str] = []
    for raw in text.splitlines():
        if raw.startswith("command: ") and " --" in raw:
            head, *flags = raw.split(" --")
            lines.append(head)
            lines.extend("    --" + flag for flag in flags)
        else:
            lines.append(raw)

    wrapped: list[str] = []
    for line in lines:
        pieces = textwrap.wrap(
            line,
            width=80,
            subsequent_indent="        ",
            break_long_words=True,
            break_on_hyphens=False,
        )
        wrapped.extend(pieces or [""])
    body = "\n".join(wrapped)
    return "{\\footnotesize\n\\begin{verbatim}\n" + body + "\n\\end{verbatim}\n}"


def render_sap(
    analysis: AnalysisResult,
    plan: TrialPlan,
    *,
    command: str = "python -m ppict.cli analyze --csv <dataset>",
    date: Optional[str] = None,
    version: str = "",
) -> str:
    """Render the appendix as a complete LaTeX document.

    ``command`` is recorded verbatim as provenance. ``date`` is injectable so the
    output is reproducible; it defaults to today.
    """
    if not version:
        from . import __version__ as version  # local import avoids a cycle

    when = date or _dt.date.today().isoformat()
    ds = analysis.dataset
    inputs = plan.inputs
    ratio_t = ds.n_treated / max(ds.n_trial, 1)

    design_notes = ""
    if plan.notes:
        joined = " ".join(tex_escape(note) for note in plan.notes)
        design_notes = f"\\noindent\\textbf{{Design note.}} {joined}\n"

    provenance = "\n".join(
        [
            f"command: {command}",
            f"dataset: {ds.source}",
            f"dataset sha256: {ds.sha256 or 'not recorded'}",
            f"trial rows: {ds.n_trial} ({ds.n_treated} treated, {ds.n_control} control)",
            f"external rows: {ds.n_external}",
            f"ppict version: {version}",
            f"alpha: {analysis.alpha:g}",
        ]
    )

    sol = (
        r"\begin{equation}"
        r"\hat{\tau} = \bar{Y}_{1} - \bar{Y}_{0}"
        r" - \hat{\lambda}\,\left(\bar{f}_{1} - \bar{f}_{0}\right)"
        r"\end{equation}"
    )
    lam = (
        r"\begin{equation}"
        r"\hat{\lambda} = "
        r"\frac{\widehat{\mathrm{Cov}}_{1}(Y,f)/n_{1} + \widehat{\mathrm{Cov}}_{0}(Y,f)/n_{0}}"
        r"{\widehat{\mathrm{Var}}_{1}(f)/n_{1} + \widehat{\mathrm{Var}}_{0}(f)/n_{0}}"
        r"\end{equation}"
    )
    var = (
        r"\begin{equation}"
        r"\widehat{\mathrm{Var}}(\hat{\tau}) = "
        r"\frac{\widehat{\mathrm{Var}}_{1}(Y - \hat{\lambda} f)}{n_{1}} + "
        r"\frac{\widehat{\mathrm{Var}}_{0}(Y - \hat{\lambda} f)}{n_{0}}"
        r"\end{equation}"
    )
    model = (
        r"\begin{equation}"
        r"Y_{i} = \alpha + \tau\, T_{i} + \beta\, f(X_{i}) + \varepsilon_{i}"
        r"\end{equation}"
    )
    design = (
        r"\begin{equation}"
        r"v = 4\sigma^{2}\left(1 - \rho^{2}\right), \qquad "
        r"n = \frac{\left(z_{1-\alpha/2} + z_{\mathrm{power}}\right)^{2} v}{\Delta^{2}}"
        r"\end{equation}"
    )

    return rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{amsmath}}
\usepackage{{booktabs}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{6pt}}

\title{{Statistical Analysis Plan Appendix\\Pre-specified Prognostic Covariate Adjustment}}
\author{{}}
\date{{{when}}}
\begin{{document}}
\maketitle

\section{{Purpose}}

This appendix fixes, before unblinding, the model and the single covariate used
in the primary analysis of the primary continuous endpoint. It is written to be
attached to the statistical analysis plan and to be read by a reviewer who wants
to check that the adjustment was specified in advance rather than chosen after
the data were seen.

It does not report the observed treatment effect. The design quantities in
Section~6 are inputs fixed from the source dataset named in Section~9, not
results.

\section{{Estimand}}

The estimand is the average treatment effect on the primary continuous endpoint
in the randomized population,
\[
\tau = E\left[Y(1) - Y(0)\right],
\]
with treatment assigned at randomization. The analysis is intention to treat.
The randomization ratio in the source dataset is {_fmt(ratio_t, 4)} treated to
{_fmt(1.0 - ratio_t, 4)} control.

\section{{Prognostic score}}

The covariate is a scalar prognostic score $f(X)$, fixed before unblinding. It
predicts the control arm outcome from baseline covariates only, and it is
constructed in one of two ways, chosen before the trial data are used:

\begin{{enumerate}}
\item \textbf{{External fit.}} The score is fitted on a cohort whose outcomes are
  independent of the trial outcomes, then applied unchanged to the trial.
\item \textbf{{Cross-fitting.}} The score is fitted inside the trial by
  $K$-fold cross-fitting, and each patient is scored by a model that did not see
  that patient's outcome.
\end{{enumerate}}

In both cases the trial outcome of a patient never contributes to the prediction
used for that patient. The score enters the primary model as a single covariate.
No further covariate selection is performed in the analysis.

\section{{Primary analysis model}}

The primary analysis is a linear model with treatment and the prognostic score,
{model}
where $\tau$ is the estimand. The treatment coefficient is estimated by least
squares, with a heteroskedasticity-consistent (HC0) standard error, and inference
uses a two-sided normal interval at level $1-\alpha$.

The same estimator has a prediction-powered form. With one tuning value shared
between arms, the external cohort cancels algebraically and
{sol}
where $\bar{{Y}}_{{w}}$ and $\bar{{f}}_{{w}}$ are arm means. Equations
(2) and (3) are the same estimator as (1) when the arms are balanced: the tuning
value is the pooled within-arm slope of the outcome on the score, which is the
regression coefficient $\beta$ after partialling out treatment.

\section{{Tuning}}

The tuning value is estimated from the trial data alone by
{lam}
with biased (divide by $n$) covariance and variance so that (2) reproduces the
least squares coefficient exactly at equal allocation. No unlabeled cohort
quantity enters (2). The interval uses
{var}
Because $\hat{{\lambda}}$ is a scalar estimated from the same data, its
contribution to the variance is of smaller order than the terms in (4), which
treat it as fixed.

If a per-arm form with arm-specific tuning values is used as a sensitivity
analysis, the external cohort no longer cancels. That form is unbiased only if
the external cohort is exchangeable with the trial population in the covariates
that enter the score. The per-arm form is not the primary analysis.

\section{{Design and sample size}}

Under the prognostic covariate adjustment model with balanced allocation, equal
arm variances $\sigma^{{2}}$ and prognostic correlation $\rho$, the per-patient
variance constant and the required total sample size are
{design}

The design inputs below are used for planning. The effect and the outcome
standard deviation are taken from the source dataset; the prognostic correlation
is {_fmt(plan.inputs.r2, 4)} in squared form, from the source named in the
sources column. These are projections under the stated model. They are not
observed savings from a completed trial.

\begin{{center}}
\begin{{tabular}}{{lr}}
\toprule
Design input & Value \\
\midrule
Two-sided alpha & {_fmt(inputs.alpha, 3)} \\
Target power & {_pct(inputs.power, 0)} \\
Assumed effect $\Delta$ & {_fmt(inputs.effect, 4)} \\
Outcome standard deviation $\sigma$ & {_fmt(inputs.outcome_sd, 4)} \\
Squared prognostic correlation $\rho^{{2}}$ & {_fmt(inputs.r2, 4)} \\
Per-patient variance, unadjusted & {_fmt(plan.variance_unadjusted, 4)} \\
Per-patient variance, adjusted & {_fmt(plan.variance_adjusted, 4)} \\
Required total $n$, unadjusted & {_fmt(plan.n_required_unadjusted, 0)} \\
Required total $n$, adjusted & {_fmt(plan.n_required_adjusted, 0)} \\
Patients saved at the target power & {_fmt(plan.patients_saved, 0)} \\
Planned trial size & {_fmt(float(inputs.n_total), 0)} \\
Interval width ratio & {_fmt(plan.width_ratio, 4)} \\
\bottomrule
\end{{tabular}}
\end{{center}}

At the planned size of {inputs.n_total} patients, the adjusted analysis carries
the information of approximately {_fmt(plan.equivalent_unadjusted_n, 0)} patients
under an unadjusted analysis at the same effect and variance.

{design_notes}
\section{{Missing data and sensitivity analyses}}

The primary analysis is complete case. No imputation is used, and the number of
excluded rows is reported. Pre-specified sensitivity analyses are: the unadjusted
difference in means; adjustment for the score with the per-arm prediction-powered
form where an external cohort is available; and, where the score was cross-fitted,
a repeat with a different fold assignment.

\section{{Reporting}}

The report gives the point estimate of $\tau$, its standard error, the two-sided
interval at level $1-\alpha$, and the corresponding p value. Subgroup analyses are
not part of this appendix and are not claimed to be adjusted by this model.

\section{{Software and provenance}}

The analysis was produced by \texttt{{ppict}} version {tex_escape(version)}. The
dataset is identified by its SHA-256 digest, so the exact input can be checked
even if the file path is temporary. The command that produced the numbers in this
appendix was:

{_provenance_block(provenance)}

\end{{document}}
"""

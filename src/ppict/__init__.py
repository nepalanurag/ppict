"""ppict: prediction-powered covariate adjustment for randomized clinical trials.

Combines a permutation-assisted group lasso prognostic score with the tuned
prediction-powered (PPI++) estimator of the average treatment effect, and the
prognostic covariate adjustment variance and allocation formulas.

See ``PLAN.md`` for the paper-versus-prompt discrepancies this package inherits
and for what is deliberately not claimed.
"""

from __future__ import annotations

import warnings

__version__ = "0.1.0"

# Suppress a known platform artifact, not a condition in this code.
#
# On macOS, the numpy build used here links Apple Accelerate for BLAS and that
# backend raises spurious floating-point flags on essentially every matmul. A
# matmul of two all-finite arrays with an all-finite result still reports
# "divide by zero", "invalid value" and "overflow" encountered in matmul. A
# matmul performs no division, so "divide by zero in matmul" cannot be a real
# signal. The filter is scoped to those four messages on the matmul ufunc rather
# than to RuntimeWarning as a class, so every other numerical warning still
# reaches the caller. Without it a multi-hundred-replicate validation run writes
# megabytes of noise that hides genuine messages. See README, Environment notes.
_BLAS_ARTIFACT_MESSAGES = (
    r"divide by zero encountered in matmul",
    r"invalid value encountered in matmul",
    r"overflow encountered in matmul",
    r"underflow encountered in matmul",
)
for _blas_msg in _BLAS_ARTIFACT_MESSAGES:
    warnings.filterwarnings("ignore", message=_blas_msg, category=RuntimeWarning)
del _blas_msg

from .analysis import (
    AnalysisResult,
    DatasetError,
    PlanInputs,
    TrialDataset,
    TrialPlan,
    analyze_trial,
    load_trial_dataset,
    plan_trial,
    pooled_sd,
    squared_correlation,
)
from .group_select import (
    GroupSelection,
    StabilitySelection,
    calibrate_lambda,
    define_groups,
    empirical_fdp,
    group_lasso_fista,
    group_lasso_path,
    group_pen_from_z,
    jaccard,
    pairwise_jaccard,
    perm_null_group_counts,
    screen_features,
    select_groups,
    selected_features,
    stability_select,
)
from .ppi import (
    AllocationResult,
    ATEEstimate,
    ate_ancova,
    ate_lasso_adjusted,
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
    z_quantile,
)
from .report import CONTRACT_VERSION, Report, build_report
from .sap import render_sap, tex_escape
from .prognostic import (
    CrossFitResult,
    PrognosticScore,
    cross_fitted_scores,
    external_score,
    fit_prognostic,
    r2_score,
)
from .simulate import (
    TrialConfig,
    TrialData,
    corrupt_score,
    make_trial_data,
    run_monte_carlo,
    run_replicate,
    summarize,
    write_run_log,
)

__all__ = [
    "__version__",
    # data in, estimate and interval out
    "AnalysisResult",
    "DatasetError",
    "PlanInputs",
    "TrialDataset",
    "TrialPlan",
    "analyze_trial",
    "load_trial_dataset",
    "plan_trial",
    "pooled_sd",
    "squared_correlation",
    # report contract
    "CONTRACT_VERSION",
    "Report",
    "build_report",
    "render_sap",
    "tex_escape",
    # group selection
    "GroupSelection",
    "StabilitySelection",
    "calibrate_lambda",
    "define_groups",
    "empirical_fdp",
    "group_lasso_fista",
    "group_lasso_path",
    "group_pen_from_z",
    "jaccard",
    "pairwise_jaccard",
    "perm_null_group_counts",
    "screen_features",
    "select_groups",
    "selected_features",
    "stability_select",
    # estimation
    "AllocationResult",
    "ATEEstimate",
    "ate_ancova",
    "ate_lasso_adjusted",
    "ate_ppi_per_arm",
    "ate_ppi_single_lambda",
    "ate_unadjusted",
    "control_arm_size",
    "optimal_allocation",
    "ppi_mean_ci",
    "ppi_mean_pointestimate",
    "ppi_mean_pvalue",
    "ppi_mean_se",
    "procova_variance",
    "required_n_total",
    "sample_size_reduction",
    "tuned_lambda_mean",
    "unadjusted_min_variance",
    "z_quantile",
    # prognostic score
    "CrossFitResult",
    "PrognosticScore",
    "cross_fitted_scores",
    "external_score",
    "fit_prognostic",
    "r2_score",
    # simulation
    "TrialConfig",
    "TrialData",
    "corrupt_score",
    "make_trial_data",
    "run_monte_carlo",
    "run_replicate",
    "summarize",
    "write_run_log",
]

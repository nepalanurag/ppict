\name{ppict-package}
\alias{ppict-package}
\alias{ppict}
\alias{define_groups}
\alias{screen_features}
\alias{group_pen_from_z}
\alias{group_lasso_fista}
\alias{group_lasso_path}
\alias{perm_null_group_counts}
\alias{calibrate_lambda}
\alias{select_groups}
\alias{selected_features}
\alias{jaccard}
\alias{pairwise_jaccard}
\alias{r2_score}
\alias{fit_prognostic}
\alias{predict_score}
\alias{cross_fitted_scores}
\alias{z_quantile}
\alias{tuned_lambda_mean}
\alias{ppi_mean_pointestimate}
\alias{ppi_mean_se}
\alias{ppi_mean_ci}
\alias{ppi_mean_pvalue}
\alias{ate_unadjusted}
\alias{ate_ancova}
\alias{ate_ppi_single_lambda}
\alias{procova_variance}
\alias{unadjusted_min_variance}
\alias{optimal_allocation}
\alias{required_n_total}
\alias{sample_size_reduction}
\alias{control_arm_size}
\alias{load_trial_dataset}
\alias{squared_correlation}
\alias{pooled_sd}
\alias{analyze_trial}
\alias{analysis_table}
\alias{analyze_trial_csv}
\docType{package}
\title{Prediction-powered inference with a group-lasso prognostic score}
\description{
A local R port of the Python \pkg{ppict} package. It covers the
permutation-assisted group-lasso selector, the prognostic score, the
shared-tuning PPI++ treatment-effect estimator and interval, and a
dataset-in, estimate-and-interval-out analysis entry point. See the repository
README for the Python and R parity contract and for what is not ported.
}
\details{
The functions use snake_case names matching the Python public API where the
arguments also match. The group-lasso solver uses FISTA with group
soft-thresholding. The permutation calibration accepts an explicit permutation
matrix so that R and Python can be compared on identical draws.
}
\keyword{package}

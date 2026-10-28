#!/usr/bin/env Rscript

# R side of the parity check. Reads only the shared fixture files, computes
# results with the installed R package, and writes one row per scalar value.
# validation/run_r_parity.py computes the Python side from the same files.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: r_parity.R <fixture_dir> <output_csv>")
fixtures <- normalizePath(args[1])
out_path <- args[2]
root <- normalizePath(file.path(fixtures, "..", ".."))
library(ppict, lib.loc = file.path(root, "r", "lib"))

read_mat <- function(name, header = TRUE) {
  as.matrix(utils::read.csv(file.path(fixtures, name), header = header,
                            check.names = FALSE))
}
read_vec <- function(name, column = NULL) {
  df <- utils::read.csv(file.path(fixtures, name), check.names = FALSE)
  if (is.null(column)) as.numeric(df[[1]]) else as.numeric(df[[column]])
}

rows <- list()
add <- function(check, key, value) {
  txt <- if (is.numeric(value)) sprintf("%.17g", value) else as.character(value)
  rows[[length(rows) + 1L]] <<- data.frame(
    check = check, key = key, value = txt, stringsAsFactors = FALSE
  )
}

read_groups <- function(name) {
  g <- utils::read.csv(file.path(fixtures, name), check.names = FALSE)
  lapply(sort(unique(g$group)), function(k) as.integer(g$feature[g$group == k]))
}

# ---------------------------------------------------------------- selector
X <- read_mat("group_X.csv")
y <- read_vec("group_y.csv")
groups <- read_groups("group_groups.csv")
lam_grid <- read_vec("group_lam_grid.csv")
perms <- as.matrix(utils::read.csv(
  file.path(fixtures, "group_permutations.csv"), header = FALSE
))

found <- define_groups(X, threshold = 0.6)
add("groups", "group_count", length(found))
for (j in seq_along(found)) {
  add("groups", paste0("group_", j), paste(found[[j]], collapse = " "))
}

screened <- screen_features(X, y, top_k = 6L)
add("screen", "screen_index", paste(screened$index, collapse = " "))
for (i in seq_along(screened$r)) {
  add("screen", paste0("screen_r_", i), screened$r[i])
}

sel_weights <- select_groups(X, y, groups = groups, lam = lam_grid[1])
for (j in seq_along(sel_weights$weights)) {
  add("weights", paste0("weight_", j), sel_weights$weights[j])
}

for (k in seq_along(lam_grid)) {
  coef <- group_lasso_fista(X, y, groups, lam_grid[k], sel_weights$weights,
                            max_iter = 500L, tol = 1e-9)
  for (i in seq_along(coef)) {
    add("solver", sprintf("coef_lam%d_f%d", k, i), coef[i])
  }
}

counts <- perm_null_group_counts(
  X, y, groups, lam_grid, sel_weights$weights,
  n_perm = nrow(perms), permutations = perms,
  max_iter = 300L, tol = 1e-8
)
for (b in seq_len(nrow(counts))) {
  for (k in seq_len(ncol(counts))) {
    add("null_counts", sprintf("null_p%d_g%d", b, k), counts[b, k])
  }
}
cal <- calibrate_lambda(
  X, y, groups, sel_weights$weights,
  q = 0.2, n_perm = nrow(perms), lam_grid = lam_grid,
  permutations = perms, max_iter = 300L, tol = 1e-8
)
add("calibration", "cal_lam", cal$lam)
for (k in seq_along(cal$mean_counts)) {
  add("calibration", paste0("cal_mean_", k), cal$mean_counts[k])
}

# ---------------------------------------------------------------- prognosis
Xp <- read_mat("prognostic_X.csv")
yp <- read_vec("prognostic_y.csv")
Xv <- read_mat("prognostic_Xval.csv")
yv <- read_vec("prognostic_yval.csv")
folds <- read_vec("prognostic_folds.csv")
pgroups <- read_groups("prognostic_groups.csv")
pweights <- select_groups(Xp, yp, groups = pgroups, lam = 0.02)$weights

score <- fit_prognostic(
  Xp, yp, groups = pgroups, weights = pweights, lam = 0.02,
  X_val = Xv, y_val = yv, max_iter = 300L, tol = 1e-8
)
add("prognostic", "prog_intercept", score$intercept)
add("prognostic", "prog_features", paste(score$feature_idx, collapse = " "))
add("prognostic", "prog_train_r2", score$train_r2)
add("prognostic", "prog_oos_r2", score$oos_r2)
for (i in seq_along(score$coef)) {
  add("prognostic", paste0("prog_coef_", i), score$coef[i])
}
preds <- predict(score, Xv)
for (i in seq_along(preds)) {
  add("prognostic", paste0("prog_pred_", i), preds[i])
}

cross <- cross_fitted_scores(
  Xp, yp, n_folds = 5L, groups = pgroups, fold_assignment = folds,
  lam = 0.02, max_iter = 300L, tol = 1e-8
)
add("crossfit", "oof_r2", cross$oof_r2)
add("crossfit", "fold_jaccard_mean", cross$mean_fold_jaccard)
for (i in seq_along(cross$oof)) {
  add("crossfit", paste0("oof_", i), cross$oof[i])
}

# ---------------------------------------------------------------- PPI
y_ppi <- read_vec("ppi_y.csv")
t_ppi <- read_vec("ppi_t.csv")
f_ppi <- read_vec("ppi_f.csv")
f_unl <- read_vec("ppi_f_unlabeled.csv")

lam <- tuned_lambda_mean(y_ppi, f_ppi, f_unl)
point <- ppi_mean_pointestimate(y_ppi, f_ppi, f_unl, lam = lam)
se <- ppi_mean_se(y_ppi, f_ppi, f_unl, lam = lam)
ci <- ppi_mean_ci(y_ppi, f_ppi, f_unl, lam = lam)
add("ppi_mean", "ppi_lam", lam)
add("ppi_mean", "ppi_point", point)
add("ppi_mean", "ppi_se", se)
add("ppi_mean", "ppi_lo", ci$low)
add("ppi_mean", "ppi_hi", ci$high)

for (method in c("unadjusted", "ancova", "ppi_single")) {
  est <- switch(method,
    unadjusted = ate_unadjusted(y_ppi, t_ppi),
    ancova = ate_ancova(y_ppi, t_ppi, matrix(f_ppi, ncol = 1)),
    ppi_single = ate_ppi_single_lambda(y_ppi, t_ppi, f_ppi)
  )
  add("ate", paste0("ate_", method, "_estimate"), est$estimate)
  add("ate", paste0("ate_", method, "_se"), est$se)
  add("ate", paste0("ate_", method, "_ci_low"), est$ci_low)
  add("ate", paste0("ate_", method, "_ci_high"), est$ci_high)
  add("ate", paste0("ate_", method, "_lam"), est$lam)
}

# ---------------------------------------------------------------- PROCOVA
procova <- utils::read.csv(
  file.path(fixtures, "procova_cases.csv"), check.names = FALSE
)
for (i in seq_len(nrow(procova))) {
  case <- procova$case[i]
  s0 <- procova$sigma0[i]
  s1 <- procova$sigma1[i]
  r0 <- procova$rho0[i]
  r1 <- procova$rho1[i]
  pi0 <- procova$pi0[i]
  effect <- procova$effect[i]
  alpha <- procova$alpha[i]
  power <- procova$power[i]
  n_total <- procova$n_total[i]

  v <- procova_variance(s0, s1, r0, r1, pi0)
  unadj <- unadjusted_min_variance(s0, s1)
  alloc <- optimal_allocation(s0, s1, r0, r1)
  add("procova", paste0("var_", case), v)
  add("procova", paste0("unadj_min_", case), unadj)
  add("procova", paste0("alloc_pi0_", case), alloc$pi0)
  add("procova", paste0("alloc_variance_", case), alloc$variance)
  add("procova", paste0("alloc_ratio_", case), alloc$variance_ratio)
  add("procova", paste0("alloc_reduction_", case), alloc$variance_reduction)
  add("procova", paste0("alloc_unadj_pi0_", case), alloc$unadjusted_pi0)
  add("procova", paste0("alloc_success_", case), as.integer(alloc$optimization_success))
  add("procova", paste0("required_n_", case),
      required_n_total(v, effect, alpha, power))
  add("procova", paste0("sample_reduction_", case),
      sample_size_reduction(unadj, v))
  add("procova", paste0("control_arm_", case),
      control_arm_size(n_total, alloc$pi0))
}

# ---------------------------------------------------------------- analysis
trial <- analyze_trial_csv(file.path(fixtures, "analysis_trial.csv"))
tab <- analysis_table(trial)
add("analysis", "analysis_score_r2_control", trial$score_r2_control)
add("analysis", "analysis_outcome_sd", trial$outcome_sd)
for (i in seq_len(nrow(tab))) {
  for (field in c("estimate", "se", "ci_low", "ci_high", "ci_width", "p_value", "lam")) {
    add("analysis", paste0("analysis_", tab$method[i], "_", field), tab[[field]][i])
  }
}

out <- do.call(rbind, rows)
utils::write.csv(out, out_path, row.names = FALSE, quote = FALSE)
cat(sprintf("wrote %d R parity values\n", nrow(out)))

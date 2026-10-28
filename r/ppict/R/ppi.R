# PPI++ means and the two-arm treatment-effect estimators.
#
# Conventions follow src/ppict/ppi.py: the mean estimator uses the reference
# implementation's biased covariance and ddof=0 interval variance; the two-arm
# estimators use ddof=1 sample variances and normal quantiles.

.sd0 <- function(x) {
  if (!length(x)) return(0)
  sqrt(mean((x - mean(x))^2))
}

.cov_biased <- function(a, b) mean((a - mean(a)) * (b - mean(b)))

#' Standard normal quantile
z_quantile <- function(p) stats::qnorm(p)

#' Plug-in tuning value for the PPI++ mean estimator
tuned_lambda_mean <- function(y, f_labeled, f_unlabeled,
                              lam_clip = c(0, 1)) {
  y <- as.numeric(y)
  f_labeled <- as.numeric(f_labeled)
  f_unlabeled <- as.numeric(f_unlabeled)
  n <- length(y)
  N <- length(f_unlabeled)
  if (n < 2) return(0)

  yc <- y - mean(y)
  fc <- f_labeled - mean(f_labeled)
  cov <- sum(yc * fc) / n
  pooled <- c(f_labeled, f_unlabeled)
  if (length(pooled) < 2) return(0)
  var_pooled <- stats::var(pooled)
  if (!is.finite(var_pooled) || var_pooled <= 0) return(0)

  lam <- cov / ((1 + n / max(N, 1)) * var_pooled)
  if (!is.null(lam_clip)) lam <- min(max(lam, lam_clip[1]), lam_clip[2])
  as.numeric(lam)
}

#' PPI++ point estimate of a mean
ppi_mean_pointestimate <- function(y, f_labeled, f_unlabeled, lam = NULL,
                                   lam_clip = c(0, 1)) {
  y <- as.numeric(y)
  f_labeled <- as.numeric(f_labeled)
  f_unlabeled <- as.numeric(f_unlabeled)
  if (is.null(lam)) {
    lam <- tuned_lambda_mean(y, f_labeled, f_unlabeled, lam_clip = lam_clip)
  }
  as.numeric(lam * mean(f_unlabeled) + mean(y - lam * f_labeled))
}

#' Standard error of the PPI++ mean, treating lambda as fixed
ppi_mean_se <- function(y, f_labeled, f_unlabeled, lam) {
  y <- as.numeric(y)
  f_labeled <- as.numeric(f_labeled)
  f_unlabeled <- as.numeric(f_unlabeled)
  n <- length(y)
  N <- length(f_unlabeled)
  rect <- if (n) .sd0(y - lam * f_labeled) / sqrt(n) else NaN
  imp <- if (N) .sd0(lam * f_unlabeled) / sqrt(N) else 0
  sqrt(rect^2 + imp^2)
}

#' Two-sided normal-approximation PPI++ interval
ppi_mean_ci <- function(y, f_labeled, f_unlabeled, lam = NULL, alpha = 0.05,
                        lam_clip = c(0, 1)) {
  if (is.null(lam)) {
    lam <- tuned_lambda_mean(y, f_labeled, f_unlabeled, lam_clip = lam_clip)
  }
  point <- ppi_mean_pointestimate(y, f_labeled, f_unlabeled, lam = lam)
  se <- ppi_mean_se(y, f_labeled, f_unlabeled, lam = lam)
  half <- z_quantile(1 - alpha / 2) * se
  list(low = point - half, high = point + half, lam = as.numeric(lam))
}

#' Two-sided normal p-value for a PPI++ mean
ppi_mean_pvalue <- function(y, f_labeled, f_unlabeled, null = 0, lam = NULL,
                            lam_clip = c(0, 1)) {
  if (is.null(lam)) {
    lam <- tuned_lambda_mean(y, f_labeled, f_unlabeled, lam_clip = lam_clip)
  }
  point <- ppi_mean_pointestimate(y, f_labeled, f_unlabeled, lam = lam)
  se <- ppi_mean_se(y, f_labeled, f_unlabeled, lam = lam)
  if (se <= 0 || !is.finite(se)) return(1)
  2 * stats::pnorm(-abs((point - null) / se))
}

.ate_result <- function(method, estimate, se, alpha = 0.05, lam = NaN,
                        n_treated = 0L, n_control = 0L, n_parameters = 0L) {
  z <- z_quantile(1 - alpha / 2)
  p <- if (!is.finite(se) || se <= 0) 1 else 2 * stats::pnorm(-abs(estimate / se))
  out <- list(
    method = method,
    estimate = as.numeric(estimate),
    se = as.numeric(se),
    alpha = alpha,
    lam = as.numeric(lam),
    n_treated = as.integer(n_treated),
    n_control = as.integer(n_control),
    n_parameters = as.integer(n_parameters),
    ci_low = as.numeric(estimate - z * se),
    ci_high = as.numeric(estimate + z * se),
    ci_width = as.numeric(2 * z * se),
    p_value = as.numeric(p),
    rejects = p < alpha
  )
  class(out) <- "ppict_ate"
  out
}

.arm_stats <- function(y, t, w) {
  m <- t == w
  n <- sum(m)
  if (n < 2) return(list(n = n, mean = if (n) mean(y[m]) else NaN, var = NaN))
  list(n = n, mean = mean(y[m]), var = stats::var(y[m]))
}

.pinv <- function(M) {
  s <- svd(M)
  tol <- max(dim(M)) * max(s$d) * .Machine$double.eps
  d <- ifelse(s$d > tol, 1 / s$d, 0)
  s$v %*% (d * t(s$u))
}

.ols_hc0 <- function(A, y, idx) {
  A <- as.matrix(A)
  y <- as.numeric(y)
  beta <- as.vector(.pinv(A) %*% y)
  resid <- y - as.vector(A %*% beta)
  bread <- .pinv(crossprod(A))
  meat <- crossprod(A, A * as.vector(resid^2))
  cov <- bread %*% meat %*% bread
  var <- cov[idx, idx]
  se <- if (var > 0 && is.finite(var)) sqrt(var) else NaN
  list(beta = beta, se = se)
}

#' Difference in means, the reference estimator
ate_unadjusted <- function(y, t, alpha = 0.05) {
  y <- as.numeric(y)
  t <- as.integer(t)
  s1 <- .arm_stats(y, t, 1L)
  s0 <- .arm_stats(y, t, 0L)
  se <- sqrt(s1$var / s1$n + s0$var / s0$n)
  .ate_result(
    "unadjusted", s1$mean - s0$mean, se, alpha,
    n_treated = s1$n, n_control = s0$n, n_parameters = 2L
  )
}

#' Linear adjustment for supplied covariates, treatment coefficient reported
ate_ancova <- function(y, t, covariates, alpha = 0.05, method = "ancova") {
  y <- as.numeric(y)
  t <- as.integer(t)
  Z <- as.matrix(covariates)
  if (nrow(Z) != length(y)) Z <- t(Z)
  A <- cbind(1, t, Z)
  fit <- .ols_hc0(A, y, idx = 2L)
  .ate_result(
    method, fit$beta[2], fit$se, alpha,
    n_treated = sum(t == 1), n_control = sum(t == 0),
    n_parameters = ncol(A)
  )
}

# --------------------------------------------------------------------------
# PROCOVA allocation and sample size
# --------------------------------------------------------------------------

#' Schuler et al. adjusted-variance bound as a per-patient constant
procova_variance <- function(sigma0, sigma1, rho0, rho1, pi0) {
  pi1 <- 1 - pi0
  if (pi0 <= 0 || pi1 <= 0) return(Inf)
  base <- sigma0^2 / pi0 + sigma1^2 / pi1
  gain <- pi0 * pi1 * (rho1 * sigma1 / pi1 + rho0 * sigma0 / pi0)^2
  max(base - gain, 0)
}

#' Minimum unadjusted variance over the allocation, (sigma0 + sigma1)^2
unadjusted_min_variance <- function(sigma0, sigma1) (sigma0 + sigma1)^2

.allocation_result <- function(pi0, variance, unadjusted_variance,
                               unadjusted_pi0, optimization_success) {
  out <- list(
    pi0 = as.numeric(pi0),
    pi1 = as.numeric(1 - pi0),
    variance = as.numeric(variance),
    unadjusted_variance = as.numeric(unadjusted_variance),
    unadjusted_pi0 = as.numeric(unadjusted_pi0),
    optimization_success = isTRUE(optimization_success)
  )
  out$variance_ratio <- if (out$unadjusted_variance > 0) {
    out$variance / out$unadjusted_variance
  } else {
    NaN
  }
  out$variance_reduction <- 1 - out$variance_ratio
  class(out) <- "ppict_allocation"
  out
}

#' Minimise the adjusted variance bound over the control allocation
optimal_allocation <- function(sigma0, sigma1, rho0, rho1) {
  unadj_pi0 <- if ((sigma0 + sigma1) > 0) sigma0 / (sigma0 + sigma1) else 0.5

  # The symmetric case has a known minimiser at 1/2. Returning it directly
  # avoids a one-patient flip in control_arm_size() when the numeric optimiser
  # stops a few ulps either side of the boundary.
  if (sigma0 == sigma1 && rho0 == rho1) {
    return(.allocation_result(
      pi0 = 0.5,
      variance = procova_variance(sigma0, sigma1, rho0, rho1, 0.5),
      unadjusted_variance = unadjusted_min_variance(sigma0, sigma1),
      unadjusted_pi0 = unadj_pi0,
      optimization_success = TRUE
    ))
  }

  objective <- function(pi0) {
    procova_variance(sigma0, sigma1, rho0, rho1, pi0)
  }
  res <- stats::optimize(objective, interval = c(1e-6, 1 - 1e-6), tol = 1e-12)
  .allocation_result(
    pi0 = res$minimum,
    variance = res$objective,
    unadjusted_variance = unadjusted_min_variance(sigma0, sigma1),
    unadjusted_pi0 = unadj_pi0,
    optimization_success = is.finite(res$objective)
  )
}

#' Total patients needed for a two-sided normal test of the effect
required_n_total <- function(variance_constant, effect, alpha = 0.05,
                             power = 0.80) {
  if (effect == 0 || !is.finite(effect)) return(Inf)
  z_a <- z_quantile(1 - alpha / 2)
  z_b <- z_quantile(power)
  (z_a + z_b)^2 * variance_constant / effect^2
}

#' Fractional reduction in required sample size from the variance ratio
sample_size_reduction <- function(variance_unadjusted, variance_adjusted) {
  if (variance_unadjusted <= 0) return(NaN)
  1 - variance_adjusted / variance_unadjusted
}

#' Control arm size implied by a total size and an allocation
control_arm_size <- function(n_total, pi0) ceiling(pi0 * n_total)

#' Tuned prediction-powered ATE with one shared tuning value
ate_ppi_single_lambda <- function(y, t, f, alpha = 0.05, lam = NULL,
                                  lam_clip = NULL) {
  y <- as.numeric(y)
  t <- as.integer(t)
  f <- as.numeric(f)
  m1 <- t == 1
  m0 <- t == 0
  n1 <- sum(m1)
  n0 <- sum(m0)

  if (is.null(lam)) {
    s1 <- .cov_biased(y[m1], f[m1]) * n1
    s0 <- .cov_biased(y[m0], f[m0]) * n0
    q1 <- mean((f[m1] - mean(f[m1]))^2) * n1
    q0 <- mean((f[m0] - mean(f[m0]))^2) * n0
    num <- s1 / max(n1, 1)^2 + s0 / max(n0, 1)^2
    den <- q1 / max(n1, 1)^2 + q0 / max(n0, 1)^2
    lam <- if (den > 0) num / den else 0
    if (!is.null(lam_clip)) lam <- min(max(lam, lam_clip[1]), lam_clip[2])
  }

  est <- mean(y[m1]) - mean(y[m0]) - lam * (mean(f[m1]) - mean(f[m0]))
  var <- stats::var(y[m1] - lam * f[m1]) / max(n1, 1) +
    stats::var(y[m0] - lam * f[m0]) / max(n0, 1)
  .ate_result(
    "ppi_single", est, sqrt(var), alpha, lam = lam,
    n_treated = n1, n_control = n0, n_parameters = 3L
  )
}

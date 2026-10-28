set.seed(11L)
n <- 300L
t <- rep(0:1, each = n / 2)
f <- rnorm(n)
y <- 0.5 * t + 0.8 * f + rnorm(n, sd = 0.6)

plain <- ate_unadjusted(y, t)
m1 <- mean(y[t == 1])
m0 <- mean(y[t == 0])
se_manual <- sqrt(var(y[t == 1]) / sum(t == 1) + var(y[t == 0]) / sum(t == 0))
stopifnot(
  abs(plain$estimate - (m1 - m0)) < 1e-12,
  abs(plain$se - se_manual) < 1e-12,
  plain$ci_low < plain$estimate,
  plain$estimate < plain$ci_high
)

zero <- ate_ppi_single_lambda(y, t, f, lam = 0)
stopifnot(
  abs(zero$estimate - plain$estimate) < 1e-14,
  abs(zero$se - plain$se) < 1e-14
)

adj <- ate_ppi_single_lambda(y, t, f)
A <- cbind(1, t, f)
beta <- as.vector(solve(crossprod(A), crossprod(A, y)))
anc <- ate_ancova(y, t, matrix(f, ncol = 1))
stopifnot(
  abs(adj$estimate - beta[2]) < 1e-9,
  abs(adj$se - anc$se) / anc$se < 0.10,
  adj$ci_width < plain$ci_width
)

shifted <- ate_ppi_single_lambda(y, t, f + 4)
flipped <- ate_ppi_single_lambda(y, t, -f)
stopifnot(
  abs(shifted$lam - adj$lam) < 1e-12,
  abs(shifted$estimate - adj$estimate) < 1e-10,
  abs(flipped$lam + adj$lam) < 1e-12,
  abs(flipped$estimate - adj$estimate) < 1e-10
)

set.seed(12L)
yl <- rnorm(120)
fl <- rnorm(120)
fu <- rnorm(400)
lam_manual <- {
  cov <- mean((yl - mean(yl)) * (fl - mean(fl)))
  cov / ((1 + length(yl) / length(fu)) * var(c(fl, fu)))
}
lam_manual <- min(max(lam_manual, 0), 1)
stopifnot(abs(tuned_lambda_mean(yl, fl, fu) - lam_manual) < 1e-14)

pt <- ppi_mean_pointestimate(yl, fl, fu, lam = 0)
ci <- ppi_mean_ci(yl, fl, fu, lam = 0)
half <- qnorm(0.975) * sqrt(mean((yl - mean(yl))^2)) / sqrt(length(yl))
stopifnot(
  abs(pt - mean(yl)) < 1e-14,
  abs(ci$low - (mean(yl) - half)) < 1e-12,
  abs(ci$high - (mean(yl) + half)) < 1e-12
)

set.seed(13L)
y2 <- 0.4 * t + 0.9 * f + rnorm(n, sd = 0.5)
good <- ate_ppi_single_lambda(y2, t, f)
stopifnot(abs(good$lam) > 0.1, good$ci_width < ate_unadjusted(y2, t)$ci_width)

for (s in c(0.5, 1, 3)) {
  for (r in c(0, 0.3, 0.6, 0.9)) {
    v <- procova_variance(s, s, r, r, 0.5)
    stopifnot(abs(v - 4 * s^2 * (1 - r^2)) < 1e-12)
  }
}
stopifnot(
  abs(procova_variance(1, 1, 0, 0, 0.5) - 4) < 1e-12,
  is.infinite(procova_variance(1, 1, 0.3, 0.3, 0)),
  is.infinite(procova_variance(1, 1, 0.3, 0.3, 1))
)

for (pars in list(c(1, 1), c(1, 2), c(0.7, 1.9))) {
  s0 <- pars[1]
  s1 <- pars[2]
  stopifnot(abs(unadjusted_min_variance(s0, s1) - (s0 + s1)^2) < 1e-12)
  pi0 <- s0 / (s0 + s1)
  stopifnot(abs(procova_variance(s0, s1, 0, 0, pi0) - (s0 + s1)^2) < 1e-12)
}

balanced <- optimal_allocation(1, 1, 0.4, 0.4)
stopifnot(
  abs(balanced$pi0 - 0.5) < 1e-3,
  abs(balanced$variance_ratio - (1 - 0.4^2)) < 1e-6,
  abs(balanced$variance_reduction - 0.16) < 1e-5,
  abs(balanced$unadjusted_pi0 - 0.5) < 1e-12
)

noisy <- optimal_allocation(2, 1, 0, 0)
stopifnot(
  abs(noisy$pi0 - 2 / 3) < 1e-3,
  abs(noisy$variance_ratio - 1) < 1e-6
)

for (pars in list(c(1, 1, .3, .3), c(1.5, 1, .5, .2),
                  c(1, 1, .8, .1), c(0.5, 2, .4, .4))) {
  alloc <- optimal_allocation(pars[1], pars[2], pars[3], pars[4])
  stopifnot(alloc$variance <= alloc$unadjusted_variance + 1e-9)
}

sigma <- 1.3
effect <- 0.4
z_a <- qnorm(0.975)
z_b <- qnorm(0.80)
textbook <- 4 * (z_a + z_b)^2 * sigma^2 / effect^2
stopifnot(abs(required_n_total(4 * sigma^2, effect, 0.05, 0.80) - textbook) < 1e-12)
stopifnot(
  required_n_total(4, 0.2) > required_n_total(4, 0.5),
  required_n_total(4, 0.4, power = 0.90) > required_n_total(4, 0.4, power = 0.80),
  is.infinite(required_n_total(4, 0)),
  abs(sample_size_reduction(4, 3) - 0.25) < 1e-15,
  control_arm_size(100, 0.5) == 50
)

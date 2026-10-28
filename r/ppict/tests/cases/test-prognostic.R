y <- c(1, 2, 3, 4)
stopifnot(
  abs(r2_score(y, y) - 1) < 1e-15,
  abs(r2_score(y, rep(mean(y), 4))) < 1e-15,
  is.nan(r2_score(rep(1, 4), y))
)

set.seed(7L)
X <- matrix(rnorm(160 * 10), 160, 10)
beta <- c(1, 0.8, 0.6, rep(0, 7))
y <- as.vector(X %*% beta) + rnorm(160, sd = 0.5)
groups <- list(1:3, 4:6, 7:10)

score <- fit_prognostic(X[1:100, , drop = FALSE], y[1:100], groups = groups, lam = 0.01)
p1 <- predict(score, X[101:160, , drop = FALSE])
p2 <- predict(score, X[101:160, , drop = FALSE])
stopifnot(
  identical(p1, p2),
  length(p1) == 60L,
  is.finite(score$train_r2),
  length(score$feature_idx) >= 1L
)

set.seed(8L)
Xc <- matrix(rnorm(120 * 10), 120, 10)
yc <- as.vector(Xc %*% beta) + rnorm(120, sd = 0.5)
fold <- rep(1:3, each = 40)
a <- cross_fitted_scores(Xc, yc, n_folds = 3L, groups = groups,
                         fold_assignment = fold, lam = 0.005)
yc2 <- yc
yc2[fold == 1] <- yc2[fold == 1] + 100
b <- cross_fitted_scores(Xc, yc2, n_folds = 3L, groups = groups,
                         fold_assignment = fold, lam = 0.005)
stopifnot(
  all(is.finite(a$oof)),
  identical(a$fold_assignment, fold),
  max(abs(a$oof[fold == 1] - b$oof[fold == 1])) < 1e-12,
  any(abs(a$oof[fold != 1] - b$oof[fold != 1]) > 1e-8)
)

set.seed(9L)
Xe <- matrix(rnorm(200 * 12), 200, 12)
ye <- as.vector(Xe %*% c(1, 1, 0.5, rep(0, 9))) + rnorm(200, sd = 0.5)
s <- fit_prognostic(
  Xe[1:120, , drop = FALSE], ye[1:120],
  groups = list(1:3, 4:6, 7:12), lam = 0.01,
  X_val = Xe[121:200, , drop = FALSE], y_val = ye[121:200]
)
stopifnot(
  length(s$feature_idx) >= 1L,
  is.finite(s$oos_r2),
  s$oos_r2 > 0.2
)

X0 <- matrix(0, 60, 5)
set.seed(10L)
y0 <- rnorm(60)
s0 <- fit_prognostic(X0, y0, lam = 1)
stopifnot(
  length(s0$feature_idx) == 0L,
  max(abs(predict(s0, matrix(1, 4, 5)) - s0$intercept)) < 1e-15
)

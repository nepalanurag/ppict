orthonormal_design <- function(m, p, seed) {
  set.seed(seed)
  Q <- qr.Q(qr(matrix(rnorm(m * p), nrow = m, ncol = p)))
  Q * sqrt(m)
}

group_soft_expected <- function(z, lam, weights, groups) {
  out <- numeric(length(z))
  for (j in seq_along(groups)) {
    g <- groups[[j]]
    nrm <- sqrt(sum(z[g]^2))
    if (nrm > lam * weights[j]) {
      out[g] <- z[g] * (1 - lam * weights[j] / nrm)
    }
  }
  out
}

m <- 60L
p <- 8L
X <- orthonormal_design(m, p, 1L)
set.seed(2L)
y <- rnorm(m)
groups <- list(c(1L, 2L), 3L, c(4L, 5L, 6L), c(7L, 8L))
weights <- c(1, 1.5, 0.5, 2)

for (lam in c(0.01, 0.05, 0.2)) {
  got <- group_lasso_fista(
    X, y, groups, lam, weights,
    max_iter = 5000L, tol = 1e-14
  )
  expected <- group_soft_expected(crossprod(X, y) / m, lam, weights, groups)
  stopifnot(max(abs(got - expected)) < 1e-6)
}

gx <- vapply(groups, function(g) sqrt(sum(crossprod(X[, g, drop = FALSE], y)^2)), numeric(1))
lam_max <- max(gx / (m * weights))
coef_hi <- group_lasso_fista(
  X, y, groups, lam_max * 1.0001, weights,
  max_iter = 2000L, tol = 1e-12
)
stopifnot(max(abs(coef_hi)) < 1e-8)

set.seed(4L)
b1 <- rnorm(60)
b2 <- rnorm(60)
b3 <- rnorm(60)
Xb <- cbind(b1, b1 + 0.01 * rnorm(60), b2, b2 + 0.01 * rnorm(60), b3, b3 + 0.01 * rnorm(60))
block_groups <- define_groups(Xb, threshold = 0.6)
stopifnot(
  length(block_groups) == 3L,
  identical(sort(unlist(block_groups, use.names = FALSE)), 1:6)
)

set.seed(5L)
Xs <- matrix(rnorm(200 * 10), 200, 10)
ys <- 3 * Xs[, 2] + rnorm(200)
screened <- screen_features(Xs, ys, top_k = 3L)
stopifnot(length(screened$index) == 3L, 2L %in% screened$index)

set.seed(6L)
Xc <- matrix(rnorm(60 * 6), 60, 6)
yc <- Xc[, 1] - Xc[, 2] + rnorm(60)
gc <- define_groups(Xc, threshold = 0.6)
wc <- rep(1, length(gc))
grid <- c(0.5, 0.2, 0.1, 0.05, 0.02)
perms <- t(replicate(5L, sample.int(60)))
counts <- perm_null_group_counts(
  Xc, yc, gc, grid, wc,
  n_perm = 5L, permutations = perms
)
stopifnot(
  identical(dim(counts), c(5L, length(grid))),
  all(counts >= 0L),
  all(counts <= length(gc))
)

cal <- calibrate_lambda(
  Xc, yc, gc, wc,
  q = 0.2, n_perm = 5L, lam_grid = grid, permutations = perms
)
mean_counts <- colMeans(counts)
budget <- 0.2 * length(gc)
ok <- which(mean_counts <= budget)
expected_lam <- if (length(ok)) grid[ok[which.min(grid[ok])]] else max(grid)
stopifnot(
  isTRUE(all.equal(cal$mean_counts, mean_counts)),
  isTRUE(all.equal(cal$lam, expected_lam))
)

path <- group_lasso_path(
  Xc, yc, gc, c(0.2, 0.05, 0.02), wc,
  max_iter = 3000L, tol = 1e-13
)
individual <- vapply(c(0.2, 0.05, 0.02), function(l) {
  group_lasso_fista(Xc, yc, gc, l, wc, max_iter = 3000L, tol = 1e-13)
}, numeric(ncol(Xc)))
stopifnot(max(abs(path - t(individual))) < 1e-5)

selection <- select_groups(Xc, yc, groups = gc, weights = wc, lam = 0.02)
stopifnot(
  selection$n_selected >= 1L,
  identical(selection$selected_idx, sort(unique(selection$selected_idx))),
  length(selection$selected_idx) == length(unique(selection$selected_idx)),
  selection$n_selected == length(selection$selected)
)

selected_counts <- vapply(c(0.5, 0.2, 0.1, 0.05, 0.02, 0.005), function(l) {
  length(select_groups(Xc, yc, groups = gc, weights = wc, lam = l)$selected)
}, numeric(1))
stopifnot(selected_counts[1] <= selected_counts[length(selected_counts)])

stopifnot(
  jaccard(integer(0), integer(0)) == 1,
  abs(jaccard(c(1, 2), c(2, 3)) - 1 / 3) < 1e-15,
  length(pairwise_jaccard(list(c(1L), c(2L), c(1L, 2L)))) == 3L
)

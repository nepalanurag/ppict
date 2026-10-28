# Permutation-assisted group lasso.
#
# This is a direct port of the deterministic parts of src/ppict/group_select.py.
# The permutation draws themselves are not bit-identical to NumPy's PCG64, so
# every function here accepts an explicit permutation matrix when the caller
# needs cross-language reproducibility. With permutations = NULL the R RNG is
# used and the result is reproducible only after set.seed().

.group_norm <- function(v) sqrt(sum(v^2))

.spectral_norm_sq <- function(X, n_iter = 100L, tol = 1e-9) {
  p <- ncol(X)
  if (p == 0L) return(0)
  v <- rep(1 / sqrt(p), p)
  prev <- NULL
  for (i in seq_len(n_iter)) {
    u <- as.vector(X %*% v)
    v_new <- as.vector(crossprod(X, u))
    nrm <- sqrt(sum(v_new^2))
    if (nrm == 0) return(0)
    v_new <- v_new / nrm
    if (!is.null(prev) && abs(sum(v_new * prev) - 1) < tol) {
      v <- v_new
      break
    }
    prev <- v
    v <- v_new
  }
  sum((as.vector(X %*% v))^2)
}

.group_soft_threshold <- function(v, thr, groups) {
  out <- numeric(length(v))
  for (j in seq_along(groups)) {
    g <- groups[[j]]
    vg <- v[g]
    nrm <- .group_norm(vg)
    if (nrm <= thr[j] || nrm == 0) next
    out[g] <- vg * (1 - thr[j] / nrm)
  }
  out
}

.group_objective <- function(X, y, b, lam, weights, groups, m) {
  r <- y - as.vector(X %*% b)
  quad <- sum(r^2) / (2 * m)
  pen <- sum(weights * vapply(groups, function(g) .group_norm(b[g]), numeric(1)))
  quad + lam * pen
}

.nonzero_groups <- function(b, groups, tol = 1e-12) {
  which(vapply(groups, function(g) .group_norm(b[g]), numeric(1)) > tol)
}

.lam_upper_bound <- function(X, y, groups, weights) {
  m <- nrow(X)
  y_norm <- sqrt(sum(y^2))
  vals <- vapply(seq_along(groups), function(j) {
    Xg <- X[, groups[[j]], drop = FALSE]
    sqrt(sum(Xg^2)) * y_norm / (m * max(weights[j], 1e-12))
  }, numeric(1))
  vals <- vals[is.finite(vals)]
  if (!length(vals)) 1 else max(vals)
}

.univariate_z <- function(X, y) {
  n <- nrow(X)
  Xc <- sweep(X, 2, colMeans(X), "-")
  yc <- y - mean(y)
  sxx <- colSums(Xc^2)
  beta <- as.vector(crossprod(Xc, yc)) / sxx
  resid <- yc - sweep(Xc, 2, beta, "*")
  dof <- max(n - 2, 1)
  s2 <- colSums(resid^2) / dof
  se <- sqrt(s2 / sxx)
  z <- beta / se
  z[!is.finite(z)] <- 0
  z
}

#' Group features by absolute correlation (connected components)
#'
#' Two features share a group when their absolute Pearson correlation is at
#' least `threshold`. Groups are ordered by their smallest feature index.
define_groups <- function(X, threshold = 0.6, max_features = 3000L) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  p <- ncol(X)
  if (p == 0L) return(list())
  if (p > max_features) {
    stop(sprintf(
      "define_groups received %d features, above max_features=%d. Screen features first.",
      p, max_features
    ))
  }
  if (p == 1L) return(list(1L))

  C <- suppressWarnings(stats::cor(X))
  C[!is.finite(C)] <- 0
  adj <- abs(C) >= threshold
  diag(adj) <- TRUE

  state <- new.env(parent = emptyenv())
  state$parent <- seq_len(p)
  find <- function(i) {
    while (state$parent[i] != i) {
      state$parent[i] <- state$parent[state$parent[i]]
      i <- state$parent[i]
    }
    i
  }
  for (i in seq_len(p)) {
    for (j in seq_len(p)) {
      if (i < j && adj[i, j]) {
        ri <- find(i)
        rj <- find(j)
        if (ri != rj) state$parent[rj] <- ri
      }
    }
  }
  labels <- vapply(seq_len(p), find, integer(1))
  groups <- lapply(sort(unique(labels)), function(k) which(labels == k))
  groups[order(vapply(groups, min, integer(1)))]
}

#' Keep the top features by absolute marginal correlation with y
screen_features <- function(X, y, top_k = 2000L) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  y <- as.numeric(y)
  if (nrow(X) != length(y)) stop("X and y must have the same number of rows")
  p <- ncol(X)
  if (p == 0L) return(list(index = integer(0), r = numeric(0)))

  Xc <- sweep(X, 2, colMeans(X), "-")
  yc <- y - mean(y)
  denom <- sqrt(colSums(Xc^2) * sum(yc^2))
  r <- abs(as.vector(crossprod(Xc, yc))) / denom
  r[!is.finite(r)] <- 0

  k <- min(as.integer(top_k), p)
  ord <- order(-r, seq_along(r), method = "radix")
  list(index = sort(ord[seq_len(k)]), r = r)
}

#' Adaptive group penalties from univariate z-scores
group_pen_from_z <- function(z, groups, gamma = 1, eps = 1e-8, normalize = TRUE) {
  raw <- vapply(groups, function(g) {
    sqrt(sum(z[g]^2) / max(length(g), 1))
  }, numeric(1))
  w <- 1 / (raw^gamma + eps)
  if (normalize && length(w) && mean(w) > 0) w <- w / mean(w)
  w
}

#' Group lasso by FISTA with exact group soft-thresholding
group_lasso_fista <- function(X, y, groups, lam, weights = NULL,
                              max_iter = 500L, tol = 1e-9, b_init = NULL) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  y <- as.numeric(y)
  m <- nrow(X)
  p <- ncol(X)
  groups <- lapply(groups, function(g) as.integer(g))
  if (is.null(weights)) weights <- rep(1, length(groups))
  weights <- as.numeric(weights)
  if (length(weights) != length(groups)) stop("weights must have one entry per group")

  L <- .spectral_norm_sq(X) / m
  if (L <= 0) return(numeric(p))
  step <- 1 / L

  b <- if (is.null(b_init)) numeric(p) else as.numeric(b_init)
  z <- b
  t <- 1
  prev_obj <- .group_objective(X, y, b, lam, weights, groups, m)

  for (iter in seq_len(max_iter)) {
    grad <- -as.vector(crossprod(X, y - as.vector(X %*% z))) / m
    v <- z - step * grad
    b_new <- .group_soft_threshold(v, step * lam * weights, groups)

    t_new <- 0.5 * (1 + sqrt(1 + 4 * t * t))
    z <- b_new + ((t - 1) / t_new) * (b_new - b)
    t <- t_new

    obj <- .group_objective(X, y, b_new, lam, weights, groups, m)
    b <- b_new
    if (abs(prev_obj - obj) <= tol * max(1, abs(prev_obj))) break
    prev_obj <- obj
  }
  b
}

#' Solve the group lasso at every penalty with warm starts
group_lasso_path <- function(X, y, groups, lam_grid, weights = NULL,
                             max_iter = 500L, tol = 1e-9) {
  lam_grid <- as.numeric(lam_grid)
  out <- matrix(0, nrow = length(lam_grid), ncol = ncol(X))
  b <- NULL
  for (pos in order(-lam_grid)) {
    b <- group_lasso_fista(
      X, y, groups, lam_grid[pos], weights,
      max_iter = max_iter, tol = tol, b_init = b
    )
    out[pos, ] <- b
  }
  out
}

#' Selected-group counts under row-permuted outcomes
#'
#' `permutations` is an optional integer matrix with one row per permutation
#' and one column per observation. Supplying it makes the calibration exactly
#' reproducible across languages; with NULL, the R RNG is used.
perm_null_group_counts <- function(X, y, groups, lam_grid, weights = NULL,
                                   n_perm = 25L, permutations = NULL,
                                   max_iter = 300L, tol = 1e-8) {
  X <- as.matrix(X)
  y <- as.numeric(y)
  lam_grid <- as.numeric(lam_grid)
  m <- nrow(X)
  if (is.null(weights)) weights <- rep(1, length(groups))
  if (is.null(permutations)) {
    permutations <- t(replicate(n_perm, sample.int(m)))
  } else {
    permutations <- as.matrix(permutations)
    if (ncol(permutations) != m) stop("permutations must have one column per observation")
    if (nrow(permutations) < n_perm) stop("not enough permutation rows")
    permutations <- permutations[seq_len(n_perm), , drop = FALSE]
  }

  counts <- matrix(0L, nrow = n_perm, ncol = length(lam_grid))
  for (b in seq_len(n_perm)) {
    path <- group_lasso_path(
      X, y[permutations[b, ]], groups, lam_grid, weights,
      max_iter = max_iter, tol = tol
    )
    for (k in seq_along(lam_grid)) {
      counts[b, k] <- length(.nonzero_groups(path[k, ], groups))
    }
  }
  counts
}

#' Smallest penalty whose permutation-null group count stays within budget
calibrate_lambda <- function(X, y, groups, weights = NULL, q = 0.10,
                             n_perm = 25L, lam_grid = NULL, n_grid = 12L,
                             permutations = NULL, max_iter = 300L, tol = 1e-8,
                             allow_all_zero = TRUE) {
  if (is.null(weights)) weights <- rep(1, length(groups))
  if (is.null(lam_grid)) {
    hi <- .lam_upper_bound(X, y, groups, weights)
    lo <- max(hi * 1e-5, 1e-15)
    lam_grid <- exp(seq(log(max(hi, 1e-12)), log(lo), length.out = n_grid))
  }
  lam_grid <- as.numeric(lam_grid)

  counts <- perm_null_group_counts(
    X, y, groups, lam_grid, weights,
    n_perm = n_perm, permutations = permutations,
    max_iter = max_iter, tol = tol
  )
  mean_counts <- colMeans(counts)
  budget <- q * length(groups)

  ok <- which(mean_counts <= budget)
  if (!length(ok)) {
    if (!allow_all_zero) {
      stop(sprintf(
        "no penalty on the grid met the permutation budget; smallest mean null count was %.3f against budget %.3f",
        min(mean_counts), budget
      ))
    }
    return(list(lam = max(lam_grid), lam_grid = lam_grid, mean_counts = mean_counts))
  }
  chosen <- ok[which.min(lam_grid[ok])]
  list(lam = lam_grid[chosen], lam_grid = lam_grid, mean_counts = mean_counts)
}

#' Calibrate the penalty, then return the selected groups and coefficients
select_groups <- function(X, y, groups = NULL, weights = NULL, q = 0.10,
                          n_perm = 25L, lam = NULL, lam_grid = NULL,
                          group_threshold = 0.6, permutations = NULL,
                          max_iter = 300L, tol = 1e-8) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  y <- as.numeric(y)
  if (is.null(groups)) groups <- define_groups(X, threshold = group_threshold)
  groups <- lapply(groups, as.integer)

  if (is.null(weights)) {
    weights <- group_pen_from_z(.univariate_z(X, y), groups)
  }
  weights <- as.numeric(weights)

  if (is.null(lam)) {
    cal <- calibrate_lambda(
      X, y, groups, weights, q = q, n_perm = n_perm,
      lam_grid = lam_grid, permutations = permutations,
      max_iter = max_iter, tol = tol
    )
    lam <- cal$lam
    grid <- cal$lam_grid
    mean_counts <- cal$mean_counts
  } else {
    grid <- as.numeric(lam)
    mean_counts <- 0
  }

  coef <- group_lasso_fista(
    X, y, groups, as.numeric(lam), weights,
    max_iter = max_iter, tol = tol
  )
  selected <- .nonzero_groups(coef, groups)
  out <- list(
    groups = groups,
    weights = weights,
    lam = as.numeric(lam),
    coef = coef,
    selected = selected,
    lam_grid = grid,
    null_mean_counts = mean_counts,
    n_perm = as.integer(n_perm)
  )
  out$selected_idx <- selected_features(groups, selected)
  out$n_selected <- length(selected)
  class(out) <- "ppict_selection"
  out
}

#' Sorted feature indices covered by the selected groups
selected_features <- function(groups, selected) {
  if (!length(selected)) return(integer(0))
  sort(unique(as.integer(unlist(groups[selected], use.names = FALSE))))
}

#' Jaccard index of two index collections; two empty sets score 1
jaccard <- function(a, b) {
  sa <- unique(as.integer(a))
  sb <- unique(as.integer(b))
  u <- union(sa, sb)
  if (!length(u)) return(1)
  length(intersect(sa, sb)) / length(u)
}

#' All pairwise Jaccard indices, flattened
pairwise_jaccard <- function(sets) {
  out <- numeric(0)
  if (length(sets) < 2) return(out)
  for (i in seq_len(length(sets) - 1)) {
    for (j in (i + 1):length(sets)) {
      out <- c(out, jaccard(sets[[i]], sets[[j]]))
    }
  }
  out
}

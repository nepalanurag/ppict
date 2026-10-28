# Prognostic score construction.
#
# The score is fitted on control outcomes, then refit by least squares on the
# selected features so that it is on the outcome scale. Predictions are frozen:
# they use only the selected columns and coefficients stored at fit time.

#' Coefficient of determination; NaN when the outcome has no variance
r2_score <- function(y_true, y_pred) {
  y_true <- as.numeric(y_true)
  y_pred <- as.numeric(y_pred)
  ss_tot <- sum((y_true - mean(y_true))^2)
  if (ss_tot <= 0) return(NaN)
  ss_res <- sum((y_true - y_pred)^2)
  1 - ss_res / ss_tot
}

.refit_ols <- function(X_sel, y) {
  if (ncol(X_sel) == 0L) {
    return(list(coef = numeric(0), intercept = mean(y)))
  }
  A <- cbind(1, X_sel)
  s <- svd(A)
  tol <- max(dim(A)) * max(s$d) * .Machine$double.eps
  d <- ifelse(s$d > tol, 1 / s$d, 0)
  beta <- as.vector(s$v %*% (d * crossprod(s$u, y)))
  list(coef = beta[-1], intercept = beta[1])
}

#' Fit a prognostic score for the control outcome
#'
#' `top_k` screens features before grouping. `permutations` is passed to the
#' selector; supply a shared matrix for cross-language reproducibility.
fit_prognostic <- function(X, y, groups = NULL, weights = NULL, q = 0.10,
                           n_perm = 25L, lam = NULL, group_threshold = 0.6,
                           top_k = NULL, permutations = NULL,
                           max_iter = 300L, tol = 1e-8,
                           X_val = NULL, y_val = NULL) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  y <- as.numeric(y)
  if (nrow(X) != length(y)) stop("X and y must have the same number of rows")

  original_idx <- seq_len(ncol(X))
  Xw <- X
  if (!is.null(top_k) && ncol(X) > top_k) {
    screened <- screen_features(X, y, top_k = top_k)
    original_idx <- screened$index
    Xw <- X[, original_idx, drop = FALSE]
  }

  if (is.null(groups)) groups <- define_groups(Xw, threshold = group_threshold)
  groups <- lapply(groups, as.integer)

  if (is.null(weights)) {
    weights <- group_pen_from_z(.univariate_z(Xw, y), groups)
  }

  sel <- select_groups(
    Xw, y, groups = groups, weights = weights, q = q, n_perm = n_perm,
    lam = lam, permutations = permutations, max_iter = max_iter, tol = tol
  )

  local_idx <- sel$selected_idx
  glob_idx <- if (length(local_idx)) original_idx[local_idx] else integer(0)
  X_sel <- X[, glob_idx, drop = FALSE]
  refit <- .refit_ols(X_sel, y)
  coef <- refit$coef
  intercept <- refit$intercept

  train_pred <- if (length(glob_idx)) {
    as.vector(X_sel %*% coef) + intercept
  } else {
    rep(intercept, nrow(X))
  }

  score <- list(
    coef = as.numeric(coef),
    intercept = as.numeric(intercept),
    feature_idx = as.integer(glob_idx),
    selected_groups = as.integer(sel$selected),
    lam = as.numeric(sel$lam),
    train_r2 = r2_score(y, train_pred),
    oos_r2 = NaN,
    info = list(
      q = as.numeric(q),
      n_perm = as.integer(n_perm),
      n_groups = length(groups),
      group_threshold = as.numeric(group_threshold),
      top_k = if (is.null(top_k)) NULL else as.integer(top_k),
      n_features_considered = ncol(Xw)
    )
  )
  class(score) <- "ppict_score"

  if (!is.null(X_val) && !is.null(y_val)) {
    score$oos_r2 <- r2_score(y_val, predict(score, X_val))
  }
  score
}

#' Apply a frozen prognostic score
predict.ppict_score <- function(object, newdata, ...) {
  X <- as.matrix(newdata)
  storage.mode(X) <- "double"
  if (length(object$feature_idx) == 0L) {
    return(rep(object$intercept, nrow(X)))
  }
  as.vector(X[, object$feature_idx, drop = FALSE] %*% object$coef) + object$intercept
}

#' Explicit prediction helper for callers that prefer a function
predict_score <- function(score, X) predict(score, X)

#' Out-of-fold scores, each from a model that excluded its own fold
#'
#' `fold_assignment` is optional and 1-based. Supplying it makes the split
#' explicit and lets the parity check use the same folds in both languages.
cross_fitted_scores <- function(X, y, n_folds = 5L, groups = NULL,
                                fold_assignment = NULL, permutations = NULL,
                                ...) {
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  y <- as.numeric(y)
  n <- nrow(X)
  n_folds <- as.integer(min(max(2L, n_folds), max(2L, n)))

  if (is.null(fold_assignment)) {
    idx <- sample.int(n)
    fold_assignment <- integer(n)
    fold_assignment[idx] <- cut(seq_len(n), breaks = n_folds, labels = FALSE)
  } else {
    fold_assignment <- as.integer(fold_assignment)
    if (length(fold_assignment) != n) stop("fold_assignment must have one entry per row")
    if (any(!fold_assignment %in% seq_len(n_folds))) {
      stop("fold_assignment values must run from 1 to n_folds")
    }
  }

  oof <- rep(NaN, n)
  models <- vector("list", n_folds)
  selected_sets <- vector("list", n_folds)
  for (k in seq_len(n_folds)) {
    train <- which(fold_assignment != k)
    test <- which(fold_assignment == k)
    if (length(train) < 2L || !length(test)) next
    model <- fit_prognostic(
      X[train, , drop = FALSE], y[train],
      groups = groups, permutations = permutations, ...
    )
    models[[k]] <- model
    selected_sets[[k]] <- model$feature_idx
    oof[test] <- predict(model, X[test, , drop = FALSE])
  }

  jac <- pairwise_jaccard(selected_sets)
  list(
    oof = oof,
    models = models,
    fold_assignment = fold_assignment,
    oof_r2 = r2_score(y, oof),
    fold_jaccard = jac,
    mean_fold_jaccard = if (length(jac)) mean(jac) else NaN
  )
}

# Data in, estimate and interval out.
#
# The accepted CSV shape matches src/ppict/analysis.py: y, t and f, optional
# cohort=trial|external, and comment lines beginning with #. The R port reports
# the shared-tuning analysis only. External rows are parsed so the error can be
# specific, but ppi_per_arm is not ported.

.normalise_name <- function(x) {
  x <- gsub("-", "_", gsub(" ", "_", tolower(trimws(x))))
  x
}

.resolve_col <- function(columns, aliases) {
  hit <- aliases[aliases %in% names(columns)]
  if (!length(hit)) return(NULL)
  unname(columns[[hit[1]]])
}

.parse_num <- function(x) {
  if (length(x) == 0 || is.na(x)) return(NA_real_)
  s <- trimws(as.character(x))
  if (s == "" || tolower(s) %in% c("na", "nan", "null", "none")) return(NA_real_)
  value <- suppressWarnings(as.numeric(s))
  if (!is.finite(value)) NA_real_ else value
}

#' Read and validate a trial CSV
load_trial_dataset <- function(path) {
  if (!file.exists(path)) stop(sprintf("no file at %s", path))
  lines <- readLines(path, warn = FALSE)
  notes <- sub("^\\s*#\\s*", "", lines[grepl("^\\s*#", lines)])
  body <- lines[!grepl("^\\s*#", lines) & nzchar(trimws(lines))]
  if (!length(body)) stop("the file has no data rows (only comments or blank lines)")

  df <- utils::read.csv(
    text = paste(body, collapse = "\n"),
    check.names = FALSE, stringsAsFactors = FALSE,
    comment.char = "", na.strings = "NA"
  )
  header <- names(df)
  columns <- setNames(header, .normalise_name(header))

  col_y <- .resolve_col(columns, c("y", "outcome", "y_i"))
  col_t <- .resolve_col(columns, c("t", "treat", "treatment", "arm", "a"))
  col_f <- .resolve_col(columns, c("f", "score", "f_hat", "fhat", "prognostic_score"))
  missing <- c(if (is.null(col_y)) "y", if (is.null(col_t)) "t", if (is.null(col_f)) "f")
  if (length(missing)) {
    stop(sprintf(
      "missing required column%s %s. Columns found: %s. The file needs y (outcome), t (arm, 0 or 1) and f (prognostic score).",
      if (length(missing) > 1) "s" else "",
      paste(missing, collapse = ", "),
      if (length(header)) paste(header, collapse = ", ") else "(none)"
    ))
  }

  col_cohort <- .resolve_col(columns, c("cohort", "set"))
  ys <- numeric(0)
  ts <- integer(0)
  fs <- numeric(0)
  ext <- numeric(0)
  dropped <- 0L

  for (i in seq_len(nrow(df))) {
    row <- df[i, , drop = FALSE]
    cohort <- "trial"
    if (!is.null(col_cohort)) {
      raw <- trimws(as.character(row[[col_cohort]]))
      if (length(raw) && nzchar(raw) && !is.na(raw)) cohort <- tolower(raw)
    }
    if (!cohort %in% c("trial", "external")) {
      stop(sprintf("unknown cohort value '%s'; use 'trial' or 'external'", cohort))
    }

    f_val <- .parse_num(row[[col_f]])
    if (is.na(f_val)) {
      if (!identical(cohort, "external")) dropped <- dropped + 1L
      next
    }
    if (identical(cohort, "external")) {
      ext <- c(ext, f_val)
      next
    }

    y_val <- .parse_num(row[[col_y]])
    t_val <- .parse_num(row[[col_t]])
    if (is.na(y_val) || is.na(t_val)) {
      dropped <- dropped + 1L
      next
    }
    if (!t_val %in% c(0, 1)) stop(sprintf("column t must be 0 or 1; found %g", t_val))
    ys <- c(ys, y_val)
    ts <- c(ts, as.integer(t_val))
    fs <- c(fs, f_val)
  }

  if (length(ys) < 16) {
    stop(sprintf(
      "only %d complete trial rows; at least 16 are needed so each arm has 8",
      length(ys)
    ))
  }
  if (sum(ts == 1) < 8 || sum(ts == 0) < 8) {
    stop(sprintf(
      "each arm needs at least 8 patients; found %d treated and %d control",
      sum(ts == 1), sum(ts == 0)
    ))
  }

  out <- list(
    y = ys,
    t = ts,
    f = fs,
    f_external = if (length(ext) >= 2) ext else NULL,
    source = path,
    notes = notes,
    n_dropped = dropped
  )
  out$n_trial <- length(ys)
  out$n_treated <- sum(ts == 1)
  out$n_control <- sum(ts == 0)
  out$n_external <- if (is.null(out$f_external)) 0L else length(out$f_external)
  class(out) <- "ppict_dataset"
  out
}

#' Squared Pearson correlation, NaN when either side has no variance
squared_correlation <- function(a, b) {
  a <- as.numeric(a)
  b <- as.numeric(b)
  if (length(a) < 3 || length(a) != length(b)) return(NaN)
  if (stats::sd(a) == 0 || stats::sd(b) == 0) return(NaN)
  stats::cor(a, b)^2
}

#' Pooled within-group standard deviation with ddof=1 in each group
pooled_sd <- function(a, b) {
  a <- as.numeric(a)
  b <- as.numeric(b)
  if (length(a) < 2 || length(b) < 2) return(NaN)
  dof <- length(a) + length(b) - 2
  var <- ((length(a) - 1) * stats::var(a) +
    (length(b) - 1) * stats::var(b)) / dof
  sqrt(var)
}

#' Estimate the treatment effect by each supported method
analyze_trial <- function(dataset, alpha = 0.05) {
  if (dataset$n_external > 0) {
    stop(
      "external cohort rows are present, but the R port implements the shared-tuning ",
      "analysis only. Remove the external rows, or use the Python package for ppi_per_arm."
    )
  }
  y <- dataset$y
  t <- dataset$t
  f <- dataset$f

  estimates <- list(
    unadjusted = ate_unadjusted(y, t, alpha = alpha),
    ancova = ate_ancova(y, t, matrix(f, ncol = 1), alpha = alpha),
    ppi_single = ate_ppi_single_lambda(y, t, f, alpha = alpha)
  )

  out <- list(
    dataset = dataset,
    estimates = estimates,
    alpha = alpha,
    score_r2_control = squared_correlation(y[t == 0], f[t == 0]),
    score_r2_treated = squared_correlation(y[t == 1], f[t == 1]),
    outcome_sd = pooled_sd(y[t == 0], y[t == 1])
  )
  class(out) <- "ppict_analysis"
  out
}

#' One row per method, with the interval and tuning value
analysis_table <- function(result) {
  rows <- lapply(result$estimates, function(e) {
    data.frame(
      method = e$method,
      estimate = e$estimate,
      se = e$se,
      ci_low = e$ci_low,
      ci_high = e$ci_high,
      ci_width = e$ci_width,
      p_value = e$p_value,
      lam = e$lam,
      n_treated = e$n_treated,
      n_control = e$n_control,
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

#' Load a CSV and run the analysis in one call
analyze_trial_csv <- function(path, alpha = 0.05) {
  analyze_trial(load_trial_dataset(path), alpha = alpha)
}

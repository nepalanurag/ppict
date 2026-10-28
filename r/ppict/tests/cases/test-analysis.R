make_csv <- function(df, comment = "# synthetic fixture") {
  path <- tempfile(fileext = ".csv")
  write.csv(df, path, row.names = FALSE)
  writeLines(c(comment, readLines(path)), path)
  path
}

set.seed(14L)
n <- 120L
t <- rep(0:1, each = n / 2)
f <- rnorm(n)
y <- 0.4 * t + 0.7 * f + rnorm(n, sd = 0.5)
path <- make_csv(data.frame(y = y, t = t, f = f))
d <- load_trial_dataset(path)
stopifnot(
  d$n_trial == n,
  d$n_treated == n / 2,
  d$n_control == n / 2,
  d$n_external == 0L,
  length(d$notes) == 1L
)

res <- analyze_trial(d)
tab <- analysis_table(res)
stopifnot(
  identical(tab$method, c("unadjusted", "ancova", "ppi_single")),
  all(tab$ci_low < tab$estimate),
  all(tab$estimate < tab$ci_high),
  all(tab$ci_width > 0),
  abs(tab$estimate[2] - tab$estimate[3]) < 1e-8,
  is.finite(res$score_r2_control),
  is.finite(res$outcome_sd)
)

alias_path <- make_csv(data.frame(outcome = y, treatment = t, score = f))
alias <- load_trial_dataset(alias_path)
stopifnot(alias$n_trial == n)

missing_path <- make_csv(data.frame(y = y, t = t))
missing_error <- tryCatch(
  {
    load_trial_dataset(missing_path)
    NULL
  },
  error = function(e) conditionMessage(e)
)
stopifnot(!is.null(missing_error), grepl("missing required", missing_error))

small_path <- make_csv(data.frame(y = y[1:10], t = t[1:10], f = f[1:10]))
small_error <- tryCatch(
  {
    load_trial_dataset(small_path)
    NULL
  },
  error = function(e) conditionMessage(e)
)
stopifnot(!is.null(small_error), grepl("at least 16", small_error))

ext_df <- data.frame(
  y = c(y, rep(NA, 5)),
  t = c(t, rep(NA, 5)),
  f = c(f, rnorm(5)),
  cohort = c(rep("trial", n), rep("external", 5))
)
ext_path <- make_csv(ext_df)
ext <- load_trial_dataset(ext_path)
stopifnot(ext$n_external == 5L)
ext_error <- tryCatch(
  {
    analyze_trial(ext)
    NULL
  },
  error = function(e) conditionMessage(e)
)
stopifnot(!is.null(ext_error), grepl("shared-tuning", ext_error))

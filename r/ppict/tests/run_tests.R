#!/usr/bin/env Rscript

# Base-R test runner. testthat is not a dependency, so each case file is a
# sequence of stopifnot() assertions and the runner reports the first failure.
library(ppict)

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
candidates <- c(
  if (length(file_arg)) dirname(normalizePath(file_arg)) else NULL,
  getwd(),
  file.path(getwd(), "tests"),
  "tests"
)
candidates <- unique(candidates[dir.exists(file.path(candidates, "cases"))])
if (!length(candidates)) stop("no tests/cases directory found; run from the package root")
case_dir <- file.path(candidates[1], "cases")
cases <- sort(list.files(case_dir, pattern = "\\.R$", full.names = TRUE))
if (!length(cases)) stop("no test cases found in ", case_dir)

failures <- 0L
for (case in cases) {
  ok <- tryCatch(
    {
      env <- new.env(parent = globalenv())
      sys.source(case, envir = env)
      TRUE
    },
    error = function(e) {
      cat("FAIL ", basename(case), ": ", conditionMessage(e), "\n", sep = "")
      FALSE
    }
  )
  if (ok) cat("ok   ", basename(case), "\n", sep = "") else failures <- failures + 1L
}

if (failures > 0L) {
  cat(sprintf("%d test file(s) failed\n", failures))
  quit(status = 1L)
}
cat(sprintf("%d test file(s) passed\n", length(cases)))

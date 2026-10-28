# ppict

Prediction-powered covariate adjustment for randomized clinical trials, with a
prognostic score built by permutation-assisted group lasso.

Who this is for: trial statisticians and reviewers who want to see the numbers
behind a claim of variance reduction, and anyone who needs to check whether a
prediction-powered analysis of a randomized trial actually holds its Type I error
rate when the prognostic model is wrong.

## The one claim

The single falsifiable claim this project exists to test:

> The tuned prediction-powered treatment effect estimator holds Type I error at
> alpha = 0.05 and produces narrower confidence intervals than an unadjusted
> randomized trial analysis, including when the prognostic score is deliberately
> corrupted.

Everything else in this repository serves that claim, and the claim is reported
as measured, including where it is limited.

## Measured results: does it hold alpha?

400 Monte Carlo replicates per scenario, 22,400 replicate rows in total. The
treatment effect is exactly zero in every scenario, so the rejection rate at
alpha = 0.05 *is* the empirical Type I error rate. Rejections out of 400 are
shown; the Monte Carlo standard error of a rate near 0.05 at 400 replicates is
0.0109.

| scenario | unadjusted | ppi_single | ppi_per_arm | lasso_l1 |
| --- | --- | --- | --- | --- |
| good_score | 0.0450 (18/400) | 0.0450 (18/400) | 0.0475 (19/400) | 0.0600 (24/400) |
| weak_score | 0.0500 (20/400) | 0.0425 (17/400) | 0.0425 (17/400) | 0.0575 (23/400) |
| poorly_estimated_score | 0.0625 (25/400) | 0.0675 (27/400) | 0.0600 (24/400) | **0.0875 (35/400)** |
| corrupt_noise | 0.0450 (18/400) | 0.0525 (21/400) | 0.0500 (20/400) | 0.0600 (24/400) |
| corrupt_shuffle | 0.0450 (18/400) | 0.0475 (19/400) | 0.0500 (20/400) | 0.0600 (24/400) |
| corrupt_bias | 0.0450 (18/400) | 0.0450 (18/400) | 0.0450 (18/400) | 0.0600 (24/400) |
| corrupt_flip | 0.0450 (18/400) | 0.0450 (18/400) | 0.0450 (18/400) | 0.0600 (24/400) |
| history_shift | 0.0450 (18/400) | 0.0400 (16/400) | 0.0400 (16/400) | 0.0600 (24/400) |

None of the 24 prediction-powered scenario and method combinations showed a
statistically significant increase over alpha under a one-sided exact binomial
test. The highest observed rate was 0.0675 (`poorly_estimated_score`,
`ppi_single`), with an inflation test p value of 0.0726. It is the weakest cell
in the table and it is reported as such rather than smoothed over.

The one method that does fail is the L1 lasso adjustment: 0.0875 in
`poorly_estimated_score`, inflation p = 0.0011. It selects covariates on the same
data it then estimates on, and it also reports the narrowest intervals of any
method. The narrow intervals and the broken error rate are the same phenomenon.
That contrast is the practical argument for fitting the score on a disjoint
cohort and tuning a single scalar instead of selecting covariates inside the
analysis.

### What the variance reduction actually was

Width ratio is the mean interval width relative to the unadjusted analysis on the
same replicates. Implied variance reduction is one minus the ratio of mean squared
standard errors, which is the same factor that multiplies required sample size at
a fixed allocation.

| scenario | width ratio | implied variance reduction |
| --- | --- | --- |
| good_score | 0.8469 | 0.2827 |
| corrupt_bias | 0.8469 | 0.2827 |
| corrupt_flip | 0.8469 | 0.2827 |
| history_shift | 0.9053 | 0.1799 |
| poorly_estimated_score | 0.9183 | 0.1560 |
| corrupt_noise | 0.9701 | 0.0588 |
| weak_score | 0.9838 | 0.0321 |
| corrupt_shuffle | 0.9984 | 0.0031 |

Read that table with these three facts:

- **`corrupt_bias` and `corrupt_flip` reproduce the clean score exactly, to four
decimals.** That is not luck. The tuning value is invariant to a constant offset
and absorbs a sign reversal, and the test suite asserts both exactly. The
estimator does not degrade on those corruptions because they are reparameterisations.
- **`corrupt_shuffle` correctly gives up.** A score with no information gets a
tuning value near zero and the analysis falls back to the unadjusted one, rather
than losing power. This is the failure mode that untuned PPI exhibits and that
tuning removes.
- **`history_shift` costs a real part of the gain**, from 0.28 to 0.18. That is
the transport penalty the FDA cautions about, measured rather than asserted.

Full tables, confidence intervals and the inflation tests for every method are in
[validation/results/NULL_REPORT.md](validation/results/NULL_REPORT.md).

## Real-data retrospective

The simulator is not the only check. [validation/run_real_data.py](validation/run_real_data.py)
downloads ACTG 175 and TCGA-BRCA, fits a prognostic score on held-out control
patients, and runs the same estimators.

On 1,054 patients from the ZDV and ddI arms of ACTG 175, 527 in the analysis
half, the three methods the spec asks for line up as follows:

| method | estimate | 95% CI | width | implied control arm |
| --- | --- | --- | --- | --- |
| unadjusted | 65.73 | 40.22 to 91.24 | 51.02 | 82 |
| L1 lasso | 65.72 | 45.53 to 85.90 | 40.37 | 51 |
| group-lasso PPI++ | 66.89 | 45.90 to 87.89 | 41.99 | 55 |

The adjusted interval was **42.0 CD4 cells wide** against **51.0**
unadjusted, a **17.7% narrowing** on one stratified split. No split-to-split
interval was computed. The implied control arm is a projection at 80% power and
the observed effect, not a measured trial size. The L1 lasso is narrower here,
but it selects covariates on the same patients it estimates on; the null sweep
records its 8.75% rejection rate in the poorly estimated score scenario, so the
narrow interval is not evidence that it is better. With 1,000 permuted-label
negative controls, the rejection rate was 5.1% unadjusted, 4.0% adjusted, and
5.1% with a corrupted score.

The TCGA-BRCA expression score is fitted two ways on 780 ER-labelled samples.
The primary score is the package's own 5-fold group lasso (AUC 0.9551, 95% CI
0.9345 to 0.9721); an L2 logistic classifier on the same folds is the benchmark
(AUC 0.9337, 0.9093 to 0.9573). Neither can be transported to ACTG 175: the
trial has no expression measurements, the cohorts share no patients, and ER
status is not on the CD4 scale. The trial adjustment uses a clinical group-lasso
score fitted on the other half of the two-arm subset, so it is an internal
split, not an external cohort; the TCGA score is external but not applicable.
The report states both. That block, not a simulated substitute, is what the
report records.

Full report: [validation/results/REAL_DATA_REPORT.md](validation/results/REAL_DATA_REPORT.md).

## Status

| Piece | State |
| --- | --- |
| `PLAN.md` with paper-versus-prompt discrepancies | written |
| Python package and command line | written |
| Unit and end-to-end tests | 101 passing |
| Null simulations with measured Type I error | run, see `validation/results/NULL_REPORT.md` |
| Interactive calculator with SAP export | built, runs locally, exact agreement with the CLI |
| Real data: ACTG 175 and TCGA-BRCA | run, see `validation/results/REAL_DATA_REPORT.md` |
| R port | written, installs from source, 4 test files pass, parity measured |

## Install and run

```bash
python3 -m venv --system-site-packages .venv
./.venv/bin/python -m pip install -e ".[dev]"

# quick end to end check
./.venv/bin/python -m ppict.cli selftest

# the full test suite
./.venv/bin/python -m pytest

# the headline experiment
./.venv/bin/python validation/run_null_simulations.py --n-rep 400 --n-jobs 9

# the real-data check
./.venv/bin/python validation/run_real_data.py --permutations 1000

```

## R port

The R package is [r/ppict](r/ppict). It uses base R plus `stats` and `utils` only,
so there is no dependency to install beyond R itself.

```bash
mkdir -p r/lib
R CMD INSTALL --library=r/lib r/ppict
R_LIBS="$PWD/r/lib" Rscript r/ppict/tests/run_tests.R
./.venv/bin/python validation/run_r_parity.py
```

The parity command installs the package, computes the R side from shared CSV
fixtures in [validation/fixtures](validation/fixtures), computes the Python side
from the same files, and writes
[validation/results/R_PARITY.md](validation/results/R_PARITY.md). Measured: 460
compared values, largest absolute difference **7.417e-09** against a stated
tolerance of 1e-6. The eight permutations are a shared fixture, not a claim that
R's Mersenne-Twister and NumPy's PCG64 produce the same draws. A source tarball
built with `R CMD build` passes `R CMD check --no-manual --no-build-vignettes`
with Status OK.

The port covers the permutation-assisted group-lasso selector, the prognostic
score including fixed-fold cross-fitting, the PPI++ mean, the unadjusted,
ANCOVA and shared-tuning PPI++ treatment effects with intervals, the PROCOVA
variance bound, optimal allocation, required sample size and control-arm size,
and the CSV-in, analysis-table-out entry point. It does not include the Monte
Carlo sweep, the calculator, the SAP renderer, or the per-arm external-cohort
estimator. The R analysis entry point stops with a named error if external rows
are present.

Then, for the calculator in a second terminal (Node 20.9 or newer):

```bash
cd calculator
npm install
npm run dev
# http://127.0.0.1:3040
```

With that server running, this compares the app, the CLI and the Python API, and
compiles the appendix:

```bash
./.venv/bin/python validation/run_calculator_checks.py
```

## The calculator

`calculator/` is a local Next.js app with one screen. The left column holds the
protocol inputs; the right holds the estimate, the interval plot and the design
projection. No statistic is computed in JavaScript. The route handler writes an
uploaded file to a temporary directory, runs `python -m ppict.cli analyze`, and
returns the JSON built by [src/ppict/report.py](src/ppict/report.py); the browser
only formats numbers and draws.

The app opens on a simulated 720-patient example, labelled as simulated in the
interface and in its own file header. It is generated by
[tools/make_demo_trial.py](tools/make_demo_trial.py) with a fixed seed, not patient
data. Upload a CSV with columns `y`, `t` and `f` to analyse another dataset;
optional `cohort=external` rows carrying only `f` enable the per-arm estimator.

The SAP button downloads a complete LaTeX appendix: the pre-specified model, the
tuning rule, the design table and the exact command. It does not contain the
observed effect, because it is an appendix fixed before unblinding.

### The app is not a second implementation

[validation/run_calculator_checks.py](validation/run_calculator_checks.py) sends
the same file through the Python API, the CLI and the app's HTTP route, and
compares every estimate, standard error, interval bound, width and projection.
Measured on the example file: the largest absolute difference was 0.0 in every
comparison, including a copy of the file uploaded through the HTTP route. The same
run compiles the appendix to 3 pages with 0 overfull boxes and no em or en
dashes. Output:
[validation/results/CALCULATOR_CHECK.md](validation/results/CALCULATOR_CHECK.md).

The sample export is
[validation/results/sap_appendix.tex](validation/results/sap_appendix.tex), with
its compiled [PDF](validation/results/sap_appendix.pdf).

## What is in the package

| Module | What it does |
| --- | --- |
| `ppict.analysis` | Data in, estimate and interval out. Loads and validates a trial CSV, runs every method on the same rows, and builds the design projection from the same variance functions the allocation code uses. |
| `ppict.report` | The JSON contract shared by the CLI and the calculator, with the report shape versioned. |
| `ppict.sap` | Renders the statistical analysis plan appendix as a standalone LaTeX document. |
| `ppict.group_select` | Group lasso by proximal gradient (FISTA) with group soft-thresholding, adaptive group weights from univariate z-scores, and a permutation calibration of the penalty. Also reports selection frequency across subsamples. |
| `ppict.prognostic` | Prognostic score for the control outcome, in the two independence regimes PPI allows: trained on a disjoint external cohort, or cross-fitted with out-of-fold predictions. |
| `ppict.ppi` | The tuned prediction-powered mean estimator, the two-arm treatment effect estimators, and the prognostic covariate adjustment variance and allocation formulas. |
| `ppict.simulate` | Seeded Monte Carlo harness with a run log, including a deliberate corruption sweep of the prognostic score. |
| `r/ppict` | R mirror of the selector, prognostic score, PPI++ shared-tuning estimator, PROCOVA design arithmetic and CSV analysis entry point, parity-checked against Python. |

### What the code deliberately does not do

These are choices, not omissions, and each is recorded in `PLAN.md`:

- It does not claim novel methodology. Adapting PPI and PPI++ to the two-arm trial
  average treatment effect, and the asymptotic equivalence to ANCOVA with the
  prognostic score as a single covariate, is established in Poulet et al. (2025).
  This project builds on that and spends its effort on a different score
  construction and on adversarial testing.
- It does not claim false-discovery-rate control. The permutation step calibrates
  a penalty; it is not a knockoff procedure. Measured false-discovery proportions
  are reported instead.
- It does not claim to produce digital twins. No individual synthetic patient is
  generated. The score is a variance-reduction device.

## Two structural facts worth knowing before reading the results

Both follow from the algebra and are checked in the test suite.

**1. With a single tuning value, the external cohort cancels out.** If the tuning
value is shared between arms, the unlabeled cohort's mean prediction appears once
with a plus sign and once with a minus sign, and

```
tau_hat = mean(Y | treated) - mean(Y | control) - lambda * (mean(f | treated) - mean(f | control))
```

contains no external quantity at all. With `ddof = 0` moments and equal arm sizes
this is *exactly* the ANCOVA coefficient for a single covariate, not merely
asymptotically equivalent to it. The per-arm form does retain a dependence on the
external cohort, which is why it requires that cohort to be exchangeable with the
trial population. `tests/test_ppi.py` asserts both.

**2. The variance reduction tends to the correlation squared.** The PPI++ variance
reduction for a mean is `rho^2 * N / (n + N)`, which tends to `rho^2` as the
unlabeled sample grows. That limit is the same `1 - R^2` factor the EMA
qualification opinion states for prognostic covariate adjustment, which is why
these two literatures meet in the middle. `tests/test_ppi.py` checks the formula
and the limit numerically.

## Environment notes

Measured on this machine, and worth knowing because it affects how you read logs:

- Python 3.9.6, numpy 2.0.2, scipy 1.13.1, scikit-learn 1.6.1, pandas 2.3.3.
- **numpy here links Apple Accelerate for BLAS, and that backend raises spurious
  floating-point flags on essentially every matmul.** A matmul of two all-finite
  arrays with an all-finite result still reports `divide by zero`, `invalid value`
  and `overflow` encountered in matmul. A matmul performs no division, so
  `divide by zero in matmul` cannot be genuine. `ppict/__init__.py` filters those
  four specific messages, and `pyproject.toml` does the same for the test run, so
  that a several-hundred-replicate validation run does not write megabytes of
  noise. Every other `RuntimeWarning` remains fatal under pytest.
- Parallelism: `run_monte_carlo(..., n_jobs=k)` spreads replicates across
  processes. Results are collected in submission order and are bit-identical to a
  serial run. This was verified by comparing the full replicate CSVs from
  `--n-jobs 1` and `--n-jobs 4`, not assumed.

## Paper versus prompt

`PLAN.md` lists every place where the brief that commissioned this project
disagrees with the primary literature or with the author's own published code, and
what was done about each. The four that changed the design most:

1. The variance reduction is `rho^2 * N / (n + N)`. An earlier and plausible
   looking derivation gives `rho^2 * n / (n + N)`, which would mean an infinite
   unlabeled cohort buys nothing. It is wrong, and the test suite pins the correct
   limit.
2. PPI++ is by Angelopoulos, Duchi and Zrnic. Candes is a co-author of the
   separate Cross-PPI paper, which is where the cross-fitting construction used
   here comes from.
3. Permutation decoys in the author's own R package are described in that
   package's README as a heuristic rather than knockoffs, so no FDR guarantee is
   claimed here. The prompt asserted one.
4. Project Data Sphere requires an account and an approval step, and this project
   is barred from creating accounts or contacting anyone. Real-trial work
   therefore uses ACTG 175, which is public, rather than pretending the access
   exists.

## Data, licensing, and what is not redistributed

No third-party data is committed. [validation/run_real_data.py](validation/run_real_data.py)
downloads ACTG 175 and TCGA-BRCA from public routes that require no login; the
URLs, sizes and hashes are recorded in
[validation/results/REAL_DATA_REPORT.md](validation/results/REAL_DATA_REPORT.md).
Downloaded files are gitignored.

The calculator's example file is simulated: 720 patients generated by
`tools/make_demo_trial.py` with seed 7, labelled as simulated in the CSV header,
the interface and this file. It is not patient data. The two typefaces are
self-hosted under the SIL Open Font License; sources and licences are in
[calculator/app/fonts/README.md](calculator/app/fonts/README.md).

Package name `ppict`: the PyPI simple index, the PyPI JSON API and the CRAN
package URL all returned 404 for this name at the time of writing, so it appears
free. Nothing is published, and the brief requires local work only.

## What is still unproven or unbuilt

Stated plainly, because a project about not overclaiming should not overclaim:

- **The real-data check is narrower than the trial.** It uses CD4 at week 20, a
  secondary endpoint, in two of ACTG 175's four arms, complete cases only. The
  TCGA expression score is fitted with the package's group lasso, so the
  selector is exercised on real expression data; an L2 logistic classifier is
  reported beside it as a fixed-budget benchmark. The expression score could
  not be transported: the trial has no expression and shares no patients with
  TCGA, so the trial adjustment uses a clinical score.
  That score is an internal split of ACTG, not an external cohort; the only
  external score (TCGA) cannot be applied. The 17.7% is one stratified split,
  not a split-to-split interval, and the negative-control interval is
  permutation noise for one fitted score. The TCGA AUC is one 5-fold split. The
  real analysis also uses the shared-tuning form, which reduces to ANCOVA at
  equal allocation; it does not test the external-cohort PPI term. That term
  remains a simulation result; the expression score is a TCGA-only result.
- **The R port is narrower than the Python package.** It covers the code paths
  behind the spec's claims, not the simulation sweep, calculator, SAP renderer
  or external-cohort per-arm estimator. Its tests use base R, not `testthat`.
  Parity is measured on shared fixtures with explicit permutations; the RNG
  streams themselves are not compared. R's `ppict` has not been published or
  submitted to CRAN.
- **The calculator is a local tool.** It needs Python and this package installed.
  It exports LaTeX rather than compiling a PDF, and the dollar and time figures
  are projections under the stated model, not observed savings. The interval
  narrowing at the top of the screen is measured on the loaded file.
- **Only continuous outcomes.** That matches the EMA qualification for prognostic
  covariate adjustment and the current code. Binary and time-to-event outcomes are
  not supported.
- **Constant treatment effects.** The simulation generates an additive treatment
  effect, which is the case where prognostic covariate adjustment is
  semiparametrically efficient. A scenario where the score is strongly prognostic
  in one arm and weakly prognostic in the other is *not* in the null preset, and
  the per-arm estimator is the form that would be exposed by it. This is the most
  important untested assumption in the repository.
- **Score-stability assumption.** Cross-fitting needs the fold models to be
  similar to each other (Cross-PPI's prediction-stability condition). The harness
  measures the mean pairwise Jaccard index of the selected feature sets across
  folds and reports it, but does not establish that the condition holds.
- **No real recruitment data.** Any cost or time saving is a translation of a
  measured variance ratio under stated assumptions, not an observed saving.

## License

MIT. The R package carries the same license in
[r/ppict/LICENSE](r/ppict/LICENSE). Downloaded datasets are not redistributed;
see the data section above.

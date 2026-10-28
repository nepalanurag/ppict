# PLAN.md

Plan for the project described in `ppi-digital-twin-deepseek-prompt.md`, including
every place where the prompt and the primary literature disagree. Written before
any package code was written.

Working name: `ppict` (prediction-powered inference for clinical trials).

## 1. What the project is

One package, one validation harness, one local calculator. The Python package
supplies a prognostic score built from high-dimensional grouped covariates with
permutation-assisted group lasso, and an estimator of the average treatment effect
(ATE) in a randomized trial that uses that score to reduce the variance of the
treatment effect estimate while keeping Type I error control. A local R package
mirrors the code paths behind those claims. The validation
harness measures, by simulation and on real data, whether that holds. The
calculator turns the measured variance reduction into recruitment cost and time.

The falsifiable claim the project must earn, not assert:

> The tuned prediction-powered ATE estimator holds Type I error at alpha = 0.05 and
> produces narrower confidence intervals than an unadjusted RCT analysis, including
> when the prognostic score is deliberately corrupted.

## 2. Verified background taken from the prompt

- Master's thesis work: permutation-assisted group lasso for multi-omics feature
  selection, in the public repo `nepalanurag/thesis-findings`, package
  `permgrouplasso` (R).
- Undergraduate work: compact CNN for malaria cell classification.
- Language preference: Python and R. Existing personal sites use Next.js.

The `permgrouplasso` package was inspected through its public GitHub files (not
cloned, not modified) to recover its real API, which this plan mirrors in Python:
`define_groups`, `consensus_groups`, `group_pen_from_z`, `valid_adaptive_pen`,
`shrunk_adaptive_pen`, `perm_select`, `stab_select`, `paglasso`, `jaccard`,
`pairwise_jaccard`, `enrichment`.

## 3. Paper versus prompt: every discrepancy found

The prompt says that where it conflicts with the papers, the papers win. These are
the conflicts, and what this project does about each.

### D1. Authorship of the PPI++ paper

Prompt: "Zrnic and Candes on PPI++ power tuning (2024)".

The paper is **Angelopoulos, Duchi, and Zrnic**, "PPI++: Efficient Prediction-Powered
Inference", arXiv:2311.01453 (2023, v2 revised March 2024). Emmanuel Candes is not an
author of PPI++. Zrnic and Candes are authors of the separate **Cross-PPI** paper
(arXiv:2309.16598, PNAS 2024). Both are cited for what they actually establish.

### D2. The variance reduction formula (prompt is silent; my first derivation was wrong)

PPI++ does not print a scalar variance-reduction formula for the mean. Specializing
PPI++ Theorem 1 Eq. (7) to Example 6.1 ("Optimal tuning for mean estimation") gives

    Var(theta_hat) = sigma_Y^2 / n * (1 - rho^2 * N / (n + N))

so the fractional **variance reduction** is `rho^2 * N / (n + N)`, with `n` the labeled
sample, `N` the unlabeled sample, and `rho = Corr(Y, f(X))`.

My own first derivation had `rho^2 * n / (n + N)`, which is wrong: it would say that an
infinite unlabeled cohort buys zero variance reduction. Recorded here because the
correct formula is load-bearing. As `N -> infinity` the reduction tends to `rho^2`,
which is exactly the `1 - R^2` factor the EMA qualification opinion states for
PROCOVA. That limit is the reason PPI++ and PROCOVA are the same object in this
setting, and it is checked numerically in the test suite.

The optimal tuning value, PPI++ Example 6.1:

    lambda* = Cov(Y, f) / ((1 + n/N) * Var(f))
    lambda_hat = Cov_n(Y, f) / ((1 + n/N) * Var_{n+N}(f))

The plug-in `lambda_hat` costs no first-order variance: PPI++ Corollary 4 gives
`sqrt(n)(theta_hat_pp(lambda_hat) - theta*) -> N(0, Sigma(lambda*))`. That is an
asymptotic statement, not a finite-sample guarantee, so this project measures
finite-sample Type I error rather than trusting it.

### D3. FDR at 0.01 cannot be claimed

Prompt: the thesis simulation study controlled FDR at 0.01.

The `permgrouplasso` package README says its permutation decoys are "a heuristic,
not model-X knockoffs" and advises reporting selection frequencies and stability
rather than claiming formal FDR guarantees. The artifact contradicts the prompt, so
the artifact wins (and this is also the more defensible position).

This project therefore does **not** claim FDR control. It calibrates the group-lasso
penalty against a permutation null at a stated nominal level, then reports the
**measured** false-discovery proportion and selection stability from its own
simulations. Any FDR number that appears anywhere in this repo is measured by a
named script.

### D4. PROCOVA regulatory status must be stated precisely

Prompt: "the EMA PROCOVA qualification opinion and the FDA guidance on covariate
adjustment in randomized trials".

- **EMA** published a CHMP qualification opinion for PROCOVA on 20 September 2022, for
  prognostic score adjustment in controlled randomised clinical trials with
  **continuous** outcomes. This is a qualification opinion, not an approval of a drug
  or a licence for a product.
- **FDA** has no guidance naming PROCOVA. The May 2023 final guidance *Adjusting for
  Covariates in Randomized Clinical Trials for Drugs and Biological Products* states
  that it does not address machine-learning methods, so it does not cover a
  machine-learning prognostic score. FDA's PROCOVA-specific position appears in a CDER
  response (reproduced publicly by Unlearn) permitting adjustment for an independently
  derived, prespecified external prognostic covariate in ANCOVA, and warning that power
  calculations can be optimistic if the score-outcome correlation does not transport
  to the trial.

Consequence for this project: outcomes are continuous, the prognostic score is
prespecified and its training data are disjoint from the trial data, and the
transport warning is treated as a first-class risk (see the domain-shift scenario in
the validation harness).

### D5. The estimator is not new, and the project will not claim it is

Prompt: build the combination and prove it works, framed as a contribution.

**Direct prior art exists.** Poulet, Tran, Tezenas du Montcel, Dubois, Durrleman, and
Jedynak, "Prediction-powered inference for clinical trials: application to linear
covariate adjustment", *BMC Medical Research Methodology* 25:204 (2025),
doi:10.1186/s12874-025-02647-6, already:

1. adapts PPI and PPI++ to the two-arm randomized trial ATE,
2. derives the tuned estimator they call PPCT and its asymptotic variance,
3. proves PPCT is asymptotically equivalent to ANCOVA with the prognostic score as the
   single covariate, in both point estimate and variance,
4. derives the optimal unbalanced control/treatment allocation,
5. reports the `1 - R_out^2` sample size relation, and
6. records that a weakly informative prognostic score *increases* variance under
   untuned PPI, which is the reason lambda must be tuned.

So the plan does not treat "PPI++ for trials" as a contribution. It cites Poulet et al.
for the framework, and puts the project's own effort where the literature is still
open:

- **(a) Score construction.** Poulet et al. use disease-progression models on
  low-dimensional clinical covariates. This project builds the score from
  **high-dimensional grouped multi-omics** covariates via permutation-assisted group
  lasso. Whether the guarantees survive a sparse stability-selected score is a real
  question, not a rhetorical one.
- **(b) Adversarial stress test.** The prompt's actual falsifiable claim is about
  misspecification and corruption. A systematic corruption sweep (noise, shuffled
  predictions, systematic bias, sign flip) with measured Type I error is not reported
  by Poulet et al. and is the core experiment here.

### D6. The n = 737 figure

Prompt: TCGA-BRCA, n = 737. That number comes from the thesis's own filtering. The
Xena `HiSeqV2` matrix has 1218 sample columns and GDC reports 1098 BRCA cases, so the
analyzed n after filtering is whatever the script actually obtains. This project
reports the n it analyzes and never copies 737.

The real-data expression score in this repository is a cross-validated `ppict`
group-lasso score on 780 ER-labelled samples (out-of-fold AUC 0.9551, 95% CI 0.9345
to 0.9721). An L2 logistic classifier on the same folds is reported beside it as a
benchmark (AUC 0.9337), not as the package method. The group-lasso selector agrees
only partly across folds (mean Jaccard 0.5966), which the report states.

### D7. "Digital twin" is a naming choice, not a mechanism

PPI does not synthesize individual patients and does not create a virtual control arm
that is added to the trial. It is a variance-reduction estimator that uses additional
covariate observations. The title keeps the prompt's wording; every method statement
describes what the code does.

One concrete consequence, derived during this plan and confirmed in tests: when the
unlabeled set is a common external cohort and a **single** lambda is used, the external
cohort's mean prediction cancels out and

    tau_hat = (mean_Y_treated - mean_Y_control) - lambda * (mean_f_treated - mean_f_control)

which contains no external-cohort quantity at all. That is exactly ANCOVA with one
constructed covariate, which is why the equivalence in D5 holds. Per-arm lambdas do
retain a dependence on the external cohort, but then the estimator is only unbiased if
the external cohort's covariate distribution matches the trial population. Both forms
are implemented, and the difference is measured rather than asserted.

### D8. Project Data Sphere is not obtainable under the hard rules

PDS requires registration and a verification/approval step ("Typically, new user
requests are approved within 24hrs", approval "once verified that the user isn't
subject to certain geographical or limited regulatory restrictions"), and some
catalogue entries need an additional NCI user agreement. There is no verified
no-login route to its patient-level trial data.

The hard rules forbid creating accounts and contacting anyone. PDS access is therefore
**blocked**, and this is reported rather than routed around. The retrospective
real-trial analysis uses **ACTG 175** instead (public, no login, real randomization:
`speff2trial::ACTG175`, n = 2139, downloadable as CSV from the UCI repository). TCGA
supplies the real high-dimensional omics data. No dataset is invented and no result
is copied.

### D9. PPI requires the predictor to be independent of the inference data

PPI (arXiv:2301.09633) Section 1.2: "we have a prediction rule f, that is independent
of the observed data. For example, it may have been trained on other data independent
from both the labeled and the unlabeled data." Cross-fitting is handled by Cross-PPI
(arXiv:2309.16598), which adds a **prediction-stability** assumption on the fold
models. Poulet et al. likewise require "that f be trained with data that is disjoint
from that of the trial."

Consequence: the `prognostic` module supports two paths, and the validation measures
both instead of assuming either is safe:

- **external**: score trained on a disjoint cohort, frozen, then applied. Matches the
  PPI assumption and the regulatory framing.
- **cross-fitted**: out-of-fold scores within the labeled data, following Cross-PPI.
  The stability assumption is stated in the docstring and its finite-sample
  consequence is measured.

### D10. Data access verified in this environment (measured, not assumed)

Checked with `curl` from this machine before planning the scripts:

| Resource | Result |
| --- | --- |
| `https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz` | HTTP 200, 64,286,337 bytes |
| `https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix` | HTTP 200, 1,698,413 bytes |
| `.../BRCA_clinicalMatrix.gz` | HTTP 403 (forbidden) |
| `https://api.gdc.cancer.gov/files` (clinical supplement query) | HTTP 200 |
| `https://api.gdc.cancer.gov/cases` (TCGA-BRCA) | HTTP 200, total 1098 |
| `https://archive.ics.uci.edu/static/public/890/data.csv` (ACTG 175) | HTTP 200, 159,675 bytes |

Note the `.gz` suffix is what breaks the clinical download; the uncompressed path
works. ER status lives in the clinical matrix as `ER_Status_nature2012`.

### D11. No em dashes

The prompt forbids em dashes in user-facing text. Applied to every file in this repo.
Hyphens and colons only.

## 4. Phases, file paths, and done criteria

| Phase | Files | Done criterion |
| --- | --- | --- |
| 0. Plan | `PLAN.md` | Every discrepancy listed above has a decision attached. |
| 1a. Scaffold | `pyproject.toml`, `.gitignore`, `src/ppict/__init__.py` | Package imports; `pip install -e .` works. |
| 1b. Group selection | `src/ppict/group_select.py` | Group lasso matches the exact closed-form solution on an orthonormal design; permutation calibration selects ~0 groups under a global null and recovers true groups under signal; adaptive weights reproduce `group_pen_from_z`. |
| 1c. Prognostic score | `src/ppict/prognostic.py` | External and cross-fitted scores both fit; out-of-sample R^2 measured and returned; no fold used to train its own prediction. |
| 1d. Estimator | `src/ppict/ppi.py` | Point estimate matches `ppi_py` (the authors' reference implementation) on identical inputs to numerical tolerance; lambda = 0 reduces exactly to the unadjusted estimator; single-lambda form equals ANCOVA with one covariate; PROCOVA allocation returns 1/2 and the `1 - rho^2` factor in the symmetric case. |
| 1e. Simulation harness | `src/ppict/simulate.py` | Fixed seeds; every run writes a log with seed, versions, and per-replicate results; same seed reproduces the same numbers. |
| 1f. Tests | `tests/` | All tests pass, including an end-to-end test on simulated trial data. |
| 1g. R port | `r/ppict/` | Installs from source; its four test files pass; `R CMD check` on the built source tarball is Status OK; the selector, prognostic score, PPI++ estimator, PROCOVA design arithmetic and analysis entry point agree with Python on shared fixtures to a measured tolerance. |
| 2a. Null simulations | `validation/run_null_simulations.py` | Measured empirical Type I error at alpha = 0.05 under true null for good, weak, and corrupted scores, with Monte Carlo uncertainty. |
| 2b. Real-data check | `validation/run_real_data.py` | ACTG 175 comparison of unadjusted, L1 lasso and group-lasso PPI++ with interval width and implied control arm; permuted-label alpha with the real and a corrupted score; cross-validated TCGA-BRCA expression score fitted with `ppict`'s group lasso, with an L2 logistic benchmark; the transport block is reported, not routed around. |
| 2c. R parity | `validation/run_r_parity.py`, `validation/r_parity.R`, `validation/fixtures/` | Both languages read the same fixtures; 460 values compared at a stated 1e-6 tolerance, largest measured difference 7.417e-09; report written to `validation/results/R_PARITY.md`. |
| 3a. Analysis surface | `src/ppict/analysis.py`, `src/ppict/report.py`, `src/ppict/sap.py` | Data in, estimate and interval out; one versioned JSON contract; a standalone LaTeX appendix that compiles. |
| 3b. Calculator | `calculator/` | Single screen; every number comes from `python -m ppict.cli analyze`; app, CLI and Python API agree exactly on the same file; the exported appendix compiles with zero overfull boxes. |
| 4. Report | `README.md` | Every number traceable to a named script and command. |

### Calculator decisions

- **No statistics in the app.** The browser owns formatting and drawing only.
  `src/ppict/analysis.py` is the data-in, estimate-and-interval-out surface,
  `src/ppict/report.py` is the JSON contract, and `ppict.cli analyze` is the single
  boundary the route handler calls. A second implementation in TypeScript would
  be a defect, so `validation/run_calculator_checks.py` compares the Python API,
  the CLI and the app's HTTP route field by field. Measured difference on the
  example file: exactly zero.
- **The appendix excludes the result.** It states the pre-specified model, the
  tuning rule, the variance model and the design arithmetic. An appendix fixed
  before unblinding that contains the observed effect is not that document.
- **Dollars and months are projections.** `plan_trial` uses the PROCOVA variance
  model with the promised inputs; the interval narrowing shown at the top of the
  screen is measured on the loaded file. The two are labelled separately because
  they are different kinds of number. The patient saving is the difference
  between the unadjusted and adjusted requirements at the stated power, not the
  gap between the planned size and the adjusted requirement, so it is
  attributable to the adjustment and does not credit an already over-sized plan.
  The planned size drives the information-equivalence line and the power check.
- **Fonts are self-hosted** so the app builds and runs without a network request.
- **The example dataset is simulated and labelled as such** in its CSV header, in
  the interface and in the README. It is never described as patient data.

## 5. Libraries

Python runtime: `numpy`, `scipy`, `scikit-learn`, `pandas`. Development: `pytest`.
`ppi-python` is a **test-only** dependency used as an independent reference
implementation; it is deliberately not a runtime dependency, so the package does not
inherit its model. The R package uses base R plus `stats` and `utils` only, and its
tests use base-R assertions rather than `testthat`.

The group lasso solver is implemented in this package with proximal gradient descent
(FISTA) and group soft-thresholding rather than pulled from a third-party package.
Reason: the permutation calibration needs to refit hundreds of times at a *fixed*
penalty on permuted data, and the orthonormal-design case has an exact closed-form
solution, which converts the solver into something testable against an analytic
answer instead of against a smoke test.

## 6. What is explicitly out of scope for this pass

Stated plainly so the next pass extends a working system rather than an outline:

- Compiling the appendix PDF inside the app. The app exports LaTeX and the check
  script invokes `latexmk`; a TeX installation is not a reasonable dependency for
  the interface.
- In the R port: the Monte Carlo sweep, the calculator, the SAP renderer, and
  the per-arm external-cohort estimator. The R analysis entry point stops with a
  named error on external rows rather than silently dropping them.

## 7. Known risks

- **Finite-sample alpha under estimated lambda.** Corollary 4 is asymptotic. Small `n`
  plus a badly scaled score could undercover. This is measured, not assumed, and if it
  inflates the project reports the failure and stops rather than tuning until it passes.
- **Stability-selected scores.** A sparse score built from a data-adaptive selection
  step may not satisfy the prediction-stability condition Cross-PPI requires.
- **Transport.** A score trained on one population applied to another is the failure
  mode FDA warns about. Domain shift is a scenario in the harness, and the negative
  control does not substitute for it.

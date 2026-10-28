# Real-data retrospective: ACTG 175 and TCGA-BRCA

Measured numbers only. Every value below is written by `validation/run_real_data.py`, which also writes `real_data_results.json`, `actg_scores.csv` and `brca_er_scores.csv`. The command is in the reproduction section.

## Bottom line

- On 527 patients from two arms of ACTG 175, the adjusted interval was **42.0 CD4 cells wide** against **51.0** unadjusted, a **17.7% narrowing** (width ratio 0.823).
- The adjusted estimate was **66.89** CD4 cells (95% CI 45.90 to 87.89); unadjusted was **65.73** (95% CI 40.22 to 91.24).
- The L1 lasso baseline was **65.72** (95% CI 45.53 to 85.90), width 40.37, the narrowest of the three. It selects covariates on the same patients it estimates on, so it is a comparison row, not a recommended analysis.
- With treatment labels permuted (1,000 permutations), the rejection rate was **5.1%** unadjusted, **4.0%** adjusted, and **5.1%** with a corrupted score. All three are consistent with alpha = 0.05.
- The TCGA-BRCA expression score is fitted with the package's 5-fold group lasso (AUC **0.955**, 95% CI 0.935 to 0.972, 83 genes in the final fit). An L2 logistic baseline scores 0.934 (95% CI 0.909 to 0.957). Neither can be transported to this trial: ACTG 175 has no expression measurements, the cohorts share no patients, and ER status is not on the CD4 scale.
- On this one stratified split, the real data are consistent with the estimator's interval narrowing and alpha control on a real randomized trial, using a held-out internal clinical score. They do not establish an external expression-derived score transported into a trial. The simulation remains the only evidence for that, and for the external-cohort PPI term.

## Data, provenance, and named blocks

| file | route | bytes | sha256 | file mtime (UTC) |
| --- | --- | --- | --- | --- |
| `validation/data/actg175.csv` | https://archive.ics.uci.edu/static/public/890/data.csv | 159,675 | `cf58638c7687764c...` | 2026-10-28 01:05 UTC |
| `validation/data/brca_hiseq.tsv.gz` | https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz | 64,286,337 | `263bf67245cc4b90...` | 2026-10-28 01:07 UTC |
| `validation/data/brca_clinical.tsv` | https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix | 1,698,413 | `39eb3be0fb86e6a5...` | 2026-10-28 01:05 UTC |

The full sha256 values are in `real_data_results.json`. Sizes are checked on every run; a mismatch stops the script instead of using a partial or changed file. Neither dataset is redistributed; both are ignored by git.

### ACTG 175

The public extract has 2,139 patients in four arms (ZDV: 532, ddI: 522, ZDV+ddI: 524, ZDV+ddC: 561). This analysis uses the two arms ZDV and ddI: 1,054 patients, 1,054 complete cases, 0 dropped for missing values. The outcome is CD4 count at week 20 (cells per microlitre).

This is a secondary endpoint, not the trial's primary AIDS/death endpoint, and it is a two-arm subset of a four-arm trial. Those are limits of what the extract supports, not claims about the trial.

### TCGA-BRCA

The expression matrix has 1,218 samples and 20,530 genes. 780 samples have known ER status; all 780 align to expression (601 positive, 179 negative). After dropping 303 constant and 0 non-finite genes, 20,227 genes are available. The group-lasso selector screens the top 300 by marginal correlation inside each training fold; the logistic baseline drops zero-variance genes inside each fold and selects 200 by ANOVA F.

### Routes that did not work or were not used

- The compressed clinical route, `.../BRCA_clinicalMatrix.gz`, returns HTTP 403. The uncompressed route used above returns the file.
- Project Data Sphere was not used. Access needs an account and an approval step, and this project does not create accounts or contact anyone. The public ACTG 175 extract supplies the randomized trial.
- Transporting the TCGA expression score to ACTG 175 is impossible with the public data: the trial records no gene expression, the cohorts have no shared patients, and the outcomes are on different scales. This is the central negative result of the pass.

## Trial score

The score is the project's group-lasso prognostic score, fitted on the 266 control patients in the training half only. It selected 1 feature: **cd40**. Training R^2 was 0.3928; out-of-sample R^2 on the 266 control patients in the analysis half was **0.4195**.

The group lasso screened 12 groups at q = 0.10 with 25 permutations and selected 1 group; the tuning value on the standardised scale was 29.53. The raw-scale coefficients are in `real_data_results.json`.

The split is stratified by arm at seed 7: 261 treated and 266 control in the analysis half, disjoint from the score's training patients. No analysis patient trained the score. This is the only split reported; no split-to-split interval is computed.

## Treatment effect and interval width

| method | estimate | SE | 95% CI | width | width vs unadj | implied control arm | p | detail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unadjusted | 65.73 | 13.01 | 40.22 to 91.24 | 51.02 | 1.000 | 82 | <0.0001 | n/a |
| L1 lasso | 65.72 | 10.30 | 45.53 to 85.90 | 40.37 | 0.791 | 51 | <0.0001 | 14 selected |
| group-lasso PPI++ | 66.89 | 10.71 | 45.90 to 87.89 | 41.99 | 0.823 | 55 | <0.0001 | lambda 1.0212 |

The implied-control-arm column is a projection, not a measured trial size: it scales each estimator's observed variance to the observed unadjusted effect at 80% power with 1:1 allocation. The L1 row is the naive post-selection baseline the spec asks for, not a recommended method.

The L1 lasso produced the narrowest interval on this split (ratio 0.7914). That is not evidence that it is the better method: it selects covariates on the same 527 patients it then estimates on, with no correction for the selection. The null sweep records the consequence, an 8.75% rejection rate in the poorly estimated score scenario (`validation/results/NULL_REPORT.md`). The group-lasso PPI++ row is the one whose score was fitted on disjoint patients.

Width ratio adjusted/unadjusted: **0.8230** (variance ratio 0.6774); L1 lasso ratio 0.7914. The point estimates differ by 1.16 CD4 cells (adjusted minus unadjusted).

The estimator used here is the shared-tuning PPI++ form. With no external unlabeled cohort, the unlabeled mean cancels between arms, so this is algebraically an ANCOVA on one fitted index when the arms are equal in size. Here the analysis arms differ by 5 patients, so the identity is approximate. The real trial therefore tests the covariate adjustment, not the external-cohort variance term. The simulation is still the evidence for that term.

## Negative control

Treatment labels in the analysis half were permuted 1,000 times. The real score stayed fixed; the corrupted score is one fixed permutation of the same analysis scores (correlation with the real score 0.0214), so its marginal distribution is unchanged and only the patient-level relationship is destroyed. Each permutation is shared by the three rows, so the comparison is paired.

| method | rejects | rate | 95% CI | MC SE | p (inflation) | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| unadjusted | 51 / 1000 | 0.0510 | 0.0382 to 0.0665 | 0.0070 | 0.4625 | holds |
| adjusted | 40 / 1000 | 0.0400 | 0.0287 to 0.0541 | 0.0062 | 0.9402 | holds |
| adjusted, corrupted score | 51 / 1000 | 0.0510 | 0.0382 to 0.0665 | 0.0070 | 0.4625 | holds |

The exact binomial test is one sided for inflation, H0: rate <= 0.05. A verdict of holds means the data are consistent with alpha; it is not proof that the rate is exactly 0.05. The intervals reflect permutation noise for this one fitted score and one corruption, not variability across scores or splits.

## TCGA-BRCA expression scores

Two models are fitted to the same 780 ER-labelled samples, and they answer different questions. The primary model is the project's own permutation-assisted group lasso, so the real-data retrospective exercises the package selector rather than a substitute. The L2 logistic model is a familiar wide-matrix benchmark, kept so the group-lasso number can be read against something. It is not the package method and none of its numbers appear in the ACTG analysis.

### Primary: ppict group lasso

5-fold cross-validation (StratifiedKFold, shuffle, seed 7) of `ppict.fit_prognostic`: inside each training fold, screen the top 300 genes by marginal correlation, group them by absolute co-expression at 0.8, calibrate the group penalty against 10 row permutations at q = 0.1, and refit OLS on the selected genes. ER status is binary, so the refit is a linear probability score; AUC is a ranking metric for it, not a calibrated probability.

Out-of-fold AUC **0.9551** (95% bootstrap CI by the percentile method 0.9345 to 0.9721, 1,000 valid resamples). Per-fold AUC: 0.9625, 0.9463, 0.9410, 0.9530, 0.9718 (mean 0.9549). Genes selected per fold: 48, 36, 88, 35, 41. Mean pairwise Jaccard across fold selections was **0.5966**, so the fold models agree only partly on which genes matter. The final fit on all 780 samples, used for naming genes only, selected 83 genes in 54 groups with training R^2 0.7732.

Largest changes in the linear ER probability score per 1 SD of the gene, final group-lasso fit: ESR1 (+0.16), FOXA1 (-0.07), TBC1D9 (-0.07), CA12 (+0.06), AGR3 (+0.05), PGR (-0.05), GREB1 (+0.05), SLC4A8 (+0.04), SLC7A2 (+0.04), FSIP1 (+0.04), KCNK15 (-0.04), VGLL1 (-0.03), TFF1 (+0.03), NPM2 (-0.03), CCNE1 (-0.03). ESR1, PGR and FOXA1 are all in that list, which is the expected biology for an ER model and a sanity check that the matrix and labels are aligned.

The refit is OLS on a binary outcome, so this is a linear probability score: it is not calibrated, and individual scores can fall outside [0, 1]. AUC uses only the ranking, which is why it is reported for this model instead of a probability-scale metric.

### Baseline: L2 logistic

A separate 5-fold pipeline on the same folds (drop zero-variance genes inside each fold, standardise, select 200 genes by ANOVA F, L2 logistic regression with balanced classes) reached AUC **0.9337** (95% bootstrap CI 0.9093 to 0.9573, 1,000 valid resamples). This is the benchmark, not the package method. The two AUCs share samples and folds, so they are paired measurements on one split; no test of their difference is computed.

Largest standardised logistic coefficients in the final fit: ESR1 (+1.50), TBC1D9 (-1.32), FOXA1 (-1.32), ANXA9 (-1.31), PGR (-1.25), DACH1 (+1.08), FAM134B (+1.06), ANKRA2 (-1.02), PCP2 (+1.01), FAM114A2 (+0.99), BBS4 (-0.93), ABAT (+0.90), TFF1 (+0.90), BCAS1 (+0.90), CA12 (+0.89). ESR1, PGR and FOXA1 are all in that list.

Both numbers come from one 5-fold split at seed 7. The bootstrap intervals resample samples, not fold seeds, so neither interval includes fold-assignment variability. The Jaccard value above is reported for the group lasso only, because that is the model whose refit depends on a selected set; the logistic model has a fixed 200-gene budget.

### What this score is not

It is not a trial prognostic score. The trial analysis uses the clinical score above. The expression model predicts ER status from gene expression in TCGA; applying it to ACTG 175 would need expression measurements for those patients and a mapping to the CD4 outcome. Neither exists in the public data. The script checks and reports 0 shared patients, 0 shared feature names, and 0 expression columns in the trial. The patient check compares identifiers from two different namespaces, so it can only rule out an accidental match; there is no record linkage between TCGA and ACTG.

Its AUC is also not a treatment-effect result. It measures how well ER status can be recovered from expression, not whether adjusting a trial for such a score narrows an interval. Only the ACTG analysis measures that.

## What this split shows, and what it does not

**Observed on this split.** A prognostic score fitted only on held-out controls narrowed the interval on this ACTG split by 17.7%. Under permuted labels the adjusted analysis rejected at 4.0%, and a corrupted score did not inflate it (5.1%). Those are measured for this split and this score, not a distribution over splits or scores.

**Not established.** An expression-derived score transported into a trial. No public route found here joins TCGA expression to ACTG 175. The simulation's transport and corruption scenarios remain simulations.

**The two conditions come apart.** The TCGA score comes from a separate cohort (0 shared identifiers) but cannot be applied to it. The clinical trial score can be applied, but it is an internal split of ACTG 175: it shares no analysis patients, yet it is not an external cohort. The real analysis is therefore a test of the estimator with a held-out internal score, not a test of an external expression score. The simulation's external-score scenarios remain the evidence for that setting.

**Compared with the simulation.** The simulation's good-score width ratio for the same estimator is 0.8469 (400 paired replicates, from `validation/results/null_width_ratios.csv`); the real trial's ratio is 0.8230. The scores have different out-of-sample R^2 and the outcomes differ, so these are two measurements, not a replication of one number. This is a comparison of magnitudes, not a controlled test.

**Still unproven.** The external-cohort PPI term, binary and time-to-event outcomes, a score that is prognostic in one arm but not the other, and the four-arm primary endpoint of ACTG 175. The repository's limits section lists these too.

## Reproduce

```bash
# from the repository root
./.venv/bin/python validation/run_real_data.py --permutations 1000
```

The script downloads the three public files if they are absent, checks their sizes, and stops on a mismatch. To reproduce from a clean result state, delete `REAL_DATA_REPORT.md`, `real_data_results.json`, `actg_scores.csv` and `brca_er_scores.csv` from `validation/results` and run the command again. The data cache can stay; the script records its hashes.

Seeds: split 7, trial score fit 8, corrupted score 9, label permutations 10; expression folds 7 to 11 (seed plus fold index), expression final fit 7, AUC bootstrap 11. Alpha 0.05.


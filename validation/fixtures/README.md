# Shared R and Python fixtures

These CSVs are inputs to both languages. They are generated once by
`validation/make_r_fixtures.py` with seed 20261027; the parity driver does not
regenerate them, and neither language reads the other's output.

Index conventions in the files are 1-based: feature columns are `x1..xp`,
group members are feature numbers, permutation entries are row numbers, and
fold labels run from 1 to 5. The Python driver subtracts one where its API is
0-based.

| file | shape | purpose |
| --- | --- | --- |
| `group_X.csv` | 96 x 14 | correlated block design for grouping and selection |
| `group_y.csv` | 96 | continuous outcome with block signal |
| `group_groups.csv` | 14 rows | group id and member feature |
| `group_lam_grid.csv` | 6 | fixed penalties for solver and calibration checks |
| `group_permutations.csv` | 8 x 96 | explicit row permutations shared by both languages |
| `prognostic_X.csv`, `prognostic_y.csv` | 140 x 12, 140 | score training data |
| `prognostic_Xval.csv`, `prognostic_yval.csv` | 50 x 12, 50 | held-out score evaluation |
| `prognostic_folds.csv` | 140 | fixed 1..5 cross-fitting assignment |
| `prognostic_groups.csv` | block members | groups used by the prognostic fit |
| `ppi_y.csv`, `ppi_t.csv`, `ppi_f.csv` | 180 each | trial rows for the PPI++ mean and ATE checks |
| `ppi_f_unlabeled.csv` | 400 | unlabeled predictions for the mean estimator |
| `procova_cases.csv` | 4 rows | arm SDs, score correlations, allocation, effect, power and size for the design arithmetic |
| `analysis_trial.csv` | 160 rows | CSV-in, analysis-table-out fixture, with a comment header |

The permutation matrix exists because NumPy PCG64 and R's Mersenne-Twister are
different generators. The parity contract is the selector and calibration given
the same permutations, not equality of the RNG streams.

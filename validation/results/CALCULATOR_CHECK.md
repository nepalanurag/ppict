# Calculator check

Produced by `validation/run_calculator_checks.py`. Every number below was
measured in that run; nothing here is typed by hand.

- ppict version: `0.1.0`
- Dataset: `calculator/data/simulated_trial.csv`
- SHA-256: `90d62a6ee95c48fc50c5c89288ab4ac18a905ecfb1bf4d33045426096cf56844`
- Trial rows: 720 (360 treated, 360 control), 600 external
- Inputs: cost per patient 4200, power 0.8, recruitment 25 per month
- App check: checked at http://127.0.0.1:3040, with and without an uploaded copy of the file

## Agreement on identical input

Each row is the largest absolute difference over all four method estimates,
standard errors, interval bounds, widths and width ratios, plus every design
projection.

| comparison | max absolute difference |
| --- | --- |
| Python API to CLI | 0.0e+00 |
| CLI to app route | 0.0e+00 |
| Python API to app route | 0.0e+00 |
| CLI to app route, file uploaded | 0.0e+00 |

Result: exact, against a tolerance of 1e-12.

## Estimates from the shared estimator

| method | estimate | 95% interval | width | width vs unadjusted |
| --- | --- | --- | --- | --- |
| unadjusted | 0.2361 | 0.0565 to 0.4157 | 0.3592 | 1.000 |
| ancova | 0.2732 | 0.1241 to 0.4222 | 0.2981 | 0.830 |
| ppi_single | 0.2732 | 0.1236 to 0.4227 | 0.2992 | 0.833 |
| ppi_per_arm | 0.2530 | 0.0982 to 0.4077 | 0.3095 | 0.862 |

## Design projection

- Adjusted interval narrowing measured on the file: 16.7%
- Required total at 80% power: 852 unadjusted, 588 adjusted
- Patients saved: 264
- Cost avoided: $1,108,800
- Recruitment time saved: 10.6 months

## Statistical analysis plan appendix

- Written to `validation/results/sap_appendix.tex`
- Compilation: passed, 3 pages, 157731 bytes, 0 overfull boxes
- PDF: `validation/results/sap_appendix.pdf`
- Em and en dashes in the appendix: none

## Commands run

```sh
./.venv/bin/python validation/run_calculator_checks.py
python -m ppict.cli analyze --csv calculator/data/simulated_trial.csv --cost 4200 --power 0.8 --patients-per-month 25
```

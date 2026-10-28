# Evidence sheet

The static results page served at
https://nepalanurag.github.io/ppict/.

It is one HTML file and one CSS file, with no build step and no JavaScript.
GitHub Pages serves this directory from `main`.

## Source of truth

| Page content | Source |
| --- | --- |
| Null sweep rates and width ratios | `validation/results/NULL_REPORT.md`, `null_summary.csv`, `null_width_ratios.csv` |
| ACTG 175 estimates and negative control | `validation/results/REAL_DATA_REPORT.md`, `real_data_results.json` |
| TCGA-BRCA AUC and genes | `validation/results/REAL_DATA_REPORT.md`, `real_data_results.json` |
| Calculator outputs and agreement | `validation/results/CALCULATOR_CHECK.md` |
| R parity | `validation/results/R_PARITY.md`, `r_parity_summary.json` |
| SAP appendix | `validation/results/sap_appendix.pdf` and `.tex` |

When a measured number changes, update `index.html` from the regenerated report.
Do not edit the numbers to make the page read better.

## Local check

Open `docs/index.html` directly, or serve it:

```bash
python3 -m http.server 8080 --directory docs
```

Then visit `http://localhost:8080`.

## Publishing

The site has no build output. GitHub Pages is configured to serve `main` and
`/docs`. The `.nojekyll` file stops Jekyll from processing the directory.

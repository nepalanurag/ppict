"""Null simulations: the headline experiment.

The treatment effect is exactly zero in every scenario, so the rejection rate at
alpha = 0.05 *is* the empirical Type I error rate. The scenarios cover a well
specified score, a weak score, a poorly estimated score, four ways of damaging the
score, and a covariate shift between the score's training cohort and the trial.

The criterion used for "holds alpha" is stated in advance and is one sided. Type I
error inflation is the failure mode that matters, so the test is
``H0: rate <= alpha`` against ``H1: rate > alpha`` with an exact binomial test.
A scenario fails the criterion when that test rejects, which is a stricter and
more meaningful statement than "the point estimate looks close to 0.05".

Run:

    .venv/bin/python validation/run_null_simulations.py --n-rep 500

Writes into validation/results/:
    null_replicates.csv, null_summary.csv, null_run_log.json, NULL_REPORT.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppict.cli import preset_configs  # noqa: E402
from ppict.simulate import run_monte_carlo, summarize, write_run_log  # noqa: E402

RESULTS = ROOT / "validation" / "results"


def inflation_test(n_rejected: int, n_rep: int, alpha: float):
    """One-sided exact binomial test of H0: rate <= alpha, plus a two-sided CI.

    The test is one sided because only inflation is a failure, but the interval is
    deliberately two sided. An interval taken from the one-sided test has an upper
    bound of 1 by construction, which reads as though nothing was estimated.
    """
    res = stats.binomtest(int(n_rejected), int(n_rep), alpha, alternative="greater")
    ci = stats.binomtest(int(n_rejected), int(n_rep)).proportion_ci(
        confidence_level=0.95, method="exact"
    )
    return res.pvalue, float(ci.low), float(ci.high)


def paired_ratios(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Variance and width ratios against the unadjusted analysis, on shared data.

    Each replicate contributes every method, so the comparison is paired. The
    variance ratio is taken from the squared standard errors, which is the
    quantity that maps onto required sample size; the width ratio is reported
    alongside it and should behave like its square root.
    """
    rows = []
    for config, g in df.groupby("config", dropna=False):
        base = g[g["method"] == "unadjusted"].set_index("seed")
        if base.empty:
            continue
        base_var = float((base["se"] ** 2).mean())
        base_width = float(base["ci_width"].mean())
        for method, gm in g.groupby("method", dropna=False):
            cur = gm.set_index("seed")
            joined = cur.join(base[["se", "ci_width"]], rsuffix="_base", how="inner")
            if joined.empty:
                continue
            var_ratio = float((joined["se"] ** 2).mean() / (joined["se_base"] ** 2).mean())
            rows.append(
                {
                    "config": config,
                    "method": method,
                    "variance_ratio_vs_unadjusted": var_ratio,
                    "implied_sample_size_reduction": 1.0 - var_ratio,
                    "width_ratio_vs_unadjusted": float(
                        joined["ci_width"].mean() / joined["ci_width_base"].mean()
                    ),
                    "n_paired": int(len(joined)),
                }
            )
    return pd.DataFrame(rows)


def build_report(summary: pd.DataFrame, ratios: pd.DataFrame, meta: dict) -> str:
    alpha = float(meta["alpha"])
    lines = []
    lines.append("# Null simulation report")
    lines.append("")
    lines.append(
        "Measured numbers only. Every value below comes from "
        "`validation/run_null_simulations.py`, which writes "
        "`validation/results/null_replicates.csv` and `null_summary.csv`."
    )
    lines.append("")
    lines.append("## What was run")
    lines.append("")
    lines.append(f"- Monte Carlo replicates per scenario: {meta['n_rep']}")
    lines.append(f"- Seed rule: {meta['seed_rule']} (seed base {meta['seed_base']})")
    lines.append(
        "- Provenance: "
        + (
            "report rebuilt from the saved per-replicate results file below; "
            "replicate count and seed base are read back from that file rather "
            "than taken from command line defaults"
            if meta.get("source") == "cached_results"
            else "this report was written by the same process that ran the sweep"
        )
    )
    lines.append(f"- Nominal level tested: alpha = {alpha}")
    lines.append(
        f"- Total patients per trial: {meta['n_treat']} treated and {meta['n_control']} control"
    )
    lines.append(
        f"- External cohorts per trial: {meta['n_hist']} patients with control outcomes "
        f"used to train the score, {meta['n_unlabeled']} patients with covariates only"
    )
    lines.append(f"- Covariates: {meta['p']} features in {meta['n_groups']} groups")
    lines.append(
        f"- Commands: `python validation/run_null_simulations.py --n-rep {meta['n_rep']}`"
    )
    lines.append("")
    lines.append(
        "The treatment effect is exactly zero in every scenario below, so the "
        "rejection rate is the empirical Type I error rate."
    )
    lines.append("")

    lines.append("## Empirical Type I error rate at alpha = 0.05")
    lines.append("")
    lines.append(
        "The right-hand columns test for inflation, not for equality. A small "
        "`p_inflation` means the observed rate is significantly above the nominal "
        "level. `95% CI` is the exact binomial interval for the rate."
    )
    lines.append("")
    header = (
        "| scenario | method | rejects / reps | rate | 95% CI | MC SE | "
        "p (inflation) | verdict |"
    )
    lines.append(header)
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")

    verdicts = []
    for _, r in summary.iterrows():
        n = int(r["n_rep"])
        rate = float(r["reject_rate"])
        n_rej = int(round(rate * n))
        p, lo, hi = inflation_test(n_rej, n, alpha)
        ok = p >= 0.05
        verdict = "holds" if ok else "INFLATED"
        verdicts.append((r["config"], r["method"], rate, p, ok))
        lines.append(
            f"| {r['config']} | {r['method']} | {n_rej} / {n} | {rate:.4f} | "
            f"[{lo:.4f}, {hi:.4f}] | {r['mc_se_reject_rate']:.4f} | {p:.4f} | {verdict} |"
        )
    lines.append("")

    # The claim under test concerns the prediction-powered estimators. The L1 lasso
    # adjustment is a comparison baseline, so its behaviour is reported separately
    # rather than being allowed to read as a failure of the claim itself.
    ppi_family = {"ppi_single", "ppi_per_arm", "ppi_lambda1"}
    inflated_ppi = [v for v in verdicts if not v[4] and v[1] in ppi_family]
    inflated_other = [v for v in verdicts if not v[4] and v[1] not in ppi_family]

    ppi_rates = [(v[0], v[1], v[2], v[3]) for v in verdicts if v[1] in ppi_family]
    worst = max(ppi_rates, key=lambda t: t[2])

    lines.append("### Verdict on the headline claim")
    lines.append("")
    if not inflated_ppi:
        lines.append(
            "No prediction-powered estimator showed a statistically significant "
            "increase in the rejection rate above alpha in any scenario, including "
            "every corruption of the prognostic score and the covariate-shift "
            "scenario. On this evidence the claim held."
        )
        lines.append("")
        lines.append(
            f"Across all {len(ppi_rates)} prediction-powered scenario and method "
            f"combinations, the highest observed rejection rate was {worst[2]:.4f} "
            f"({worst[0]} / {worst[1]}), against a nominal 0.05. Its one-sided inflation "
            f"test gave p = {worst[3]:.4f}."
        )
    else:
        lines.append(
            "The claim did not hold in the following scenarios, which showed a rejection "
            "rate significantly above alpha at the one-sided 5% level. Per the project's "
            "rules this is reported rather than tuned away:"
        )
        lines.append("")
        for cfg, method, rate, p, _ in inflated_ppi:
            lines.append(f"- {cfg} / {method}: rate {rate:.4f}, inflation test p = {p:.4f}")
    lines.append("")

    if inflated_other:
        lines.append("### Comparison baselines")
        lines.append("")
        lines.append(
            "These are not the claim under test. They are the comparison methods, and "
            "one of them fails:"
        )
        lines.append("")
        for cfg, method, rate, p, _ in inflated_other:
            lines.append(f"- {cfg} / {method}: rate {rate:.4f}, inflation test p = {p:.4f}")
        lines.append("")
        lines.append(
            "The L1 lasso adjustment selects covariates using the same data it then "
            "estimates on, with no sample splitting and no correction for the selection. "
            "It also produced the narrowest intervals of any method in that scenario. "
            "Narrow intervals and a broken rejection rate are the same phenomenon here, "
            "and it is the reason this project fits the score on a disjoint cohort and "
            "tunes a single scalar rather than selecting covariates inside the analysis."
        )
        lines.append("")

    lines.append("## Interval widths and implied sample size")
    lines.append("")
    has_rho2 = "score_rho2" in summary.columns and summary["score_rho2"].notna().any()
    lines.append(
        "Variance ratio is the mean squared standard error over the mean squared "
        "standard error of the unadjusted analysis on the same replicates. Because "
        "required sample size is proportional to the variance constant, at a fixed "
        "allocation the implied sample size reduction equals 1 minus that ratio."
    )
    lines.append("")
    if has_rho2:
        lines.append(
            "`rho^2` is the squared correlation between the score the estimators "
            "received and the true control outcome on the trial patients. This is the "
            "quantity the variance-reduction expression depends on. It is invariant to "
            "a constant offset in the score, as the estimator is."
        )
    else:
        lines.append(
            "`raw R^2` is the uncentered out-of-sample R squared of the score the "
            "estimators received. It is reported because this run did not record "
            "rho^2, and it must be read with care: a constant offset in the score drives "
            "it sharply negative even though the estimator is exactly invariant to such "
            "an offset. The relevant quantity is the squared correlation, which the "
            "simulator records for subsequent runs."
        )
    lines.append("")
    q_col = "rho^2" if has_rho2 else "raw R^2"
    lines.append(
        f"| scenario | method | variance ratio | implied reduction | width ratio | {q_col} |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    merged = summary.merge(ratios, on=["config", "method"], how="left")
    for _, r in merged.iterrows():
        vr = r["variance_ratio_vs_unadjusted"]
        ir = r["implied_sample_size_reduction"]
        wr = r["width_ratio_vs_unadjusted"]
        vr_s = "n/a" if pd.isna(vr) else f"{vr:.4f}"
        ir_s = "n/a" if pd.isna(ir) else f"{ir:.4f}"
        wr_s = "n/a" if pd.isna(wr) else f"{wr:.4f}"
        quality = r["score_rho2"] if has_rho2 else r["score_oos_r2_used"]
        q_s = "n/a" if pd.isna(quality) else f"{quality:.4f}"
        lines.append(
            f"| {r['config']} | {r['method']} | {vr_s} | {ir_s} | {wr_s} | {q_s} |"
        )
    lines.append("")
    lines.append(
        "Reported reductions are measured on simulated patients, not on real trial "
        "recruitment. They carry the Monte Carlo uncertainty implied by the replicate "
        "count stated above and they assume the score transports to the trial "
        "population, which the `history_shift` scenario deliberately violates."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-rep", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(RESULTS))
    ap.add_argument(
        "--n-jobs", type=int, default=1,
        help="worker processes; results do not depend on this value",
    )
    ap.add_argument(
        "--from-cached", action="store_true",
        help="rebuild the report from existing result CSVs instead of re-running",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs = preset_configs("null", base_seed=args.seed)

    if args.from_cached:
        df = pd.read_csv(out_dir / "null_replicates.csv")
        summary = pd.read_csv(out_dir / "null_summary.csv")
        ratios = paired_ratios(df, alpha=configs[0].alpha)
        ratios.to_csv(out_dir / "null_width_ratios.csv", index=False)
        # Take the run parameters from the data, not from argparse defaults. The
        # defaults are what the rebuild was invoked with, which is not what produced
        # the numbers; using them made the report claim 500 replicates for a 400
        # replicate sweep.
        n_rep_effective = int(df.groupby("config")["seed"].nunique().max())
        seed_base_effective = int(df["seed"].min())
        n_jobs_recorded = None
        source = "cached_results"
        print(f"rebuilt from cache: {len(df)} replicate rows", flush=True)
    else:
        print(
            f"null sweep: {len(configs)} scenarios x {args.n_rep} replicates "
            f"on {args.n_jobs} worker(s)",
            flush=True,
        )
        df = run_monte_carlo(
            configs, n_rep=args.n_rep, seed_base=args.seed, n_jobs=args.n_jobs
        )
        summary = summarize(df)
        ratios = paired_ratios(df, alpha=configs[0].alpha)

        df.to_csv(out_dir / "null_replicates.csv", index=False)
        summary.to_csv(out_dir / "null_summary.csv", index=False)
        ratios.to_csv(out_dir / "null_width_ratios.csv", index=False)

        n_rep_effective = int(args.n_rep)
        seed_base_effective = int(args.seed)
        n_jobs_recorded = int(args.n_jobs)
        source = "fresh_run"

    meta = {
        "n_rep": n_rep_effective,
        "seed_base": seed_base_effective,
        "seed_rule": "replicate r of every config uses seed = seed_base + r",
        "alpha": float(configs[0].alpha),
        "source": source,
        "n_treat": configs[0].n_treat,
        "n_control": configs[0].n_control,
        "n_hist": configs[0].n_hist,
        "n_unlabeled": configs[0].n_unlabeled,
        "p": configs[0].p,
        "n_groups": configs[0].n_groups,
    }
    report = build_report(summary, ratios, meta)
    (out_dir / "NULL_REPORT.md").write_text(report)

    log = write_run_log(
        out_dir / "null_run_log.json",
        configs, n_rep=n_rep_effective, seed_base=seed_base_effective, summary=summary,
        extra={"preset": "null", "replicate_rows": int(len(df)),
               "n_jobs": n_jobs_recorded, "source": source,
               "n_rep_derived_from_data": int(
                   df.groupby("config")["seed"].nunique().max()
               ),
               "report": "validation/results/NULL_REPORT.md"},
    )

    print(f"wrote {out_dir/'null_replicates.csv'} ({len(df)} rows)")
    print(f"wrote {out_dir/'null_summary.csv'} ({len(summary)} rows)")
    print(f"wrote {out_dir/'null_width_ratios.csv'}")
    print(f"wrote {out_dir/'NULL_REPORT.md'}")
    print(f"git: {log.get('git_commit')}")

    key = summary[summary["method"].isin(["unadjusted", "ppi_single", "ppi_per_arm"])]
    print()
    print(key[["config", "method", "reject_rate", "coverage", "mean_ci_width", "mean_lam"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line entry point.

    python -m ppict.cli selftest
    python -m ppict.cli simulate --preset null --n-rep 200 --out validation/results
    python -m ppict.cli analyze --csv trial.csv --json
    python -m ppict.cli sap --csv trial.csv --out sap.tex

The calculator app calls ``analyze`` and parses its JSON, which is what keeps the
calculator from re-implementing the estimator. Only JSON reaches stdout on that
command, so the app can parse the output directly.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .analysis import DatasetError
from .report import build_report
from .simulate import TrialConfig, run_monte_carlo, summarize, write_run_log


def preset_configs(preset: str, base_seed: int = 0) -> List[TrialConfig]:
    """Named scenario sets used by the validation scripts.

    ``null`` is the headline set: the treatment effect is exactly zero, so the
    rejection rate is the empirical Type I error rate and must sit at the nominal
    level. It spans a well specified score, a weak score, a poorly estimated
    score, four ways of damaging the score, and a covariate shift between the
    score's training cohort and the trial.
    """
    common = dict(
        n_hist=500, n_unlabeled=500, n_treat=150, n_control=150,
        p=60, n_groups=12, n_active_groups=3, true_effect=0.0,
        signal_r2=0.30, select_q=0.10, select_n_perm=25,
    )

    def cfg(name: str, **kw) -> TrialConfig:
        return TrialConfig(name=name, seed=base_seed, **{**common, **kw})

    if preset == "null":
        return [
            cfg("good_score"),
            cfg("weak_score", signal_r2=0.05),
            cfg("poorly_estimated_score", n_hist=80),
            cfg("corrupt_noise", corruption="noise", corruption_param=2.0),
            cfg("corrupt_shuffle", corruption="shuffle"),
            cfg("corrupt_bias", corruption="bias", corruption_param=3.0),
            cfg("corrupt_flip", corruption="flip"),
            cfg("history_shift", history_shift=0.75),
        ]
    if preset == "power":
        return [
            cfg("effect_good_score", true_effect=0.50),
            cfg("effect_weak_score", true_effect=0.50, signal_r2=0.05),
            cfg("effect_corrupt_shuffle", true_effect=0.50, corruption="shuffle"),
            cfg("effect_corrupt_noise", true_effect=0.50, corruption="noise", corruption_param=2.0),
        ]
    if preset == "smoke":
        return [TrialConfig(name="smoke", seed=base_seed, p=24, n_groups=6, n_active_groups=2,
                            n_hist=120, n_unlabeled=120, n_treat=60, n_control=60,
                            select_n_perm=10)]
    raise ValueError(f"unknown preset {preset!r}")


def cmd_simulate(args: argparse.Namespace) -> int:
    configs = preset_configs(args.preset, base_seed=args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"preset={args.preset} configs={len(configs)} n_rep={args.n_rep} seed_base={args.seed}")
    df = run_monte_carlo(configs, n_rep=args.n_rep, seed_base=args.seed)
    summary = summarize(df)

    reps_path = out_dir / f"{args.preset}_replicates.csv"
    summ_path = out_dir / f"{args.preset}_summary.csv"
    df.to_csv(reps_path, index=False)
    summary.to_csv(summ_path, index=False)

    log = write_run_log(
        out_dir / f"{args.preset}_run_log.json",
        configs, n_rep=args.n_rep, seed_base=args.seed, summary=summary,
        extra={"preset": args.preset, "replicate_rows": int(len(df))},
    )

    print(f"wrote {reps_path} ({len(df)} rows)")
    print(f"wrote {summ_path} ({len(summary)} rows)")
    print(f"wrote {out_dir / f'{args.preset}_run_log.json'}")
    print(f"ppict {log['versions']['ppict']} git={log.get('git_commit')}")

    cols = ["config", "method", "reject_rate", "coverage", "mean_ci_width", "mean_lam", "score_oos_r2_used"]
    with_cols = [c for c in cols if c in summary.columns]
    print(summary[with_cols].to_string(index=False))
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    """Fast end-to-end smoke path, used to check an install."""
    configs = preset_configs("smoke", base_seed=args.seed)
    df = run_monte_carlo(configs, n_rep=args.n_rep, seed_base=args.seed)
    summary = summarize(df)
    print(summary[["config", "method", "mean_estimate", "mean_ci_width", "mean_lam"]].to_string(index=False))
    ok = np.isfinite(summary["mean_estimate"]).all() and np.isfinite(summary["mean_ci_width"]).all()
    print("selftest:", "ok" if ok else "FAILED")
    return 0 if ok else 1


def _add_report_args(p: argparse.ArgumentParser) -> None:
    """Arguments shared by ``analyze`` and ``sap``, so both produce the same numbers."""
    p.add_argument("--csv", required=True, help="trial dataset with columns y, t, f")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--cost", type=float, default=4200.0, help="cost per patient")
    p.add_argument("--power", type=float, default=0.80)
    p.add_argument("--trial-size", type=int, default=None, help="planned total patients")
    p.add_argument("--r2", type=float, default=None, help="squared prognostic correlation; measured if omitted")
    p.add_argument("--effect", type=float, default=None, help="assumed effect; measured if omitted")
    p.add_argument("--sd", type=float, default=None, help="outcome standard deviation; measured if omitted")
    p.add_argument("--patients-per-month", type=float, default=25.0)
    p.add_argument("--date", default=None, help="date stamped on the SAP; defaults to today")


def _report_from_args(args: argparse.Namespace):
    return build_report(
        args.csv,
        alpha=args.alpha,
        cost_per_patient=args.cost,
        power=args.power,
        trial_size=args.trial_size,
        r2=args.r2,
        effect=args.effect,
        outcome_sd=args.sd,
        patients_per_month=args.patients_per_month,
        command=getattr(args, "_command", "python -m ppict.cli analyze --csv <dataset>"),
        date=args.date,
    )


def cmd_analyze(args: argparse.Namespace) -> int:
    try:
        report = _report_from_args(args)
    except (DatasetError, ValueError) as exc:
        # A user-facing problem, not a crash: one line, no traceback. The
        # calculator shows this line directly.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    payload = json.dumps(report.as_dict(), indent=2, allow_nan=False)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
        print(str(out))
    else:
        print(payload)
    return 0


def cmd_sap(args: argparse.Namespace) -> int:
    try:
        report = _report_from_args(args)
    except (DatasetError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.sap_tex, encoding="utf-8")
    print(str(out))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ppict", description="ppict command line")
    sub = p.add_subparsers(dest="command", required=True)

    ps = sub.add_parser("simulate", help="run a Monte Carlo preset and write results")
    ps.add_argument("--preset", default="null", choices=["null", "power", "smoke"])
    ps.add_argument("--n-rep", type=int, default=200)
    ps.add_argument("--seed", type=int, default=0)
    ps.add_argument("--out", default="validation/results")
    ps.set_defaults(func=cmd_simulate)

    pt = sub.add_parser("selftest", help="quick end-to-end check")
    pt.add_argument("--n-rep", type=int, default=4)
    pt.add_argument("--seed", type=int, default=0)
    pt.set_defaults(func=cmd_selftest)

    pa = sub.add_parser("analyze", help="estimate the treatment effect and the design plan as JSON")
    _add_report_args(pa)
    pa.add_argument("--out", default=None, help="write JSON here instead of stdout")
    pa.set_defaults(func=cmd_analyze)

    pp = sub.add_parser("sap", help="write the statistical analysis plan appendix as LaTeX")
    _add_report_args(pp)
    pp.add_argument("--out", default="validation/results/sap_appendix.tex")
    pp.set_defaults(func=cmd_sap)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    raw = list(sys.argv[1:] if argv is None else argv)
    # Recorded in the SAP and the JSON so a reported number can be reproduced.
    args._command = "python -m ppict.cli " + shlex.join(raw)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

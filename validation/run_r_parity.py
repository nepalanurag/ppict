"""Measure agreement between the Python and R ppict implementations.

Both sides read the same fixture files. The Python results are computed here;
the R results are computed by validation/r_parity.R after a source install.
The check writes validation/results/R_PARITY.md and fails if any compared
value is outside the stated tolerance.

The permutation matrix is a fixture rather than an RNG stream. NumPy's PCG64
and R's Mersenne-Twister do not produce the same draws, so parity is asserted
for the deterministic selector given identical permutations, not for the
draws themselves.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppict.analysis import analyze_trial, load_trial_dataset  # noqa: E402
from ppict.group_select import (  # noqa: E402
    _nonzero_groups,
    _univariate_z,
    define_groups,
    group_lasso_fista,
    group_lasso_path,
    group_pen_from_z,
    pairwise_jaccard,
    screen_features,
)
from ppict.ppi import (  # noqa: E402
    ate_ancova,
    ate_ppi_single_lambda,
    ate_unadjusted,
    control_arm_size,
    optimal_allocation,
    ppi_mean_ci,
    ppi_mean_pointestimate,
    ppi_mean_se,
    procova_variance,
    required_n_total,
    sample_size_reduction,
    tuned_lambda_mean,
    unadjusted_min_variance,
)
from ppict.prognostic import fit_prognostic  # noqa: E402

FIXTURES = ROOT / "validation" / "fixtures"
R_OUT = ROOT / "validation" / "results" / "r_parity_values.csv"
REPORT = ROOT / "validation" / "results" / "R_PARITY.md"
TOL = 1e-6


def load(name: str, skip_header: bool = True) -> np.ndarray:
    return np.loadtxt(FIXTURES / name, delimiter=",", skiprows=1 if skip_header else 0)


def load_groups(name: str) -> List[np.ndarray]:
    raw = np.loadtxt(FIXTURES / name, delimiter=",", skiprows=1, dtype=int)
    return [raw[raw[:, 0] == g, 1] - 1 for g in sorted(set(raw[:, 0]))]


def fmt(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.17g}"


def strings(values: np.ndarray) -> str:
    return " ".join(str(int(v) + 1) for v in values)


def python_values() -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    X = load("group_X.csv")
    y = load("group_y.csv").ravel()
    groups = load_groups("group_groups.csv")
    lam_grid = load("group_lam_grid.csv").ravel()
    perms = load("group_permutations.csv", skip_header=False).astype(int)

    found = define_groups(X, threshold=0.6)
    out["groups|group_count"] = len(found)
    for j, g in enumerate(found):
        out[f"groups|group_{j + 1}"] = strings(g)

    idx, r = screen_features(X, y, top_k=6)
    out["screen|screen_index"] = strings(idx)
    for i, value in enumerate(r):
        out[f"screen|screen_r_{i + 1}"] = value

    weights = group_pen_from_z(_univariate_z(X, y), groups)
    for j, value in enumerate(weights):
        out[f"weights|weight_{j + 1}"] = value

    for k, lam in enumerate(lam_grid, start=1):
        coef = group_lasso_fista(X, y, groups, lam, weights, max_iter=500, tol=1e-9)
        for i, value in enumerate(coef, start=1):
            out[f"solver|coef_lam{k}_f{i}"] = value

    counts = np.zeros((perms.shape[0], len(lam_grid)), dtype=int)
    for b, perm in enumerate(perms):
        path = group_lasso_path(X, y[perm - 1], groups, lam_grid, weights)
        for k in range(len(lam_grid)):
            counts[b, k] = _nonzero_groups(path[k], groups).size
    for b in range(counts.shape[0]):
        for k in range(counts.shape[1]):
            out[f"null_counts|null_p{b + 1}_g{k + 1}"] = int(counts[b, k])

    mean_counts = counts.mean(axis=0)
    budget = 0.2 * len(groups)
    ok = np.flatnonzero(mean_counts <= budget)
    chosen = float(lam_grid[np.argmin(lam_grid[ok])]) if ok.size else float(lam_grid.max())
    out["calibration|cal_lam"] = chosen
    for k, value in enumerate(mean_counts, start=1):
        out[f"calibration|cal_mean_{k}"] = value

    Xp = load("prognostic_X.csv")
    yp = load("prognostic_y.csv").ravel()
    Xv = load("prognostic_Xval.csv")
    yv = load("prognostic_yval.csv").ravel()
    folds = load("prognostic_folds.csv").ravel().astype(int)
    pgroups = load_groups("prognostic_groups.csv")
    pweights = group_pen_from_z(_univariate_z(Xp, yp), pgroups)

    score = fit_prognostic(
        Xp, yp, groups=pgroups, weights=pweights, lam=0.02,
        X_val=Xv, y_val=yv,
    )
    out["prognostic|prog_intercept"] = score.intercept
    out["prognostic|prog_features"] = strings(score.feature_idx)
    out["prognostic|prog_train_r2"] = score.train_r2
    out["prognostic|prog_oos_r2"] = score.oos_r2
    for i, value in enumerate(score.coef, start=1):
        out[f"prognostic|prog_coef_{i}"] = value
    for i, value in enumerate(score.predict(Xv), start=1):
        out[f"prognostic|prog_pred_{i}"] = value

    oof = np.full(yp.size, np.nan)
    sets: List[np.ndarray] = []
    for k in range(1, 6):
        train = folds != k
        test = folds == k
        model = fit_prognostic(
            Xp[train], yp[train], groups=pgroups, weights=pweights, lam=0.02
        )
        oof[test] = model.predict(Xp[test])
        sets.append(model.feature_idx)
    jac = pairwise_jaccard(sets)
    out["crossfit|oof_r2"] = 1 - np.sum((yp - oof) ** 2) / np.sum((yp - yp.mean()) ** 2)
    out["crossfit|fold_jaccard_mean"] = float(jac.mean()) if jac.size else float("nan")
    for i, value in enumerate(oof, start=1):
        out[f"crossfit|oof_{i}"] = value

    y_ppi = load("ppi_y.csv").ravel()
    t_ppi = load("ppi_t.csv").ravel().astype(int)
    f_ppi = load("ppi_f.csv").ravel()
    f_unl = load("ppi_f_unlabeled.csv").ravel()
    lam = tuned_lambda_mean(y_ppi, f_ppi, f_unl)
    ci = ppi_mean_ci(y_ppi, f_ppi, f_unl, lam=lam)
    out["ppi_mean|ppi_lam"] = lam
    out["ppi_mean|ppi_point"] = ppi_mean_pointestimate(y_ppi, f_ppi, f_unl, lam=lam)
    out["ppi_mean|ppi_se"] = ppi_mean_se(y_ppi, f_ppi, f_unl, lam=lam)
    out["ppi_mean|ppi_lo"] = ci[0]
    out["ppi_mean|ppi_hi"] = ci[1]

    estimates = {
        "unadjusted": ate_unadjusted(y_ppi, t_ppi),
        "ancova": ate_ancova(y_ppi, t_ppi, f_ppi.reshape(-1, 1)),
        "ppi_single": ate_ppi_single_lambda(y_ppi, t_ppi, f_ppi),
    }
    for method, est in estimates.items():
        for field in ("estimate", "se", "ci_low", "ci_high", "lam"):
            out[f"ate|ate_{method}_{field}"] = getattr(est, field)

    with (FIXTURES / "procova_cases.csv").open() as fh:
        for row in csv.DictReader(fh):
            case = row["case"]
            s0 = float(row["sigma0"])
            s1 = float(row["sigma1"])
            r0 = float(row["rho0"])
            r1 = float(row["rho1"])
            pi0 = float(row["pi0"])
            effect = float(row["effect"])
            alpha = float(row["alpha"])
            power = float(row["power"])
            n_total = float(row["n_total"])

            v = procova_variance(s0, s1, r0, r1, pi0)
            unadj = unadjusted_min_variance(s0, s1)
            alloc = optimal_allocation(s0, s1, r0, r1)
            out[f"procova|var_{case}"] = v
            out[f"procova|unadj_min_{case}"] = unadj
            out[f"procova|alloc_pi0_{case}"] = alloc.pi0
            out[f"procova|alloc_variance_{case}"] = alloc.variance
            out[f"procova|alloc_ratio_{case}"] = alloc.variance_ratio
            out[f"procova|alloc_reduction_{case}"] = alloc.variance_reduction
            out[f"procova|alloc_unadj_pi0_{case}"] = alloc.unadjusted_pi0
            out[f"procova|alloc_success_{case}"] = 1 if alloc.optimization_success else 0
            out[f"procova|required_n_{case}"] = required_n_total(
                v, effect, alpha, power
            )
            out[f"procova|sample_reduction_{case}"] = sample_size_reduction(unadj, v)
            out[f"procova|control_arm_{case}"] = control_arm_size(n_total, alloc.pi0)

    dataset = load_trial_dataset(FIXTURES / "analysis_trial.csv")
    result = analyze_trial(dataset)
    out["analysis|analysis_score_r2_control"] = result.score_r2_control
    out["analysis|analysis_outcome_sd"] = result.outcome_sd
    for name, est in result.estimates.items():
        for field in ("estimate", "se", "ci_low", "ci_high", "ci_width", "p_value", "lam"):
            out[f"analysis|analysis_{name}_{field}"] = getattr(est, field)
    return out


def run_r() -> str:
    (ROOT / "r" / "lib").mkdir(parents=True, exist_ok=True)
    install = subprocess.run(
        ["R", "CMD", "INSTALL", "--library", str(ROOT / "r" / "lib"), str(ROOT / "r" / "ppict")],
        cwd=ROOT, capture_output=True, text=True,
    )
    if install.returncode != 0:
        raise RuntimeError("R CMD INSTALL failed:\n" + install.stdout + install.stderr)
    run = subprocess.run(
        ["Rscript", str(ROOT / "validation" / "r_parity.R"), str(FIXTURES), str(R_OUT)],
        cwd=ROOT, capture_output=True, text=True,
    )
    if run.returncode != 0:
        raise RuntimeError("r_parity.R failed:\n" + run.stdout + run.stderr)
    version = subprocess.run(
        ["Rscript", "-e", "cat(R.version.string)"],
        cwd=ROOT, capture_output=True, text=True,
    )
    return version.stdout.strip() or "unknown"


def read_r() -> Dict[str, str]:
    with R_OUT.open() as fh:
        return {
            f"{row['check']}|{row['key']}": row["value"]
            for row in csv.DictReader(fh)
        }


def compare(expected: Dict[str, Any], actual: Dict[str, str]) -> Tuple[List[Dict[str, Any]], bool]:
    rows: List[Dict[str, Any]] = []
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    by_check: Dict[str, List[Tuple[str, Any, str]]] = {}
    for key, exp in expected.items():
        check = key.split("|", 1)[0]
        by_check.setdefault(check, []).append((key, exp, actual.get(key, "")))

    overall = not missing and not extra
    for check in sorted(by_check):
        items = by_check[check]
        numeric_diffs: List[float] = []
        exact_ok = 0
        failures: List[str] = []
        for key, exp, raw in items:
            if isinstance(exp, str):
                if raw == exp:
                    exact_ok += 1
                else:
                    failures.append(f"{key}: expected {exp!r}, got {raw!r}")
                continue
            try:
                got = float(raw)
            except (TypeError, ValueError):
                failures.append(f"{key}: non-numeric R value {raw!r}")
                continue
            if math.isnan(float(exp)) and math.isnan(got):
                exact_ok += 1
                continue
            diff = abs(got - float(exp))
            numeric_diffs.append(diff)
            if not math.isclose(got, float(exp), rel_tol=TOL, abs_tol=TOL):
                failures.append(f"{key}: expected {exp!r}, got {got!r}, diff {diff:.3e}")
        passed = not failures
        overall = overall and passed
        rows.append(
            {
                "check": check,
                "values": len(items),
                "numeric": len(numeric_diffs),
                "max_abs_diff": max(numeric_diffs) if numeric_diffs else 0.0,
                "failures": failures,
                "passed": passed,
            }
        )
    if missing:
        overall = False
    if extra:
        overall = False
    return rows, overall


def fixture_table() -> List[Tuple[str, int]]:
    out = []
    for path in sorted(FIXTURES.glob("*.csv")):
        if path.is_file():
            with path.open("rb") as fh:
                rows = sum(1 for _ in fh)
            out.append((path.name, rows))
    return out


def main() -> int:
    expected = python_values()
    try:
        r_version = run_r()
        actual = read_r()
    except Exception as exc:  # noqa: BLE001 - report the blocker, do not hide it
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(
            "# R parity check\n\n"
            "The R side could not be run, so no agreement is claimed.\n\n"
            f"```\n{exc}\n```\n",
            encoding="utf-8",
        )
        print(f"BLOCKED: {exc}")
        return 1

    rows, passed = compare(expected, actual)
    total_values = sum(row["values"] for row in rows)
    max_diff = max((row["max_abs_diff"] for row in rows), default=0.0)

    lines = [
        "# R parity check",
        "",
        "Measured agreement between the Python package and the local R port.",
        "Both sides read the same CSV fixtures; neither reads the other's output.",
        "",
        f"- Result: **{'PASS' if passed else 'FAIL'}**",
        f"- Compared values: {total_values}",
        f"- Tolerance: abs and rel {TOL:g} for numeric values, exact for group and feature sets",
        f"- Largest numeric difference: {max_diff:.3e}",
        f"- R: {r_version}",
        f"- Command: `./.venv/bin/python validation/run_r_parity.py`",
        "",
        "## Checks",
        "",
        "| check | values | numeric | largest abs difference | result |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['check']} | {row['values']} | {row['numeric']} | "
            f"{row['max_abs_diff']:.3e} | {'pass' if row['passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## What is compared",
        "",
        "- Group definitions from absolute correlation, and the top-six screening order.",
        "- Adaptive group weights, FISTA coefficients at six fixed penalties, and the",
        "  selected-group counts under the same eight explicit permutations.",
        "- The calibration rule applied to those counts, including the chosen penalty.",
        "- Prognostic coefficients, selected features, out-of-fold predictions and",
        "  out-of-fold R squared on fixed folds.",
        "- PPI++ mean tuning, point estimate, standard error and interval.",
        "- Unadjusted, ANCOVA and shared-tuning PPI++ treatment effects with intervals.",
        "- The PROCOVA variance bound, optimal allocation, required total, variance",
        "  reduction and control-arm size on four design cases.",
        "- The CSV-in, analysis-table-out entry point on the same trial file.",
        "",
        "## What is not compared, and why",
        "",
        "- The permutation draws themselves. NumPy PCG64 and R's Mersenne-Twister",
        "  are different generators, so the fixture supplies one permutation matrix",
        "  to both languages. The calibration arithmetic and the selector are",
        "  compared; the RNG streams are not claimed to match.",
        "- The per-arm external-cohort estimator. It is not ported; the R analysis",
        "  entry point stops with a named error if external rows are present.",
        "- The simulation sweep, the calculator and the LaTeX export. They remain",
        "  Python-only.",
        "- The largest numeric difference is the symmetric allocation, where R returns",
        "  the exact minimiser 0.5 and Python's bounded optimiser stops a few ulps away.",
        "  R uses the exact value so that control_arm_size does not flip by one patient",
        "  at an integer boundary.",
        "",
        "## Fixtures",
        "",
        "Generated by `validation/make_r_fixtures.py` with seed 20261027.",
        "",
        "| file | lines |",
        "| --- | --- |",
    ]
    for name, lines_count in fixture_table():
        lines.append(f"| `{name}` | {lines_count} |")
    lines += [
        "",
        "The R package was installed from `r/ppict` into `r/lib` by this check.",
        "Its own tests are run separately:",
        "",
        "```bash",
        "R CMD INSTALL --library=r/lib r/ppict",
        "R_LIBS=\"$PWD/r/lib\" Rscript r/ppict/tests/run_tests.R",
        "```",
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    with (ROOT / "validation" / "results" / "r_parity_summary.json").open("w") as fh:
        json.dump(
            {
                "passed": passed,
                "tolerance": TOL,
                "values": total_values,
                "max_abs_diff": max_diff,
                "checks": [
                    {k: v for k, v in row.items() if k != "failures"} for row in rows
                ],
                "r_version": r_version,
                "fixtures": [
                    {
                        "name": name,
                        "sha256": hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest(),
                    }
                    for name, _ in fixture_table()
                ],
            },
            fh,
            indent=2,
        )
        fh.write("\n")

    if not passed:
        print(f"FAIL: see {REPORT}")
        for row in rows:
            for failure in row["failures"][:5]:
                print("  " + failure)
        return 1
    print(f"PASS: {total_values} values, largest difference {max_diff:.3e}")
    print(f"Wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

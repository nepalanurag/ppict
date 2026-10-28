"""Write the shared fixtures used by the R and Python parity checks.

Every fixture is deterministic from one seed. The files are inputs only: no
expected outputs are written here, so both languages compute their own results.
Feature indices, group ids, permutation entries and fold labels are 1-based in
the CSVs; the Python driver subtracts one where its API is 0-based.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppict.group_select import define_groups  # noqa: E402

FIXTURES = ROOT / "validation" / "fixtures"
SEED = 20261027


def _save(name: str, array: np.ndarray, header: str | None = None) -> None:
    np.savetxt(
        FIXTURES / name,
        np.atleast_2d(array),
        delimiter=",",
        header=header or "",
        comments="",
        fmt="%.17g",
    )


def make_group_fixture(rng: np.random.Generator) -> None:
    n, p = 96, 14
    latent = rng.normal(size=(n, 5))
    load = np.zeros((5, p))
    load[0, 0:3] = [1.0, 0.9, 0.8]
    load[1, 3:6] = [1.0, 0.85, 0.75]
    load[2, 6:9] = [1.0, 0.9, 0.7]
    load[3, 9:11] = [1.0, 0.8]
    load[4, 11:14] = [1.0, 0.9, 0.85]
    X = latent @ load + 0.15 * rng.normal(size=(n, p))
    y = 0.9 * X[:, 0] - 0.7 * X[:, 4] + 0.5 * X[:, 7] + rng.normal(size=n)
    groups = define_groups(X, threshold=0.6)
    group_rows = [
        (j + 1, i + 1) for j, g in enumerate(groups) for i in g
    ]
    lam_grid = np.array([0.5, 0.2, 0.1, 0.05, 0.02, 0.005])
    permutations = np.vstack([rng.permutation(n) + 1 for _ in range(8)])

    _save("group_X.csv", X, ",".join(f"x{i+1}" for i in range(p)))
    _save("group_y.csv", y[:, None], "y")
    np.savetxt(
        FIXTURES / "group_groups.csv",
        np.asarray(group_rows, dtype=int),
        delimiter=",",
        header="group,feature",
        comments="",
        fmt="%d",
    )
    _save("group_lam_grid.csv", lam_grid[:, None], "lam")
    np.savetxt(
        FIXTURES / "group_permutations.csv",
        permutations.astype(int),
        delimiter=",",
        comments="",
        fmt="%d",
    )
    print(f"group fixture: {n} x {p}, {len(groups)} groups")


def make_prognostic_fixture(rng: np.random.Generator) -> None:
    n, p = 140, 12
    latent = rng.normal(size=(n, 3))
    load = np.zeros((3, p))
    load[0, 0:4] = [1.0, 0.9, 0.8, 0.75]
    load[1, 4:8] = [1.0, 0.9, 0.8, 0.7]
    load[2, 8:12] = [1.0, 0.85, 0.8, 0.75]
    X = latent @ load + 0.15 * rng.normal(size=(n, p))
    y = 1.1 * X[:, 0] - 0.8 * X[:, 4] + 0.6 * X[:, 8] + 0.4 * rng.normal(size=n)
    groups = define_groups(X, threshold=0.6)

    n_val = 50
    X_val = latent[:n_val] @ load + 0.15 * rng.normal(size=(n_val, p))
    y_val = (
        1.1 * X_val[:, 0] - 0.8 * X_val[:, 4] + 0.6 * X_val[:, 8]
        + 0.4 * rng.normal(size=n_val)
    )
    folds = np.empty(n, dtype=int)
    order = rng.permutation(n)
    labels = np.tile(np.arange(1, 6), int(np.ceil(n / 5)))[:n]
    folds[order] = labels[order]

    _save("prognostic_X.csv", X, ",".join(f"x{i+1}" for i in range(p)))
    _save("prognostic_y.csv", y[:, None], "y")
    _save("prognostic_Xval.csv", X_val, ",".join(f"x{i+1}" for i in range(p)))
    _save("prognostic_yval.csv", y_val[:, None], "y")
    _save("prognostic_folds.csv", folds[:, None], "fold")
    np.savetxt(
        FIXTURES / "prognostic_groups.csv",
        np.asarray([(j + 1, i + 1) for j, g in enumerate(groups) for i in g]),
        delimiter=",",
        header="group,feature",
        comments="",
        fmt="%d",
    )
    print(f"prognostic fixture: {n} x {p}, {len(groups)} groups, folds 1..5")


def make_ppi_fixture(rng: np.random.Generator) -> None:
    n, N = 180, 400
    t = np.repeat([0, 1], n // 2)
    f = rng.normal(size=n)
    y = 0.45 * t + 0.85 * f + 0.55 * rng.normal(size=n)
    f_unl = rng.normal(size=N)
    _save("ppi_y.csv", y[:, None], "y")
    _save("ppi_t.csv", t[:, None], "t")
    _save("ppi_f.csv", f[:, None], "f")
    _save("ppi_f_unlabeled.csv", f_unl[:, None], "f")
    print(f"ppi fixture: n={n}, N={N}")


def make_procova_fixture() -> None:
    rows = [
        ("balanced", 1.0, 1.0, 0.4, 0.4, 0.5, 0.4, 0.05, 0.80, 100.0),
        ("unequal", 2.0, 1.0, 0.3, 0.2, 0.4, 0.5, 0.05, 0.80, 160.0),
        ("no_signal", 1.5, 1.2, 0.0, 0.0, 0.5, 0.35, 0.05, 0.90, 120.0),
        ("high_signal", 1.0, 1.0, 0.8, 0.8, 0.5, 0.25, 0.05, 0.80, 80.0),
    ]
    lines = ["case,sigma0,sigma1,rho0,rho1,pi0,effect,alpha,power,n_total"]
    lines += [
        f"{case},{s0:.17g},{s1:.17g},{r0:.17g},{r1:.17g},{pi0:.17g},"
        f"{effect:.17g},{alpha:.17g},{power:.17g},{n_total:.17g}"
        for case, s0, s1, r0, r1, pi0, effect, alpha, power, n_total in rows
    ]
    (FIXTURES / "procova_cases.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"procova fixture: {len(rows)} design cases")


def make_analysis_fixture(rng: np.random.Generator) -> None:
    n = 160
    t = np.repeat([0, 1], n // 2)
    f = rng.normal(size=n)
    y = 0.5 * t + 0.8 * f + 0.6 * rng.normal(size=n)
    path = FIXTURES / "analysis_trial.csv"
    lines = ["# synthetic parity fixture, no patient data", "y,t,f"]
    lines += [f"{y[i]:.17g},{t[i]},{f[i]:.17g}" for i in range(n)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"analysis fixture: {n} rows")


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    make_group_fixture(rng)
    make_prognostic_fixture(rng)
    make_ppi_fixture(rng)
    make_procova_fixture()
    make_analysis_fixture(rng)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

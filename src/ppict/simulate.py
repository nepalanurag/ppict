"""Monte Carlo harness for the trial scenarios.

Design of the generated data, stated once so every scenario can be read against
it:

* Patients are drawn i.i.d. from one covariate distribution with block structure:
  features inside a group share a latent factor, so groups are the natural unit
  of selection. Every feature has unit marginal variance and the within-group
  correlation is exactly ``within_group_rho``.
* The control outcome is ``Y0 = scale * g(X) + eps``, where ``g`` is a sparse
  group function supported on a seed-chosen set of groups. ``scale`` is set
  numerically so that ``Var(g) / Var(Y0)`` equals ``signal_r2`` before the score
  is estimated. The treatment effect is a constant shift, which is the case where
  prognostic covariate adjustment is semiparametrically efficient
  (arXiv:2012.09935).
* The external cohort has two disjoint parts: a training cohort with control
  outcomes (the digital twin registry) and an unlabeled cohort with covariates
  only. Both are drawn from the trial's covariate distribution unless
  ``history_shift`` moves them, which is the transport failure mode.
* All patients in the trial have observed outcomes, because it is a completed
  trial. The unlabeled sample is external, which is why ``ppi_per_arm`` retains a
  dependence on it while ``ppi_single`` does not.

Every replicate writes the seed it used. Results are only ever reported as
measured frequencies and measured interval widths.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .group_select import group_lasso_fista, empirical_fdp
from .ppi import (
    ATEEstimate,
    ate_ancova,
    ate_lasso_adjusted,
    ate_ppi_per_arm,
    ate_ppi_single_lambda,
    ate_unadjusted,
)
from .prognostic import cross_fitted_scores, fit_prognostic, r2_score

__all__ = [
    "TrialConfig",
    "TrialData",
    "corrupt_score",
    "make_trial_data",
    "run_replicate",
    "run_monte_carlo",
    "summarize",
    "write_run_log",
]

CORRUPTION_MODES = ("none", "noise", "shuffle", "bias", "flip", "scale")
SCORE_MODES = ("external", "cross_fitted")


@dataclass(frozen=True)
class TrialConfig:
    """Everything that defines one simulation scenario.

    Frozen so that a config can be hashed into a run log and cannot be mutated
    halfway through a sweep.
    """

    name: str
    seed: int = 0
    n_hist: int = 500
    n_unlabeled: int = 500
    n_treat: int = 150
    n_control: int = 150
    p: int = 60
    n_groups: int = 12
    within_group_rho: float = 0.5
    n_active_groups: int = 3
    signal_r2: float = 0.30
    true_effect: float = 0.0
    outcome_sd: float = 1.0
    corruption: str = "none"
    corruption_param: float = 0.0
    history_shift: float = 0.0
    alpha: float = 0.05
    score_mode: str = "external"
    select_q: float = 0.10
    select_n_perm: int = 25
    select_top_k: Optional[int] = None
    stability_B: int = 0

    def __post_init__(self) -> None:
        if self.corruption not in CORRUPTION_MODES:
            raise ValueError(f"corruption must be one of {CORRUPTION_MODES}")
        if self.score_mode not in SCORE_MODES:
            raise ValueError(f"score_mode must be one of {SCORE_MODES}")
        if self.n_treat < 4 or self.n_control < 4:
            raise ValueError("each arm needs at least 4 patients")
        if self.n_groups > self.p:
            raise ValueError("n_groups cannot exceed p")


@dataclass
class TrialData:
    """One generated trial plus its external cohorts."""

    X_trial: np.ndarray
    t: np.ndarray
    y: np.ndarray
    y0: np.ndarray
    f: np.ndarray  # prognostic score as used by the estimators (possibly corrupted)
    f_unlabeled: np.ndarray
    f_oracle: np.ndarray  # true E[Y0 | X], the ceiling any score can reach
    active_groups: np.ndarray
    active_features: np.ndarray  # feature indices carrying the signal
    group_sizes: np.ndarray
    score_oos_r2: float  # R^2 of the uncorrupted score against the true control outcome
    score_oos_r2_used: float  # same, for the score the estimators actually saw
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def corrupt_score(
    f: np.ndarray,
    mode: str,
    param: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Deliberately damage a prognostic score.

    The corruption is a corruption of the *mapping*, so it is applied with the
    same recipe to the trial scores and to the unlabeled cohort scores, each with
    its own randomness. Constant bias cancels algebraically in the single-tuning
    estimator and is shift-invariant in the tuning value, so it is included to
    confirm that invariance rather than to create a failure.
    """
    if mode not in CORRUPTION_MODES:
        raise ValueError(f"corruption must be one of {CORRUPTION_MODES}")
    f = np.asarray(f, dtype=float).ravel()
    if f.size == 0:
        return f.copy()
    sd = float(np.std(f))
    if mode == "none":
        return f.copy()
    if mode == "noise":
        return f + param * sd * rng.standard_normal(f.size)
    if mode == "shuffle":
        return rng.permutation(f)
    if mode == "bias":
        return f + param * sd
    if mode == "flip":
        return -f
    if mode == "scale":
        return f * param
    raise ValueError(mode)  # pragma: no cover


def _draw_X(
    n: int,
    group_sizes: Sequence[int],
    rho: float,
    rng: np.random.Generator,
    active: Sequence[int] = (),
    shift: float = 0.0,
) -> np.ndarray:
    """Block-correlated covariates, unit marginal variance, exact within-group rho."""
    p = int(np.sum(group_sizes))
    X = np.empty((n, p), dtype=float)
    start = 0
    active_set = set(int(a) for a in active)
    for gi, s in enumerate(group_sizes):
        u = rng.standard_normal((n, 1))
        e = rng.standard_normal((n, s))
        block = np.sqrt(rho) * u + np.sqrt(1.0 - rho) * e
        if shift and gi in active_set:
            block = block + shift
        X[:, start : start + s] = block
        start += s
    return X


def make_trial_data(cfg: TrialConfig) -> TrialData:
    """Generate one trial under ``cfg``.

    Everything random in a config is driven by a single generator seeded from
    ``cfg.seed``, so a config plus a seed fully determines the dataset.
    """
    rng = np.random.default_rng(cfg.seed)

    # Group structure: contiguous blocks of near-equal size.
    base = cfg.p // cfg.n_groups
    rem = cfg.p % cfg.n_groups
    group_sizes = np.array([base + (1 if i < rem else 0) for i in range(cfg.n_groups)], dtype=int)
    group_sizes = group_sizes[group_sizes > 0]

    active = np.sort(rng.choice(len(group_sizes), size=min(cfg.n_active_groups, len(group_sizes)), replace=False))
    weights = [np.ones(int(s), dtype=float) / np.sqrt(float(s)) for s in group_sizes]

    # Feature indices per simulated group, and the union of the signal-carrying
    # features. False-discovery diagnostics are computed on features rather than on
    # group indices, because the grouping the score discovers by correlation is not
    # the grouping the data was generated with, and comparing indices across two
    # different partitions is meaningless.
    group_feature_idx = []
    _start = 0
    for _s in group_sizes:
        group_feature_idx.append(np.arange(_start, _start + int(_s), dtype=int))
        _start += int(_s)
    active_feature_idx = (
        np.sort(np.concatenate([group_feature_idx[int(a)] for a in active]))
        if len(active)
        else np.empty(0, dtype=int)
    )

    def g(X: np.ndarray) -> np.ndarray:
        out = np.zeros(X.shape[0], dtype=float)
        start = 0
        for gi, s in enumerate(group_sizes):
            if gi in set(int(a) for a in active):
                out += X[:, start : start + int(s)] @ weights[gi]
            start += int(s)
        return out

    # Scale the sparse function so that R^2 of the control outcome equals signal_r2.
    X_ref = _draw_X(20000, group_sizes, cfg.within_group_rho, rng, active)
    g_ref = g(X_ref)
    sd_g = float(np.std(g_ref))
    target_var = cfg.signal_r2 / max(1.0 - cfg.signal_r2, 1e-9) * cfg.outcome_sd**2
    scale = float(np.sqrt(target_var) / sd_g) if sd_g > 0 else 0.0

    def make_controls(X: np.ndarray) -> np.ndarray:
        eps = rng.standard_normal(X.shape[0]) * cfg.outcome_sd
        return scale * g(X) + eps

    # External cohorts: training cohort has control outcomes, the other is unlabeled.
    X_hist = _draw_X(cfg.n_hist, group_sizes, cfg.within_group_rho, rng, active, shift=cfg.history_shift)
    y_hist = make_controls(X_hist)
    X_unl = _draw_X(cfg.n_unlabeled, group_sizes, cfg.within_group_rho, rng, active)

    # The trial.
    n_total = cfg.n_treat + cfg.n_control
    X_trial = _draw_X(n_total, group_sizes, cfg.within_group_rho, rng, active)
    y0_trial = make_controls(X_trial)
    y1_trial = y0_trial + cfg.true_effect
    t = np.concatenate([np.ones(cfg.n_treat, dtype=int), np.zeros(cfg.n_control, dtype=int)])
    rng.shuffle(t)
    y = np.where(t == 1, y1_trial, y0_trial)

    # The prognostic score.
    diag: Dict[str, Any] = {}
    if cfg.score_mode == "external":
        score = fit_prognostic(
            X_hist, y_hist,
            q=cfg.select_q, n_perm=cfg.select_n_perm, top_k=cfg.select_top_k,
            rng=rng, stability_B=cfg.stability_B,
        )
        f_trial_raw = score.predict(X_trial)
        f_unl_raw = score.predict(X_unl)
        diag["n_features_selected"] = score.n_features
        diag["n_groups_selected"] = int(score.selected_groups.size)
        diag["train_r2"] = float(score.train_r2)
        diag["lam_select"] = float(score.lam)
        diag["feature_fdp"] = float(
            empirical_fdp([int(x) for x in score.feature_idx], [int(x) for x in active_feature_idx])
        )
    else:
        cf = cross_fitted_scores(
            X_trial, y, n_folds=5, q=cfg.select_q, n_perm=cfg.select_n_perm,
            top_k=cfg.select_top_k, rng=rng,
        )
        f_trial_raw = cf.oof
        # Fold models averaged for the unlabeled cohort, since no fold owns it.
        f_unl_raw = np.mean([m.predict(X_unl) for m in cf.models], axis=0)
        diag["n_features_selected"] = float(np.mean([m.n_features for m in cf.models]))
        diag["oof_r2"] = float(cf.oof_r2)
        diag["mean_fold_jaccard"] = float(cf.mean_fold_jaccard)
        diag["feature_fdp"] = float(
            np.mean(
                [
                    empirical_fdp([int(x) for x in m.feature_idx], [int(x) for x in active_feature_idx])
                    for m in cf.models
                ]
            )
        )

    f_oracle = scale * g(X_trial)

    # Corruption, applied to both sets with the same recipe but independent draws.
    crng = np.random.default_rng(cfg.seed + 991)
    f_trial = corrupt_score(f_trial_raw, cfg.corruption, cfg.corruption_param, crng)
    f_unl = corrupt_score(f_unl_raw, cfg.corruption, cfg.corruption_param, crng)

    return TrialData(
        X_trial=X_trial, t=t, y=y, y0=y0_trial, f=f_trial, f_unlabeled=f_unl,
        f_oracle=f_oracle, active_groups=active, active_features=active_feature_idx,
        group_sizes=group_sizes,
        score_oos_r2=r2_score(y0_trial, f_trial_raw),
        score_oos_r2_used=r2_score(y0_trial, f_trial),
        diagnostics=diag,
    )


def _squared_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Squared Pearson correlation, or nan if either input has no variance."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        r = float(np.corrcoef(a, b)[0, 1])
    return r * r if np.isfinite(r) else float("nan")


def run_replicate(cfg: TrialConfig) -> List[Dict[str, Any]]:
    """Run every estimator on one generated trial and return one row per method."""
    data = make_trial_data(cfg)
    alpha = cfg.alpha

    estimates: List[ATEEstimate] = [
        ate_unadjusted(data.y, data.t, alpha=alpha),
        ate_ancova(data.y, data.t, data.f, alpha=alpha, method="ancova"),
        ate_ppi_single_lambda(data.y, data.t, data.f, alpha=alpha),
        ate_ppi_per_arm(data.y, data.t, data.f, data.f_unlabeled, alpha=alpha),
        ate_ppi_per_arm(
            data.y, data.t, data.f, data.f_unlabeled, alpha=alpha,
            lam=(1.0, 1.0), method="ppi_lambda1",
        ),
        ate_lasso_adjusted(data.y, data.t, data.X_trial, alpha=alpha, seed=cfg.seed),
        ate_ancova(
            data.y, data.t, data.f_oracle, alpha=alpha, method="oracle_ancova"
        ),
    ]

    rows: List[Dict[str, Any]] = []
    for est in estimates:
        row = est.as_row()
        row.update(
            {
                "config": cfg.name,
                "seed": cfg.seed,
                "true_effect": cfg.true_effect,
                "covered": float(est.ci_low <= cfg.true_effect <= est.ci_high),
                "rejected": float(est.p_value < alpha),
                "signed_error": float(est.estimate - cfg.true_effect),
                "squared_error": float((est.estimate - cfg.true_effect) ** 2),
                "score_oos_r2": data.score_oos_r2,
                "score_oos_r2_used": data.score_oos_r2_used,
                # Squared correlation between the score the estimators received and
                # the true control outcome. This, not the raw R squared, is the
                # quantity the PPI variance-reduction expression and the
                # prognostic covariate adjustment factor both depend on, and unlike
                # raw R squared it is invariant to a constant offset in the score.
                "score_rho2": _squared_correlation(data.f, data.y0),
                "corruption": cfg.corruption,
                "corruption_param": cfg.corruption_param,
                "score_mode": cfg.score_mode,
                "signal_r2": cfg.signal_r2,
                "history_shift": cfg.history_shift,
                "select_q": cfg.select_q,
                "select_n_perm": cfg.select_n_perm,
                "within_group_rho": cfg.within_group_rho,
                "n_groups": int(len(data.group_sizes)),
                "n_active_groups": int(len(data.active_groups)),
            }
        )
        for k, v in data.diagnostics.items():
            row.setdefault(f"score_{k}", v)
        rows.append(row)
    return rows


def _replicate_task(task: Tuple[Dict[str, Any], int]) -> List[Dict[str, Any]]:
    """One replicate, as a module-level function so that it can be pickled.

    macOS starts worker processes with ``spawn``, so the task and its arguments
    must both be importable and picklable. A config dict plus a seed is.
    """
    cfg_dict, seed = task
    return run_replicate(TrialConfig(**{**cfg_dict, "seed": int(seed)}))


def run_monte_carlo(
    configs: Sequence[TrialConfig],
    n_rep: int,
    seed_base: int = 1000,
    verbose: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Run ``n_rep`` replicates of every config.

    Replicate ``r`` of a config uses seed ``seed_base + r``, so the whole sweep is
    reproducible from ``seed_base`` and ``n_rep`` alone. ``n_jobs`` above 1 spreads
    replicates across processes. Results are collected in submission order, so the
    numbers do not depend on ``n_jobs``.
    """
    if n_rep < 1:
        raise ValueError("n_rep must be at least 1")
    n_jobs = int(n_jobs)
    rows: List[Dict[str, Any]] = []
    executor = None
    try:
        if n_jobs > 1:
            from concurrent.futures import ProcessPoolExecutor

            executor = ProcessPoolExecutor(max_workers=n_jobs)
        for cfg in configs:
            t0 = time.time()
            cfg_dict = asdict(cfg)
            seeds = [int(seed_base + r) for r in range(n_rep)]
            tasks = [(cfg_dict, s) for s in seeds]
            if executor is not None:
                chunk = max(1, n_rep // (n_jobs * 4))
                results = list(executor.map(_replicate_task, tasks, chunksize=chunk))
            else:
                results = [_replicate_task(t) for t in tasks]
            for res in results:
                rows.extend(res)
            if verbose:
                print(
                    f"  {cfg.name}: {n_rep} replicates in {time.time() - t0:.1f}s",
                    flush=True,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return pd.DataFrame(rows)


def _nanmean_or_nan(values: np.ndarray) -> float:
    """Mean ignoring nan, returning nan when every value is nan.

    ``np.nanmean`` on an all-nan slice raises a RuntimeWarning; the unadjusted
    estimator legitimately has no tuning value, so an all-nan column is expected
    rather than exceptional.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def summarize(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Measured summary per config and method, with Monte Carlo uncertainty.

    Coverage, rejection rate, interval width, bias and the ratio of the reported
    standard error to the empirically observed spread are all measured from the
    replicate results. Nothing here is analytic.
    """

    def _one(g: pd.DataFrame) -> pd.Series:
        n = len(g)
        est = g["estimate"].to_numpy(dtype=float)
        se = g["se"].to_numpy(dtype=float)
        covered = g["covered"].to_numpy(dtype=float)
        rejected = g["rejected"].to_numpy(dtype=float)
        width = g["ci_width"].to_numpy(dtype=float)
        empirical_sd = float(np.std(est, ddof=1)) if n > 1 else float("nan")
        cov = float(covered.mean())
        rej = float(rejected.mean())
        return pd.Series(
            {
                "n_rep": n,
                "true_effect": float(g["true_effect"].iloc[0]),
                "mean_estimate": float(est.mean()),
                "bias": float(est.mean() - g["true_effect"].iloc[0]),
                "rmse": float(np.sqrt(np.mean(g["squared_error"]))),
                "empirical_sd": empirical_sd,
                "mean_se": float(np.nanmean(se)),
                "se_ratio": float(np.nanmean(se) / empirical_sd) if empirical_sd else float("nan"),
                "coverage": cov,
                "mc_se_coverage": float(np.sqrt(max(cov * (1 - cov), 0.0) / n)),
                "reject_rate": rej,
                "mc_se_reject_rate": float(np.sqrt(max(rej * (1 - rej), 0.0) / n)),
                "mean_ci_width": float(np.nanmean(width)),
                "mean_lam": _nanmean_or_nan(g["lam"].to_numpy(dtype=float)),
                "score_oos_r2": float(g["score_oos_r2"].iloc[0]),
                "score_oos_r2_used": float(g["score_oos_r2_used"].iloc[0]),
                "score_rho2": (
                    float(g["score_rho2"].iloc[0]) if "score_rho2" in g.columns else float("nan")
                ),
            }
        )

    keys = [
        "config", "method", "corruption", "corruption_param", "score_mode",
        "signal_r2", "history_shift", "alpha",
    ]
    df = df.copy()
    if "alpha" not in df.columns:
        df["alpha"] = alpha

    # Looped rather than DataFrameGroupBy.apply so that the grouping columns are
    # not passed into the summary function.
    rows: List[Dict[str, Any]] = []
    for key, group in df.groupby(keys, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        rec: Dict[str, Any] = dict(zip(keys, key_tuple))
        rec.update(_one(group).to_dict())
        rows.append(rec)
    return pd.DataFrame(rows)


def write_run_log(
    path: Path,
    configs: Sequence[TrialConfig],
    n_rep: int,
    seed_base: int,
    summary: Optional[pd.DataFrame] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write the log that makes a run traceable to a command and a seed."""
    import sklearn
    import scipy

    from . import __version__

    log: Dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "cwd": str(Path.cwd()),
        "n_rep": int(n_rep),
        "seed_base": int(seed_base),
        "seed_rule": "replicate r of every config uses seed = seed_base + r",
        "configs": [asdict(c) for c in configs],
        "versions": {
            "ppict": __version__,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "platform": platform.platform(),
    }
    try:
        log["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # pragma: no cover - git may be unavailable
        log["git_commit"] = None
    if summary is not None:
        log["summary_rows"] = int(len(summary))
    if extra:
        log.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(log, fh, indent=2, sort_keys=True, default=str)
    return log

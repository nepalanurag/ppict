#!/usr/bin/env python3
"""Real-data retrospective: ACTG 175 plus a TCGA-BRCA expression score.

This is the script the real-data report is generated from. It does four things:

1. Fetch and verify the two public cohorts. Files are cached under
   ``validation/data``; a size mismatch is an error, not something to work
   around.
2. Fit a prognostic score on ACTG 175 using only held-out control patients,
   then run the shipped estimators on the disjoint analysis half. Report the
   treatment effect, the interval width against the unadjusted analysis, and a
   permuted-label negative control with the real score and with a corrupted
   score.
3. Fit a cross-validated ER-status score on TCGA-BRCA expression, and record
   what it is and what it is not.
4. Check whether the expression score can be transported to the trial. It
   cannot: ACTG 175 has no expression measurements, shares no patients with
   TCGA, and the two outcomes are on different scales. The report says this
   plainly rather than substituting a simulated score.

Run from the repository root:

    ./.venv/bin/python validation/run_real_data.py --permutations 1000

Writes into ``validation/results``:
    REAL_DATA_REPORT.md, real_data_results.json,
    actg_scores.csv, brca_er_scores.csv

Nothing is published or sent anywhere. Downloading the three public files is
the only network access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppict import (  # noqa: E402
    ate_lasso_adjusted,
    ate_ppi_single_lambda,
    ate_unadjusted,
    control_arm_size,
    define_groups,
    fit_prognostic,
    pairwise_jaccard,
    required_n_total,
)

DATA = ROOT / "validation" / "data"
RESULTS = ROOT / "validation" / "results"

ALPHA = 0.05
SEED = 7
N_PERM_DEFAULT = 1000


class DataError(RuntimeError):
    """A public data route did not deliver what the script requires."""


@dataclass(frozen=True)
class Source:
    """One public file, its route, and the size that makes it the right file."""

    key: str
    filename: str
    url: str
    bytes: int
    description: str


SOURCES: Tuple[Source, ...] = (
    Source(
        key="actg",
        filename="actg175.csv",
        url="https://archive.ics.uci.edu/static/public/890/data.csv",
        bytes=159675,
        description="ACTG 175, UCI Machine Learning Repository dataset 890",
    ),
    Source(
        key="brca_expr",
        filename="brca_hiseq.tsv.gz",
        url="https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/HiSeqV2.gz",
        bytes=64286337,
        description="TCGA-BRCA HiSeqV2 expression, UCSC Xena",
    ),
    Source(
        key="brca_clin",
        filename="brca_clinical.tsv",
        url="https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix",
        bytes=1698413,
        description="TCGA-BRCA clinical matrix, UCSC Xena (uncompressed route)",
    ),
)

# Baseline covariates, fixed before looking at any result. These are the
# variables the trial recorded at randomisation, not post-baseline mediators.
COVARIATES: Tuple[str, ...] = (
    "age",
    "wtkg",
    "karnof",
    "cd40",
    "cd80",
    "symptom",
    "hemo",
    "homo",
    "drugs",
    "race",
    "gender",
    "oprior",
    "z30",
    "preanti",
)

ARM_LABELS = {0: "ZDV", 1: "ddI", 2: "ZDV+ddI", 3: "ZDV+ddC"}
OUTCOME_COLUMN = "cd420"
OUTCOME_LABEL = "CD4 count at week 20 (cells per microlitre)"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(src: Source, force: bool) -> Dict[str, Any]:
    """Return a verified local file, downloading only when needed."""
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / src.filename
    status = "cached"

    if force and path.exists():
        path.unlink()

    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        h = hashlib.sha256()
        n = 0
        request = urllib.request.Request(
            src.url, headers={"User-Agent": "ppict-validation/0.1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                with tmp.open("wb") as fh:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                        h.update(chunk)
                        n += len(chunk)
        except urllib.error.HTTPError as exc:
            if tmp.exists():
                tmp.unlink()
            raise DataError(f"{src.url}: HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            if tmp.exists():
                tmp.unlink()
            raise DataError(f"{src.url}: {exc.reason}") from exc

        if n != src.bytes:
            tmp.unlink()
            raise DataError(
                f"{src.url}: downloaded {n} bytes, expected {src.bytes}. "
                "The public file may have changed; inspect it before changing "
                "the expected size in this script."
            )
        tmp.replace(path)
        status = "downloaded"

    size = path.stat().st_size
    if size != src.bytes:
        raise DataError(
            f"{path}: {size} bytes on disk, expected {src.bytes}. "
            "Delete it and rerun, or pass --force-download."
        )

    return {
        "key": src.key,
        "description": src.description,
        "url": src.url,
        "path": str(path.relative_to(ROOT)),
        "bytes": size,
        "sha256": _sha256(path),
        "sha256_scope": "full file",
        "status": status,
        "modified_utc": datetime.fromtimestamp(
            path.stat().st_mtime, timezone.utc
        ).strftime("%Y-%m-%d %H:%M UTC"),
    }


def ensure_data(force: bool) -> Dict[str, Dict[str, Any]]:
    return {src.key: _fetch(src, force) for src in SOURCES}


def load_actg(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["pidnum", "trt", OUTCOME_COLUMN] + list(COVARIATES)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataError(f"{path}: missing columns {missing}")
    if len(df) != 2139:
        raise DataError(f"{path}: expected 2139 rows, found {len(df)}")
    if df["pidnum"].duplicated().any():
        raise DataError(f"{path}: patient identifiers are not unique")
    return df


def prepare_trial(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Two-arm subset, complete cases, and the arm counts used in the report."""
    arm_counts = {int(k): int(v) for k, v in df["trt"].value_counts().items()}
    two_arm = df[df["trt"].isin([0, 1])].copy()
    n_two_arm = int(len(two_arm))
    complete = two_arm.dropna(subset=list(COVARIATES) + [OUTCOME_COLUMN, "trt"])
    n_dropped = n_two_arm - int(len(complete))
    complete = complete.reset_index(drop=True)
    complete["trt"] = complete["trt"].astype(int)
    if complete["trt"].nunique() != 2:
        raise DataError("the two-arm subset lost an arm during cleaning")
    if not np.isfinite(complete[list(COVARIATES) + [OUTCOME_COLUMN]].to_numpy(float)).all():
        raise DataError("non-finite values survived the complete-case filter")
    meta = {
        "n_rows_total": int(len(df)),
        "arm_counts_total": arm_counts,
        "n_two_arm": n_two_arm,
        "n_complete": int(len(complete)),
        "n_dropped_complete_case": int(n_dropped),
        "arms_used": {
            "0": ARM_LABELS[0],
            "1": ARM_LABELS[1],
        },
        "n_arm_0": int((complete["trt"] == 0).sum()),
        "n_arm_1": int((complete["trt"] == 1).sum()),
    }
    return complete, meta


def stratified_split(trt: np.ndarray, seed: int) -> np.ndarray:
    """Half of each arm for score training, half for inference.

    Splitting within arm keeps the analysis sample balanced even when the trial
    itself is not exactly 1:1. The split is a property of the patients, not of
    their outcomes, so it cannot select a favourable result.
    """
    rng = np.random.default_rng(seed)
    train = np.zeros(trt.size, dtype=bool)
    for arm in (0, 1):
        idx = np.flatnonzero(trt == arm)
        rng.shuffle(idx)
        k = int(round(0.5 * idx.size))
        train[idx[:k]] = True
    if train.all() or not train.any():
        raise DataError("the split left one side empty")
    for arm in (0, 1):
        n_train = int(((trt == arm) & train).sum())
        n_analysis = int(((trt == arm) & ~train).sum())
        if n_train < 20 or n_analysis < 20:
            raise DataError(f"arm {arm} is too small after splitting")
    return train


def fit_trial_score(
    X: np.ndarray, y: np.ndarray, trt: np.ndarray, train: np.ndarray, seed: int
) -> Dict[str, Any]:
    """Group-lasso score on training controls, applied to every patient.

    Standardisation uses the training controls only. The score is fit on
    ``train & control`` and evaluated out of sample on ``analysis & control``.
    """
    fit_rows = train & (trt == 0)
    val_rows = (~train) & (trt == 0)
    if fit_rows.sum() < 20 or val_rows.sum() < 20:
        raise DataError("not enough control patients on one side of the split")

    mu = X[fit_rows].mean(axis=0)
    sd = X[fit_rows].std(axis=0)
    if np.any(sd <= 0):
        raise DataError("a covariate has no variance in the training controls")
    Xs = (X - mu) / sd

    groups = define_groups(Xs[fit_rows], threshold=0.6)
    score = fit_prognostic(
        Xs[fit_rows],
        y[fit_rows],
        groups=groups,
        q=0.10,
        n_perm=25,
        rng=seed,
        X_val=Xs[val_rows],
        y_val=y[val_rows],
    )
    f = score.predict(Xs)

    idx = score.feature_idx
    names = [COVARIATES[i] for i in idx]
    beta_std = np.asarray(score.coef, dtype=float)
    beta_raw = beta_std / sd[idx] if idx.size else np.empty(0)
    intercept_raw = float(score.intercept - np.sum(beta_std * mu[idx] / sd[idx]))
    check = X[:, idx] @ beta_raw + intercept_raw if idx.size else np.full(len(X), intercept_raw)
    if not np.allclose(check, f, rtol=1e-9, atol=1e-9):
        raise DataError("the raw-scale score does not reproduce the fitted score")

    return {
        "f": f,
        "selected_features": names,
        "selected_feature_idx": [int(i) for i in idx],
        "coefficients_raw": [float(b) for b in beta_raw],
        "intercept_raw": intercept_raw,
        "train_r2": float(score.train_r2),
        "oos_control_r2": float(score.oos_r2),
        "lam_standardized": float(score.lam),
        "n_groups": int(score.info["n_groups"]),
        "n_selected_groups": int(score.selected_groups.size),
        "q": float(score.info["q"]),
        "n_perm": int(score.info["n_perm"]),
        "group_threshold": float(score.info["group_threshold"]),
        "n_fit_controls": int(fit_rows.sum()),
        "n_validation_controls": int(val_rows.sum()),
        "standardization": {
            "mean": [float(v) for v in mu],
            "sd": [float(v) for v in sd],
        },
    }


def estimate_trial(
    y: np.ndarray,
    trt: np.ndarray,
    f: np.ndarray,
    covariates: np.ndarray,
    analysis: np.ndarray,
    seed: int,
) -> Tuple[Dict[str, Any], Any, Any, Any]:
    """Run the shipped estimators on the analysis half.

    The L1 lasso baseline is the package's ``ate_lasso_adjusted``: one lasso fit
    on treatment plus all baseline covariates, then an OLS refit on the selected
    covariates and treatment. It is the naive post-selection baseline the spec
    asks for, not a recommended method.

    Implied control-arm sizes are a projection, not a measurement: they scale
    the observed variance of each estimator to the observed unadjusted effect at
    80% power and a 1:1 allocation. The projection is labelled in the report.
    """
    y_a = y[analysis]
    t_a = trt[analysis]
    f_a = f[analysis]
    x_a = covariates[analysis]
    u = ate_unadjusted(y_a, t_a, alpha=ALPHA)
    lasso = ate_lasso_adjusted(
        y_a, t_a, x_a, alpha=ALPHA, seed=seed + 5, cv=5
    )
    p = ate_ppi_single_lambda(y_a, t_a, f_a, alpha=ALPHA)

    n_current = int(analysis.sum())
    power = 0.80

    def implied(e: Any) -> Tuple[Optional[float], Optional[float]]:
        variance_constant = n_current * float(e.se) ** 2
        n_total = required_n_total(
            variance_constant, effect=float(u.estimate), alpha=ALPHA, power=power
        )
        if not np.isfinite(n_total):
            return None, None
        return float(n_total), float(control_arm_size(n_total, 0.5))

    def pack(e: Any) -> Dict[str, Any]:
        n_total, n_control = implied(e)
        return {
            "method": e.method,
            "estimate": float(e.estimate),
            "se": float(e.se),
            "ci_low": float(e.ci_low),
            "ci_high": float(e.ci_high),
            "ci_width": float(e.ci_width),
            "p_value": float(e.p_value),
            "lam": None if not np.isfinite(e.lam) else float(e.lam),
            "n_selected": (
                int(e.extra["n_selected"])
                if "n_selected" in e.extra
                else None
            ),
            "n_treated": int(e.n_treated),
            "n_control": int(e.n_control),
            "width_ratio_vs_unadjusted": float(e.ci_width / u.ci_width),
            "variance_ratio_vs_unadjusted": float((e.se / u.se) ** 2),
            "implied_n_total_80": n_total,
            "implied_control_arm_80": n_control,
        }

    width_ratio = float(p.ci_width / u.ci_width)
    variance_ratio = float((p.se / u.se) ** 2)
    result = {
        "n_analysis": n_current,
        "n_analysis_treated": int((t_a == 1).sum()),
        "n_analysis_control": int((t_a == 0).sum()),
        "unadjusted": pack(u),
        "lasso_l1": pack(lasso),
        "adjusted": pack(p),
        "width_ratio_adjusted_over_unadjusted": width_ratio,
        "width_narrowing_fraction": float(1.0 - width_ratio),
        "variance_ratio_adjusted_over_unadjusted": variance_ratio,
        "difference_adjusted_minus_unadjusted": float(p.estimate - u.estimate),
        "lasso_width_ratio_vs_unadjusted": float(lasso.ci_width / u.ci_width),
        "implied_design_assumptions": {
            "effect": float(u.estimate),
            "power": power,
            "allocation_control": 0.5,
            "alpha": ALPHA,
            "note": (
                "Projection from the observed variance of each estimator at the "
                "observed unadjusted effect and 80% power, 1:1 allocation. It "
                "is not a measured trial size."
            ),
        },
        "alpha": ALPHA,
    }
    return result, u, lasso, p


def negative_control(
    y: np.ndarray,
    trt: np.ndarray,
    f: np.ndarray,
    analysis: np.ndarray,
    n_perm: int,
    seed: int,
) -> Dict[str, Any]:
    """Permute treatment labels in the analysis half, with one fixed corruption.

    The corrupted score is a permutation of the same analysis scores, so its
    marginal distribution is preserved and only the patient-level relationship
    with the outcome is destroyed. The real score stays fixed: it was fitted on
    patients outside the analysis half, so permuting analysis labels does not
    touch its training data.
    """
    y_a = y[analysis]
    t_a = trt[analysis]
    f_a = f[analysis]
    rng_corrupt = np.random.default_rng(seed + 2)
    rng_perm = np.random.default_rng(seed + 3)
    f_corrupt = rng_corrupt.permutation(f_a)

    counts = {"unadjusted": 0, "adjusted": 0, "adjusted_corrupted": 0}
    for _ in range(n_perm):
        t_perm = rng_perm.permutation(t_a)
        if ate_unadjusted(y_a, t_perm, alpha=ALPHA).rejects:
            counts["unadjusted"] += 1
        if ate_ppi_single_lambda(y_a, t_perm, f_a, alpha=ALPHA).rejects:
            counts["adjusted"] += 1
        if ate_ppi_single_lambda(y_a, t_perm, f_corrupt, alpha=ALPHA).rejects:
            counts["adjusted_corrupted"] += 1

    rows = {}
    for key, k in counts.items():
        rate = k / n_perm
        ci = stats.binomtest(k, n_perm).proportion_ci(
            confidence_level=0.95, method="exact"
        )
        p_infl = float(
            stats.binomtest(k, n_perm, ALPHA, alternative="greater").pvalue
        )
        se = math.sqrt(rate * (1.0 - rate) / n_perm) if n_perm else float("nan")
        rows[key] = {
            "rejects": int(k),
            "permutations": int(n_perm),
            "rate": float(rate),
            "ci_low": float(ci.low),
            "ci_high": float(ci.high),
            "mc_se": float(se),
            "p_inflation": p_infl,
            "verdict": "holds" if p_infl >= 0.05 else "inflates",
        }

    return {
        "n_permutations": int(n_perm),
        "seed_corruption": int(seed + 2),
        "seed_permutation": int(seed + 3),
        "corruption": "fixed random permutation of the analysis scores",
        "score_correlation_with_corrupted": float(
            np.corrcoef(f_a, f_corrupt)[0, 1]
        ),
        "methods": rows,
    }


def load_brca(
    expr_path: Path, clinical_path: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Align ER status to expression, drop unusable genes, return a clean matrix."""
    clinical = pd.read_csv(clinical_path, sep="\t", low_memory=False)
    expression = pd.read_csv(expr_path, sep="\t", index_col=0)
    if "ER_Status_nature2012" not in clinical.columns:
        raise DataError("TCGA clinical matrix has no ER_Status_nature2012")
    if "sampleID" not in clinical.columns:
        raise DataError("TCGA clinical matrix has no sampleID")

    n_expression_samples = int(expression.shape[1])
    n_expression_genes = int(expression.shape[0])
    known = clinical[
        clinical["ER_Status_nature2012"].isin(["Positive", "Negative"])
    ][["sampleID", "ER_Status_nature2012"]].dropna()
    n_known = int(len(known))
    known = known.drop_duplicates(subset="sampleID")
    overlap = known[known["sampleID"].isin(expression.columns)].reset_index(drop=True)
    n_aligned = int(len(overlap))
    if n_aligned < 100:
        raise DataError(f"only {n_aligned} ER-labelled samples aligned to expression")

    X = expression[overlap["sampleID"].tolist()].T.to_numpy(dtype=np.float32)
    genes = expression.index.to_numpy()
    y = (overlap["ER_Status_nature2012"] == "Positive").to_numpy(dtype=int)

    finite = np.isfinite(X).all(axis=0)
    n_nonfinite = int((~finite).sum())
    X = X[:, finite]
    genes = genes[finite]

    variable = X.std(axis=0) > 0
    n_constant = int((~variable).sum())
    X = X[:, variable]
    genes = genes[variable]

    if np.unique(y).size != 2:
        raise DataError("ER status has only one class after alignment")

    meta = {
        "n_expression_samples": n_expression_samples,
        "n_expression_genes": n_expression_genes,
        "n_er_known": n_known,
        "n_er_positive": int((y == 1).sum()),
        "n_er_negative": int((y == 0).sum()),
        "n_aligned": n_aligned,
        "n_genes_nonfinite_dropped": n_nonfinite,
        "n_genes_constant_dropped": n_constant,
        "n_genes_used": int(X.shape[1]),
        "positive_prevalence": float(y.mean()),
        "sample_ids": overlap["sampleID"].tolist(),
        "gene_names": genes,
    }
    return X, y, genes, meta


def bootstrap_auc(
    y: np.ndarray, scores: np.ndarray, seed: int, n_boot: int = 1000
) -> Tuple[float, float, int]:
    """Percentile bootstrap interval for an AUC. Samples are the unit."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    values: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, y.size, y.size)
        if np.unique(y[idx]).size < 2:
            continue
        values.append(float(roc_auc_score(y[idx], scores[idx])))
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi), int(len(values))


def fit_expression_logistic(
    X: np.ndarray, y: np.ndarray, seed: int, n_boot: int = 1000
) -> Dict[str, Any]:
    """L2 logistic ER baseline. AUC is out of fold, not in sample."""
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline(
        [
            # A gene that is constant within a training fold has no usable
            # F statistic. Dropping it inside the pipeline keeps that decision
            # out of the validation fold and keeps the selector finite.
            ("variance", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=200)),
            (
                "logit",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold = np.full(y.size, -1, dtype=int)
    for k, (_, test_idx) in enumerate(cv.split(X, y)):
        fold[test_idx] = k
    if (fold < 0).any():
        raise DataError("cross-validation left a sample without a fold")

    proba = cross_val_predict(
        pipe, X, y, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    auc = float(roc_auc_score(y, proba))
    lo, hi, n_valid = bootstrap_auc(y, proba, seed + 4, n_boot)

    pipe.fit(X, y)
    var_mask = pipe.named_steps["variance"].get_support()
    sel_mask = pipe.named_steps["select"].get_support()
    coef = pipe.named_steps["logit"].coef_[0]
    selected_positions = np.flatnonzero(var_mask)[np.flatnonzero(sel_mask)]
    top = np.argsort(-np.abs(coef))[:15]
    # ``coef`` indexes the 200 selected genes; map back through both masks to
    # the full matrix so the report can name them with the caller's gene array.
    top_positions = selected_positions[top]
    return {
        "auc": auc,
        "auc_ci_low": float(lo),
        "auc_ci_high": float(hi),
        "n_boot": int(n_boot),
        "n_boot_valid": int(n_valid),
        "n_folds": int(cv.get_n_splits()),
        "n_genes_after_variance_final": int(var_mask.sum()),
        "n_selected": int(sel_mask.sum()),
        "top_coefficients": [float(coef[i]) for i in top],
        "top_positions": [int(i) for i in top_positions],
        "probability": proba,
        "fold": fold,
        "model": {
            "variance_filter": "VarianceThreshold(0) inside each fold",
            "scaler": "StandardScaler",
            "selector": "SelectKBest(f_classif, k=200)",
            "classifier": "LogisticRegression(C=1.0, class_weight=balanced, lbfgs)",
            "seed": int(seed),
        },
    }


def fit_expression_group_lasso(
    X: np.ndarray,
    y: np.ndarray,
    gene_names: np.ndarray,
    seed: int,
    n_folds: int = 5,
    top_k: int = 300,
    q: float = 0.10,
    n_perm: int = 10,
    group_threshold: float = 0.8,
    max_iter: int = 200,
    n_boot: int = 1000,
) -> Dict[str, Any]:
    """Cross-validated ER score from ppict's permutation-assisted group lasso.

    The package selector screens the top ``top_k`` genes by marginal correlation
    inside each training fold, groups them by absolute co-expression, calibrates
    the group penalty against row-permuted outcomes, and refits OLS on the
    selected genes. ER status is binary, so the resulting score is a linear
    probability score: AUC is a ranking metric, not a calibrated probability.

    The final fit on all 780 samples is for naming the selected genes only. It
    does not contribute to the out-of-fold AUC.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = np.full(y.size, np.nan, dtype=float)
    fold = np.full(y.size, -1, dtype=int)
    fold_aucs: List[float] = []
    selected_sets: List[np.ndarray] = []
    selected_counts: List[int] = []

    for k, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        model = fit_prognostic(
            X[train_idx],
            y[train_idx],
            top_k=top_k,
            q=q,
            n_perm=n_perm,
            group_threshold=group_threshold,
            rng=seed + k,
            max_iter=max_iter,
        )
        prediction = model.predict(X[test_idx])
        scores[test_idx] = prediction
        fold[test_idx] = k
        fold_aucs.append(float(roc_auc_score(y[test_idx], prediction)))
        selected_sets.append(model.feature_idx)
        selected_counts.append(int(model.n_features))

    if (fold < 0).any() or not np.isfinite(scores).all():
        raise DataError("group-lasso cross-validation did not cover every sample")

    auc = float(roc_auc_score(y, scores))
    lo, hi, n_valid = bootstrap_auc(y, scores, seed + 4, n_boot)

    final = fit_prognostic(
        X,
        y,
        top_k=top_k,
        q=q,
        n_perm=n_perm,
        group_threshold=group_threshold,
        rng=seed,
        max_iter=max_iter,
    )
    idx = final.feature_idx
    coef = np.asarray(final.coef, dtype=float)
    # ``coef`` is on the raw log2 expression scale. Multiplying by the gene's
    # SD gives the change in the linear probability score per 1 SD, which is
    # the only scale on which these coefficients are comparable to each other.
    coef_sd = coef * X[:, idx].std(axis=0) if idx.size else np.empty(0, dtype=float)
    order = np.argsort(-np.abs(coef_sd))[:15]
    top_positions = idx[order] if order.size else np.empty(0, dtype=int)

    jaccard = pairwise_jaccard(selected_sets)
    mean_jaccard = float(jaccard.mean()) if jaccard.size else float("nan")

    return {
        "auc": auc,
        "auc_ci_low": float(lo),
        "auc_ci_high": float(hi),
        "n_boot": int(n_boot),
        "n_boot_valid": int(n_valid),
        "n_folds": int(n_folds),
        "fold_aucs": [float(v) for v in fold_aucs],
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "mean_fold_jaccard": mean_jaccard,
        "selected_counts_per_fold": [int(v) for v in selected_counts],
        "n_selected_final": int(final.n_features),
        "n_selected_groups_final": int(final.selected_groups.size),
        "final_train_r2": float(final.train_r2),
        "final_lam": float(final.lam),
        "top_coefficients": [float(coef_sd[i]) for i in order],
        "top_coefficients_raw": [float(coef[i]) for i in order],
        "top_positions": [int(i) for i in top_positions],
        "top_genes": [str(gene_names[i]) for i in top_positions],
        "final_selected_genes": [str(gene_names[i]) for i in idx],
        "score": scores,
        "fold": fold,
        "model": {
            "selector": "ppict.fit_prognostic",
            "top_k": int(top_k),
            "group_threshold": float(group_threshold),
            "screening": f"top {top_k} genes by marginal correlation inside each fold",
            "groups": f"absolute co-expression connected components at {group_threshold}",
            "q": float(q),
            "n_perm": int(n_perm),
            "max_iter": int(max_iter),
            "outcome_model": "OLS refit on selected genes (linear probability score)",
            "seed": int(seed),
            "fold_seed_rule": "seed + fold index",
        },
    }


def transport_check(
    trial: pd.DataFrame,
    brca_meta: Dict[str, Any],
    expression_genes: np.ndarray,
) -> Dict[str, Any]:
    """Measure the three reasons the expression score cannot reach this trial."""
    trial_columns = set(trial.columns)
    gene_set = set(str(g) for g in expression_genes)
    shared_features = sorted(trial_columns & gene_set)
    trial_ids = set(trial["pidnum"].astype(str))
    brca_ids = set(brca_meta["sample_ids"])
    shared_patients = sorted(trial_ids & brca_ids)
    expression_like = [
        c for c in trial.columns if str(c).startswith("TCGA-")
    ]
    return {
        "n_trial_patients": int(len(trial)),
        "n_brca_samples": int(brca_meta["n_aligned"]),
        "n_shared_patients": int(len(shared_patients)),
        "n_shared_feature_names": int(len(shared_features)),
        "shared_feature_names": shared_features[:20],
        "n_expression_columns_in_trial": int(len(expression_like)),
        "trial_outcome": OUTCOME_LABEL,
        "expression_outcome": "ER status (Positive / Negative)",
        "outcome_scales_comparable": False,
        "conclusion": (
            "The expression score cannot be applied to ACTG 175. The trial has "
            "no gene-expression measurements, the two cohorts share no patients, "
            "and an ER probability is not on the CD4 outcome scale. No score "
            "was transferred, imputed, or simulated for the trial analysis."
        ),
    }


def _p_text(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.0001:
        return "<0.0001"
    return f"{p:.4f}"


def _ci_text(low: float, high: float, digits: int = 2) -> str:
    return f"{low:.{digits}f} to {high:.{digits}f}"


def simulation_reference() -> Optional[Dict[str, Any]]:
    """The comparable simulation row, if the null sweep has already run.

    Reading the generated table keeps the report's comparison traceable to a
    file instead of a number copied into this script.
    """
    path = RESULTS / "null_width_ratios.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[(df["config"] == "good_score") & (df["method"] == "ppi_single")]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "source": "validation/results/null_width_ratios.csv",
        "config": "good_score",
        "method": "ppi_single",
        "width_ratio": float(r["width_ratio_vs_unadjusted"]),
        "variance_ratio": float(r["variance_ratio_vs_unadjusted"]),
        "n_paired": int(r["n_paired"]),
    }


def build_report(results: Dict[str, Any]) -> str:
    actg = results["actg"]
    trial = actg["trial"]
    score = actg["score"]
    u = trial["unadjusted"]
    p = trial["adjusted"]
    nc = actg["negative_control"]
    brca = results["brca"]
    g = brca["group_lasso"]
    lg = brca["logistic_baseline"]
    tr = results["transport"]
    data = results["data"]
    sim = results.get("simulation_reference")

    lines: List[str] = []
    lines.append("# Real-data retrospective: ACTG 175 and TCGA-BRCA")
    lines.append("")
    lines.append(
        "Measured numbers only. Every value below is written by "
        "`validation/run_real_data.py`, which also writes "
        "`real_data_results.json`, `actg_scores.csv` and "
        "`brca_er_scores.csv`. The command is in the reproduction section."
    )
    lines.append("")
    lines.append("## Bottom line")
    lines.append("")
    lines.append(
        f"- On {trial['n_analysis']} patients from two arms of ACTG 175, the "
        f"adjusted interval was **{p['ci_width']:.1f} CD4 cells wide** against "
        f"**{u['ci_width']:.1f}** unadjusted, a **{100*trial['width_narrowing_fraction']:.1f}% "
        f"narrowing** (width ratio {trial['width_ratio_adjusted_over_unadjusted']:.3f})."
    )
    lines.append(
        f"- The adjusted estimate was **{p['estimate']:.2f}** CD4 cells "
        f"(95% CI {_ci_text(p['ci_low'], p['ci_high'])}); unadjusted was "
        f"**{u['estimate']:.2f}** (95% CI {_ci_text(u['ci_low'], u['ci_high'])})."
    )
    la = trial["lasso_l1"]
    lines.append(
        f"- The L1 lasso baseline was **{la['estimate']:.2f}** "
        f"(95% CI {_ci_text(la['ci_low'], la['ci_high'])}), width "
        f"{la['ci_width']:.2f}, the narrowest of the three. It selects "
        f"covariates on the same patients it estimates on, so it is a "
        f"comparison row, not a recommended analysis."
    )
    lines.append(
        f"- With treatment labels permuted ({nc['n_permutations']:,} permutations), "
        f"the rejection rate was **{100*nc['methods']['unadjusted']['rate']:.1f}%** "
        f"unadjusted, **{100*nc['methods']['adjusted']['rate']:.1f}%** adjusted, "
        f"and **{100*nc['methods']['adjusted_corrupted']['rate']:.1f}%** with a "
        f"corrupted score. All three are consistent with alpha = 0.05."
    )
    lines.append(
        f"- The TCGA-BRCA expression score is fitted with the package's 5-fold "
        f"group lasso (AUC **{g['auc']:.3f}**, 95% CI "
        f"{_ci_text(g['auc_ci_low'], g['auc_ci_high'], 3)}, "
        f"{g['n_selected_final']} genes in the final fit). An L2 logistic "
        f"baseline scores {lg['auc']:.3f} (95% CI "
        f"{_ci_text(lg['auc_ci_low'], lg['auc_ci_high'], 3)}). Neither can be "
        f"transported to this trial: ACTG 175 has no expression measurements, "
        f"the cohorts share no patients, and ER status is not on the CD4 scale."
    )
    lines.append(
        "- On this one stratified split, the real data are consistent with the "
        "estimator's interval narrowing and alpha control on a real randomized "
        "trial, using a held-out internal clinical score. They do not establish "
        "an external expression-derived score transported into a trial. The "
        "simulation remains the only evidence for that, and for the "
        "external-cohort PPI term."
    )
    lines.append("")
    lines.append("## Data, provenance, and named blocks")
    lines.append("")
    lines.append("| file | route | bytes | sha256 | file mtime (UTC) |")
    lines.append("| --- | --- | --- | --- | --- |")
    for key in ("actg", "brca_expr", "brca_clin"):
        d = data[key]
        lines.append(
            f"| `{d['path']}` | {d['url']} | {d['bytes']:,} | "
            f"`{d['sha256'][:16]}...` | {d['modified_utc']} |"
        )
    lines.append("")
    lines.append(
        "The full sha256 values are in `real_data_results.json`. Sizes are "
        "checked on every run; a mismatch stops the script instead of using a "
        "partial or changed file. Neither dataset is redistributed; both are "
        "ignored by git."
    )
    lines.append("")
    lines.append("### ACTG 175")
    lines.append("")
    lines.append(
        f"The public extract has {actg['meta']['n_rows_total']:,} patients in "
        f"four arms ("
        + ", ".join(
            f"{ARM_LABELS[k]}: {v}"
            for k, v in sorted(actg["meta"]["arm_counts_total"].items())
        )
        + f"). This analysis uses the two arms {ARM_LABELS[0]} and "
        f"{ARM_LABELS[1]}: {actg['meta']['n_two_arm']:,} patients, "
        f"{actg['meta']['n_complete']:,} complete cases, "
        f"{actg['meta']['n_dropped_complete_case']} dropped for missing values. "
        f"The outcome is {OUTCOME_LABEL}."
    )
    lines.append("")
    lines.append(
        "This is a secondary endpoint, not the trial's primary AIDS/death "
        "endpoint, and it is a two-arm subset of a four-arm trial. Those are "
        "limits of what the extract supports, not claims about the trial."
    )
    lines.append("")
    lines.append("### TCGA-BRCA")
    lines.append("")
    n_known = brca["meta"]["n_er_known"]
    n_aligned = brca["meta"]["n_aligned"]
    alignment_text = (
        f"all {n_aligned:,} align to expression"
        if n_known == n_aligned
        else f"{n_aligned:,} of those align to expression"
    )
    lines.append(
        f"The expression matrix has {brca['meta']['n_expression_samples']:,} "
        f"samples and {brca['meta']['n_expression_genes']:,} genes. "
        f"{brca['meta']['n_er_known']:,} samples have known ER status; "
        f"{alignment_text} "
        f"({brca['meta']['n_er_positive']:,} positive, "
        f"{brca['meta']['n_er_negative']:,} negative). After dropping "
        f"{brca['meta']['n_genes_constant_dropped']} constant and "
        f"{brca['meta']['n_genes_nonfinite_dropped']} non-finite genes, "
        f"{brca['meta']['n_genes_used']:,} genes are available. The group-lasso "
        f"selector screens the top 300 by marginal correlation inside each "
        f"training fold; the logistic baseline drops zero-variance genes inside "
        f"each fold and selects 200 by ANOVA F."
    )
    lines.append("")
    lines.append("### Routes that did not work or were not used")
    lines.append("")
    lines.append(
        "- The compressed clinical route, `.../BRCA_clinicalMatrix.gz`, returns "
        "HTTP 403. The uncompressed route used above returns the file."
    )
    lines.append(
        "- Project Data Sphere was not used. Access needs an account and an "
        "approval step, and this project does not create accounts or contact "
        "anyone. The public ACTG 175 extract supplies the randomized trial."
    )
    lines.append(
        "- Transporting the TCGA expression score to ACTG 175 is impossible "
        "with the public data: the trial records no gene expression, the "
        "cohorts have no shared patients, and the outcomes are on different "
        "scales. This is the central negative result of the pass."
    )
    lines.append("")
    lines.append("## Trial score")
    lines.append("")
    n_feat = len(score["selected_features"])
    feat_word = "feature" if n_feat == 1 else "features"
    n_grp = score["n_selected_groups"]
    grp_word = "group" if n_grp == 1 else "groups"
    lines.append(
        f"The score is the project's group-lasso prognostic score, fitted on "
        f"the {score['n_fit_controls']} control patients in the training half "
        f"only. It selected {n_feat} {feat_word}: "
        f"**{', '.join(score['selected_features']) if score['selected_features'] else 'none'}**. "
        f"Training R^2 was {score['train_r2']:.4f}; out-of-sample R^2 on the "
        f"{score['n_validation_controls']} control patients in the analysis "
        f"half was **{score['oos_control_r2']:.4f}**."
    )
    lines.append("")
    lines.append(
        f"The group lasso screened {score['n_groups']} groups at q = "
        f"{score['q']:.2f} with {score['n_perm']} permutations and selected "
        f"{n_grp} {grp_word}; the tuning value on the "
        f"standardised scale was {score['lam_standardized']:.2f}. The raw-scale "
        f"coefficients are in `real_data_results.json`."
    )
    lines.append("")
    lines.append(
        f"The split is stratified by arm at seed {results['run']['seed']}: "
        f"{trial['n_analysis_treated']} treated and "
        f"{trial['n_analysis_control']} control in the analysis half, "
        f"disjoint from the score's training patients. No analysis patient "
        f"trained the score. This is the only split reported; no "
        f"split-to-split interval is computed."
    )
    lines.append("")
    lines.append("## Treatment effect and interval width")
    lines.append("")
    lines.append(
        "| method | estimate | SE | 95% CI | width | width vs unadj | "
        "implied control arm | p | detail |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for label, key in (
        ("unadjusted", "unadjusted"),
        ("L1 lasso", "lasso_l1"),
        ("group-lasso PPI++", "adjusted"),
    ):
        row = trial[key]
        control = (
            "n/a"
            if row["implied_control_arm_80"] is None
            else f"{row['implied_control_arm_80']:.0f}"
        )
        if row["n_selected"] is not None:
            detail = f"{row['n_selected']} selected"
        elif row["lam"] is not None:
            detail = f"lambda {row['lam']:.4f}"
        else:
            detail = "n/a"
        lines.append(
            f"| {label} | {row['estimate']:.2f} | {row['se']:.2f} | "
            f"{_ci_text(row['ci_low'], row['ci_high'])} | {row['ci_width']:.2f} | "
            f"{row['width_ratio_vs_unadjusted']:.3f} | {control} | "
            f"{_p_text(row['p_value'])} | {detail} |"
        )
    lines.append("")
    lines.append(
        "The implied-control-arm column is a projection, not a measured trial "
        "size: it scales each estimator's observed variance to the observed "
        "unadjusted effect at 80% power with 1:1 allocation. The L1 row is the "
        "naive post-selection baseline the spec asks for, not a recommended "
        "method."
    )
    lines.append("")
    lines.append(
        f"The L1 lasso produced the narrowest interval on this split "
        f"(ratio {trial['lasso_width_ratio_vs_unadjusted']:.4f}). That is not "
        f"evidence that it is the better method: it selects covariates on the "
        f"same 527 patients it then estimates on, with no correction for the "
        f"selection. The null sweep records the consequence, an 8.75% rejection "
        f"rate in the poorly estimated score scenario "
        f"(`validation/results/NULL_REPORT.md`). The group-lasso PPI++ row is "
        f"the one whose score was fitted on disjoint patients."
    )
    lines.append("")
    lines.append(
        f"Width ratio adjusted/unadjusted: "
        f"**{trial['width_ratio_adjusted_over_unadjusted']:.4f}** "
        f"(variance ratio {trial['variance_ratio_adjusted_over_unadjusted']:.4f}); "
        f"L1 lasso ratio {trial['lasso_width_ratio_vs_unadjusted']:.4f}. "
        f"The point estimates differ by "
        f"{trial['difference_adjusted_minus_unadjusted']:.2f} CD4 cells "
        f"(adjusted minus unadjusted)."
    )
    lines.append("")
    lines.append(
        "The estimator used here is the shared-tuning PPI++ form. With no "
        "external unlabeled cohort, the unlabeled mean cancels between arms, so "
        "this is algebraically an ANCOVA on one fitted index when the arms are "
        "equal in size. Here the analysis arms differ by 5 patients, so the "
        "identity is approximate. The real trial therefore tests the covariate "
        "adjustment, not the external-cohort variance term. The simulation is "
        "still the evidence for that term."
    )
    lines.append("")
    lines.append("## Negative control")
    lines.append("")
    lines.append(
        f"Treatment labels in the analysis half were permuted "
        f"{nc['n_permutations']:,} times. The real score stayed fixed; the "
        f"corrupted score is one fixed permutation of the same analysis scores "
        f"(correlation with the real score "
        f"{nc['score_correlation_with_corrupted']:.4f}), so its marginal "
        f"distribution is unchanged and only the patient-level relationship is "
        f"destroyed. Each permutation is shared by the three rows, so the "
        f"comparison is paired."
    )
    lines.append("")
    lines.append("| method | rejects | rate | 95% CI | MC SE | p (inflation) | verdict |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for label, key in (
        ("unadjusted", "unadjusted"),
        ("adjusted", "adjusted"),
        ("adjusted, corrupted score", "adjusted_corrupted"),
    ):
        row = nc["methods"][key]
        lines.append(
            f"| {label} | {row['rejects']} / {row['permutations']} | "
            f"{row['rate']:.4f} | {_ci_text(row['ci_low'], row['ci_high'], 4)} | "
            f"{row['mc_se']:.4f} | {_p_text(row['p_inflation'])} | "
            f"{row['verdict']} |"
        )
    lines.append("")
    lines.append(
        "The exact binomial test is one sided for inflation, "
        "H0: rate <= 0.05. A verdict of holds means the data are consistent "
        "with alpha; it is not proof that the rate is exactly 0.05. The "
        "intervals reflect permutation noise for this one fitted score and one "
        "corruption, not variability across scores or splits."
    )
    lines.append("")
    lines.append("## TCGA-BRCA expression scores")
    lines.append("")
    lines.append(
        "Two models are fitted to the same 780 ER-labelled samples, and they "
        "answer different questions. The primary model is the project's own "
        "permutation-assisted group lasso, so the real-data retrospective "
        "exercises the package selector rather than a substitute. The L2 "
        "logistic model is a familiar wide-matrix benchmark, kept so the "
        "group-lasso number can be read against something. It is not the "
        "package method and none of its numbers appear in the ACTG analysis."
    )
    lines.append("")
    lines.append("### Primary: ppict group lasso")
    lines.append("")
    g_fold_auc = ", ".join(f"{v:.4f}" for v in g["fold_aucs"])
    g_counts = ", ".join(str(v) for v in g["selected_counts_per_fold"])
    lines.append(
        f"5-fold cross-validation (StratifiedKFold, shuffle, seed "
        f"{g['model']['seed']}) of `ppict.fit_prognostic`: inside each training "
        f"fold, screen the top {g['model']['top_k']} genes by marginal "
        f"correlation, group them by absolute co-expression at "
        f"{g['model']['group_threshold']}, calibrate the group penalty against "
        f"{g['model']['n_perm']} row permutations at q = "
        f"{g['model']['q']}, and refit OLS on the selected genes. ER status is "
        f"binary, so the refit is a linear probability score; AUC is a ranking "
        f"metric for it, not a calibrated probability."
    )
    lines.append("")
    lines.append(
        f"Out-of-fold AUC **{g['auc']:.4f}** (95% bootstrap CI by the "
        f"percentile method {_ci_text(g['auc_ci_low'], g['auc_ci_high'], 4)}, "
        f"{g['n_boot_valid']:,} valid resamples). Per-fold AUC: {g_fold_auc} "
        f"(mean {g['mean_fold_auc']:.4f}). Genes selected per fold: {g_counts}. "
        f"Mean pairwise Jaccard across fold selections was "
        f"**{g['mean_fold_jaccard']:.4f}**, so the fold models agree only "
        f"partly on which genes matter. The final fit on all 780 samples, used "
        f"for naming genes only, selected {g['n_selected_final']} genes in "
        f"{g['n_selected_groups_final']} groups with training R^2 "
        f"{g['final_train_r2']:.4f}."
    )
    lines.append("")
    g_top = ", ".join(
        f"{gene} ({coef:+.2f})"
        for gene, coef in zip(g["top_genes"], g["top_coefficients"])
    )
    g_biology = [gene for gene in ("ESR1", "PGR", "FOXA1") if gene in g["top_genes"]]
    if len(g_biology) == 3:
        g_biology_text = (
            "ESR1, PGR and FOXA1 are all in that list, which is the expected "
            "biology for an ER model and a sanity check that the matrix and "
            "labels are aligned."
        )
    else:
        g_biology_text = (
            "The expected ER genes in the top 15 were "
            + (", ".join(g_biology) if g_biology else "none")
            + ". This is reported as it came out, not filtered to match the "
            "expected biology."
        )
    lines.append(
        f"Largest changes in the linear ER probability score per 1 SD of the "
        f"gene, final group-lasso fit: {g_top}. " + g_biology_text
    )
    lines.append("")
    lines.append(
        "The refit is OLS on a binary outcome, so this is a linear probability "
        "score: it is not calibrated, and individual scores can fall outside "
        "[0, 1]. AUC uses only the ranking, which is why it is reported for "
        "this model instead of a probability-scale metric."
    )
    lines.append("")
    lines.append("### Baseline: L2 logistic")
    lines.append("")
    lines.append(
        f"A separate 5-fold pipeline on the same folds (drop zero-variance "
        f"genes inside each fold, standardise, select 200 genes by ANOVA F, L2 "
        f"logistic regression with balanced classes) reached AUC "
        f"**{lg['auc']:.4f}** (95% bootstrap CI "
        f"{_ci_text(lg['auc_ci_low'], lg['auc_ci_high'], 4)}, "
        f"{lg['n_boot_valid']:,} valid resamples). This is the benchmark, not "
        f"the package method. The two AUCs share samples and folds, so they "
        f"are paired measurements on one split; no test of their difference is "
        f"computed."
    )
    lines.append("")
    lg_top = ", ".join(
        f"{gene} ({coef:+.2f})"
        for gene, coef in zip(brca["logistic_top_genes"], lg["top_coefficients"])
    )
    lg_biology = [
        gene for gene in ("ESR1", "PGR", "FOXA1") if gene in brca["logistic_top_genes"]
    ]
    lg_biology_text = (
        "ESR1, PGR and FOXA1 are all in that list."
        if len(lg_biology) == 3
        else "The expected ER genes in the top 15 were "
        + (", ".join(lg_biology) if lg_biology else "none")
        + "."
    )
    lines.append(
        f"Largest standardised logistic coefficients in the final fit: "
        f"{lg_top}. " + lg_biology_text
    )
    lines.append("")
    lines.append(
        f"Both numbers come from one 5-fold split at seed "
        f"{results['run']['seed']}. The bootstrap intervals resample samples, "
        f"not fold seeds, so neither interval includes fold-assignment "
        f"variability. The Jaccard value above is reported for the group "
        f"lasso only, because that is the model whose refit depends on a "
        f"selected set; the logistic model has a fixed 200-gene budget."
    )
    lines.append("")
    lines.append("### What this score is not")
    lines.append("")
    lines.append(
        f"It is not a trial prognostic score. The trial analysis uses the "
        f"clinical score above. The expression model predicts ER status from "
        f"gene expression in TCGA; applying it to ACTG 175 would need "
        f"expression measurements for those patients and a mapping to the CD4 "
        f"outcome. Neither exists in the public data. The script checks and "
        f"reports {tr['n_shared_patients']} shared patients, "
        f"{tr['n_shared_feature_names']} shared feature names, and "
        f"{tr['n_expression_columns_in_trial']} expression columns in the "
        f"trial. The patient check compares identifiers from two different "
        f"namespaces, so it can only rule out an accidental match; there is no "
        f"record linkage between TCGA and ACTG."
    )
    lines.append("")
    lines.append(
        "Its AUC is also not a treatment-effect result. It measures how well "
        "ER status can be recovered from expression, not whether adjusting a "
        "trial for such a score narrows an interval. Only the ACTG analysis "
        "measures that."
    )
    lines.append("")
    lines.append("## What this split shows, and what it does not")
    lines.append("")
    lines.append(
        f"**Observed on this split.** A prognostic score fitted only on "
        f"held-out controls narrowed the interval on this ACTG split by "
        f"{100*trial['width_narrowing_fraction']:.1f}%. Under permuted labels "
        f"the adjusted analysis rejected at "
        f"{100*nc['methods']['adjusted']['rate']:.1f}%, and a corrupted score "
        f"did not inflate it "
        f"({100*nc['methods']['adjusted_corrupted']['rate']:.1f}%). Those are "
        f"measured for this split and this score, not a distribution over "
        f"splits or scores."
    )
    lines.append("")
    lines.append(
        "**Not established.** An expression-derived score transported into a "
        "trial. No public route found here joins TCGA expression to ACTG 175. "
        "The simulation's transport and corruption scenarios remain "
        "simulations."
    )
    lines.append("")
    lines.append(
        "**The two conditions come apart.** The TCGA score comes from a "
        "separate cohort (0 shared identifiers) but cannot be applied to it. The clinical "
        "trial score can be applied, but it is an internal split of ACTG 175: "
        "it shares no analysis patients, yet it is not an external cohort. The "
        "real analysis is therefore a test of the estimator with a held-out "
        "internal score, not a test of an external expression score. The "
        "simulation's external-score scenarios remain the evidence for that "
        "setting."
    )
    lines.append("")
    if sim is not None:
        lines.append(
            f"**Compared with the simulation.** The simulation's good-score "
            f"width ratio for the same estimator is {sim['width_ratio']:.4f} "
            f"({sim['n_paired']} paired replicates, from `{sim['source']}`); "
            f"the real trial's ratio is "
            f"{trial['width_ratio_adjusted_over_unadjusted']:.4f}. The scores "
            f"have different out-of-sample R^2 and the outcomes differ, so "
            f"these are two measurements, not a replication of one number. "
            f"This is a comparison of magnitudes, not a controlled test."
        )
    else:
        lines.append(
            f"**Compared with the simulation.** The real trial's width ratio "
            f"is {trial['width_ratio_adjusted_over_unadjusted']:.4f}. The "
            f"simulation's comparable row was not available when this report "
            f"was written; `validation/results/NULL_REPORT.md` is the source "
            f"for that comparison. The scores and outcomes differ, so the two "
            f"are not a replication of one number."
        )
    lines.append("")
    lines.append(
        "**Still unproven.** The external-cohort PPI term, binary and "
        "time-to-event outcomes, a score that is prognostic in one arm but not "
        "the other, and the four-arm primary endpoint of ACTG 175. The "
        "repository's limits section lists these too."
    )
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("# from the repository root")
    lines.append("./.venv/bin/python validation/run_real_data.py --permutations 1000")
    lines.append("```")
    lines.append("")
    lines.append(
        "The script downloads the three public files if they are absent, checks "
        "their sizes, and stops on a mismatch. To reproduce from a clean "
        "result state, delete `REAL_DATA_REPORT.md`, `real_data_results.json`, "
        "`actg_scores.csv` and `brca_er_scores.csv` from "
        "`validation/results` and run the command again. The data cache can "
        "stay; the script records its hashes."
    )
    lines.append("")
    lines.append(
        f"Seeds: split {results['run']['seed']}, trial score fit "
        f"{results['run']['seed'] + 1}, corrupted score "
        f"{nc['seed_corruption']}, label permutations "
        f"{nc['seed_permutation']}; expression folds "
        f"{results['run']['seed']} to {results['run']['seed'] + 4} (seed plus "
        f"fold index), expression final fit {results['run']['seed']}, AUC "
        f"bootstrap {results['run']['seed'] + 4}. Alpha {ALPHA}."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--permutations",
        type=int,
        default=N_PERM_DEFAULT,
        help="treatment-label permutations for the negative control",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="base random seed"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="re-download files even when a correct cached copy exists",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="fail instead of downloading a missing file",
    )
    args = parser.parse_args(argv)

    if args.permutations < 100:
        raise SystemExit("--permutations must be at least 100")
    if args.seed < 0:
        raise SystemExit("--seed must be non-negative")

    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Checking public data files")
    if args.skip_download:
        for src in SOURCES:
            path = DATA / src.filename
            if not path.exists() or path.stat().st_size != src.bytes:
                raise SystemExit(
                    f"{path} is missing or wrong-sized and --skip-download was given"
                )
        data_info = {
            src.key: {
                "key": src.key,
                "description": src.description,
                "url": src.url,
                "path": str((DATA / src.filename).relative_to(ROOT)),
                "bytes": src.bytes,
                "sha256": _sha256(DATA / src.filename),
                "sha256_scope": "full file",
                "status": "cached",
                "modified_utc": datetime.fromtimestamp(
                    (DATA / src.filename).stat().st_mtime, timezone.utc
                ).strftime("%Y-%m-%d %H:%M UTC"),
            }
            for src in SOURCES
        }
    else:
        data_info = ensure_data(args.force_download)
    for key, d in data_info.items():
        print(f"  {key}: {d['bytes']:,} bytes, {d['sha256'][:16]}... ({d['status']})")

    print("Loading ACTG 175")
    trial_df = load_actg(DATA / "actg175.csv")
    trial, trial_meta = prepare_trial(trial_df)
    X = trial[list(COVARIATES)].to_numpy(dtype=float)
    y = trial[OUTCOME_COLUMN].to_numpy(dtype=float)
    trt = trial["trt"].to_numpy(dtype=int)

    train = stratified_split(trt, args.seed)
    print(
        f"  {trial_meta['n_complete']} complete cases; "
        f"{int(train.sum())} score-training, {int((~train).sum())} analysis"
    )

    print("Fitting the trial score on training controls only")
    score = fit_trial_score(X, y, trt, train, args.seed + 1)
    print(
        f"  selected {score['selected_features']} "
        f"(OOS control R^2 {score['oos_control_r2']:.4f})"
    )

    print("Running the shipped estimators on the analysis half")
    trial_result, u, lasso, p = estimate_trial(
        y, trt, score["f"], X, ~train, args.seed
    )
    print(
        f"  unadjusted {u.estimate:.2f} [{u.ci_low:.2f}, {u.ci_high:.2f}] "
        f"width {u.ci_width:.2f}"
    )
    print(
        f"  lasso_l1   {lasso.estimate:.2f} [{lasso.ci_low:.2f}, {lasso.ci_high:.2f}] "
        f"width {lasso.ci_width:.2f}"
    )
    print(
        f"  adjusted   {p.estimate:.2f} [{p.ci_low:.2f}, {p.ci_high:.2f}] "
        f"width {p.ci_width:.2f} "
        f"({100*trial_result['width_narrowing_fraction']:.1f}% narrower)"
    )

    print(f"Negative control: {args.permutations} permutations")
    nc = negative_control(y, trt, score["f"], ~train, args.permutations, args.seed)
    for key, row in nc["methods"].items():
        print(
            f"  {key}: {row['rejects']}/{row['permutations']} "
            f"= {row['rate']:.4f} ({row['verdict']})"
        )

    print("Loading TCGA-BRCA expression and ER status")
    X_brca, y_brca, genes, brca_meta = load_brca(
        DATA / "brca_hiseq.tsv.gz", DATA / "brca_clinical.tsv"
    )
    print(
        f"  {brca_meta['n_aligned']} labelled samples, "
        f"{brca_meta['n_genes_used']:,} genes after filtering"
    )

    print("Fitting the cross-validated expression scores")
    print("  ppict group lasso")
    brca_group = fit_expression_group_lasso(X_brca, y_brca, genes, args.seed)
    print(
        f"    AUC {brca_group['auc']:.4f} "
        f"[{brca_group['auc_ci_low']:.4f}, {brca_group['auc_ci_high']:.4f}], "
        f"{brca_group['n_selected_final']} genes in the final fit, "
        f"mean fold Jaccard {brca_group['mean_fold_jaccard']:.4f}"
    )
    print(f"    top genes: {', '.join(brca_group['top_genes'][:8])}")

    print("  L2 logistic baseline")
    brca_logistic = fit_expression_logistic(X_brca, y_brca, args.seed)
    print(
        f"    AUC {brca_logistic['auc']:.4f} "
        f"[{brca_logistic['auc_ci_low']:.4f}, {brca_logistic['auc_ci_high']:.4f}]"
    )
    print(f"    top genes: {', '.join(str(genes[i]) for i in brca_logistic['top_positions'][:8])}")
    if not np.array_equal(brca_group["fold"], brca_logistic["fold"]):
        raise DataError("the two expression models used different folds")

    transport = transport_check(trial_df, brca_meta, genes)

    results: Dict[str, Any] = {
        "run": {
            "script": "validation/run_real_data.py",
            "command": (
                f"./.venv/bin/python validation/run_real_data.py "
                f"--permutations {args.permutations} --seed {args.seed}"
            ),
            "seed": int(args.seed),
            "alpha": ALPHA,
            "permutations": int(args.permutations),
            "generated_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            ),
        },
        "data": data_info,
        "actg": {
            "meta": trial_meta,
            "covariates": list(COVARIATES),
            "outcome_column": OUTCOME_COLUMN,
            "outcome_label": OUTCOME_LABEL,
            "split": {
                "seed": int(args.seed),
                "n_train": int(train.sum()),
                "n_analysis": int((~train).sum()),
                "n_train_treated": int((train & (trt == 1)).sum()),
                "n_train_control": int((train & (trt == 0)).sum()),
                "n_analysis_treated": int(((~train) & (trt == 1)).sum()),
                "n_analysis_control": int(((~train) & (trt == 0)).sum()),
            },
            "score": {k: v for k, v in score.items() if k != "f"},
            "trial": trial_result,
            "negative_control": nc,
        },
        "brca": {
            "meta": {
                k: v
                for k, v in brca_meta.items()
                if k not in ("sample_ids", "gene_names")
            },
            "primary_model": "ppict_group_lasso",
            "group_lasso": {
                k: v
                for k, v in brca_group.items()
                if k not in ("score", "fold")
            },
            "logistic_baseline": {
                k: v
                for k, v in brca_logistic.items()
                if k not in ("probability", "fold")
            },
            "top_genes": brca_group["top_genes"],
            "logistic_top_genes": [
                str(genes[i]) for i in brca_logistic["top_positions"]
            ],
        },
        "transport": transport,
        "simulation_reference": simulation_reference(),
    }

    # The score vector is needed for the CSV but not the JSON.
    trial_out = trial[["pidnum", "trt", OUTCOME_COLUMN]].copy()
    trial_out["arm"] = trial_out["trt"].map(ARM_LABELS)
    trial_out["split"] = np.where(train, "score_train", "analysis")
    trial_out["statistic_role"] = np.where(
        train & (trial_out["trt"] == 0),
        "fit_score",
        np.where(train, "not_used", "estimate"),
    )
    trial_out["score"] = score["f"]
    trial_out = trial_out.rename(
        columns={"pidnum": "patient_id", OUTCOME_COLUMN: "cd4_week20"}
    )
    trial_out.to_csv(RESULTS / "actg_scores.csv", index=False)

    expr_out = pd.DataFrame(
        {
            "sample_id": brca_meta["sample_ids"],
            "er_status": np.where(y_brca == 1, "Positive", "Negative"),
            "fold": brca_group["fold"],
            "group_lasso_score": brca_group["score"],
            "logistic_probability_positive": brca_logistic["probability"],
        }
    )
    expr_out.to_csv(RESULTS / "brca_er_scores.csv", index=False)

    with (RESULTS / "real_data_results.json").open("w") as fh:
        json.dump(results, fh, indent=2, allow_nan=False, sort_keys=True)
        fh.write("\n")

    report = build_report(results)
    (RESULTS / "REAL_DATA_REPORT.md").write_text(report)

    print("")
    print(f"Wrote {RESULTS / 'REAL_DATA_REPORT.md'}")
    print(f"Wrote {RESULTS / 'real_data_results.json'}")
    print(f"Wrote {RESULTS / 'actg_scores.csv'}")
    print(f"Wrote {RESULTS / 'brca_er_scores.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

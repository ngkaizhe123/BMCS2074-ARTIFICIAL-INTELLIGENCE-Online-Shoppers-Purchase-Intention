"""
svm_model.py
------------
Support Vector Machine (SVM) module for the Online Shoppers Purchasing
Intention classification task — pipeline construction, hyperparameter tuning
via RandomizedSearchCV, probability calibration, OOF threshold optimisation,
cross-validation, and SHAP model interpretability.

Pipeline: preprocessor (StandardScaler + OneHotEncoder) -> SMOTENC/class_weight -> SVC
Probability Calibration: CalibratedClassifierCV(ensemble=False)

Key changes vs. original
--------------------------
* SMOTE replaced with SMOTENC (prevents fractional interpolation of OHE columns).
* Search space covers three *distinct* imbalance strategies:
    (A) SMOTENC only,  (B) class_weight only,  (C) Both combined.
* n_iter raised to 60, scoring changed to "average_precision" (PR-AUC).
* OOF threshold scanner (np.arange(0.20, 0.61, 0.02)) maximises F1 on
  cross-validation data; the frozen threshold is then applied to the test set.
* Import time module; wall-clock duration printed at the very end.

CPU Optimisations (Intel Core Ultra 5 125H — Meteor Lake hybrid)
-----------------------------------------------------------------
* Hybrid topology: 4 P-cores (8 HT) + 8 E-cores + 2 LP-E-cores = 18 threads.
* N_JOBS = 12 — saturates P-cores + half the E-cores; leaves LP-E-cores and
  some P-threads free for OS scheduling and memory bandwidth tasks.
* OOF fold loop parallelised with joblib.Parallel (loky backend, spawn-safe).
* SVC max_iter capped at 10 000 — high enough for extreme C/gamma combos
  sampled during RandomizedSearch to converge; still prevents truly pathological
  cases from hanging on E-cores. tol=1e-3 relaxed to aid early stopping.
* joblib parallel_backend set globally to 'loky' with prefer='processes' so
  each worker gets a dedicated OS thread that the scheduler can pin to a
  P-core or E-core as appropriate.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_SCRIPT_START = time.perf_counter()  # record start immediately

# ---------------------------------------------------------------------------
# CPU topology constants — Intel Core Ultra 5 125H (Meteor Lake)
# ---------------------------------------------------------------------------
# Physical layout: 4 P-cores (8 HT threads) + 8 E-cores + 2 LP-E-cores = 18 threads.
# We target 12 parallel workers:
#   • All 4 P-cores  (8 logical threads via HT)
#   • 4 of the 8 E-cores
# Leaving 4 E-cores + 2 LP-E-cores free for the OS, memory controller,
# and background Streamlit/Python overhead.
_CPU_P_THREADS  = 8   # P-core hyper-threads
_CPU_E_CORES    = 8   # E-core count
_CPU_LPE_CORES  = 2   # Low-Power E-cores (keep free)
_N_JOBS: int    = min(12, os.cpu_count() or 12)  # gracefully caps on smaller machines

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.svm import SVC

from src.data_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
    preprocess_data,
)
from src.utils import (
    generate_shap_explanation,
    print_metrics,
    save_metrics,
    save_model,
    split_dataset,
)


# ---------------------------------------------------------------------------
# Hyperparameter search distributions — three imbalance strategies
# ---------------------------------------------------------------------------

def _make_svm_param_distributions() -> tuple[list, list, list]:
    """
    Build three separate param-dict lists for RandomizedSearchCV.

    Strategy A – SMOTENC only (class_weight=None):
        Pipeline must contain a "smotenc" step.

    Strategy B – class_weight only (no SMOTENC step):
        Pipeline has no oversampler; uses SVC(class_weight=...).

    Strategy C – SMOTENC + class_weight combined.
        Pipeline must contain a "smotenc" step.
    """
    # ── Search space bounds ──────────────────────────────────────────────────
    # C upper bound reduced 300 → 100: very large C causes ill-conditioned dual
    # problems that never converge regardless of max_iter.
    # gamma lower bound raised 1e-4 → 1e-3: extremely small gamma flattens the
    # RBF kernel, making the decision boundary insensitive and slow to converge.
    # Both changes keep the space practically useful while eliminating the
    # combinations that trigger ConvergenceWarning.
    rbf_C = loguniform(0.1, 100)
    rbf_g = loguniform(1e-3, 1)
    lin_C = loguniform(0.01, 100)

    dist_A = [
        {
            "smotenc__k_neighbors": [3, 5, 7],
            "svm__kernel":          ["rbf"],
            "svm__C":               rbf_C,
            "svm__gamma":           rbf_g,
            "svm__class_weight":    [None],
        },
        {
            "smotenc__k_neighbors": [3, 5, 7],
            "svm__kernel":          ["linear"],
            "svm__C":               lin_C,
            "svm__class_weight":    [None],
        },
    ]

    dist_B = [
        {
            "svm__kernel":       ["rbf"],
            "svm__C":            rbf_C,
            "svm__gamma":        rbf_g,
            "svm__class_weight": ["balanced", {0: 1, 1: 2}, {0: 1, 1: 3}],
        },
        {
            "svm__kernel":       ["linear"],
            "svm__C":            lin_C,
            "svm__class_weight": ["balanced", {0: 1, 1: 2}, {0: 1, 1: 3}],
        },
    ]

    dist_C = [
        {
            "smotenc__k_neighbors": [3, 5, 7],
            "svm__kernel":          ["rbf"],
            "svm__C":               rbf_C,
            "svm__gamma":           rbf_g,
            "svm__class_weight":    ["balanced", {0: 1, 1: 2}, {0: 1, 1: 3}],
        },
        {
            "smotenc__k_neighbors": [3, 5, 7],
            "svm__kernel":          ["linear"],
            "svm__C":               lin_C,
            "svm__class_weight":    ["balanced", {0: 1, 1: 2}, {0: 1, 1: 3}],
        },
    ]

    return dist_A, dist_B, dist_C


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------

def _build_pipeline_with_smotenc(
    smotenc_cat_indices: list[int],
    random_state: int = 42,
) -> Pipeline:
    """Preprocessor -> SMOTENC -> CalibratedClassifierCV(SVC)  (imblearn Pipeline)."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numerical=True)),
            (
                "smotenc",
                SMOTENC(
                    categorical_features=smotenc_cat_indices,
                    random_state=random_state,
                ),
            ),
            # CalibratedClassifierCV(ensemble=False) wraps SVC to expose
            # predict_proba via Platt scaling.  SVC(probability=True) is
            # deprecated in sklearn 1.9+ — use this pattern instead.
            # max_iter=10_000 + cache_size=512 MB: larger kernel cache reduces
            # the number of SMO re-computations, so fewer iterations are needed.
            # tol=1e-3: relaxed to aid early stopping on E-cores.
            (
                "svm",
                CalibratedClassifierCV(
                    SVC(random_state=random_state, max_iter=10_000, tol=1e-3, cache_size=512),
                    ensemble=False,
                ),
            ),
        ]
    )


def _build_pipeline_no_smote(random_state: int = 42) -> Pipeline:
    """Preprocessor -> CalibratedClassifierCV(SVC)  (class_weight handles imbalance; no oversampler)."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numerical=True)),
            # CalibratedClassifierCV(ensemble=False) wraps SVC to expose
            # predict_proba via Platt scaling.  SVC(probability=True) is
            # deprecated in sklearn 1.9+ — use this pattern instead.
            # max_iter=10_000 + cache_size=512 MB: larger kernel cache reduces
            # the number of SMO re-computations, so fewer iterations are needed.
            # tol=1e-3: relaxed to aid early stopping on E-cores.
            (
                "svm",
                CalibratedClassifierCV(
                    SVC(random_state=random_state, max_iter=10_000, tol=1e-3, cache_size=512),
                    ensemble=False,
                ),
            ),
        ]
    )


# ---------------------------------------------------------------------------
# OOF threshold scanner
# ---------------------------------------------------------------------------

def find_optimal_threshold_oof(
    raw_pipeline,
    X_train: pd.DataFrame,
    y_train,
    thresholds: np.ndarray | None = None,
    metric: str = "f1",
    cv: int = 5,
    random_state: int = 42,
) -> float:
    """
    Find the optimal decision threshold via Out-Of-Fold (OOF) probabilities.

    For each stratified fold, a *fresh clone* of raw_pipeline is calibrated on
    the in-fold data and produces probabilities for the held-out fold.  The
    pooled OOF probabilities are scanned across `thresholds` to find the cutoff
    that maximises `metric`.  This threshold is then *frozen* and applied to
    the untouched test set — no test-set information leaks into the decision.

    Parameters
    ----------
    raw_pipeline : Unfitted best estimator from RandomizedSearchCV.best_estimator_.
    X_train      : Full training feature DataFrame.
    y_train      : Full training labels (array-like).
    thresholds   : Candidate cutoffs (default: np.arange(0.20, 0.61, 0.02)).
    metric       : Optimisation target — 'f1' (default) or 'recall'.
    cv           : Number of stratified CV folds.
    random_state : RNG seed for fold splitting.

    Returns
    -------
    float : Frozen optimal threshold to apply to the test set.
    """
    if thresholds is None:
        thresholds = np.arange(0.20, 0.61, 0.02)

    y_arr = np.asarray(y_train)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    splits = list(skf.split(X_train, y_arr))

    print(
        f"\n[OOF Threshold Scanner] {cv}-fold OOF | "
        f"{len(thresholds)} thresholds | optimising '{metric}' ..."
        f"  (parallel n_jobs={_N_JOBS}, backend=loky)"
    )

    # ── Parallel OOF fold fitting (loky, spawn-safe) ─────────────────────────
    # Each fold clones and fits independently — no shared mutable state.
    def _fit_fold(fold_idx: int, tr_idx, val_idx):
        X_fold_tr  = X_train.iloc[tr_idx]
        y_fold_tr  = y_arr[tr_idx]
        X_fold_val = X_train.iloc[val_idx]

        fold_pipe = clone(raw_pipeline)
        fold_pipe.fit(X_fold_tr, y_fold_tr)
        # The pipeline's final step is CalibratedClassifierCV(SVC(), ensemble=False)
        # which exposes predict_proba directly — no extra wrapping needed.

        proba_val = fold_pipe.predict_proba(X_fold_val)[:, 1]
        print(f"  Fold {fold_idx + 1}/{cv} completed.")
        return val_idx, proba_val

    with joblib.parallel_backend("loky", n_jobs=_N_JOBS):
        fold_results = joblib.Parallel(verbose=0)(
            joblib.delayed(_fit_fold)(fold_idx, tr_idx, val_idx)
            for fold_idx, (tr_idx, val_idx) in enumerate(splits)
        )

    oof_proba = np.zeros(len(y_arr), dtype=np.float64)
    for val_idx, proba_val in fold_results:
        oof_proba[val_idx] = proba_val

    best_thresh, best_score = 0.5, -1.0
    for thr in thresholds:
        y_pred = (oof_proba >= thr).astype(int)
        if metric == "f1":
            score = f1_score(y_arr, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_arr, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported metric '{metric}'. Use 'f1' or 'recall'.")

        if score > best_score:
            best_score  = score
            best_thresh = thr

    print(
        f"[OOF Threshold Scanner] Optimal threshold = {best_thresh:.2f} "
        f"(OOF {metric} = {best_score:.4f})\n"
    )
    return float(best_thresh)


def predict_with_threshold(model, X, threshold: float = 0.5) -> np.ndarray:
    """Return hard-label predictions applying a custom probability threshold."""
    return (model.predict_proba(X)[:, 1] >= threshold).astype(int)


def evaluate_model_with_threshold(model, X_test, y_test, threshold: float = 0.5) -> dict:
    """Evaluate a calibrated model at a custom decision threshold."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    return {
        "Threshold":             threshold,
        "Accuracy":              accuracy_score(y_test, y_pred),
        "Precision":             precision_score(y_test, y_pred, zero_division=0),
        "Recall":                recall_score(y_test, y_pred, zero_division=0),
        "F1":                    f1_score(y_test, y_pred, zero_division=0),
        "AUC":                   roc_auc_score(y_test, y_proba),
        "PR_AUC":                average_precision_score(y_test, y_proba),
        "Confusion Matrix":      confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Inspecting search results
# ---------------------------------------------------------------------------

def get_grid_search_results(search_obj) -> pd.DataFrame:
    """Return RandomizedSearchCV cv_results_ as a tidy, sorted DataFrame."""
    results = pd.DataFrame(search_obj.cv_results_)
    param_cols = [c for c in results.columns if c.startswith("param_")]
    keep_cols  = param_cols + ["mean_test_score", "std_test_score", "rank_test_score"]
    tidy = results[keep_cols].sort_values("rank_test_score").reset_index(drop=True)
    tidy.columns = [
        c.replace("param_svm__", "").replace("param_smotenc__", "smotenc__")
        for c in tidy.columns
    ]
    return tidy


# ---------------------------------------------------------------------------
# K-fold cross-validation of the final model
# ---------------------------------------------------------------------------

def cross_validate_svm(model, X, y, cv: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Run stratified k-fold CV on the tuned SVM pipeline and return metrics."""
    stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy":          "accuracy",
        "precision":         "precision",
        "recall":            "recall",
        "f1":                "f1",
        "roc_auc":           "roc_auc",
        "average_precision": "average_precision",
    }
    # n_jobs=_N_JOBS: pin to P-cores + E-cores, spare LP-E-cores for OS work.
    cv_results = cross_validate(
        model, X, y,
        cv=stratified_cv,
        scoring=scoring,
        n_jobs=_N_JOBS,
        return_train_score=False,
    )
    rows = []
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        rows.append(
            {
                "Metric": metric.replace("_", " ").title(),
                **{f"Fold {i + 1}": s for i, s in enumerate(scores)},
                "Mean": scores.mean(),
                "Std":  scores.std(),
            }
        )
    return pd.DataFrame(rows).set_index("Metric")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_svm(
    X_train,
    y_train,
    scoring: str = "average_precision",
    cv: int = 5,
    n_iter: int = 60,
    random_state: int = 42,
    output_path: str | Path | None = None,
    verbose: int = 2,
    n_jobs: int = _N_JOBS,  # default: 12 workers for Core Ultra 5 125H
    threshold_metric: str = "f1",
    oof_cv: int = 5,
) -> tuple[CalibratedClassifierCV, dict]:
    """
    Tune SVM hyperparameters across three imbalance strategies, calibrate the
    winner, and find an optimal decision threshold via OOF scanning.

    Strategies compared
    -------------------
    A (SMOTENC only)        — oversampling, no class penalty
    B (class_weight only)   — no oversampling, asymmetric cost
    C (SMOTENC + weight)    — both combined

    Parameters
    ----------
    X_train          : Training features (raw DataFrame, pre-transform).
    y_train          : Training labels.
    scoring          : Search metric (default: 'average_precision' = PR-AUC).
    cv               : CV folds for hyperparameter search.
    n_iter           : RandomizedSearchCV iterations (default: 60).
    random_state     : RNG seed.
    output_path      : .pkl save path for the calibrated model.
    verbose          : Verbosity for RandomizedSearchCV.
    n_jobs           : Parallel workers.
    threshold_metric : OOF optimisation target ('f1' | 'recall').
    oof_cv           : CV folds for OOF threshold scan.

    Returns
    -------
    (calibrated_model, result_dict)
        result_dict keys:
          'search_smotenc'    – RandomizedSearchCV result for Strategy A+C
          'search_no_smote'   – RandomizedSearchCV result for Strategy B
          'best_strategy'     – winning strategy label string
          'best_search'       – winning RandomizedSearchCV object
          'optimal_threshold' – frozen threshold (float) from OOF scan
    """
    if output_path is None:
        output_path = project_root / "saved_models" / "svm_model.pkl"

    stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    dist_A, dist_B, dist_C = _make_svm_param_distributions()

    # ── Determine transformed-space categorical indices for SMOTENC ──────────
    # After the ColumnTransformer, OHE expands CATEGORICAL_FEATURES into a
    # contiguous block of binary columns placed *before* the scaled numerical
    # columns.  SMOTENC must know which of those transformed columns are still
    # categorical (i.e., must not be fractionally interpolated).
    _probe_pp = build_preprocessor(scale_numerical=True)
    _probe_pp.fit(X_train, y_train)
    _n_total_transformed = _probe_pp.transform(X_train.iloc[:1]).shape[1]
    _n_num               = len(NUMERICAL_FEATURES)
    _n_ohe_cols          = _n_total_transformed - _n_num
    smotenc_cat_idx      = list(range(_n_ohe_cols))   # first N cols = OHE block

    print("\n" + "=" * 70)
    print(" SVM Hyperparameter Search  (RandomizedSearchCV)")
    print(f" scoring={scoring!r}  n_iter={n_iter}  cv={cv}")
    print(f" n_jobs={n_jobs}  (CPU: Intel Core Ultra 5 125H — {os.cpu_count()} logical threads)")
    print(f" SMOTENC categorical block: {_n_ohe_cols} columns (indices 0..{_n_ohe_cols-1})")
    print("=" * 70)

    # ── Strategy A + C: pipelines that include SMOTENC ───────────────────────
    print("\n[train_svm] Searching Strategy A (SMOTENC only) + C (SMOTENC + weight)...")
    pipeline_smotenc = _build_pipeline_with_smotenc(smotenc_cat_idx, random_state)
    search_smotenc = RandomizedSearchCV(
        estimator=pipeline_smotenc,
        param_distributions=dist_A + dist_C,
        n_iter=n_iter,
        scoring=scoring,
        cv=stratified_cv,
        verbose=verbose,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search_smotenc.fit(X_train, y_train)
    print(f"[train_svm] Strategy A+C | best {scoring}: {search_smotenc.best_score_:.4f}")
    print(f"[train_svm] Strategy A+C | best params:    {search_smotenc.best_params_}")

    # ── Strategy B: class_weight only ────────────────────────────────────────
    print("\n[train_svm] Searching Strategy B (class_weight only, no SMOTENC)...")
    pipeline_no_smote = _build_pipeline_no_smote(random_state)
    search_no_smote = RandomizedSearchCV(
        estimator=pipeline_no_smote,
        param_distributions=dist_B,
        n_iter=n_iter,
        scoring=scoring,
        cv=stratified_cv,
        verbose=verbose,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )
    search_no_smote.fit(X_train, y_train)
    print(f"[train_svm] Strategy B    | best {scoring}: {search_no_smote.best_score_:.4f}")
    print(f"[train_svm] Strategy B    | best params:    {search_no_smote.best_params_}")

    # ── Pick the overall winner ───────────────────────────────────────────────
    if search_smotenc.best_score_ >= search_no_smote.best_score_:
        best_search   = search_smotenc
        best_strategy = "A+C (SMOTENC)"
    else:
        best_search   = search_no_smote
        best_strategy = "B (class_weight only)"

    print(f"\n[train_svm] Winning strategy : {best_strategy}")
    print(f"[train_svm] Best {scoring}  : {best_search.best_score_:.4f}")
    print(f"[train_svm] Best params      : {best_search.best_params_}")

    raw_best_pipeline = best_search.best_estimator_

    # ── Probability calibration (prefit on full training set) ─────────────────
    calibrated_model = CalibratedClassifierCV(
        estimator=raw_best_pipeline,
        ensemble=False,
        cv=5,
    )
    calibrated_model.fit(X_train, y_train)

    # ── OOF threshold scan (uses CV on training data only — test untouched) ───
    optimal_threshold = find_optimal_threshold_oof(
        raw_pipeline=raw_best_pipeline,
        X_train=X_train,
        y_train=np.asarray(y_train),
        thresholds=np.arange(0.20, 0.61, 0.02),
        metric=threshold_metric,
        cv=oof_cv,
        random_state=random_state,
    )

    if output_path:
        save_model(calibrated_model, output_path)

    return calibrated_model, {
        "search_smotenc":    search_smotenc,
        "search_no_smote":   search_no_smote,
        "best_strategy":     best_strategy,
        "best_search":       best_search,
        "optimal_threshold": optimal_threshold,
    }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── 1. Load & split data ─────────────────────────────────────────────────
    data_path = str(project_root / "data" / "raw" / "online_shoppers_intention.csv")
    df = preprocess_data(filepath=data_path)
    X_train, X_test, y_train, y_test = split_dataset(df)

    # ── 2. Train & save ──────────────────────────────────────────────────────
    save_path = project_root / "saved_models" / "svm_model.pkl"
    model, result = train_svm(
        X_train,
        y_train,
        output_path=save_path,
    )

    optimal_threshold = result["optimal_threshold"]
    best_strategy     = result["best_strategy"]

    print(f"\n[main] Winning imbalance strategy : {best_strategy}")
    print(f"[main] Frozen decision threshold   : {optimal_threshold:.2f}")

    # ── 3. Evaluate at optimal threshold on untouched test set ───────────────
    metrics = evaluate_model_with_threshold(model, X_test, y_test, threshold=optimal_threshold)
    print_metrics("SVM Model", metrics)

    # ── 4. Persist metrics ───────────────────────────────────────────────────
    metrics_output_path = project_root / "report_assets" / "metrics.json"
    save_metrics("SVM Model", "svm", metrics, metrics_output_path)

    # ── 5. SHAP Interpretability ─────────────────────────────────────────────
    print("\n[SHAP] Generating SVM SHAP explanation plots...")
    try:
        plot_dir = str(project_root / "report_assets" / "plots")
        generate_shap_explanation(
            model=model,
            X_test=X_test,
            save_dir=plot_dir,
            prefix="svm_",
            show=False,
        )
        print("[SHAP] Plots saved successfully.")
    except Exception as exc:
        print(f"[SHAP] Skipped: {exc}")

    # ── 6. Wall-clock duration ───────────────────────────────────────────────
    _total_seconds = time.perf_counter() - _SCRIPT_START
    print(
        f"\n[main] Total run duration: {_total_seconds:.1f} s "
        f"({_total_seconds / 60:.2f} min)"
    )

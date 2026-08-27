"""
svm_model.py
------------
Support Vector Machine (SVM) pipeline for the Online Shoppers Purchasing
Intention classification task.

Key Components:
- Cleaner:      TrainFittedDataCleaner (imputation, duration fix, rare-cat grouping)
               fitted on training data only — no pre-split leakage.
- Preprocessor: StandardScaler (numerical) + OneHotEncoder (categorical).
- Imbalance Handling: Compares SMOTENC, class_weight, and combined strategies.
- Hyperparameter Tuning: RandomizedSearchCV optimizing PR-AUC (average_precision).
- Calibration: Post-hoc Platt scaling via CalibratedClassifierCV.
- Threshold Tuning: Leak-free OOF threshold optimization using harmonic
  Precision-Recall gap minimization within the 99% max-F1 band.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_SCRIPT_START = time.perf_counter()
_N_JOBS: int = min(12, os.cpu_count() or 12)

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend — safe for CLI + Streamlit
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform
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
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.svm import SVC

from src.data_preprocessing import (
    NUMERICAL_FEATURES,
    build_preprocessor,
    preprocess_data,
    remove_outliers_iqr_train,
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
    # Moderately widened parameter space — slightly wider than the original
    # bounds to explore more configurations, without spreading 120 iterations
    # too thin across the space.
    rbf_C = loguniform(0.05, 500)
    rbf_g = loguniform(5e-4, 1)
    lin_C = loguniform(0.01, 5)
    dist_A = [
        {
            "smotenc__k_neighbors": [3, 5, 7],
            "svm__kernel": ["rbf"],
            "svm__C": rbf_C,
            "svm__gamma": rbf_g,
            "svm__class_weight": [None],
        },
        {
            "smotenc__k_neighbors": [3, 5, 7],
            "svm__kernel": ["linear"],
            "svm__C": lin_C,
            "svm__class_weight": [None],
        },
    ]

    dist_B = [
        {
            "svm__kernel": ["rbf"],
            "svm__C": rbf_C,
            "svm__gamma": rbf_g,
            "svm__class_weight": ["balanced", {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}],
        },
        {
            "svm__kernel": ["linear"],
            "svm__C": lin_C,
            "svm__class_weight": ["balanced", {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}],
        },
    ]

    dist_C = [
        {
            "smotenc__k_neighbors": [3, 5, 7],
            "svm__kernel": ["rbf"],
            "svm__C": rbf_C,
            "svm__gamma": rbf_g,
            "svm__class_weight": ["balanced", {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}],
        },
        {
            "smotenc__k_neighbors": [3, 5, 7],
            "svm__kernel": ["linear"],
            "svm__C": lin_C,
            "svm__class_weight": ["balanced", {0: 1, 1: 2}, {0: 1, 1: 3}, {0: 1, 1: 4}],
        },
    ]

    return dist_A, dist_B, dist_C


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def _build_pipeline_with_smotenc(
    smotenc_cat_indices: list[int],
    random_state: int = 42,
    outlier_method: str = "iqr",
) -> Pipeline:
    """Build TrainFittedDataCleaner -> (OutlierFilter) -> Preprocessor -> SMOTENC -> SVC pipeline."""
    steps = [
        ("preprocessor", build_preprocessor(scale_numerical=True)),
        (
            "smotenc",
            SMOTENC(
                categorical_features=smotenc_cat_indices,
                random_state=random_state,
            ),
        ),
        (
            "svm",
            SVC(
                random_state=random_state,
                max_iter=100_000,
                tol=1e-3,
                cache_size=1024,
            ),
        ),
    ]

    pipeline_smotenc = Pipeline(steps=steps)
    return pipeline_smotenc


def _build_pipeline_no_smote(
    random_state: int = 42,
    outlier_method: str = "iqr",
) -> Pipeline:
    """Build TrainFittedDataCleaner -> (OutlierFilter) -> Preprocessor -> SVC pipeline (class_weight only)."""
    steps = [
        ("preprocessor", build_preprocessor(scale_numerical=True)),
        (
            "svm",
            SVC(
                random_state=random_state,
                max_iter=100_000,
                tol=1e-3,
                cache_size=1024,
            ),
        ),
    ]

    pipeline_no_smote = Pipeline(steps=steps)
    return pipeline_no_smote


# ---------------------------------------------------------------------------
# OOF threshold scanner
# ---------------------------------------------------------------------------


def find_optimal_threshold_oof(
    raw_pipeline,
    X_train: pd.DataFrame,
    y_train,
    thresholds: np.ndarray | None = None,
    cv: int = 5,
    random_state: int = 42,
) -> tuple[float, pd.DataFrame]:
    """
    Find the optimal decision threshold via Out-Of-Fold (OOF) probabilities
    using balanced selection — 99% F1-band tie-breaking.

    For each stratified fold a *fresh clone* of ``raw_pipeline`` is fitted on
    the in-fold data and produces probabilities for the held-out fold.  The
    pooled OOF probabilities are scanned across ``thresholds``.  **No test-set
    data is used anywhere in this function.**

    Balanced selection algorithm
    ---------------------------
    1. Compute Precision / Recall / F1 for every candidate threshold.
    2. Identify the maximum OOF F1.
    3. Keep all thresholds where  F1 >= 0.99 × max_F1  ("99 % band").
    4. Among those, choose the threshold with:
       a) highest Precision,
       b) highest Recall       (tie-breaker),
       c) highest F1           (tie-breaker),
       d) closest to 0.50      (final tie-breaker).

    Parameters
    ----------
    raw_pipeline : Unfitted / clonable estimator (e.g. CalibratedClassifierCV
                   wrapping the best pipeline from RandomizedSearchCV).
    X_train      : Full training feature DataFrame.
    y_train      : Full training labels (array-like).
    thresholds   : Candidate cutoffs (default: np.arange(0.10, 0.81, 0.01)).
    cv           : Number of stratified CV folds.
    random_state : RNG seed for fold splitting.

    Returns
    -------
    (selected_threshold, threshold_results_df)
        selected_threshold : float — frozen threshold for test-set evaluation.
        threshold_results_df : pd.DataFrame with columns
            [Threshold, Precision, Recall, F1].
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.81, 0.01)

    if len(thresholds) == 0:
        raise ValueError("Threshold array is empty — cannot scan.")

    y_arr = np.asarray(y_train)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    print(
        f"\n[OOF Threshold Scanner] {cv}-fold OOF | "
        f"{len(thresholds)} thresholds | balanced selection ..."
        f"  (parallel n_jobs={_N_JOBS})"
    )

    # ── Out-of-fold probability estimation via cross_val_predict ─────────────
    oof_proba = cross_val_predict(
        raw_pipeline,
        X_train,
        y_arr,
        cv=skf,
        method="predict_proba",
        n_jobs=_N_JOBS,
    )[:, 1]

    # ── Compute P / R / F1 for every threshold ───────────────────────────────
    rows = []
    for thr in thresholds:
        y_pred = (oof_proba >= thr).astype(int)
        rows.append(
            {
                "Threshold": round(float(thr), 4),
                "Precision": precision_score(y_arr, y_pred, zero_division=0),
                "Recall": recall_score(y_arr, y_pred, zero_division=0),
                "F1": f1_score(y_arr, y_pred, zero_division=0),
            }
        )

    df_thr = pd.DataFrame(rows)

    if df_thr.empty or df_thr["F1"].max() == 0:
        print("[WARNING] No valid threshold found — falling back to 0.50.")
        return 0.50, df_thr

    # ── Balanced selection within 99% F1 band ────────────────────────────────
    max_f1 = df_thr["F1"].max()
    band = df_thr[df_thr["F1"] >= 0.99 * max_f1].copy()

    # Minimum precision/recall constraints
    min_precision = 0.60
    min_recall = 0.60

    constrained = band[
        (band["Precision"] >= min_precision) & (band["Recall"] >= min_recall)
    ].copy()

    if constrained.empty:
        print(
            "[WARNING] No threshold satisfies minimum precision/recall constraints. "
            "Using full 99% F1 band."
        )
        constrained = band.copy()

    # Harmonic balance: minimize precision-recall gap, then maximize F1/Recall
    constrained["_pr_gap"] = (constrained["Precision"] - constrained["Recall"]).abs()

    constrained["_dist_05"] = (constrained["Threshold"] - 0.50).abs()

    constrained = constrained.sort_values(
        by=["_pr_gap", "F1", "Recall", "_dist_05"],
        ascending=[True, False, False, True],
    )

    selected = constrained.iloc[0]
    selected_threshold = float(selected["Threshold"])

    print(
        f"[OOF Threshold Scanner] max OOF F1 = {max_f1:.4f} "
        f"| 99% band: {len(band)} thresholds"
    )
    print(
        f"[OOF Threshold Scanner] Selected threshold = {selected_threshold:.2f}  "
        f"(P={selected['Precision']:.4f}  R={selected['Recall']:.4f}  "
        f"F1={selected['F1']:.4f})"
    )

    return selected_threshold, df_thr


# ---------------------------------------------------------------------------
# Threshold analysis plots
# ---------------------------------------------------------------------------


def _plot_threshold_metrics(
    df_thr: pd.DataFrame,
    selected_threshold: float,
    save_path: str | Path,
) -> None:
    """Plot Precision, Recall, F1 vs Threshold and mark the selected point."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_thr["Threshold"], df_thr["Precision"], label="Precision", linewidth=1.5)
    ax.plot(df_thr["Threshold"], df_thr["Recall"], label="Recall", linewidth=1.5)
    ax.plot(df_thr["Threshold"], df_thr["F1"], label="F1", linewidth=2.0)

    # Mark the selected threshold
    sel_row = df_thr.loc[(df_thr["Threshold"] - selected_threshold).abs().idxmin()]
    ax.axvline(
        selected_threshold,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Selected = {selected_threshold:.2f}",
    )
    ax.scatter([selected_threshold], [sel_row["F1"]], color="red", s=80, zorder=5)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("SVM — Threshold Metrics (OOF, training data only)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"[Threshold Plot] Saved: {save_path}")


def _plot_precision_recall_threshold(
    df_thr: pd.DataFrame,
    selected_threshold: float,
    save_path: str | Path,
) -> None:
    """Plot Precision vs Recall curve and mark the selected threshold."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df_thr["Recall"], df_thr["Precision"], linewidth=1.5, color="steelblue")

    # Mark the selected threshold
    sel_row = df_thr.loc[(df_thr["Threshold"] - selected_threshold).abs().idxmin()]
    ax.scatter(
        [sel_row["Recall"]],
        [sel_row["Precision"]],
        color="red",
        s=100,
        zorder=5,
        label=f"Threshold = {selected_threshold:.2f}",
    )
    ax.annotate(
        f"  t={selected_threshold:.2f}",
        xy=(sel_row["Recall"], sel_row["Precision"]),
        fontsize=9,
        color="red",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("SVM — Precision vs Recall (OOF, training data only)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
    print(f"[PR Plot] Saved: {save_path}")


def predict_with_threshold(model, X, threshold: float = 0.5) -> np.ndarray:
    """Return hard-label predictions applying a custom probability threshold."""
    return (model.predict_proba(X)[:, 1] >= threshold).astype(int)


def evaluate_model_with_threshold(
    model, X_test, y_test, threshold: float = 0.5
) -> dict:
    """Evaluate a calibrated model at a custom decision threshold."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_test, y_proba),
        "PR_AUC": average_precision_score(y_test, y_proba),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Inspecting search results
# ---------------------------------------------------------------------------


def get_grid_search_results(search_obj) -> pd.DataFrame:
    """Return RandomizedSearchCV cv_results_ as a tidy, sorted DataFrame."""
    results = pd.DataFrame(search_obj.cv_results_)
    param_cols = [c for c in results.columns if c.startswith("param_")]
    keep_cols = param_cols + ["mean_test_score", "std_test_score", "rank_test_score"]
    tidy = results[keep_cols].sort_values("rank_test_score").reset_index(drop=True)
    tidy.columns = [
        c.replace("param_svm__", "").replace("param_smotenc__", "smotenc__")
        for c in tidy.columns
    ]
    return tidy


# ---------------------------------------------------------------------------
# K-fold cross-validation of the final model
# ---------------------------------------------------------------------------


def cross_validate_svm(
    model, X, y, cv: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Run stratified k-fold CV on the tuned SVM pipeline and return metrics."""
    stratified_cv = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=random_state
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
    }
    # n_jobs=_N_JOBS: pin to P-cores + E-cores, spare LP-E-cores for OS work.
    cv_results = cross_validate(
        model,
        X,
        y,
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
                "Std": scores.std(),
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
    n_jobs: int = _N_JOBS,
    oof_cv: int = 5,
    outlier_method: str = "iqr",
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
    oof_cv           : CV folds for OOF threshold scan.
    outlier_method   : 'iqr' | 'zscore' | 'none'  (passed to TrainingOutlierFilter).

    Returns
    -------
    (calibrated_model, result_dict)
        result_dict keys:
          'search_smotenc'    – RandomizedSearchCV result for Strategy A+C
          'search_no_smote'   – RandomizedSearchCV result for Strategy B
          'best_strategy'     – winning strategy label string
          'best_search'       – winning RandomizedSearchCV object
          'optimal_threshold' – frozen threshold (float) from OOF scan
          'threshold_results' – pd.DataFrame of per-threshold P/R/F1
    """
    if output_path is None:
        output_path = project_root / "saved_models" / "svm_model.pkl"

    stratified_cv = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=random_state
    )
    dist_A, dist_B, dist_C = _make_svm_param_distributions()

    # ── Determine transformed-space categorical indices for SMOTENC ──────────
    # After TrainFittedDataCleaner + ColumnTransformer, OHE expands
    # CATEGORICAL_FEATURES into a contiguous block of binary columns placed
    # *before* the scaled numerical columns.  SMOTENC must know which of those
    # transformed columns are still categorical (must not be fractionally
    # interpolated).
    # We fit a standalone cleaner + preprocessor probe on training data to
    # count the OHE output columns correctly.
    X_train_clean, y_train_clean = remove_outliers_iqr_train(X_train, y_train)
    _probe_pp = build_preprocessor(scale_numerical=True)
    _probe_pp.fit(X_train_clean)
    _n_total_transformed = _probe_pp.transform(X_train_clean.iloc[:1]).shape[1]
    _n_num = len(NUMERICAL_FEATURES)
    _n_ohe_cols = _n_total_transformed - _n_num
    smotenc_cat_idx = list(range(_n_ohe_cols))  # first N cols = OHE block

    print("\n" + "=" * 70)
    print(" SVM Hyperparameter Search  (RandomizedSearchCV)")
    print(f" scoring={scoring!r}  n_iter={n_iter}  cv={cv}")
    print(f" n_jobs={n_jobs}{os.cpu_count()} logical threads)")
    print(f" outlier_method={outlier_method!r}")
    print(
        f" SMOTENC categorical block: {_n_ohe_cols} columns (indices 0..{_n_ohe_cols-1})"
    )
    print("=" * 70)

    # ── Strategy A + C: pipelines that include SMOTENC ───────────────────────
    print("\n[train_svm] Searching Strategy A (SMOTENC only) + C (SMOTENC + weight)...")
    pipeline_smotenc = _build_pipeline_with_smotenc(
        smotenc_cat_idx, random_state, outlier_method=outlier_method
    )
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
    search_smotenc.fit(X_train_clean, y_train_clean)
    print(
        f"[train_svm] Strategy A+C | best {scoring}: {search_smotenc.best_score_:.4f}"
    )
    print(f"[train_svm] Strategy A+C | best params:    {search_smotenc.best_params_}")

    top_smotenc = get_grid_search_results(search_smotenc)
    print("\n[train_svm] Strategy A+C — top 15 trials by mean_test_score:")
    print(top_smotenc.head(15).to_string(index=False))

    # ── Strategy B: class_weight only ────────────────────────────────────────
    print("\n[train_svm] Searching Strategy B (class_weight only, no SMOTENC)...")
    pipeline_no_smote = _build_pipeline_no_smote(
        random_state, outlier_method=outlier_method
    )
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
    search_no_smote.fit(X_train_clean, y_train_clean)
    print(
        f"[train_svm] Strategy B    | best {scoring}: {search_no_smote.best_score_:.4f}"
    )
    print(f"[train_svm] Strategy B    | best params:    {search_no_smote.best_params_}")

    top_no_smote = get_grid_search_results(search_no_smote)
    print("\n[train_svm] Strategy B — top 15 trials by mean_test_score:")
    print(top_no_smote.head(15).to_string(index=False))

    # ── Pick the overall winner ───────────────────────────────────────────────
    if search_smotenc.best_score_ >= search_no_smote.best_score_:
        best_search = search_smotenc

        # Distinguish Strategy A from Strategy C
        best_class_weight = search_smotenc.best_params_.get("svm__class_weight")

        if best_class_weight is None:
            best_strategy = "A (SMOTENC only)"
        else:
            best_strategy = "C (SMOTENC + class_weight)"

    else:
        best_search = search_no_smote
        best_strategy = "B (class_weight only)"

    print(f"\n[train_svm] Winning strategy : {best_strategy}")
    print(f"[train_svm] Best {scoring}  : {best_search.best_score_:.4f}")
    print(f"[train_svm] Best params      : {best_search.best_params_}")

    raw_best_pipeline = best_search.best_estimator_

    # ── Probability Calibration Layer ────────────────────────────────────────
    calibrated_model = CalibratedClassifierCV(
        estimator=raw_best_pipeline,
        ensemble=False,
        cv=5,
    )
    calibrated_model.fit(X_train_clean, y_train_clean)

    # ── Convergence guard ────────────────────────────────────────────────────
    try:
        inner_svc = calibrated_model.calibrated_classifiers_[0].estimator.named_steps[
            "svm"
        ]
        if hasattr(inner_svc, "n_iter_"):
            n_iter_arr = np.asarray(inner_svc.n_iter_)
            if np.any(n_iter_arr >= 100_000):
                print(
                    f"[WARNING] SVC hit max_iter=100,000 (n_iter_={n_iter_arr.tolist()}). "
                    f"Consider raising max_iter or relaxing C/tol."
                )
            else:
                print(
                    f"[train_svm] SVC converged within max_iter (n_iter_={n_iter_arr.tolist()})."
                )
    except (AttributeError, IndexError, KeyError):
        pass

    # ── OOF threshold scan ────────────────────────────────────────────────────
    # Pass calibrated_model (not raw_best_pipeline) so the probability
    # architecture used to select the threshold is identical to the one used
    # for test-set prediction — both go through the same single calibration.
    optimal_threshold, threshold_results_df = find_optimal_threshold_oof(
        raw_pipeline=calibrated_model,
        X_train=X_train_clean,
        y_train=np.asarray(y_train_clean),
        thresholds=np.arange(0.10, 0.81, 0.01),
        cv=oof_cv,
        random_state=random_state,
    )

    # ── Persist threshold analysis artefacts ──────────────────────────────────
    thr_dir = project_root / "report_assets" / "threshold_analysis"
    plot_dir = project_root / "report_assets" / "plots"
    thr_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # CSV — full threshold table
    csv_path = thr_dir / "svm_threshold_results.csv"
    threshold_results_df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"[train_svm] Threshold table saved: {csv_path}")

    # JSON — summary of the selected threshold
    summary = {
        "selected_threshold": round(optimal_threshold, 4),
        "max_oof_f1": round(float(threshold_results_df["F1"].max()), 4),
        "band_99pct_count": int(
            (
                threshold_results_df["F1"] >= 0.99 * threshold_results_df["F1"].max()
            ).sum()
        ),
        "selected_precision": round(
            float(
                threshold_results_df.loc[
                    (threshold_results_df["Threshold"] - optimal_threshold)
                    .abs()
                    .idxmin(),
                    "Precision",
                ]
            ),
            4,
        ),
        "selected_recall": round(
            float(
                threshold_results_df.loc[
                    (threshold_results_df["Threshold"] - optimal_threshold)
                    .abs()
                    .idxmin(),
                    "Recall",
                ]
            ),
            4,
        ),
        "selected_f1": round(
            float(
                threshold_results_df.loc[
                    (threshold_results_df["Threshold"] - optimal_threshold)
                    .abs()
                    .idxmin(),
                    "F1",
                ]
            ),
            4,
        ),
    }
    json_path = thr_dir / "svm_threshold_summary.json"
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[train_svm] Threshold summary saved: {json_path}")

    # Plots
    _plot_threshold_metrics(
        threshold_results_df,
        optimal_threshold,
        save_path=plot_dir / "svm_threshold_metrics.png",
    )
    _plot_precision_recall_threshold(
        threshold_results_df,
        optimal_threshold,
        save_path=plot_dir / "svm_precision_recall_threshold.png",
    )

    calibrated_model.optimal_threshold_ = float(optimal_threshold)

    if output_path:
        save_model(calibrated_model, output_path)

    return calibrated_model, {
        "search_smotenc": search_smotenc,
        "search_no_smote": search_no_smote,
        "best_strategy": best_strategy,
        "best_search": best_search,
        "optimal_threshold": optimal_threshold,
        "threshold_results": threshold_results_df,
    }


if __name__ == "__main__":
    # ── 1. Load & split data ─────────────────────────────────────────────────
    # preprocess_data() only applies deterministic pre-split cleaning (no
    # outlier removal). Outlier removal is handled inside the training pipeline
    # by TrainingOutlierFilter, which is fitted on training folds only.
    data_path = str(project_root / "data" / "raw" / "online_shoppers_intention.csv")
    df = preprocess_data(filepath=data_path, outlier_method="none")
    X_train, X_test, y_train, y_test = split_dataset(df)

    # ── 2. Train & save ──────────────────────────────────────────────────────
    save_path = project_root / "saved_models" / "svm_model.pkl"
    model, result = train_svm(
        X_train,
        y_train,
        output_path=save_path,
        outlier_method="iqr",  # TrainingOutlierFilter fitted on training folds only
    )

    optimal_threshold = result["optimal_threshold"]
    best_strategy = result["best_strategy"]

    print(f"\n[main] Winning imbalance strategy : {best_strategy}")
    print(f"[main] Frozen decision threshold   : {optimal_threshold:.2f}")

    # ── 3. Evaluate at optimal threshold on untouched test set ───────────────
    metrics = evaluate_model_with_threshold(
        model, X_test, y_test, threshold=optimal_threshold
    )
    print_metrics("Svm Model", metrics)

    # ── 4. Persist metrics ───────────────────────────────────────────────────
    metrics_output_path = project_root / "report_assets" / "metrics.json"
    save_metrics("Svm Model", "svm", metrics, metrics_output_path)

    # ── 5. SHAP Interpretability ──────────────────────────────────
    try:
        plot_dir = str(project_root / "report_assets" / "plots")
        generate_shap_explanation(
            model=model,
            X_test=X_test,
            save_dir=plot_dir,
            prefix="svm_",
            show=False,
        )
    except Exception as exc:
        print(f"[SHAP] Skipped: {exc}")

    # ── 6. Wall-clock duration ───────────────────────────────────────────────
    _total_seconds = time.perf_counter() - _SCRIPT_START
    print(
        f"\n[main] Total run duration: {_total_seconds:.1f} s "
        f"({_total_seconds / 60:.2f} min)"
    )

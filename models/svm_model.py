"""
svm_model.py
------------
Support Vector Machine (SVM) pipeline with hyperparameter tuning,
probability calibration, and Out-of-Fold (OOF) decision threshold optimization.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

_SCRIPT_START = time.perf_counter()
_N_JOBS: int = min(12, os.cpu_count() or 12)

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

matplotlib.use("Agg")

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
# Hyperparameter Search Space
# ---------------------------------------------------------------------------


def _make_svm_param_distributions() -> tuple[list, list, list]:
    """Define hyperparameter distributions for Strategy A (SMOTENC), B (class_weight), and C (Combined)."""
    rbf_C = loguniform(0.1, 100)
    rbf_g = loguniform(1e-3, 1)
    lin_C = loguniform(0.01, 0.15)

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
# Pipeline Builders
# ---------------------------------------------------------------------------


def _build_pipeline_with_smotenc(
    smotenc_cat_indices: list[int],
    random_state: int = 42,
) -> Pipeline:
    """Construct imblearn Pipeline with Preprocessor, SMOTENC, and SVC."""
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
    )


def _build_pipeline_no_smotenc(
    random_state: int = 42,
) -> Pipeline:
    """Construct standard Pipeline with Preprocessor and SVC (cost-sensitive weighting)."""
    return Pipeline(
        steps=[
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
    )


# ---------------------------------------------------------------------------
# Out-of-Fold (OOF) Decision Threshold Optimization
# ---------------------------------------------------------------------------


def find_optimal_threshold_oof(
    raw_pipeline,
    X_train: pd.DataFrame,
    y_train,
    thresholds: np.ndarray | None = None,
    cv: int = 5,
    random_state: int = 42,
) -> tuple[float, pd.DataFrame]:
    """Find the optimal decision threshold via OOF cross-validation on training data."""
    if thresholds is None:
        thresholds = np.arange(0.21, 0.71, 0.01)

    if len(thresholds) == 0:
        raise ValueError("Threshold array is empty.")

    y_arr = np.asarray(y_train)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    print(
        f"\n[OOF Threshold Scanner] {cv}-fold OOF | "
        f"{len(thresholds)} thresholds | balanced selection (n_jobs={_N_JOBS})..."
    )

    oof_proba = cross_val_predict(
        raw_pipeline,
        X_train,
        y_arr,
        cv=skf,
        method="predict_proba",
        n_jobs=_N_JOBS,
    )[:, 1]

    rows = []
    for thr in thresholds:
        y_pred = (oof_proba >= thr).astype(int)
        rows.append(
            {
                "Threshold": round(float(thr), 4),
                "Accuracy": accuracy_score(y_arr, y_pred),
                "Precision": precision_score(y_arr, y_pred, zero_division=0),
                "Recall": recall_score(y_arr, y_pred, zero_division=0),
                "F1": f1_score(y_arr, y_pred, zero_division=0),
            }
        )

    df_thr = pd.DataFrame(rows)
    if df_thr.empty or df_thr["F1"].max() == 0:
        print("[WARNING] No valid threshold found — fallback to 0.50.")
        return 0.50, df_thr

    # Select threshold within 99% of max F1 that minimizes Precision-Recall gap
    max_f1 = df_thr["F1"].max()
    band = df_thr[df_thr["F1"] >= 0.99 * max_f1].copy()

    constrained = band[(band["Precision"] >= 0.60) & (band["Recall"] >= 0.60)].copy()
    if constrained.empty:
        constrained = band.copy()

    constrained["_pr_gap"] = (constrained["Precision"] - constrained["Recall"]).abs()
    constrained["_dist_05"] = (constrained["Threshold"] - 0.50).abs()
    constrained = constrained.sort_values(
        by=["_pr_gap", "F1", "Recall", "_dist_05"],
        ascending=[True, False, False, True],
    )

    selected = constrained.iloc[0]
    selected_threshold = float(selected["Threshold"])

    print(
        f"[OOF Threshold Scanner] Max OOF F1: {max_f1:.4f} | "
        f"Selected threshold: {selected_threshold:.2f} "
        f"(P={selected['Precision']:.4f}, R={selected['Recall']:.4f}, F1={selected['F1']:.4f})"
    )

    return selected_threshold, df_thr


# ---------------------------------------------------------------------------
# Evaluation & Visualizations
# ---------------------------------------------------------------------------


def _plot_threshold_metrics(
    df_thr: pd.DataFrame,
    selected_threshold: float,
    save_path: str | Path,
) -> None:
    """Plot Precision, Recall, and F1 across decision thresholds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_thr["Threshold"], df_thr["Accuracy"], label="Accuracy", linewidth=1.5)
    ax.plot(df_thr["Threshold"], df_thr["Precision"], label="Precision", linewidth=1.5)
    ax.plot(df_thr["Threshold"], df_thr["Recall"], label="Recall", linewidth=1.5)
    ax.plot(df_thr["Threshold"], df_thr["F1"], label="F1", linewidth=2.0)

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
    ax.set_title("SVM — Threshold Metrics (OOF)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def _plot_precision_recall_threshold(
    df_thr: pd.DataFrame,
    selected_threshold: float,
    save_path: str | Path,
) -> None:
    """Plot Precision vs Recall curve with the selected threshold marked."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(df_thr["Recall"], df_thr["Precision"], linewidth=1.5, color="steelblue")

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
    ax.set_title("SVM — Precision vs Recall (OOF)")
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


def get_grid_search_results(search_obj) -> pd.DataFrame:
    """Format RandomizedSearchCV cv_results_ into a sorted DataFrame."""
    results = pd.DataFrame(search_obj.cv_results_)
    param_cols = [c for c in results.columns if c.startswith("param_")]
    keep_cols = param_cols + ["mean_test_score", "std_test_score", "rank_test_score"]
    tidy = results[keep_cols].sort_values("rank_test_score").reset_index(drop=True)
    tidy.columns = [
        c.replace("param_svm__", "").replace("param_smotenc__", "smotenc__")
        for c in tidy.columns
    ]
    return tidy


def cross_validate_svm(
    model, X, y, cv: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Run stratified k-fold CV on the tuned SVM pipeline."""
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
# Training Routine
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
    Train and tune SVM across imbalance strategies, calibrate probabilities,
    and find the optimal decision threshold via OOF CV.
    """
    if output_path is None:
        output_path = project_root / "saved_models" / "svm_model.pkl"

    stratified_cv = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=random_state
    )
    dist_A, dist_B, dist_C = _make_svm_param_distributions()

    # Pre-calculate categorical column indices for SMOTENC
    X_train_clean, y_train_clean = remove_outliers_iqr_train(X_train, y_train)
    _probe_pp = build_preprocessor(scale_numerical=True)
    _probe_pp.fit(X_train_clean)
    _n_total_transformed = _probe_pp.transform(X_train_clean.iloc[:1]).shape[1]
    _n_num = len(NUMERICAL_FEATURES)
    _n_ohe_cols = _n_total_transformed - _n_num
    smotenc_cat_idx = list(range(_n_ohe_cols))

    print("\n" + "=" * 70)
    print(" SVM Hyperparameter Search (RandomizedSearchCV)")
    print(f" scoring={scoring!r} | n_iter={n_iter} | cv={cv} | n_jobs={n_jobs}")
    print("=" * 70)

    # 1. Strategy A & C: Pipelines with SMOTENC
    print(
        "\n[train_svm] Searching Strategy A (SMOTENC) & C (SMOTENC + class_weight)..."
    )
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
    search_smotenc.fit(X_train_clean, y_train_clean)
    print(
        f"[train_svm] Strategy A&C | Best {scoring}: {search_smotenc.best_score_:.4f}"
    )

    # 2. Strategy B: Cost-sensitive class_weight only
    print("\n[train_svm] Searching Strategy B (class_weight only)...")
    pipeline_no_smotenc = _build_pipeline_no_smotenc(random_state)
    search_no_smotenc = RandomizedSearchCV(
        estimator=pipeline_no_smotenc,
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
    search_no_smotenc.fit(X_train_clean, y_train_clean)
    print(
        f"[train_svm] Strategy B   | Best {scoring}: {search_no_smotenc.best_score_:.4f}"
    )

    # 3. Strategy Selection
    if search_smotenc.best_score_ >= search_no_smotenc.best_score_:
        best_search = search_smotenc
        best_class_weight = search_smotenc.best_params_.get("svm__class_weight")
        best_strategy = (
            "A (SMOTENC only)"
            if best_class_weight is None
            else "C (SMOTENC + class_weight)"
        )
    else:
        best_search = search_no_smotenc
        best_strategy = "B (class_weight only)"

    print(f"\n[train_svm] Winning strategy : {best_strategy}")
    print(f"[train_svm] Best {scoring}  : {best_search.best_score_:.4f}")
    print(f"[train_svm] Best params      : {best_search.best_params_}")

    raw_best_pipeline = best_search.best_estimator_

    # 4. Probability Calibration (Platt Scaling)
    calibrated_model = CalibratedClassifierCV(
        estimator=raw_best_pipeline,
        ensemble=False,
        cv=5,
    )
    calibrated_model.fit(X_train_clean, y_train_clean)

    # 5. OOF Decision Threshold Scan
    optimal_threshold, threshold_results_df = find_optimal_threshold_oof(
        raw_pipeline=calibrated_model,
        X_train=X_train_clean,
        y_train=np.asarray(y_train_clean),
        thresholds=np.arange(0.21, 0.71, 0.01),
        cv=oof_cv,
        random_state=random_state,
    )

    # 6. Save Artifacts & Plots
    thr_dir = project_root / "report_assets" / "threshold_analysis"
    plot_dir = project_root / "report_assets" / "plots"
    thr_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    csv_path = thr_dir / "svm_threshold_results.csv"
    threshold_results_df.to_csv(csv_path, index=False, float_format="%.4f")

    summary = {
        "selected_threshold": round(optimal_threshold, 4),
        "max_oof_f1": round(float(threshold_results_df["F1"].max()), 4),
        "band_99pct_count": int(
            (
                threshold_results_df["F1"] >= 0.99 * threshold_results_df["F1"].max()
            ).sum()
        ),
        "selected_accuracy": round(
            float(
                threshold_results_df.loc[
                    (threshold_results_df["Threshold"] - optimal_threshold)
                    .abs()
                    .idxmin(),
                    "Accuracy",
                ]
            ),
            4,
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
    with open(thr_dir / "svm_threshold_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

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

    setattr(calibrated_model, "optimal_threshold_", float(optimal_threshold))

    if output_path:
        save_model(calibrated_model, output_path)

    return calibrated_model, {
        "search_smotenc": search_smotenc,
        "search_no_smotenc": search_no_smotenc,
        "best_strategy": best_strategy,
        "best_search": best_search,
        "optimal_threshold": optimal_threshold,
        "threshold_results": threshold_results_df,
    }


if __name__ == "__main__":
    # 1. Load and split dataset
    data_path = str(project_root / "data" / "raw" / "online_shoppers_intention.csv")
    df = preprocess_data(filepath=data_path, outlier_method="none")
    X_train, X_test, y_train, y_test = split_dataset(df)

    # 2. Train, calibrate, and optimize threshold
    save_path = project_root / "saved_models" / "svm_model.pkl"
    model, result = train_svm(
        X_train,
        y_train,
        output_path=save_path,
        outlier_method="iqr",
    )

    optimal_threshold = result["optimal_threshold"]
    best_strategy = result["best_strategy"]

    print(f"\n[main] Winning strategy: {best_strategy}")
    print(f"[main] Optimal threshold: {optimal_threshold:.2f}")

    # 3. Evaluate on untouched test set
    metrics = evaluate_model_with_threshold(
        model, X_test, y_test, threshold=optimal_threshold
    )
    print_metrics("SVM Model", metrics)

    # 4. Save metrics
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
        f"\n[main] Total run duration: {_total_seconds:.1f} s ({_total_seconds / 60:.2f} min)"
    )

"""
svm_model.py
------------
Support Vector Machine (SVM) module for the Online Shoppers Purchasing
Intention classification task — training, tuning, cross-validation, and
SVM-specific evaluation/visualisation, all in one file, matching the
structure of knn_model.py and xgboost_model.py.

Generic evaluation (accuracy/precision/recall/F1/AUC, plain confusion
matrix, plain ROC curve) already lives in src/utils.py and is reused here
via evaluate_model() / print_metrics() — it is NOT duplicated below.
Everything in this file is here specifically because it doesn't fit the
generic helpers:
  - SVC has no native `.feature_importances_`, so importance has to be
    derived differently depending on kernel (coefficients for linear,
    permutation importance otherwise).
  - The C/gamma hyperparameter heatmap only makes sense for SVM's own
    hyperparameter grid.
  - Precision-Recall curve and learning curve aren't in src/utils.py at
    all yet (only plain confusion matrix / ROC curve are).

Probability Calibration
-----------------------
SVC is initialised with ``probability=False`` and then wrapped in
``CalibratedClassifierCV(cv="prefit")`` after hyperparameter search, so
that ``predict_proba()`` is available for:
  - AUC-ROC computation in evaluate_model() / generate_svm_report()
  - The Live Prediction probability meter (pages/3_Live_Prediction.py)
  - The Model Comparison page (pages/2_Model_Comparison.py)

SHAP Interpretability
---------------------
``generate_shap_explanation`` from ``src.utils`` is imported and called
in the ``__main__`` block to produce Beeswarm, Feature Importance Bar,
and Waterfall plots saved to ``report_assets/plots/`` with the prefix
``svm_``.  The filenames therefore match the pattern ``svm_shap_*.png``
expected by ``pages/2_Model_Comparison.py``.

Usage
-----
    from models.svm_model import train_svm, generate_svm_report

    model, search_obj = train_svm(X_train, y_train)
    metrics = generate_svm_report(model, X_test, y_test, search_obj=search_obj)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    learning_curve,
)
from sklearn.svm import SVC

from src.data_preprocessing import build_preprocessor, preprocess_data
from src.utils import (
    evaluate_model,
    generate_shap_explanation,
    print_metrics,
    save_model,
)
from scipy.stats import loguniform

sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# Default hyperparameter search space
# ---------------------------------------------------------------------------
# NOTE: 'poly' is intentionally left out of the default grid — it rarely
# beats rbf/linear on this dataset and multiplies search time via the extra
# 'degree' axis. Pass a custom param_grid to explore it if needed.
SVM_PARAM_DISTRIBUTIONS = [
    # RBF kernel
    {
        "svm__kernel": ["rbf"],
        "svm__C": loguniform(0.1, 300),
        "svm__gamma": loguniform(1e-4, 1),
        "svm__class_weight": [
            None,
            "balanced",
            {0: 1, 1: 1.5},
            {0: 1, 1: 2},
            {0: 1, 1: 3},
        ],
    },
    # Linear kernel
    {
        "svm__kernel": ["linear"],
        "svm__C": loguniform(0.01, 300),
        "svm__class_weight": [
            None,
            "balanced",
            {0: 1, 1: 1.5},
            {0: 1, 1: 2},
            {0: 1, 1: 3},
        ],
    },
]


def _save_show(fig: plt.Figure, name: str, save_dir: str | None, show: bool) -> None:
    """Save a figure to save_dir and/or display it — same convention as src/eda.py."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Pipeline & training
# ---------------------------------------------------------------------------


def build_svm_pipeline(use_smote: bool = True, random_state: int = 42) -> Pipeline:
    """Build a base SVM pipeline (preprocessor → optional SMOTE → SVC).

    SVC is created without the deprecated ``probability`` parameter; Platt
    scaling is applied externally via
    ``CalibratedClassifierCV(estimator, ensemble=False)`` after tuning so
    that calibration does not interfere with the hyperparameter search
    (sklearn ≥ 1.9 API).
    """
    preprocessor = build_preprocessor(scale_numerical=True)
    # max_iter=-1 lets the solver converge naturally based on tol
    # probability= param is deprecated in sklearn 1.9; calibration is handled
    # by CalibratedClassifierCV(ensemble=False) in train_svm() instead.
    svm = SVC(
        random_state=random_state,
        max_iter=-1,
        tol=1e-3,
    )

    steps = [("preprocessor", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("svm", svm))
    return Pipeline(steps=steps)


def train_svm(
    X_train,
    y_train,
    use_smote: bool = True,
    param_grid: dict | None = None,
    scoring: str = "f1",
    cv: int = 5,
    search: str = "random",
    n_iter: int = 8,
    random_state: int = 42,
    output_path: str | Path | None = "saved_models/svm_model.pkl",
    verbose: int = 2,
    n_jobs: int = -2,
):
    pipeline = build_svm_pipeline(use_smote=use_smote, random_state=random_state)
    grid = param_grid or SVM_PARAM_DISTRIBUTIONS
    stratified_cv = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=random_state
    )

    if search == "random":
        search_obj = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=stratified_cv,
            verbose=verbose,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    else:
        search_obj = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            scoring=scoring,
            cv=stratified_cv,
            verbose=verbose,
            n_jobs=n_jobs,
        )

    search_obj.fit(X_train, y_train)
    raw_best_pipeline = search_obj.best_estimator_

    print(f"\n[train_svm] Best SVM params: {search_obj.best_params_}")
    print(f"[train_svm] Best CV {scoring} score: {search_obj.best_score_:.4f}")

    # Calibrate the winning pipeline to safely expose predict_proba().
    # sklearn ≥ 1.9: cv="prefit" was removed; use ensemble=False instead
    # to wrap a pre-fitted estimator without re-fitting.
    calibrated_model = CalibratedClassifierCV(
        estimator=raw_best_pipeline,
        ensemble=False,
    )
    calibrated_model.fit(X_train, y_train)

    if output_path:
        save_model(calibrated_model, output_path)

    return calibrated_model, search_obj


# ---------------------------------------------------------------------------
# 2. Inspecting the hyperparameter search
# ---------------------------------------------------------------------------


def get_grid_search_results(search_obj) -> pd.DataFrame:
    """Return GridSearchCV/RandomizedSearchCV cv_results_ as a tidy, sorted
    DataFrame — handy for a 'hyperparameter tuning results' table in the
    Methodology or Results section of the report.
    """
    results = pd.DataFrame(search_obj.cv_results_)
    param_cols = [c for c in results.columns if c.startswith("param_")]
    keep_cols = param_cols + ["mean_test_score", "std_test_score", "rank_test_score"]
    tidy = results[keep_cols].sort_values("rank_test_score").reset_index(drop=True)
    tidy.columns = [c.replace("param_svm__", "") for c in tidy.columns]
    return tidy


def plot_svm_hyperparameter_heatmap(search_obj, save_dir=None, show=True):
    """Visualise mean CV F1 score across C x gamma for rbf-kernel runs only
    (linear-kernel runs don't have a gamma axis and are excluded here).
    SVM-specific because the axes are SVM's own hyperparameters.

    Returns None gracefully when no rbf results are present or if any
    unexpected error occurs during pivot/render.
    """
    try:
        results = pd.DataFrame(search_obj.cv_results_)
        rbf_results = results[results["param_svm__kernel"] == "rbf"].copy()

        if rbf_results.empty:
            print(
                "[plot_svm_hyperparameter_heatmap] No rbf-kernel results found — skipping."
            )
            return None

        pivot = rbf_results.pivot_table(
            index="param_svm__C",
            columns="param_svm__gamma",
            values="mean_test_score",
            aggfunc="mean",
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        ax.set_title(
            "SVM (RBF kernel) — Mean CV F1 Score by C and gamma",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("gamma")
        ax.set_ylabel("C")
        plt.tight_layout()
        _save_show(fig, "13_svm_hyperparameter_heatmap", save_dir, show)
        return pivot
    except Exception as exc:
        print(f"[plot_svm_hyperparameter_heatmap] Skipped due to error: {exc}")
        return None


# ---------------------------------------------------------------------------
# 3. K-fold cross-validation of the final model (for robust reporting)
# ---------------------------------------------------------------------------


def cross_validate_svm(
    model: Pipeline, X, y, cv: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Run stratified k-fold CV on the already-tuned SVM pipeline and return
    per-fold + mean/std Accuracy, Precision, Recall, F1 and AUC.

    Distinct from the tuning step above: tuning selects hyperparameters
    using CV on the training set only; this re-validates the *final* chosen
    pipeline so the report can state e.g. "F1 = 0.71 +/- 0.03 across 5
    folds" instead of a single point estimate.

    AUC scoring requires ``predict_proba``; because the pipeline is wrapped
    in ``CalibratedClassifierCV`` this is always available.
    """
    stratified_cv = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=random_state
    )
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=stratified_cv,
        scoring=scoring,
        n_jobs=-2,
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
# 4. Feature importance (SVM has no native .feature_importances_)
# ---------------------------------------------------------------------------


def get_svm_feature_importance(
    model: Pipeline,
    X_sample,
    y_sample=None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return a DataFrame of feature importances for the fitted SVM pipeline.

    The model is expected to be a ``CalibratedClassifierCV`` wrapping the raw
    SVM pipeline produced by ``train_svm()``.  The inner pipeline is accessed
    via ``model.estimator`` before inspecting individual named steps.

    - Linear kernel  -> uses |coefficient| from svm.coef_ directly (fast,
                        exact, directly interpretable).
    - Non-linear     -> falls back to permutation importance on the whole
                        model (works for any kernel, needs y_sample, slower).

    Edge-case handling
    ------------------
    Column names are inferred from ``X_sample.columns`` when X_sample is a
    DataFrame, or auto-generated as ``feature_0``, ``feature_1``, … when it
    is a numpy array.  This prevents an AttributeError when the test set has
    been pre-transformed to a numpy matrix.
    """
    # Unwrap CalibratedClassifierCV to reach the raw pipeline
    inner_pipeline = getattr(model, "estimator", model)
    svm_step = inner_pipeline.named_steps["svm"]

    if svm_step.kernel == "linear":
        preprocessor = inner_pipeline.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
        coefs = np.abs(svm_step.coef_).ravel()
        importance_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": coefs})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
        importance_df["Method"] = "Linear SVM |coefficient|"
        return importance_df

    if y_sample is None:
        raise ValueError(
            "y_sample is required to compute permutation importance for a "
            "non-linear kernel (rbf/poly)."
        )

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-2,
    )

    # Safely retrieve feature names from DataFrame or generate generic names
    if hasattr(X_sample, "columns"):
        feature_names = list(X_sample.columns)
    else:
        feature_names = [f"feature_{i}" for i in range(X_sample.shape[1])]

    importance_df = (
        pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": result.importances_mean,
                "Std": result.importances_std,
            }
        )
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["Method"] = f"Permutation importance ({svm_step.kernel} kernel)"
    return importance_df


def plot_svm_feature_importance(
    importance_df: pd.DataFrame, top_n: int = 15, save_dir=None, show=True
):
    """Plot a horizontal bar chart of the top-N SVM feature importances.

    Handles the case where importance_df has fewer rows than top_n without
    raising an IndexError (iloc[::-1] on a shorter frame still works).
    """
    try:
        top = importance_df.head(top_n).iloc[::-1]
        method_label = (
            importance_df["Method"].iloc[0] if "Method" in importance_df.columns else ""
        )

        fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
        ax.barh(top["Feature"], top["Importance"], color="#4C72B0")
        ax.set_xlabel("Importance")
        ax.set_title(
            f"SVM — Feature Importance\n({method_label})",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()
        _save_show(fig, "15_svm_feature_importance", save_dir, show)
    except Exception as exc:
        print(f"[plot_svm_feature_importance] Skipped due to error: {exc}")


# ---------------------------------------------------------------------------
# 5. Extra evaluation plots not covered by src/utils.py
# ---------------------------------------------------------------------------
# src/utils.py already has a generic plot_confusion_matrix() / plot_roc_curve()
# usable for any model. The two below are kept here because they aren't in
# src/utils.py at all yet (Precision-Recall curve, learning curve) — if you'd
# rather have them apply to every model, they belong in src/utils.py instead;
# they're written model-agnostically enough to move there as-is.


def plot_svm_precision_recall_curve(model, X_test, y_test, save_dir=None, show=True):
    """More informative than ROC on an imbalanced target (~85/15 split).

    Requires predict_proba() — always available when the pipeline is built
    with probability=True.  Returns None gracefully on any runtime error.
    """
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        ap = average_precision_score(y_test, y_prob)

        fig, ax = plt.subplots(figsize=(7, 6))
        PrecisionRecallDisplay.from_predictions(
            y_test, y_prob, ax=ax, name=f"SVM (AP = {ap:.3f})"
        )
        ax.set_title("SVM — Precision-Recall Curve", fontsize=12, fontweight="bold")
        plt.tight_layout()
        _save_show(fig, "12_svm_precision_recall_curve", save_dir, show)
        return ap
    except Exception as exc:
        print(f"[plot_svm_precision_recall_curve] Skipped due to error: {exc}")
        return None


def plot_svm_learning_curve(
    model, X, y, cv: int = 5, scoring: str = "f1", save_dir=None, show=True
):
    """Plot training vs. cross-validation score across increasing training set sizes.

    Returns (None, None, None) on any runtime error so the caller can
    continue without crashing.
    """
    try:
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-2,
            train_sizes=np.linspace(0.1, 1.0, 6),
            random_state=42,
        )

        train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
        val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(train_sizes, train_mean, "o-", color="#4C72B0", label="Training score")
        ax.fill_between(
            train_sizes,
            train_mean - train_std,
            train_mean + train_std,
            alpha=0.15,
            color="#4C72B0",
        )
        ax.plot(
            train_sizes, val_mean, "o-", color="#DD8452", label="Cross-validation score"
        )
        ax.fill_between(
            train_sizes,
            val_mean - val_std,
            val_mean + val_std,
            alpha=0.15,
            color="#DD8452",
        )
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel(scoring.upper())
        ax.set_title("SVM — Learning Curve", fontsize=12, fontweight="bold")
        ax.legend(loc="best")
        plt.tight_layout()
        _save_show(fig, "14_svm_learning_curve", save_dir, show)
        return train_sizes, train_mean, val_mean
    except Exception as exc:
        print(f"[plot_svm_learning_curve] Skipped due to error: {exc}")
        return None, None, None


# ---------------------------------------------------------------------------
# 6. Master report function — everything in one call
# ---------------------------------------------------------------------------


def generate_svm_report(
    model,
    X_test,
    y_test,
    search_obj=None,
    X_importance=None,
    y_importance=None,
    save_dir: str | Path | None = "report_assets/plots",
    show: bool = False,
) -> dict:
    """Run the full SVM evaluation suite and return a metrics dict for the
    Results & Discussion section.

    Reuses src/utils.py's evaluate_model() for the core metrics + generic
    confusion matrix data, then adds the SVM-specific plots on top.

    All individual plot calls are wrapped in try/except blocks so that an
    isolated failure (e.g. a degenerate test fold during unit testing) does
    not abort the entire report generation.

    Probability arrays
    ------------------
    ``evaluate_model()`` calls ``model.predict_proba()`` when available.
    Because the pipeline is wrapped in ``CalibratedClassifierCV``, this is
    always present for SVM models produced by ``train_svm()``.
    """
    # --- Core metrics -------------------------------------------------------
    base_metrics = evaluate_model(model, X_test, y_test)

    # Safely extract probability estimates for downstream plots
    try:
        y_prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )
    except Exception as exc:
        print(
            f"[generate_svm_report] predict_proba failed: {exc}. AUC plots will be skipped."
        )
        y_prob = None

    # --- Confusion Matrix ---------------------------------------------------
    print("[generate_svm_report] Plotting confusion matrix...")
    try:
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix=base_metrics["Confusion Matrix"],
            display_labels=["No Purchase", "Purchase"],
        ).plot(cmap="Blues", ax=ax, colorbar=False)
        ax.set_title("SVM — Confusion Matrix", fontsize=12, fontweight="bold")
        plt.tight_layout()
        _save_show(fig, "10_svm_confusion_matrix", save_dir, show)
    except Exception as exc:
        print(f"[generate_svm_report] Confusion matrix plot failed: {exc}")

    # --- ROC Curve ----------------------------------------------------------
    if y_prob is not None:
        print("[generate_svm_report] Plotting ROC curve...")
        try:
            auc_val = base_metrics.get("AUC")
            auc_str = f"{auc_val:.3f}" if auc_val is not None else "N/A"
            fig, ax = plt.subplots(figsize=(7, 6))
            RocCurveDisplay.from_predictions(
                y_test, y_prob, ax=ax, name=f"SVM (AUC = {auc_str})"
            )
            ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
            ax.set_title("SVM — ROC Curve", fontsize=12, fontweight="bold")
            ax.legend()
            plt.tight_layout()
            _save_show(fig, "11_svm_roc_curve", save_dir, show)
        except Exception as exc:
            print(f"[generate_svm_report] ROC curve plot failed: {exc}")

        # --- Precision-Recall Curve -----------------------------------------
        print("[generate_svm_report] Plotting Precision-Recall curve...")
        ap = plot_svm_precision_recall_curve(model, X_test, y_test, save_dir, show)
        if ap is not None:
            base_metrics["Average Precision"] = ap

    # --- Hyperparameter Heatmap ---------------------------------------------
    if search_obj is not None:
        print("[generate_svm_report] Plotting hyperparameter heatmap...")
        plot_svm_hyperparameter_heatmap(search_obj, save_dir, show)

    # --- Feature Importance -------------------------------------------------
    if X_importance is not None:
        print("[generate_svm_report] Computing & plotting feature importance...")
        try:
            importance_df = get_svm_feature_importance(
                model, X_importance, y_importance
            )
            base_metrics["Feature Importance"] = importance_df
            plot_svm_feature_importance(importance_df, save_dir=save_dir, show=show)
        except Exception as exc:
            print(f"[generate_svm_report] Feature importance failed: {exc}")

    return base_metrics


# ---------------------------------------------------------------------------
# CLI entry point — mirrors knn_model.py / xgboost_model.py conventions
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── 1. Load & split data ────────────────────────────────────────────────
    data_path = project_root / "data" / "raw" / "online_shoppers_intention.csv"
    X_train, X_test, y_train, y_test, _ = preprocess_data(
        filepath=data_path, transform=False
    )

    # ── 2. Train & save ─────────────────────────────────────────────────────
    save_path = project_root / "saved_models" / "svm_model.pkl"
    model, search_obj = train_svm(
        X_train,
        y_train,
        output_path=save_path,
        param_grid=SVM_PARAM_DISTRIBUTIONS,
    )

    # ── 3. Core metrics ─────────────────────────────────────────────────────
    metrics = evaluate_model(model, X_test, y_test)
    print_metrics("SVM Classifier", metrics)

    # ── 4. Full SVM report (confusion matrix, ROC, PR curve, heatmap, …) ───
    PLOT_DIR = str(project_root / "report_assets" / "plots")
    generate_svm_report(
        model,
        X_test,
        y_test,
        search_obj=search_obj,
        X_importance=X_test,
        y_importance=y_test,
        save_dir=PLOT_DIR,
        show=False,
    )

    # ── 5. SHAP Interpretability ─────────────────────────────────────────────
    # Saves Beeswarm, Feature Importance Bar, and Waterfall plots with the
    # prefix ``svm_`` → filenames become:
    #   svm_shap_beeswarm.png
    #   svm_shap_feature_importance.png
    #   svm_shap_waterfall.png
    #
    # These match the glob pattern ``{stem}_shap_*.png`` (stem = "svm") used
    # by pages/2_Model_Comparison.py to auto-discover SHAP plots.
    #
    # generate_shap_explanation() automatically uses KernelExplainer for SVC
    # (non-tree estimator), operating on already-transformed numpy arrays
    # (post-preprocessor) so no DataFrame column passthrough is needed.
    print("\n[__main__] Generating SVM SHAP explanation plots...")
    try:
        generate_shap_explanation(
            model=model,
            X_test=X_test,
            max_display=15,
            save_dir=PLOT_DIR,
            prefix="svm_",
            show=False,
        )
        print("[__main__] SHAP plots saved successfully.")
    except Exception as exc:
        print(f"[__main__] SHAP explanation failed (non-fatal): {exc}")

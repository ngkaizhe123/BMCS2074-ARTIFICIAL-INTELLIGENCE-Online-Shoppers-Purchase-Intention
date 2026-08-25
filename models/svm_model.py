"""
svm_model.py
------------
Support Vector Machine (SVM) module for the Online Shoppers Purchasing
Intention classification task — pipeline construction, hyperparameter tuning
via RandomizedSearchCV / GridSearchCV, probability calibration, cross-validation,
and SHAP model interpretability.

Pipeline: preprocessor (StandardScaler + OneHotEncoder) -> SMOTE -> SVC
Probability Calibration: CalibratedClassifierCV(ensemble=False)
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.svm import SVC

from src.data_preprocessing import build_preprocessor, preprocess_data
from src.utils import (
    evaluate_model,
    generate_shap_explanation,
    print_metrics,
    save_metrics,
    save_model,
    split_dataset,
)

# ---------------------------------------------------------------------------
# Default hyperparameter search space
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 1. Pipeline & training
# ---------------------------------------------------------------------------


def build_svm_pipeline(use_smote: bool = True, random_state: int = 42) -> Pipeline:
    """Build a base SVM pipeline (preprocessor -> optional SMOTE -> SVC)."""
    preprocessor = build_preprocessor(scale_numerical=True)
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
    param_grid: dict | list | None = None,
    scoring: str = "f1",
    cv: int = 5,
    search: str = "random",
    n_iter: int = 8,
    random_state: int = 42,
    output_path: str | Path | None = None,
    verbose: int = 2,
    n_jobs: int = -2,
) -> tuple[CalibratedClassifierCV, RandomizedSearchCV | GridSearchCV]:
    """Tune hyperparameters via RandomizedSearchCV / GridSearchCV, fit SVM, and calibrate."""
    if output_path is None:
        output_path = project_root / "saved_models" / "svm_model.pkl"

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

    # Calibrate winning pipeline to expose predict_proba()
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
    """Return GridSearchCV/RandomizedSearchCV cv_results_ as a tidy, sorted DataFrame."""
    results = pd.DataFrame(search_obj.cv_results_)
    param_cols = [c for c in results.columns if c.startswith("param_")]
    keep_cols = param_cols + ["mean_test_score", "std_test_score", "rank_test_score"]
    tidy = results[keep_cols].sort_values("rank_test_score").reset_index(drop=True)
    tidy.columns = [c.replace("param_svm__", "") for c in tidy.columns]
    return tidy


# ---------------------------------------------------------------------------
# 3. K-fold cross-validation of the final model
# ---------------------------------------------------------------------------


def cross_validate_svm(
    model: Pipeline, X, y, cv: int = 5, random_state: int = 42
) -> pd.DataFrame:
    """Run stratified k-fold CV on the already-tuned SVM pipeline and return metrics."""
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
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── 1. Load & split data ────────────────────────────────────────────────
    data_path = str(project_root / "data" / "raw" / "online_shoppers_intention.csv")
    df = preprocess_data(filepath=data_path, outlier_method="iqr")
    X_train, X_test, y_train, y_test = split_dataset(df)

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
    print_metrics("SVM Model", metrics)

    # ── 4. Persist metrics ──────────────────────────────────────────────────
    metrics_output_path = project_root / "report_assets" / "metrics.json"
    save_metrics("SVM Model", "svm", metrics, metrics_output_path)

    # ── 5. SHAP Interpretability ────────────────────────────────────────────
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

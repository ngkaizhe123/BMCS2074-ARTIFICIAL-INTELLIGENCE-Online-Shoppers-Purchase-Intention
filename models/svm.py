"""
svm.py
------

SVM-RFE feature selection + Bayesian-optimised SVM classification pipeline
for the Online Shoppers Purchasing Intention dataset.

Enhancement approach
--------------------
    SVM-RFE (feature selection)  +  Optuna Bayesian optimisation (tuning)

Stage 1 — SVM-RFE
    RFECV wrapping ``LinearSVC`` on preprocessed features
    (``build_preprocessor()`` from ``src/data_preprocessing.py``) to select
    the most informative feature subset.  Returns the selected feature mask
    and names of eliminated columns.

Stage 2 — Bayesian Optimisation (Optuna)
    ``optuna.Study(direction="maximize")`` with ``TPESampler(seed=42)``,
    ~25 trials.  Tunes ``SVC(kernel="rbf")`` hyperparameters:
        - C            (log-uniform 0.01 – 300)
        - gamma        (log-uniform 1e-4 – 1)
        - class_weight (None | "balanced" | {0:1, 1:1.5} | …)
    Objective = mean 5-fold stratified CV PR-AUC (``average_precision_score``).
    SMOTE is applied **inside each fold only** (no leakage).
    Returns ``best_params``, ``best_score``, and the Optuna ``study`` object.

Final pipeline
--------------
    ``build_final_pipeline()`` → imblearn ``Pipeline``:
        preprocessor → feature-mask selector → SMOTE → SVC(**best_params)

    Exposes ``fit`` / ``predict`` / ``predict_proba``, fully compatible with
    ``evaluate_model()`` / ``print_metrics()`` in ``src/utils.py`` and the
    plotting functions in ``svm_model.py``.

Important
---------
    This module is a **pure-SVM** pipeline.  It does NOT import or combine
    with XGBoost, KNN, RF, or any other classifier family.

Usage
-----
    from models.svm import run_svm_rfe_bayesopt, build_final_pipeline

    rfe_result  = run_svm_rfe(X_train, y_train, preprocessor)
    bayes_result = run_bayesian_optimisation(X_train, y_train, preprocessor,
                                             rfe_result.support_mask)
    pipeline = build_final_pipeline(preprocessor, rfe_result.support_mask,
                                     bayes_result.best_params)
    pipeline.fit(X_train, y_train)
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import optuna
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFECV
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.base import clone

# ============================================================================
# Project root
# ============================================================================

project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ============================================================================
# Project-specific helpers
# ============================================================================

from src.data_preprocessing import (
    build_preprocessor,
    get_smote,
    preprocess_data,
)

from src.utils import (
    evaluate_model,
    generate_shap_explanation,
    print_metrics,
    save_model,
)


# ============================================================================
# FEATURE-MASK SELECTOR (sklearn transformer)
# ============================================================================


class FeatureMaskSelector(BaseEstimator, TransformerMixin):
    """Select columns from a 2-D array/matrix by a boolean mask.

    This lightweight transformer is inserted into the final imblearn
    ``Pipeline`` between the preprocessor and SMOTE so that the same
    feature subset chosen by SVM-RFE is applied automatically during
    both ``fit`` and ``predict`` / ``predict_proba``.

    Parameters
    ----------
    support_mask : np.ndarray
        1-D boolean array — ``True`` for features to keep.
    """

    def __init__(self, support_mask: np.ndarray) -> None:
        self.support_mask = support_mask

    def fit(self, X, y=None):  # noqa: D401
        """No-op (stateless transformer)."""
        return self

    def transform(self, X):
        """Apply the boolean mask to select columns."""
        if hasattr(X, "toarray"):
            X = X.toarray()
        if hasattr(X, "iloc"):
            return X.iloc[:, self.support_mask]
        return X[:, self.support_mask]

    def get_feature_names_out(self, input_features=None):
        """Forward only the selected feature names."""
        if input_features is not None:
            input_features = np.asarray(input_features)
            return input_features[self.support_mask]
        return None


# ============================================================================
# STAGE 1 — SVM-RFE RESULT
# ============================================================================


@dataclass
class RFEResult:
    """Container for SVM-RFE feature-selection results."""

    support_mask: np.ndarray
    """Boolean array — True for selected features."""

    ranking: np.ndarray
    """Integer ranking — 1 = selected, >1 = elimination order."""

    n_features_selected: int
    """Number of features retained."""

    eliminated_columns: list[str]
    """Names of features that were eliminated."""

    selected_columns: list[str]
    """Names of features that were selected."""

    elapsed_seconds: float
    """Wall-clock time for the RFE stage."""


# ============================================================================
# STAGE 1 — SVM-RFE FEATURE SELECTION
# ============================================================================


def run_svm_rfe(
    X_train,
    y_train,
    preprocessor,
    *,
    min_features_to_select: int = 5,
    cv: int = 5,
    scoring: str = "average_precision",
    random_state: int = 42,
    verbose: bool = True,
) -> RFEResult:
    """Run Recursive Feature Elimination with Cross-Validation (RFECV)
    using ``LinearSVC`` on preprocessed features.

    The preprocessor (``build_preprocessor()``) is fitted on ``X_train``
    here to produce the transformed feature matrix that ``RFECV`` operates
    on.  The fitted preprocessor is **not** modified — downstream code
    must still include it as a pipeline step.

    Parameters
    ----------
    X_train : pd.DataFrame
        Raw training features (pre-preprocessing).
    y_train : pd.Series
        Binary target labels.
    preprocessor : ColumnTransformer
        As returned by ``build_preprocessor(scale_numerical=True)``.
    min_features_to_select : int
        Minimum number of features to keep.
    cv : int
        Number of cross-validation folds.
    scoring : str
        Scoring metric for RFECV (default: ``average_precision``).
    random_state : int
        Seed for reproducibility.
    verbose : bool
        Whether to print progress banners.

    Returns
    -------
    RFEResult
        Dataclass containing the boolean support mask, eliminated column
        names, and timing information.
    """

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    if verbose:
        print("\n" + "=" * 70)
        print(" Stage 1 — SVM-RFE Feature Selection")
        print("=" * 70)

    # ------------------------------------------------------------------
    # Preprocess X_train to get feature matrix + names
    # ------------------------------------------------------------------

    prep = clone(preprocessor)
    X_transformed = prep.fit_transform(X_train)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = np.array(
        prep.get_feature_names_out()
        if hasattr(prep, "get_feature_names_out")
        else [f"feature_{i}" for i in range(X_transformed.shape[1])]
    )

    if verbose:
        print(f"Preprocessed features : {X_transformed.shape[1]}")

    # ------------------------------------------------------------------
    # RFECV with LinearSVC
    # ------------------------------------------------------------------

    estimator = LinearSVC(
        dual="auto",
        max_iter=10_000,
        random_state=random_state,
        class_weight="balanced",
    )

    skf = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=random_state,
    )

    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=skf,
        scoring=scoring,
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
    )

    if verbose:
        print("Running RFECV (this may take a moment) ...")

    rfecv.fit(X_transformed, y_train)

    # ------------------------------------------------------------------
    # Collect results
    # ------------------------------------------------------------------

    support_mask = rfecv.support_
    ranking = rfecv.ranking_

    selected_cols = feature_names[support_mask].tolist()
    eliminated_cols = feature_names[~support_mask].tolist()

    elapsed = time.perf_counter() - t_start

    if verbose:
        print(f"\nFeatures selected     : {rfecv.n_features_}")
        print(f"Features eliminated   : {len(eliminated_cols)}")
        print(f"Eliminated features   : {eliminated_cols}")
        print(f"Stage 1 elapsed time  : {elapsed:.2f}s")
        print("=" * 70)

    return RFEResult(
        support_mask=support_mask,
        ranking=ranking,
        n_features_selected=int(rfecv.n_features_),
        eliminated_columns=eliminated_cols,
        selected_columns=selected_cols,
        elapsed_seconds=elapsed,
    )


# ============================================================================
# STAGE 2 — BAYESIAN OPTIMISATION RESULT
# ============================================================================


@dataclass
class BayesOptResult:
    """Container for Bayesian optimisation results."""

    best_params: dict[str, Any]
    """Best hyperparameters found."""

    best_score: float
    """Best mean CV PR-AUC achieved."""

    study: optuna.Study
    """The Optuna study object (contains all trial info)."""

    elapsed_seconds: float
    """Wall-clock time for the optimisation stage."""


# ============================================================================
# STAGE 2 — BAYESIAN OPTIMISATION (OPTUNA)
# ============================================================================


def run_bayesian_optimisation(
    X_train,
    y_train,
    preprocessor,
    support_mask: np.ndarray,
    *,
    n_trials: int = 25,
    cv: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> BayesOptResult:
    """Bayesian-optimise ``SVC(kernel="rbf")`` via Optuna.

    Objective = mean 5-fold stratified CV PR-AUC (``average_precision_score``).
    SMOTE is applied **inside each fold** to prevent data leakage.

    Parameters
    ----------
    X_train : pd.DataFrame
        Raw training features (pre-preprocessing).
    y_train : pd.Series
        Binary target labels.
    preprocessor : ColumnTransformer
        As returned by ``build_preprocessor(scale_numerical=True)``.
    support_mask : np.ndarray
        Boolean feature mask from SVM-RFE (Stage 1).
    n_trials : int
        Number of Optuna trials (default: 25).
    cv : int
        Number of cross-validation folds.
    random_state : int
        Seed for reproducibility.
    verbose : bool
        Whether to print progress banners.

    Returns
    -------
    BayesOptResult
        Dataclass with ``best_params``, ``best_score``, ``study``, and
        timing information.
    """

    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    if verbose:
        print("\n" + "=" * 70)
        print(" Stage 2 — Bayesian Optimisation (Optuna)")
        print("=" * 70)
        print(f"Trials               : {n_trials}")
        print(f"CV folds             : {cv}")
        print(f"Objective            : PR-AUC (average_precision_score)")
        print(f"Sampler              : TPESampler(seed={random_state})")
        print("=" * 70)

    # ------------------------------------------------------------------
    # Pre-transform once — the preprocessor step is deterministic, so
    # we can fit-transform up front and reuse the result across all
    # Optuna trials for speed.
    # ------------------------------------------------------------------

    prep = clone(preprocessor)
    X_transformed = prep.fit_transform(X_train)

    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    # Apply feature mask from RFE
    X_selected = X_transformed[:, support_mask]

    # ------------------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------------------

    class_weight_options: list[Optional[dict | str]] = [
        None,
        "balanced",
        {0: 1, 1: 1.5},
        {0: 1, 1: 2},
        {0: 1, 1: 3},
    ]

    def objective(trial: optuna.Trial) -> float:
        """Optuna trial objective — mean 5-fold stratified CV PR-AUC."""

        C = trial.suggest_float("C", 0.01, 300.0, log=True)
        gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        cw_idx = trial.suggest_categorical(
            "class_weight_idx", list(range(len(class_weight_options)))
        )
        class_weight = class_weight_options[cw_idx]

        skf = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=random_state,
        )

        smote = get_smote(random_state=random_state)

        fold_scores: list[float] = []

        for train_idx, val_idx in skf.split(X_selected, y_train):
            X_tr = X_selected[train_idx]
            X_val = X_selected[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            # SMOTE inside the fold only
            X_tr_resampled, y_tr_resampled = smote.fit_resample(X_tr, y_tr)

            svc = SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                class_weight=class_weight,
                probability=True,
                random_state=random_state,
                max_iter=-1,
                tol=1e-3,
            )

            svc.fit(X_tr_resampled, y_tr_resampled)

            val_proba = svc.predict_proba(X_val)[:, 1]

            fold_pr_auc = average_precision_score(y_val, val_proba)
            fold_scores.append(fold_pr_auc)

        return float(np.mean(fold_scores))

    # ------------------------------------------------------------------
    # Run study
    # ------------------------------------------------------------------

    optuna_verbosity = optuna.logging.INFO if verbose else optuna.logging.WARNING
    optuna.logging.set_verbosity(optuna_verbosity)

    sampler = optuna.samplers.TPESampler(seed=random_state)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=verbose,
    )

    # ------------------------------------------------------------------
    # Decode best params
    # ------------------------------------------------------------------

    best_trial = study.best_trial
    best_params = {
        "C": best_trial.params["C"],
        "gamma": best_trial.params["gamma"],
        "class_weight": class_weight_options[
            best_trial.params["class_weight_idx"]
        ],
    }

    elapsed = time.perf_counter() - t_start

    if verbose:
        print("\n" + "=" * 70)
        print(" Bayesian Optimisation Completed")
        print("=" * 70)
        print(f"Best CV PR-AUC       : {study.best_value:.4f}")
        print(f"\nBest Parameters:")
        for name, value in best_params.items():
            print(f"    {name}: {value}")
        print(f"\nStage 2 elapsed time : {elapsed:.2f}s")
        print("=" * 70)

    return BayesOptResult(
        best_params=best_params,
        best_score=study.best_value,
        study=study,
        elapsed_seconds=elapsed,
    )


# ============================================================================
# BUILD FINAL PIPELINE
# ============================================================================


def build_final_pipeline(
    preprocessor,
    support_mask: np.ndarray,
    best_params: dict[str, Any],
    *,
    random_state: int = 42,
) -> ImbPipeline:
    """Build the production pipeline for SVM-RFE + Bayesian-optimised SVM.

    Pipeline layout
    ---------------
    preprocessor → FeatureMaskSelector → SMOTE → SVC(**best_params)

    The pipeline exposes ``fit``, ``predict``, and ``predict_proba``
    and is fully compatible with ``evaluate_model()`` / ``print_metrics()``
    in ``src/utils.py`` and the plotting functions in ``svm_model.py``.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        As returned by ``build_preprocessor(scale_numerical=True)``.
    support_mask : np.ndarray
        Boolean feature mask from SVM-RFE (Stage 1).
    best_params : dict
        Best hyperparameters from Bayesian optimisation (Stage 2).
        Expected keys: ``C``, ``gamma``, ``class_weight``.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    ImbPipeline
        Ready-to-fit imblearn Pipeline.
    """

    svc = SVC(
        kernel="rbf",
        C=best_params["C"],
        gamma=best_params["gamma"],
        class_weight=best_params.get("class_weight"),
        probability=True,
        random_state=random_state,
        max_iter=-1,
        tol=1e-3,
    )

    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("feature_selector", FeatureMaskSelector(support_mask)),
            ("smote", get_smote(random_state=random_state)),
            ("svc", svc),
        ]
    )

    return pipeline


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    # ========================================================================
    # Load data
    #
    # transform=False is important because preprocessing is handled
    # inside the pipeline.
    # ========================================================================

    print("\n" + "=" * 70)
    print(" SVM-RFE + Bayesian-Optimised SVM Pipeline")
    print("=" * 70)

    data_path = project_root / "data" / "raw" / "online_shoppers_intention.csv"

    X_train, X_test, y_train, y_test, _ = preprocess_data(
        filepath=data_path,
        transform=False,
    )

    # SVM requires scaled features
    preprocessor = build_preprocessor(scale_numerical=True)

    # ========================================================================
    # STAGE 1 — SVM-RFE Feature Selection
    # ========================================================================

    t_total_start = time.perf_counter()

    rfe_result = run_svm_rfe(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        min_features_to_select=5,
        cv=5,
        scoring="average_precision",
        random_state=42,
        verbose=True,
    )

    # ========================================================================
    # STAGE 2 — Bayesian Optimisation (Optuna)
    # ========================================================================

    bayes_result = run_bayesian_optimisation(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        support_mask=rfe_result.support_mask,
        n_trials=25,
        cv=5,
        random_state=42,
        verbose=True,
    )

    # ========================================================================
    # BUILD & FIT FINAL PIPELINE
    # ========================================================================

    print("\n" + "=" * 70)
    print(" Building & Fitting Final Pipeline")
    print("=" * 70)

    final_pipeline = build_final_pipeline(
        preprocessor=preprocessor,
        support_mask=rfe_result.support_mask,
        best_params=bayes_result.best_params,
        random_state=42,
    )

    print("Fitting final pipeline on full training data ...")
    final_pipeline.fit(X_train, y_train)
    print("Done.")

    t_total_elapsed = time.perf_counter() - t_total_start

    print(f"\nTotal elapsed time   : {t_total_elapsed:.2f}s")
    print(f"  Stage 1 (RFE)      : {rfe_result.elapsed_seconds:.2f}s")
    print(f"  Stage 2 (Optuna)   : {bayes_result.elapsed_seconds:.2f}s")
    print(
        f"  Final fit          : "
        f"{t_total_elapsed - rfe_result.elapsed_seconds - bayes_result.elapsed_seconds:.2f}s"
    )

    # ========================================================================
    # FINAL TEST EVALUATION
    # ========================================================================

    print("\n" + "=" * 70)
    print(" Final Test Evaluation")
    print("=" * 70)

    metrics = evaluate_model(
        final_pipeline,
        X_test,
        y_test,
    )

    print_metrics(
        "SVM-RFE + Bayesian-Optimised SVM",
        metrics,
    )

    # ========================================================================
    # SAVE MODEL
    # ========================================================================

    save_path = project_root / "saved_models" / "svm_rfe_bayesopt.pkl"

    save_model(
        final_pipeline,
        save_path,
    )

    print(f"[__main__] Model saved to: {save_path}")

    # ========================================================================
    # SHAP INTERPRETABILITY
    # ========================================================================

    PLOT_DIR = str(project_root / "report_assets" / "plots")

    print("\n[__main__] Generating SHAP explanation plots ...")

    try:

        generate_shap_explanation(
            model=final_pipeline,
            X_test=X_test,
            max_display=15,
            save_dir=PLOT_DIR,
            prefix="svm_rfe_bayesopt_",
            show=False,
        )

        print("[__main__] SHAP plots saved successfully.")

    except Exception as exc:

        print(f"[__main__] SHAP explanation failed (non-fatal): {exc}")

    # ========================================================================
    # DONE
    # ========================================================================

    print("\n" + "=" * 70)
    print(" SVM-RFE + Bayesian-Optimised SVM — Complete")
    print("=" * 70)

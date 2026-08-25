"""
fsvm.py
-------
Asymmetric Fuzzy Support Vector Machine (FSVM) module for the Online
Shoppers Purchasing Intention classification task — training, fuzzy membership
assignment, cost-sensitive weighting, hyperparameter tuning via
RandomizedSearchCV, probability calibration, OOF threshold optimisation, and
SHAP model interpretability.

Methodology (Lin & Wang, 2002 style)
--------------------------------------
Traditional SVM treats every training sample equally.  FSVM assigns a
**fuzzy membership** s_i in (0, 1] to each sample, reflecting how
"representative" it is of its own class.  Samples far from their class
centroid (potential noise / borderline points) receive low membership and
are effectively down-weighted during training.

Parameterised membership formula
---------------------------------
    raw_s_i = 1 - fuzzy_strength * (dist_i / (max_dist_in_class + eps))
    s_i     = clip(raw_s_i, membership_floor, 1.0)

``fuzzy_strength`` controls how aggressively distant points are suppressed
(1.0 = classic Lin & Wang; <1.0 = softer suppression).
``membership_floor`` prevents borderline minority (Purchase) samples from
being driven to near-zero weight, protecting recall on the minority class.

Asymmetric class cost
---------------------
Because Revenue=True is the minority class (~15.5 %), an asymmetric
penalty is applied so that misclassifying a purchaser costs more:

    class_cost(y_i) = cost_ratio   if y_i == 1
                    = 1.0          if y_i == 0

If cost_ratio is None it is auto-inferred as (n_majority / n_minority).

Final per-sample weight
-----------------------
    sample_weight_i = s_i * class_cost(y_i)

This vector is passed to ``SVC.fit(X, y, sample_weight=...)`` with an RBF
kernel.  Probability calibration uses CalibratedClassifierCV(ensemble=False).

Key changes vs. original
--------------------------
* Removed hardcoded C=10.0, gamma=0.01, and fixed cost_ratio.
* FuzzySVM now exposes fuzzy_strength and membership_floor as tunable params.
* compute_fuzzy_membership() accepts fuzzy_strength and membership_floor.
* train_fsvm() uses RandomizedSearchCV over C, gamma, cost_ratio,
  fuzzy_strength, and membership_floor.
* OOF threshold scanner (np.arange(0.20, 0.61, 0.02)) maximises F1 on
  cross-validation data; the frozen threshold is applied to the test set.
* import time; _SCRIPT_START recorded immediately; wall-clock duration
  printed at the very end.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

_SCRIPT_START = time.perf_counter()  # record start immediately

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Register module alias so pickle can serialise FuzzySVM without PicklingError
sys.modules["models.fsvm"] = sys.modules[__name__]

import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.data_preprocessing import (
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


# ============================================================================
# FUZZY MEMBERSHIP COMPUTATION
# ============================================================================


def compute_fuzzy_membership(
    X: np.ndarray,
    y: np.ndarray,
    fuzzy_strength: float = 1.0,
    membership_floor: float = 0.05,
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute distance-based fuzzy membership for every training sample.

    Parameters
    ----------
    X               : Pre-processed feature array, shape (n, d).
    y               : Integer class labels, shape (n,).
    fuzzy_strength  : Controls suppression intensity for outlier samples.
                      1.0 = classic Lin & Wang (full suppression);
                      <1.0 = softer; >1.0 = stronger.
    membership_floor: Lower bound applied via np.clip so that no sample,
                      especially minority-class borderline points, is driven
                      to effectively zero weight.  Typical values: 0.05, 0.1,
                      0.2.
    eps             : Small constant to avoid division by zero.

    Returns
    -------
    membership : np.ndarray of shape (n,) with values in
                 [membership_floor, 1.0].
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    membership = np.ones(len(X), dtype=np.float64)

    for label in np.unique(y):
        mask = y == label
        X_class = X[mask]

        centroid  = X_class.mean(axis=0)
        distances = np.linalg.norm(X_class - centroid, axis=1)
        max_dist  = distances.max()

        raw_s = 1.0 - fuzzy_strength * (distances / (max_dist + eps))
        s = np.clip(raw_s, membership_floor, 1.0)
        membership[mask] = s

    return membership


# ============================================================================
# ASYMMETRIC CLASS COST COMPUTATION
# ============================================================================


def compute_class_costs(
    y: np.ndarray,
    cost_ratio: Optional[float] = None,
) -> np.ndarray:
    """Compute per-sample class cost for asymmetric penalisation.

    Parameters
    ----------
    y           : Integer class labels (0 = majority, 1 = minority).
    cost_ratio  : Penalty multiplier for the minority class (y==1).
                  If None, auto-inferred as n_majority / n_minority.

    Returns
    -------
    costs : np.ndarray of shape (n,).
    """
    y = np.asarray(y)

    n_majority = int(np.sum(y == 0))
    n_minority = int(np.sum(y == 1))

    if cost_ratio is None:
        cost_ratio = n_majority / max(n_minority, 1)

    costs = np.where(y == 1, cost_ratio, 1.0)
    return costs


# ============================================================================
# FUZZY SVM CUSTOM ESTIMATOR
# ============================================================================


class FuzzySVM(BaseEstimator, ClassifierMixin):
    """Asymmetric Fuzzy SVM classifier (Lin & Wang, 2002 style).

    Hyperparameters
    ---------------
    C               : SVC regularisation parameter.
    gamma           : RBF kernel bandwidth.
    cost_ratio      : Asymmetric class penalty for the minority class.
                      None → auto-inferred from class counts.
    fuzzy_strength  : Strength of distance-based suppression in [0, ∞).
                      1.0 = classic Lin & Wang.
    membership_floor: Minimum membership value after clipping; prevents
                      borderline minority samples from getting near-zero
                      weight.  Typical range: 0.05 – 0.2.
    eps             : Numerical stability epsilon.
    random_state    : RNG seed for SVC.
    """

    __module__ = "models.fsvm"

    def __init__(
        self,
        C: float = 1.0,
        gamma: float = 0.1,
        cost_ratio: Optional[float] = None,
        fuzzy_strength: float = 1.0,
        membership_floor: float = 0.05,
        eps: float = 1e-10,
        random_state: int = 42,
    ) -> None:
        self.C                = C
        self.gamma            = gamma
        self.cost_ratio       = cost_ratio
        self.fuzzy_strength   = fuzzy_strength
        self.membership_floor = membership_floor
        self.eps              = eps
        self.random_state     = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FuzzySVM":
        """Fit Fuzzy SVM with parameterised distance-based sample weights."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        # 1. Fuzzy membership (parameterised strength + floor)
        self.fuzzy_membership_ = compute_fuzzy_membership(
            X,
            y,
            fuzzy_strength=self.fuzzy_strength,
            membership_floor=self.membership_floor,
            eps=self.eps,
        )

        # 2. Asymmetric class cost
        class_costs = compute_class_costs(y, cost_ratio=self.cost_ratio)

        # 3. Combined per-sample weight
        self.sample_weights_ = self.fuzzy_membership_ * class_costs

        # 4. Fit underlying SVC
        self.svc_ = SVC(
            kernel="rbf",
            C=self.C,
            gamma=self.gamma,
            random_state=self.random_state,
            max_iter=-1,
            tol=1e-3,
        )
        self.svc_.fit(X, y, sample_weight=self.sample_weights_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X = np.asarray(X, dtype=np.float64)
        return self.svc_.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute signed distance to the separating hyperplane."""
        X = np.asarray(X, dtype=np.float64)
        return self.svc_.decision_function(X)


# ============================================================================
# PIPELINE BUILDER
# ============================================================================


def build_fsvm_pipeline(
    C: float = 1.0,
    gamma: float = 0.1,
    cost_ratio: Optional[float] = None,
    fuzzy_strength: float = 1.0,
    membership_floor: float = 0.05,
    eps: float = 1e-10,
    random_state: int = 42,
) -> Pipeline:
    """Build a sklearn Pipeline: preprocessor → FuzzySVM.

    Parameters mirror FuzzySVM.__init__ for direct use inside
    RandomizedSearchCV via set_params().
    """
    preprocessor = build_preprocessor(scale_numerical=True)
    fsvm = FuzzySVM(
        C=C,
        gamma=gamma,
        cost_ratio=cost_ratio,
        fuzzy_strength=fuzzy_strength,
        membership_floor=membership_floor,
        eps=eps,
        random_state=random_state,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("fsvm", fsvm)])


# ============================================================================
# HYPERPARAMETER SEARCH DISTRIBUTIONS
# ============================================================================


def _make_fsvm_param_distributions() -> list[dict]:
    """Return the RandomizedSearchCV param-distribution list for FSVM.

    Covers
    ------
    C               : log-uniform in [0.1, 500]
    gamma           : log-uniform in [1e-4, 1]
    cost_ratio      : uniform in [1.0, 7.0] — minority-class penalty multiplier
    fuzzy_strength  : uniform in [0.5, 2.0] — membership suppression intensity
    membership_floor: categorical {0.05, 0.10, 0.20} — minority protection floor
    """
    return [
        {
            "fsvm__C":                loguniform(0.1, 500),
            "fsvm__gamma":            loguniform(1e-4, 1),
            "fsvm__cost_ratio":       uniform(1.0, 6.0),   # samples in [1.0, 7.0]
            "fsvm__fuzzy_strength":   uniform(0.5, 1.5),   # samples in [0.5, 2.0]
            "fsvm__membership_floor": [0.05, 0.10, 0.20],
        }
    ]


# ============================================================================
# OOF THRESHOLD SCANNER
# ============================================================================


def find_optimal_threshold_oof(
    raw_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    metric: str = "f1",
    cv: int = 5,
    random_state: int = 42,
) -> float:
    """Find the optimal decision threshold via Out-Of-Fold (OOF) probabilities.

    For each stratified fold, a *fresh clone* of raw_pipeline is calibrated on
    the in-fold data and produces probabilities for the held-out fold.  The
    pooled OOF probabilities are scanned across ``thresholds`` to find the
    cutoff that maximises ``metric``.  This threshold is then *frozen* and
    applied to the untouched test set — no test-set information leaks.

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
    skf   = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    print(
        f"\n[OOF Threshold Scanner] {cv}-fold OOF | "
        f"{len(thresholds)} thresholds | optimising '{metric}' ..."
    )

    oof_proba = np.zeros(len(y_arr), dtype=np.float64)

    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_arr)):
        X_fold_tr  = X_train.iloc[tr_idx]
        y_fold_tr  = y_arr[tr_idx]
        X_fold_val = X_train.iloc[val_idx]

        fold_pipe = clone(raw_pipeline)
        fold_cal  = CalibratedClassifierCV(estimator=fold_pipe, ensemble=False, cv="prefit")
        fold_pipe.fit(X_fold_tr, y_fold_tr)
        fold_cal.fit(X_fold_tr, y_fold_tr)

        oof_proba[val_idx] = fold_cal.predict_proba(X_fold_val)[:, 1]
        print(f"  Fold {fold_idx + 1}/{cv} completed.")

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


# ============================================================================
# MAIN TRAINING ROUTINE  (RandomizedSearchCV)
# ============================================================================


def train_fsvm(
    X_train,
    y_train,
    scoring: str = "average_precision",
    cv: int = 5,
    n_iter: int = 40,
    random_state: int = 42,
    output_path: str | Path | None = None,
    verbose: int = 2,
    n_jobs: int = -2,
    threshold_metric: str = "f1",
    oof_cv: int = 5,
) -> tuple[CalibratedClassifierCV, dict]:
    """Tune FSVM hyperparameters with RandomizedSearchCV, calibrate, and
    find an optimal decision threshold via OOF scanning.

    Search space
    ------------
    C               : log-uniform [0.1, 500]
    gamma           : log-uniform [1e-4, 1]
    cost_ratio      : uniform [1.0, 7.0]
    fuzzy_strength  : uniform [0.5, 2.0]
    membership_floor: {0.05, 0.10, 0.20}

    Parameters
    ----------
    X_train          : Training features (raw DataFrame, pre-transform).
    y_train          : Training labels.
    scoring          : Search metric (default: 'average_precision' = PR-AUC).
    cv               : CV folds for hyperparameter search.
    n_iter           : RandomizedSearchCV iterations (default: 40).
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
          'search'            – RandomizedSearchCV object
          'best_params'       – best hyperparameters dict
          'best_score'        – best CV score (PR-AUC by default)
          'optimal_threshold' – frozen threshold (float) from OOF scan
    """
    if output_path is None:
        output_path = project_root / "saved_models" / "svm_fsvm.pkl"

    # Build base pipeline (default params; RandomizedSearchCV will override via set_params)
    base_pipeline = build_fsvm_pipeline(random_state=random_state)
    param_dists   = _make_fsvm_param_distributions()
    stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    print("\n" + "=" * 70)
    print(" FSVM Hyperparameter Search  (RandomizedSearchCV)")
    print(f" scoring={scoring!r}  n_iter={n_iter}  cv={cv}")
    print(" Search space:")
    print("   C               : log-uniform [0.1, 500]")
    print("   gamma           : log-uniform [1e-4, 1]")
    print("   cost_ratio      : uniform [1.0, 7.0]")
    print("   fuzzy_strength  : uniform [0.5, 2.0]")
    print("   membership_floor: {0.05, 0.10, 0.20}")
    print("=" * 70)

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_dists,
        n_iter=n_iter,
        scoring=scoring,
        cv=stratified_cv,
        verbose=verbose,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )

    t_search_start = time.perf_counter()
    print("\n[train_fsvm] Running RandomizedSearchCV...")
    search.fit(X_train, y_train)
    t_search_end = time.perf_counter()

    print(f"\n[train_fsvm] Search completed in {t_search_end - t_search_start:.1f} s")
    print(f"[train_fsvm] Best {scoring}  : {search.best_score_:.4f}")
    print(f"[train_fsvm] Best params      : {search.best_params_}")

    raw_best_pipeline = search.best_estimator_

    # ── Probability calibration (prefit on full training set) ────────────────
    calibrated_model = CalibratedClassifierCV(
        estimator=raw_best_pipeline,
        ensemble=False,
        cv="prefit",
    )
    calibrated_model.fit(X_train, y_train)

    # ── OOF threshold scan (training data only — test set untouched) ─────────
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
        "search":            search,
        "best_params":       search.best_params_,
        "best_score":        search.best_score_,
        "optimal_threshold": optimal_threshold,
    }


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # ── 1. Load & split data ─────────────────────────────────────────────────
    data_path = str(project_root / "data" / "raw" / "online_shoppers_intention.csv")
    df = preprocess_data(filepath=data_path)
    X_train, X_test, y_train, y_test = split_dataset(df)

    # ── 2. Tune & save ───────────────────────────────────────────────────────
    save_path = project_root / "saved_models" / "svm_fsvm.pkl"
    model, result = train_fsvm(
        X_train,
        y_train,
        output_path=save_path,
    )

    optimal_threshold = result["optimal_threshold"]

    print(f"\n[main] Best CV PR-AUC          : {result['best_score']:.4f}")
    print(f"[main] Best params             : {result['best_params']}")
    print(f"[main] Frozen decision threshold: {optimal_threshold:.2f}")

    # ── 3. Evaluate at optimal threshold on untouched test set ───────────────
    metrics = evaluate_model_with_threshold(model, X_test, y_test, threshold=optimal_threshold)
    print_metrics("Asymmetric Fuzzy SVM (FSVM)", metrics)

    # ── 4. Persist metrics ───────────────────────────────────────────────────
    metrics_output_path = project_root / "report_assets" / "metrics.json"
    save_metrics("Asymmetric Fuzzy SVM (FSVM)", "fsvm", metrics, metrics_output_path)

    # ── 5. SHAP Interpretability ─────────────────────────────────────────────
    print("\n[SHAP] Generating FSVM SHAP explanation plots...")
    try:
        plot_dir = str(project_root / "report_assets" / "plots")
        generate_shap_explanation(
            model=model,
            X_test=X_test,
            save_dir=plot_dir,
            prefix="fsvm_",
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

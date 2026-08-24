"""
fsvm.py
-------
Asymmetric Fuzzy Support Vector Machine (FSVM) module for the Online
Shoppers Purchasing Intention classification task.

Methodology (Lin & Wang, 2002 style)
--------------------------------------
Traditional SVM treats every training sample equally.  FSVM assigns a
**fuzzy membership** s_i ∈ (0, 1] to each sample, reflecting how
"representative" it is of its own class.  Samples far from their class
centroid (potential noise / borderline points) receive low membership and
are effectively down-weighted during training.

    s_i = 1 − (dist_i / (max_dist_in_class + ε))

where ``dist_i`` is the Euclidean distance from sample *i* to the mean
feature vector (centroid) of its class **in preprocessed space**.

Asymmetric class cost
---------------------
Because Revenue=True is the minority class (~15.5 %), an asymmetric
penalty is applied so that misclassifying a purchaser costs more:

    class_cost(y_i) = C_minority   if y_i == 1
                    = C_majority   if y_i == 0

where C_minority / C_majority ≈ majority_count / minority_count (or a
user-specified ratio).

Final per-sample weight
-----------------------
    sample_weight_i = s_i × class_cost(y_i)

This vector is passed to ``SVC.fit(X, y, sample_weight=...)`` — no
custom kernel is needed; the standard RBF kernel is used.

Why SMOTE is NOT used
---------------------
FSVM's fuzzy + asymmetric weighting directly addresses class imbalance
at the optimisation level.  Combining it with SMOTE is redundant (and
potentially harmful, as synthetic samples would receive fuzzy
memberships based on distances to centroids that were computed before
oversampling).  The pipeline therefore omits SMOTE entirely.

Probability calibration
-----------------------
``SVC`` is created **without** the deprecated ``probability`` parameter
(sklearn ≥ 1.9).  Probabilities are obtained by wrapping the fitted
model in ``CalibratedClassifierCV(ensemble=False)``, identical to the
approach used in ``svm_model.py``.

Hyperparameters
---------------
* ``C`` and ``gamma`` are **not** re-tuned here.  They are accepted as
  arguments — the caller is expected to pass the best values already
  found by ``svm_model.py``'s ``RandomizedSearchCV``.

Usage
-----
    from models.fsvm import FuzzySVM, build_pipeline

    pipeline = build_pipeline(
        params={"C": 10.0, "gamma": 0.01},
        preprocessor=preprocessor,
    )
    pipeline.fit(X_train, y_train)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

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
    preprocess_data,
)

from src.utils import (
    evaluate_model,
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
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute distance-based fuzzy membership for every training sample.

    For each sample *i* the membership is:

        s_i = 1 − (dist_i / (max_dist_in_class + ε))

    where ``dist_i`` is the Euclidean distance from *i* to the centroid
    (mean feature vector) of its own class, and ``max_dist_in_class`` is
    the maximum such distance observed for that class.

    The result is clipped to (ε, 1] so that no sample has zero weight
    (which would cause numerical issues in SVC).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        **Preprocessed** (scaled / encoded) training features.
    y : ndarray of shape (n_samples,)
        Binary class labels (0 / 1).
    eps : float, default=1e-10
        Small constant to avoid division by zero and to ensure a
        strictly positive lower bound on membership.

    Returns
    -------
    membership : ndarray of shape (n_samples,)
        Fuzzy membership values in (0, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    membership = np.ones(len(X), dtype=np.float64)

    for label in np.unique(y):
        mask = y == label
        X_class = X[mask]

        # Class centroid (mean feature vector)
        centroid = X_class.mean(axis=0)

        # Euclidean distance from each sample to its class centroid
        distances = np.linalg.norm(X_class - centroid, axis=1)

        max_dist = distances.max()

        # s_i = 1 − (dist_i / (max_dist + ε))
        s = 1.0 - (distances / (max_dist + eps))

        # Clip to (eps, 1] — avoids zero-weight samples
        s = np.clip(s, eps, 1.0)

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

    The minority class (label=1, Revenue=True) receives a higher cost
    than the majority class (label=0) so that misclassifying a purchaser
    is penalised more heavily.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Binary class labels (0 / 1).
    cost_ratio : float or None, default=None
        Ratio ``C_minority / C_majority``.  If ``None``, this is
        automatically set to ``majority_count / minority_count``
        (equivalent to sklearn's ``class_weight='balanced'`` scaling).

    Returns
    -------
    costs : ndarray of shape (n_samples,)
        Per-sample class cost.  Majority-class samples get 1.0;
        minority-class samples get ``cost_ratio``.
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
    __module__ = "models.fsvm"

    """Asymmetric Fuzzy SVM classifier (Lin & Wang, 2002 style).

    This estimator wraps ``sklearn.svm.SVC(kernel="rbf")`` and injects
    per-sample weights computed from **fuzzy membership × class cost**
    into the ``sample_weight`` argument of ``SVC.fit()``.

    It is designed to be placed as the **last step** of a
    ``sklearn.pipeline.Pipeline`` (after a ``ColumnTransformer``
    preprocessor).  The ``fit()`` method receives already-transformed
    features and computes fuzzy memberships in that preprocessed space.

    Parameters
    ----------
    C : float, default=10.0
        Regularisation parameter for the SVM.
    gamma : float, default=0.01
        Kernel coefficient for the RBF kernel.
    cost_ratio : float or None, default=None
        Ratio ``C_minority / C_majority``.  ``None`` → auto-compute
        from class frequencies (≈ ``majority / minority``).
    eps : float, default=1e-10
        Small constant for fuzzy membership computation.
    random_state : int, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    svc_ : SVC
        The fitted SVC instance.
    classes_ : ndarray
        Unique class labels observed during ``fit``.
    fuzzy_membership_ : ndarray
        Fuzzy memberships computed during the last ``fit`` call.
    sample_weights_ : ndarray
        Final sample weights (membership × class cost) used during the
        last ``fit`` call.
    """

    def __init__(
        self,
        C: float = 10.0,
        gamma: float = 0.01,
        cost_ratio: Optional[float] = None,
        eps: float = 1e-10,
        random_state: int = 42,
    ) -> None:
        self.C = C
        self.gamma = gamma
        self.cost_ratio = cost_ratio
        self.eps = eps
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "FuzzySVM":
        """Fit the Fuzzy SVM on preprocessed training data.

        Steps:
            1. Compute fuzzy membership s_i for each sample.
            2. Compute asymmetric class cost for each sample.
            3. Multiply: sample_weight_i = s_i × class_cost_i.
            4. Fit ``SVC(kernel="rbf")`` with ``sample_weight``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Preprocessed training features (already scaled / encoded).
        y : array-like of shape (n_samples,)
            Binary target labels (0 / 1).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        # ── 1. Fuzzy membership ────────────────────────────────────────
        self.fuzzy_membership_ = compute_fuzzy_membership(
            X,
            y,
            eps=self.eps,
        )

        # ── 2. Asymmetric class cost ───────────────────────────────────
        class_costs = compute_class_costs(
            y,
            cost_ratio=self.cost_ratio,
        )

        # ── 3. Combined sample weight ─────────────────────────────────
        self.sample_weights_ = self.fuzzy_membership_ * class_costs

        # ── 4. Fit SVC ─────────────────────────────────────────────────
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

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for *X*.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        X = np.asarray(X, dtype=np.float64)
        return self.svc_.predict(X)

    # ------------------------------------------------------------------
    # Decision function (required by CalibratedClassifierCV)
    # ------------------------------------------------------------------

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute signed distance to the separating hyperplane.

        Required by ``CalibratedClassifierCV`` to produce calibrated
        probability estimates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        scores : ndarray of shape (n_samples,) for binary classification
        """
        X = np.asarray(X, dtype=np.float64)
        return self.svc_.decision_function(X)


# ============================================================================
# BUILD PIPELINE
# ============================================================================


def build_pipeline(
    params: dict,
    preprocessor,
    random_state: int = 42,
) -> Pipeline:
    """Build a sklearn Pipeline: preprocessor → FuzzySVM.

    SMOTE is intentionally **omitted** — the fuzzy membership +
    asymmetric class cost weighting replaces synthetic oversampling.

    Parameters
    ----------
    params : dict
        Must contain at least ``C`` and ``gamma``.
        May optionally contain ``cost_ratio`` and ``eps``.
    preprocessor : sklearn ColumnTransformer
        The shared project preprocessor (from ``build_preprocessor``).
    random_state : int, default=42
        Seed for the classifier.

    Returns
    -------
    Pipeline
        A pipeline exposing ``fit`` / ``predict`` / (after calibration)
        ``predict_proba``, compatible with ``evaluate_model()`` and
        ``print_metrics()`` from ``src/utils.py``.
    """
    fsvm = FuzzySVM(
        C=params["C"],
        gamma=params["gamma"],
        cost_ratio=params.get("cost_ratio"),
        eps=params.get("eps", 1e-10),
        random_state=random_state,
    )

    steps: list[tuple] = [
        ("preprocessor", clone(preprocessor)),
        ("fsvm", fsvm),
    ]

    return Pipeline(steps=steps)


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    # ── 1. Load & split data ────────────────────────────────────────────────
    data_path = project_root / "data" / "raw" / "online_shoppers_intention.csv"
    df = preprocess_data(filepath=data_path)
    X_train, X_test, y_train, y_test = split_dataset(df)

    # ── 2. Best C / gamma from svm_model.py's RandomizedSearchCV ───────────
    # Paste the values printed by svm_model.py's "[train_svm] Best SVM
    # params:" output.  These are NOT re-tuned here.
    BEST_C: float = 10.0
    BEST_GAMMA: float = 0.01

    # ── 3. Build pipeline ──────────────────────────────────────────────────
    preprocessor = build_preprocessor(scale_numerical=True)

    final_params: dict = {
        "C": BEST_C,
        "gamma": BEST_GAMMA,
        # cost_ratio=None → auto-compute from class frequencies
    }

    final_pipeline = build_pipeline(
        params=final_params,
        preprocessor=preprocessor,
        random_state=42,
    )

    # ── 4. Train final model on the full training set ───────────────────────
    print("\n" + "=" * 70)
    print(" Training Asymmetric Fuzzy SVM (FSVM)")
    print("=" * 70)

    print(f"C (fixed)        : {BEST_C}")
    print(f"gamma (fixed)    : {BEST_GAMMA}")
    print(f"cost_ratio       : auto (majority / minority)")
    print(f"SMOTE            : disabled (replaced by fuzzy weighting)")
    print("=" * 70)

    t_start = time.perf_counter()

    print("\n[__main__] Fitting FSVM on full training data...")
    final_pipeline.fit(X_train, y_train)

    t_end = time.perf_counter()
    train_duration = t_end - t_start

    # Print fuzzy-membership statistics for diagnostics
    fsvm_step: FuzzySVM = final_pipeline.named_steps["fsvm"]

    print(f"\n[__main__] Fuzzy membership statistics:")
    print(f"    min  = {fsvm_step.fuzzy_membership_.min():.6f}")
    print(f"    max  = {fsvm_step.fuzzy_membership_.max():.6f}")
    print(f"    mean = {fsvm_step.fuzzy_membership_.mean():.6f}")
    print(f"    std  = {fsvm_step.fuzzy_membership_.std():.6f}")

    print(f"\n[__main__] Sample weight statistics:")
    print(f"    min  = {fsvm_step.sample_weights_.min():.6f}")
    print(f"    max  = {fsvm_step.sample_weights_.max():.6f}")
    print(f"    mean = {fsvm_step.sample_weights_.mean():.6f}")
    print(f"    std  = {fsvm_step.sample_weights_.std():.6f}")

    # ── 5. Wrap in CalibratedClassifierCV for predict_proba ─────────────────
    calibrated_model = CalibratedClassifierCV(
        estimator=final_pipeline,
        ensemble=False,
    )
    calibrated_model.fit(X_train, y_train)

    # ── 6. Evaluate on the held-out test set ────────────────────────────────
    metrics = evaluate_model(calibrated_model, X_test, y_test)
    print_metrics("Asymmetric Fuzzy SVM (FSVM)", metrics)

    # ── 7. Save model ──────────────────────────────────────────────────────
    save_path = project_root / "saved_models" / "svm_fsvm.pkl"
    save_model(calibrated_model, save_path)

    # ── 8. Save metrics to metrics.json ─────────────────────────────────────
    metrics_output_path = project_root / "report_assets" / "metrics.json"
    save_metrics("Asymmetric Fuzzy SVM (FSVM)", "fsvm", metrics, metrics_output_path)

    # ── 9. Print training duration ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f" FSVM Training Duration: {train_duration:.2f} seconds")
    print("=" * 70)

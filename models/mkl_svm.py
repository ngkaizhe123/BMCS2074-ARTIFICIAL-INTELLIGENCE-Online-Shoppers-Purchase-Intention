"""
mkl_svm.py
----------
Multiple Kernel Learning (MKL) SVM module for the Online Shoppers
Purchasing Intention classification task.

This model blends a **linear** and **RBF** kernel into a single combined
Gram matrix and trains an SVM on the result via ``kernel="precomputed"``.

Combined kernel
---------------
    K(x, x') = lam * linear_kernel(x, x')
             + (1 - lam) * rbf_kernel(x, x', gamma)

where ``lam`` in [0, 1] controls the blend ratio.  When ``lam = 1`` the
model reduces to a pure linear SVM; when ``lam = 0`` it is a pure RBF SVM.

Tuning strategy
---------------
* **C and gamma** are NOT re-tuned here.  They are accepted as function
  arguments — the caller is expected to pass the best values already
  found by ``svm_model.py``'s ``RandomizedSearchCV``.
* **lam** is the only hyperparameter searched.  A 5-fold stratified
  cross-validation grid search over ``[0.0, 0.25, 0.5, 0.75, 1.0]``
  using mean PR-AUC (``average_precision_score``) selects the best blend.
  SMOTE is applied inside each fold only (no leakage).  That is 5 lambda
  candidates × 5 folds = **25 fits** total — intentionally cheap.

Probability calibration
-----------------------
``SVC(kernel="precomputed")`` does not support the deprecated
``probability`` parameter.  Probabilities are obtained by wrapping the
fitted model in ``CalibratedClassifierCV(ensemble=False)`` (sklearn ≥ 1.9
API), identical to the approach used in ``svm_model.py``.

Limitations (for the report)
----------------------------
* ``kernel="precomputed"`` requires the **full training Gram matrix** to
  be materialised in memory — an (n_train × n_train) dense float64 array.
  For the Online Shoppers dataset (~9 900 training rows after the 80/20
  split) this is ≈ 750 MB, which is manageable but would not scale to
  datasets with hundreds of thousands of rows.
* At **prediction time** the model must recompute the kernel between every
  test sample and every stored training sample, so inference cost scales
  linearly with training-set size — O(n_test × n_train × d).

Usage
-----
    from models.mkl_svm import tune_mkl_lambda, build_pipeline, HybridKernelSVC

    best_lam, cv_results = tune_mkl_lambda(
        X_train, y_train, best_C=10.0, best_gamma=0.01,
    )
    pipeline = build_pipeline(
        params={"C": 10.0, "gamma": 0.01, "lam": best_lam},
        preprocessor=preprocessor,
    )
    pipeline.fit(X_train, y_train)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.model_selection import StratifiedKFold
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
    get_smote,
    preprocess_data,
)

from src.utils import (
    evaluate_model,
    print_metrics,
    save_model,
)


# ============================================================================
# HYBRID KERNEL SVC ESTIMATOR
# ============================================================================


class HybridKernelSVC(BaseEstimator, ClassifierMixin):
    """SVM classifier with a convex combination of linear and RBF kernels.

    The combined kernel is defined as::

        K(x, x') = lam * linear_kernel(x, x')
                 + (1 - lam) * rbf_kernel(x, x', gamma)

    Internally this class computes the combined Gram matrix and delegates
    to ``SVC(kernel="precomputed")``.

    Parameters
    ----------
    C : float, default=1.0
        Regularisation parameter for the SVM.
    gamma : float, default=1.0
        Bandwidth parameter for the RBF component of the kernel.
    lam : float, default=0.5
        Blend ratio.  ``lam = 1`` → pure linear; ``lam = 0`` → pure RBF.
    class_weight : dict, "balanced", or None, default=None
        Passed directly to ``SVC``.
    random_state : int or None, default=42
        Random seed for reproducibility.

    Attributes
    ----------
    X_train_ : ndarray of shape (n_samples, n_features)
        Stored training data — required at prediction time to compute the
        kernel between new samples and the training set.
    svc_ : SVC
        The fitted ``SVC(kernel="precomputed")`` instance.
    classes_ : ndarray
        Unique class labels observed during ``fit``.

    Notes
    -----
    * ``kernel="precomputed"`` means the **entire training set** is stored
      (via ``self.X_train_``).  Memory usage is O(n² · 8) bytes for the
      Gram matrix during fit, plus O(n · d) for the stored training data.
    * Prediction cost scales **linearly** with training-set size because
      the kernel must be evaluated between each test sample and all
      training samples: O(n_test × n_train × d).

    These trade-offs should be mentioned in the *Limitations* section of
    the report.
    """

    def __init__(
        self,
        C: float = 1.0,
        gamma: float = 1.0,
        lam: float = 0.5,
        class_weight: Optional[dict | str] = None,
        random_state: int = 42,
    ) -> None:
        self.C = C
        self.gamma = gamma
        self.lam = lam
        self.class_weight = class_weight
        self.random_state = random_state

    # ------------------------------------------------------------------
    # Kernel computation
    # ------------------------------------------------------------------

    def _combined_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute the blended linear + RBF Gram matrix.

        Parameters
        ----------
        X : ndarray of shape (n_X, d)
        Y : ndarray of shape (n_Y, d)

        Returns
        -------
        K : ndarray of shape (n_X, n_Y)
        """
        K_lin = linear_kernel(X, Y)
        K_rbf = rbf_kernel(X, Y, gamma=self.gamma)
        return self.lam * K_lin + (1.0 - self.lam) * K_rbf

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HybridKernelSVC":
        """Fit the hybrid-kernel SVM on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix (already preprocessed / scaled).
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.X_train_ = X.copy()
        self.classes_ = np.unique(y)

        K_train = self._combined_kernel(X, X)

        self.svc_ = SVC(
            kernel="precomputed",
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.svc_.fit(K_train, y)
        return self

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
        K_test = self._combined_kernel(X, self.X_train_)
        return self.svc_.predict(K_test)

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
        K_test = self._combined_kernel(X, self.X_train_)
        return self.svc_.decision_function(K_test)


# ============================================================================
# BUILD PIPELINE
# ============================================================================


def build_pipeline(
    params: dict,
    preprocessor,
    use_smote: bool = True,
    random_state: int = 42,
) -> ImbPipeline:
    """Build an imblearn Pipeline: preprocessor → (optional SMOTE) → HybridKernelSVC.

    Parameters
    ----------
    params : dict
        Must contain at least ``C``, ``gamma``, ``lam``.
        May optionally contain ``class_weight``.
    preprocessor : sklearn ColumnTransformer
        The shared project preprocessor (from ``build_preprocessor``).
    use_smote : bool, default=True
        Whether to include a SMOTE step between preprocessing and the
        classifier.
    random_state : int, default=42
        Seed for SMOTE and the classifier.

    Returns
    -------
    ImbPipeline
        A pipeline exposing ``fit`` / ``predict`` / (after calibration)
        ``predict_proba``, compatible with ``evaluate_model()`` and
        ``print_metrics()`` from ``src/utils.py``.
    """
    mkl_svc = HybridKernelSVC(
        C=params["C"],
        gamma=params["gamma"],
        lam=params["lam"],
        class_weight=params.get("class_weight"),
        random_state=random_state,
    )

    steps: list[tuple] = [("preprocessor", clone(preprocessor))]

    if use_smote:
        steps.append(("smote", get_smote(random_state=random_state)))

    steps.append(("mkl_svc", mkl_svc))

    return ImbPipeline(steps=steps)


# ============================================================================
# LAMBDA GRID SEARCH (PR-AUC, SMOTE inside folds)
# ============================================================================


def tune_mkl_lambda(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_C: float,
    best_gamma: float,
    lam_grid: list[float] | None = None,
    class_weight: Optional[dict | str] = None,
    n_splits: int = 5,
    use_smote: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[float, pd.DataFrame]:
    """Grid-search the kernel-blend ratio ``lam`` using stratified CV.

    Only ``lam`` is searched; ``C`` and ``gamma`` are fixed to the best
    values found by ``svm_model.py``'s ``RandomizedSearchCV``.

    Scoring metric is **mean PR-AUC** (``average_precision_score``),
    which is more informative than ROC-AUC on the imbalanced Online
    Shoppers dataset (~85 / 15 class split).

    SMOTE is applied **inside** each CV fold to prevent data leakage.

    Parameters
    ----------
    X_train : DataFrame
        Raw (un-transformed) training features.
    y_train : Series
        Binary target labels (0 / 1).
    best_C : float
        Best C from ``svm_model.py``'s tuning.
    best_gamma : float
        Best gamma from ``svm_model.py``'s tuning.
    lam_grid : list of float or None
        Lambda candidates.  Defaults to ``[0.0, 0.25, 0.5, 0.75, 1.0]``.
    class_weight : dict, "balanced", or None
        Passed to ``HybridKernelSVC``.
    n_splits : int, default=5
        Number of stratified CV folds.
    use_smote : bool, default=True
        Whether to apply SMOTE inside each fold.
    random_state : int, default=42
        Seed for reproducibility.
    verbose : bool, default=True
        Whether to print progress and results.

    Returns
    -------
    best_lam : float
        The lambda value with the highest mean CV PR-AUC.
    results_df : DataFrame
        Per-lambda mean and std PR-AUC scores.
    """
    if lam_grid is None:
        lam_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

    preprocessor = build_preprocessor(scale_numerical=True)
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    if verbose:
        print("\n" + "=" * 70)
        print(" MKL-SVM Lambda Grid Search")
        print("=" * 70)
        print(f"C (fixed)        : {best_C}")
        print(f"gamma (fixed)    : {best_gamma}")
        print(f"Lambda grid      : {lam_grid}")
        print(f"CV folds         : {n_splits}")
        print(f"Scoring          : PR-AUC (average_precision_score)")
        print(f"SMOTE            : {use_smote}")
        print("=" * 70)

    rows: list[dict] = []

    for lam in lam_grid:
        params = {
            "C": best_C,
            "gamma": best_gamma,
            "lam": lam,
            "class_weight": class_weight,
        }

        fold_scores: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(
            skf.split(X_train, y_train), start=1
        ):
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

            pipeline = build_pipeline(
                params=params,
                preprocessor=preprocessor,
                use_smote=use_smote,
                random_state=random_state,
            )
            pipeline.fit(X_tr, y_tr)

            # Wrap in CalibratedClassifierCV to obtain probabilities
            # for PR-AUC scoring.  ensemble=False wraps without re-fitting
            # (sklearn ≥ 1.9 API — replaces deprecated cv="prefit").
            calibrated = CalibratedClassifierCV(
                estimator=pipeline,
                ensemble=False,
            )
            calibrated.fit(X_tr, y_tr)

            y_prob = calibrated.predict_proba(X_val)[:, 1]
            fold_prauc = average_precision_score(y_val, y_prob)
            fold_scores.append(fold_prauc)

        mean_prauc = float(np.mean(fold_scores))
        std_prauc = float(np.std(fold_scores))

        rows.append(
            {
                "lam": lam,
                "mean_pr_auc": mean_prauc,
                "std_pr_auc": std_prauc,
            }
        )

        if verbose:
            print(
                f"  lam={lam:.2f}  →  "
                f"mean PR-AUC = {mean_prauc:.4f} ± {std_prauc:.4f}"
            )

    results_df = pd.DataFrame(rows).sort_values(
        "mean_pr_auc", ascending=False
    ).reset_index(drop=True)

    best_lam = float(results_df.iloc[0]["lam"])

    if verbose:
        print("-" * 70)
        print(f"  ✓ Best lam = {best_lam:.2f}  "
              f"(mean PR-AUC = {results_df.iloc[0]['mean_pr_auc']:.4f})")
        print("=" * 70)

    return best_lam, results_df


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    # ── 1. Load & split data ────────────────────────────────────────────────
    data_path = project_root / "data" / "raw" / "online_shoppers_intention.csv"
    X_train, X_test, y_train, y_test, _ = preprocess_data(
        filepath=data_path, transform=False
    )

    # ── 2. Best C / gamma from svm_model.py's RandomizedSearchCV ───────────
    # Paste the values printed by svm_model.py's "[train_svm] Best SVM
    # params:" output.  These are NOT re-tuned here.
    BEST_C: float = 10.0
    BEST_GAMMA: float = 0.01

    # ── 3. Grid-search lambda ───────────────────────────────────────────────
    best_lam, cv_results = tune_mkl_lambda(
        X_train=X_train,
        y_train=y_train,
        best_C=BEST_C,
        best_gamma=BEST_GAMMA,
    )

    print("\n[__main__] Lambda CV results:")
    print(cv_results.to_string(index=False))

    # ── 4. Train final model on the full training set ───────────────────────
    preprocessor = build_preprocessor(scale_numerical=True)

    final_params = {
        "C": BEST_C,
        "gamma": BEST_GAMMA,
        "lam": best_lam,
    }

    final_pipeline = build_pipeline(
        params=final_params,
        preprocessor=preprocessor,
        use_smote=True,
        random_state=42,
    )

    print("\n[__main__] Fitting final MKL-SVM on full training data...")
    final_pipeline.fit(X_train, y_train)

    # ── 5. Wrap in CalibratedClassifierCV for predict_proba ─────────────────
    calibrated_model = CalibratedClassifierCV(
        estimator=final_pipeline,
        ensemble=False,
    )
    calibrated_model.fit(X_train, y_train)

    # ── 6. Evaluate on the held-out test set ────────────────────────────────
    metrics = evaluate_model(calibrated_model, X_test, y_test)
    print_metrics("MKL-SVM (Hybrid Kernel)", metrics)

    # ── 7. Save model ──────────────────────────────────────────────────────
    save_path = project_root / "saved_models" / "svm_mkl.pkl"
    save_model(calibrated_model, save_path)

"""
fsvm.py
-------
Asymmetric Fuzzy Support Vector Machine (FSVM) module for the Online
Shoppers Purchasing Intention classification task — training, fuzzy membership
assignment, cost-sensitive weighting, and SHAP model interpretability.

Methodology (Lin & Wang, 2002 style)
--------------------------------------
Traditional SVM treats every training sample equally.  FSVM assigns a
**fuzzy membership** s_i in (0, 1] to each sample, reflecting how
"representative" it is of its own class.  Samples far from their class
centroid (potential noise / borderline points) receive low membership and
are effectively down-weighted during training.

    s_i = 1 - (dist_i / (max_dist_in_class + eps))

where ``dist_i`` is the Euclidean distance from sample *i* to the mean
feature vector (centroid) of its class **in preprocessed space**.

Asymmetric class cost
---------------------
Because Revenue=True is the minority class (~15.5 %), an asymmetric
penalty is applied so that misclassifying a purchaser costs more:

    class_cost(y_i) = C_minority   if y_i == 1
                    = C_majority   if y_i == 0

Final per-sample weight
-----------------------
    sample_weight_i = s_i * class_cost(y_i)

This vector is passed to ``SVC.fit(X, y, sample_weight=...)`` with standard RBF kernel.
Probability Calibration: CalibratedClassifierCV(ensemble=False)
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
from imblearn.pipeline import Pipeline
from sklearn.svm import SVC

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Register module alias in sys.modules so pickle can serialize FuzzySVM without PicklingError
sys.modules["models.fsvm"] = sys.modules[__name__]

from src.data_preprocessing import (
    build_preprocessor,
    get_smote,
    preprocess_data,
    TrainFittedDataCleaner,
    TrainingOutlierFilter,
)
from src.utils import (
    evaluate_model,
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
    eps: float = 1e-10,
) -> np.ndarray:
    """Compute distance-based fuzzy membership for every training sample."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)

    membership = np.ones(len(X), dtype=np.float64)

    for label in np.unique(y):
        mask = y == label
        X_class = X[mask]

        centroid = X_class.mean(axis=0)
        distances = np.linalg.norm(X_class - centroid, axis=1)
        max_dist = distances.max()

        s = 1.0 - (distances / (max_dist + eps))
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
    """Compute per-sample class cost for asymmetric penalisation."""
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

    """Asymmetric Fuzzy SVM classifier (Lin & Wang, 2002 style)."""

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

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "FuzzySVM":
        """Fit Fuzzy SVM with distance-based sample weights."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        # 1. Fuzzy membership
        self.fuzzy_membership_ = compute_fuzzy_membership(
            X,
            y,
            eps=self.eps,
        )

        # 2. Asymmetric class cost
        class_costs = compute_class_costs(
            y,
            cost_ratio=self.cost_ratio,
        )

        # 3. Combined sample weight
        self.sample_weights_ = self.fuzzy_membership_ * class_costs

        # 4. Fit SVC
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
# BUILD PIPELINE & TRAIN
# ============================================================================


def build_pipeline(
    params: dict,
    preprocessor,
    random_state: int = 42,
) -> Pipeline:
    """Build a leakage-safe pipeline: clean -> IQR -> scale -> SMOTE -> FSVM."""
    fsvm = FuzzySVM(
        C=params["C"],
        gamma=params["gamma"],
        cost_ratio=params.get("cost_ratio"),
        eps=params.get("eps", 1e-10),
        random_state=random_state,
    )

    steps: list[tuple] = [
        ("iqr", TrainingOutlierFilter(method="iqr")),
        ("cleaner", TrainFittedDataCleaner()),
        ("preprocessor", clone(preprocessor)),
        ("smote", get_smote(random_state)),
        ("fsvm", fsvm),
    ]

    return Pipeline(steps=steps)


def train_fsvm(
    X_train,
    y_train,
    C: float = 10.0,
    gamma: float = 0.01,
    cost_ratio: Optional[float] = None,
    eps: float = 1e-10,
    random_state: int = 42,
    output_path: str | Path | None = None,
) -> tuple[CalibratedClassifierCV, Pipeline]:
    """Train an Asymmetric Fuzzy SVM (FSVM) and calibrate probabilities."""
    if output_path is None:
        output_path = project_root / "saved_models" / "svm_fsvm.pkl"

    preprocessor = build_preprocessor(scale_numerical=True)
    pipeline = build_pipeline(
        params={"C": C, "gamma": gamma, "cost_ratio": cost_ratio, "eps": eps},
        preprocessor=preprocessor,
        random_state=random_state,
    )

    print("\n" + "=" * 70)
    print(" Training Asymmetric Fuzzy SVM (FSVM)")
    print("=" * 70)
    print(f"C (fixed)        : {C}")
    print(f"gamma (fixed)    : {gamma}")
    print(f"cost_ratio       : auto (majority / minority)")
    print(f"SMOTE            : enabled inside the training pipeline")
    print("=" * 70)

    t_start = time.perf_counter()
    print("\n[train_fsvm] Fitting FSVM on full training data...")
    pipeline.fit(X_train, y_train)
    iqr = pipeline.named_steps["iqr"]
    print(
        f"[train_fsvm] Training rows before/after IQR: "
        f"{iqr.n_samples_before_} -> {iqr.n_samples_after_}; "
        "test rows are never removed."
    )
    t_end = time.perf_counter()

    print(f"[train_fsvm] Fitted in {t_end - t_start:.2f} seconds.")

    # Wrap in CalibratedClassifierCV for probability calibration
    calibrated_model = CalibratedClassifierCV(
        estimator=pipeline,
        ensemble=False,
    )
    calibrated_model.fit(X_train, y_train)
    calibrated_model.leakage_safe_protocol_ = "fixed-split-v1"

    if output_path:
        save_model(calibrated_model, output_path)

    return calibrated_model, pipeline


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # ── 1. Load & split data ────────────────────────────────────────────────
    data_path = str(project_root / "data" / "raw" / "online_shoppers_intention.csv")
    df = preprocess_data(filepath=data_path)
    X_train, X_test, y_train, y_test = split_dataset(df)

    # ── 2. Train & save ─────────────────────────────────────────────────────
    save_path = project_root / "saved_models" / "svm_fsvm.pkl"
    calibrated_model, raw_pipeline = train_fsvm(
        X_train,
        y_train,
        output_path=save_path,
    )

    # ── 3. Evaluate ─────────────────────────────────────────────────────────
    metrics = evaluate_model(calibrated_model, X_test, y_test)
    print_metrics("Asymmetric Fuzzy SVM (FSVM)", metrics)

    # ── 4. Save metrics ─────────────────────────────────────────────────────
    metrics_output_path = project_root / "report_assets" / "metrics.json"
    save_metrics("Asymmetric Fuzzy SVM (FSVM)", "fsvm", metrics, metrics_output_path)

    # ── 5. SHAP Interpretability ───────────────────────────────────────────
    print("\n[SHAP] Generating FSVM SHAP explanation plots...")
    try:
        plot_dir = str(project_root / "report_assets" / "plots")
        generate_shap_explanation(
            model=calibrated_model,
            X_test=X_test,
            save_dir=plot_dir,
            prefix="fsvm_",
            show=False,
        )
        print("[SHAP] Plots saved successfully.")
    except Exception as exc:
        print(f"[SHAP] Skipped: {exc}")

"""
xgboost_model.py
----------------

XGBoost training pipeline with optional PSO hyperparameter optimization.

Main proposed model:
    SMOTE + PSO + XGBoost

PSO optimizes the following XGBoost hyperparameters:
    - n_estimators
    - max_depth
    - learning_rate
    - subsample
    - colsample_bytree
    - min_child_weight
    - gamma
    - reg_lambda

Optimization objective:
    Mean 5-fold Cross-Validation PR-AUC (Average Precision)

Important:
    SMOTE is applied INSIDE each CV training fold to prevent
    data leakage.

Final evaluation is performed only once on the untouched test set.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier

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
# PSO SEARCH SPACE
# ============================================================================

SEARCH_SPACE: dict[str, tuple[float, float, bool]] = {
    # Main boosting parameters
    "n_estimators": (200, 800, True),
    "max_depth": (3, 12, True),
    "learning_rate": (0.005, 0.10, False),
    # Sampling parameters
    "subsample": (0.60, 1.00, False),
    "colsample_bytree": (0.60, 1.00, False),
    # Tree complexity / regularization
    "min_child_weight": (1, 10, True),
    "gamma": (0.00, 0.50, False),
    "reg_lambda": (0.10, 10.00, False),
}


# ============================================================================
# Decode PSO particle
# ============================================================================


def _decode(position: np.ndarray) -> dict:
    """
    Convert a continuous PSO position vector into valid
    XGBoost hyperparameters.
    """

    params = {}

    for value, (name, (low, high, is_integer)) in zip(
        position,
        SEARCH_SPACE.items(),
    ):
        clipped = np.clip(value, low, high)

        if is_integer:
            params[name] = int(round(clipped))
        else:
            params[name] = float(clipped)

    return params


# ============================================================================
# Build XGBoost pipeline
# ============================================================================


def _build_pipeline(
    params: dict,
    preprocessor,
    use_smote: bool = True,
    scale_pos_weight: Optional[float] = None,
    random_state: int = 42,
) -> ImbPipeline:
    """
    Build an imblearn Pipeline.

    If use_smote=True:
        preprocessing -> SMOTE -> XGBoost

    If use_smote=False:
        preprocessing -> XGBoost with scale_pos_weight
    """

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": random_state,
        "n_jobs": -1,
        **params,
    }

    if scale_pos_weight is not None:
        xgb_params["scale_pos_weight"] = scale_pos_weight

    xgb = XGBClassifier(**xgb_params)

    steps = [
        ("preprocessor", clone(preprocessor)),
    ]

    if use_smote:
        steps.append(("smote", get_smote()))

    steps.append(("xgb", xgb))

    return ImbPipeline(steps=steps)


# ============================================================================
# PSO FITNESS FUNCTION
# ============================================================================


def _fitness(
    position: np.ndarray,
    X_train,
    y_train,
    preprocessor,
    use_smote: bool = True,
    n_splits: int = 5,
    random_state: int = 42,
) -> float:
    """
    Calculate mean CV PR-AUC for one PSO particle.

    Higher PR-AUC = better particle.

    SMOTE is performed ONLY on each fold's training data.
    Validation data is never SMOTE-resampled.
    """

    params = _decode(position)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    # Calculate scale_pos_weight only when SMOTE is disabled
    scale_pos_weight = None

    if not use_smote:
        class_counts = y_train.value_counts()

        if 0 not in class_counts.index or 1 not in class_counts.index:
            raise ValueError("y_train must contain binary labels 0 and 1.")

        scale_pos_weight = class_counts[0] / class_counts[1]

    fold_scores = []

    for fold_number, (train_idx, val_idx) in enumerate(
        skf.split(X_train, y_train),
        start=1,
    ):

        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]

        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        pipeline = _build_pipeline(
            params=params,
            preprocessor=preprocessor,
            use_smote=use_smote,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
        )

        # IMPORTANT:
        #
        # If SMOTE is enabled:
        #     preprocessing -> SMOTE -> XGBoost
        #
        # SMOTE only sees X_tr/y_tr.
        #
        # X_val remains completely untouched.
        pipeline.fit(
            X_tr,
            y_tr,
        )

        # Probability prediction is required for PR-AUC
        val_proba = pipeline.predict_proba(X_val)[:, 1]

        fold_pr_auc = average_precision_score(
            y_val,
            val_proba,
        )

        fold_scores.append(fold_pr_auc)

    return float(np.mean(fold_scores))


# ============================================================================
# PSO RESULT
# ============================================================================


@dataclass
class PSOResult:
    best_params: dict
    best_score: float
    history: list = field(default_factory=list)


# ============================================================================
# PSO SEARCH
# ============================================================================


def pso_search_xgb(
    X_train,
    y_train,
    preprocessor,
    use_smote: bool = True,
    n_particles: int = 10,
    n_iterations: int = 10,
    n_splits: int = 5,
    w: float = 0.6,
    c1: float = 1.5,
    c2: float = 1.5,
    random_state: int = 42,
    verbose: bool = True,
) -> PSOResult:
    """
    Particle Swarm Optimization for XGBoost.

    Objective:
        Maximize mean CV PR-AUC.

    PSO parameters:
        n_particles
        n_iterations
        w
        c1
        c2
    """

    rng = np.random.default_rng(random_state)

    n_dims = len(SEARCH_SPACE)

    bounds_low = np.array(
        [value[0] for value in SEARCH_SPACE.values()],
        dtype=float,
    )

    bounds_high = np.array(
        [value[1] for value in SEARCH_SPACE.values()],
        dtype=float,
    )

    # ------------------------------------------------------------------------
    # Initialize particles
    # ------------------------------------------------------------------------

    positions = rng.uniform(
        low=bounds_low,
        high=bounds_high,
        size=(n_particles, n_dims),
    )

    # Initial velocities
    velocities = (
        rng.uniform(
            low=-1.0,
            high=1.0,
            size=(n_particles, n_dims),
        )
        * (bounds_high - bounds_low)
        * 0.10
    )

    # ------------------------------------------------------------------------
    # Personal best
    # ------------------------------------------------------------------------

    pbest_positions = positions.copy()

    pbest_scores = np.array(
        [
            _fitness(
                position=particle,
                X_train=X_train,
                y_train=y_train,
                preprocessor=preprocessor,
                use_smote=use_smote,
                n_splits=n_splits,
                random_state=random_state,
            )
            for particle in positions
        ]
    )

    # ------------------------------------------------------------------------
    # Global best
    # ------------------------------------------------------------------------

    gbest_idx = int(np.argmax(pbest_scores))

    gbest_position = pbest_positions[gbest_idx].copy()

    gbest_score = float(pbest_scores[gbest_idx])

    history = [gbest_score]

    # ------------------------------------------------------------------------
    # Initial output
    # ------------------------------------------------------------------------

    if verbose:

        print("\n" + "=" * 70)
        print(" PSO-XGBoost Hyperparameter Optimization")
        print("=" * 70)

        print(f"Particles        : {n_particles}")
        print(f"Iterations       : {n_iterations}")
        print(f"CV folds         : {n_splits}")
        print(f"Optimization     : PR-AUC")
        print(f"SMOTE            : {use_smote}")
        print(f"Initial Best PR-AUC: {gbest_score:.4f}")

        print(f"Initial Best Params:")
        for name, value in _decode(gbest_position).items():
            print(f"    {name}: {value}")

        print("=" * 70)

    # ------------------------------------------------------------------------
    # PSO main loop
    # ------------------------------------------------------------------------

    for iteration in range(n_iterations):

        # Random coefficients
        r1 = rng.uniform(
            0,
            1,
            size=(n_particles, n_dims),
        )

        r2 = rng.uniform(
            0,
            1,
            size=(n_particles, n_dims),
        )

        # --------------------------------------------------------------------
        # Velocity update
        # --------------------------------------------------------------------

        velocities = (
            w * velocities
            + c1 * r1 * (pbest_positions - positions)
            + c2 * r2 * (gbest_position - positions)
        )

        # --------------------------------------------------------------------
        # Position update
        # --------------------------------------------------------------------

        positions = positions + velocities

        # Keep particles inside search space
        positions = np.clip(
            positions,
            bounds_low,
            bounds_high,
        )

        # --------------------------------------------------------------------
        # Evaluate new particles
        # --------------------------------------------------------------------

        scores = np.array(
            [
                _fitness(
                    position=particle,
                    X_train=X_train,
                    y_train=y_train,
                    preprocessor=preprocessor,
                    use_smote=use_smote,
                    n_splits=n_splits,
                    random_state=random_state,
                )
                for particle in positions
            ]
        )

        # --------------------------------------------------------------------
        # Update personal best
        # --------------------------------------------------------------------

        improved = scores > pbest_scores

        pbest_positions[improved] = positions[improved]

        pbest_scores[improved] = scores[improved]

        # --------------------------------------------------------------------
        # Update global best
        # --------------------------------------------------------------------

        current_best_idx = int(np.argmax(pbest_scores))

        current_best_score = float(pbest_scores[current_best_idx])

        if current_best_score > gbest_score:

            gbest_score = current_best_score

            gbest_position = pbest_positions[current_best_idx].copy()

        # --------------------------------------------------------------------
        # Save history
        # --------------------------------------------------------------------

        history.append(gbest_score)

        # --------------------------------------------------------------------
        # Print progress
        # --------------------------------------------------------------------

        if verbose:
            print(f"\n[PSO] Iteration " f"{iteration + 1}/{n_iterations}")
            print(f"      Best CV PR-AUC: " f"{gbest_score:.4f}")
            print(f"      Best Params: " f"{_decode(gbest_position)}")

    # ------------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------------

    best_params = _decode(gbest_position)

    if verbose:

        print("\n" + "=" * 70)
        print(" PSO Optimization Completed")
        print("=" * 70)
        print(f"Best CV PR-AUC: " f"{gbest_score:.4f}")
        print("\nBest Parameters:")
        for name, value in best_params.items():
            print(f"    {name}: {value}")
        print("=" * 70)

    return PSOResult(
        best_params=best_params,
        best_score=gbest_score,
        history=history,
    )


# ============================================================================
# THRESHOLD ANALYSIS
# ============================================================================


def threshold_analysis(
    model,
    X_test,
    y_test,
):
    """
    Evaluate different probability thresholds.

    IMPORTANT:
        This function is for diagnostic/reporting purposes.

        Do NOT use the test set to choose the final production
        threshold. A validation set or out-of-fold predictions
        should be used for final threshold selection.
    """

    probabilities = model.predict_proba(X_test)[:, 1]

    thresholds = [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    print("\n" + "=" * 70)
    print(" Threshold Analysis")
    print("=" * 70)

    print(f"{'Threshold':<12}" f"{'Precision':<12}" f"{'Recall':<12}" f"{'F1':<12}")

    print("-" * 70)

    for threshold in thresholds:

        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.4f}"
            f"{recall:<12.4f}"
            f"{f1:<12.4f}"
        )

    print("=" * 70)


# ============================================================================
# FINAL TEST EVALUATION
# ============================================================================


def evaluate_final_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate the final model on the untouched test set.

    Includes:
        Accuracy
        Precision
        Recall
        F1
        ROC-AUC
        PR-AUC
        Confusion Matrix
    """

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (probabilities >= 0.50).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print("\n" + "=" * 70)
    print(" Model Evaluation Metrics: PSO-XGBoost")
    print("=" * 70)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("=" * 70)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
    }


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================


def train_xgboost(
    X_train,
    y_train,
    use_smote: bool = True,
    output_path: Optional[str | Path] = ("saved_models/xgboost_pso.pkl"),
    use_pso: bool = True,
    pso_n_particles: int = 10,
    pso_n_iterations: int = 10,
    pso_cv_folds: int = 5,
) -> ImbPipeline:
    """
    Train XGBoost using PSO or RandomizedSearchCV.

    Main recommended configuration:

        use_smote=True
        use_pso=True
        pso_n_particles=10
        pso_n_iterations=10
        pso_cv_folds=5
    """

    preprocessor = build_preprocessor(scale_numerical=False)

    # ========================================================================
    # PSO MODE
    # ========================================================================

    if use_pso:

        print("\n")
        print("=" * 70)
        print(" Training PSO-XGBoost")
        print("=" * 70)

        result = pso_search_xgb(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            use_smote=use_smote,
            n_particles=pso_n_particles,
            n_iterations=pso_n_iterations,
            n_splits=pso_cv_folds,
            random_state=42,
            verbose=True,
        )

        best_params = result.best_params

        print("\n[train_xgboost]")
        print(f"PSO Best CV PR-AUC: " f"{result.best_score:.4f}")

        print(f"PSO Best Parameters: " f"{best_params}")

        # --------------------------------------------------------------------
        # Calculate scale_pos_weight only if SMOTE disabled
        # --------------------------------------------------------------------

        scale_pos_weight = None

        if not use_smote:

            class_counts = y_train.value_counts()

            scale_pos_weight = class_counts[0] / class_counts[1]

        # --------------------------------------------------------------------
        # Build final model
        # --------------------------------------------------------------------

        final_pipeline = _build_pipeline(
            params=best_params,
            preprocessor=preprocessor,
            use_smote=use_smote,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        )

        print("\n[train_xgboost] " "Fitting final model on full training data...")

        final_pipeline.fit(
            X_train,
            y_train,
        )

        # --------------------------------------------------------------------
        # Save model
        # --------------------------------------------------------------------

        if output_path:

            save_model(
                final_pipeline,
                output_path,
            )

            print(f"[train_xgboost] " f"Model saved to: {output_path}")

        # --------------------------------------------------------------------
        # Print PSO convergence
        # --------------------------------------------------------------------

        print("\nPSO Convergence History:")

        for iteration, score in enumerate(result.history):
            print(f"Iteration {iteration}: " f"PR-AUC = {score:.4f}")

        return final_pipeline

    # ========================================================================
    # RANDOMIZED SEARCH MODE
    # ========================================================================

    else:

        print("\n")
        print("=" * 70)
        print(" Training XGBoost with RandomizedSearchCV")
        print("=" * 70)

        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

        if use_smote:

            smote = get_smote()

            pipeline = ImbPipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor,
                    ),
                    (
                        "smote",
                        smote,
                    ),
                    (
                        "xgb",
                        xgb,
                    ),
                ]
            )

        else:

            class_counts = y_train.value_counts()

            scale_pos_weight = class_counts[0] / class_counts[1]

            xgb.set_params(scale_pos_weight=scale_pos_weight)

            pipeline = ImbPipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor,
                    ),
                    (
                        "xgb",
                        xgb,
                    ),
                ]
            )

        param_dist = {
            "xgb__n_estimators": [
                100,
                200,
                300,
                400,
                500,
            ],
            "xgb__max_depth": [
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
            ],
            "xgb__learning_rate": [
                0.01,
                0.03,
                0.05,
                0.08,
                0.10,
                0.15,
                0.20,
            ],
            "xgb__subsample": [
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
            ],
            "xgb__colsample_bytree": [
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
            ],
            "xgb__min_child_weight": [
                1,
                2,
                3,
                5,
                7,
                10,
            ],
            "xgb__gamma": [
                0.0,
                0.1,
                0.2,
                0.3,
                0.5,
            ],
        }

        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=30,
            scoring="f1",
            cv=5,
            verbose=1,
            random_state=42,
            n_jobs=-1,
        )

        print("Fitting RandomizedSearchCV...")

        random_search.fit(
            X_train,
            y_train,
        )

        best_model = random_search.best_estimator_

        print("\nBest Parameters:")
        print(random_search.best_params_)

        print(f"Best CV F1: " f"{random_search.best_score_:.4f}")

        if output_path:
            save_model(
                best_model,
                output_path,
            )

        return best_model


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------------
    # Load data
    #
    # transform=False is important because preprocessing is handled
    # inside the pipeline.
    # ------------------------------------------------------------------------

    X_train, X_test, y_train, y_test, _ = preprocess_data(transform=False)

    # ------------------------------------------------------------------------
    # Recommended configuration
    # ------------------------------------------------------------------------

    USE_PSO = True
    USE_SMOTE = True

    model = train_xgboost(
        X_train=X_train,
        y_train=y_train,
        use_smote=USE_SMOTE,
        use_pso=USE_PSO,
        output_path=("saved_models/" "xgboost_pso.pkl"),
        # PSO configuration
        pso_n_particles=10,
        pso_n_iterations=10,
        pso_cv_folds=5,
    )

    # ========================================================================
    # FINAL TEST EVALUATION
    # ========================================================================

    print("\n")
    print("=" * 70)
    print(" Final Test Evaluation")
    print("=" * 70)

    metrics = evaluate_final_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    # Optional: keep existing project evaluation
    try:

        project_metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        print_metrics(
            "XGBoost (PSO)",
            project_metrics,
        )

    except Exception as e:

        print("\n[Evaluation] " f"Project evaluation skipped: {e}")

    # ========================================================================
    # THRESHOLD ANALYSIS
    # ========================================================================

    threshold_analysis(
        model=model,
        X_test=X_test,
        y_test=y_test,
    )

    # ========================================================================
    # SHAP
    # ========================================================================

    print("\n[SHAP] " "Generating SHAP explanations...")

    try:

        generate_shap_explanation(
            model=model,
            X_test=X_test,
            save_dir=("report_assets/plots"),
            prefix="xgboost_pso_",
            show=False,
        )

        print("[SHAP] " "SHAP explanations generated successfully.")

    except Exception as e:

        print("[SHAP] " f"Skipped SHAP generation: {e}")

    print("\n")
    print("=" * 70)
    print(" Training and Evaluation Completed")
    print("=" * 70)

import sys
from pathlib import Path
import optuna
import numpy as np

# This lets the script find the project's other folders (like "src") when
# run directly, e.g. "python models/knn_rf_ensemble_model.py"
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from src.data_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
    get_smote,
    preprocess_data,
    remove_outliers_iqr_train,
)
from src.utils import (
    evaluate_model,
    generate_shap_explanation,
    print_metrics,
    save_model,
    split_dataset,
)

# Fixed seed reused everywhere in this file (data split, SMOTE, PCA, RF, and
# now Optuna's sampler too) so that re-running this script reproduces the
# same tuning result, not just the same data split.
RANDOM_STATE = 42


def evaluate_with_nested_cv(X, y, n_trials=5, outer_splits=3, inner_splits=3):
    """
    [Advanced Technique 1: Nested Cross-Validation]
    This is the "Gold Standard" way to evaluate an AI model in academia.
    It splits the data into two completely separate loops:
    1. Inner Loop: Only used to find the best settings (hyperparameters).
    2. Outer Loop: Only used to test how good those settings actually are on unseen data.
    This scientifically guarantees our final score isn't overly optimistic or biased.

    FIX: `inner_splits` is now actually passed through to
    `train_knn_rf_ensemble`, which uses it to build its inner StratifiedKFold.
    Previously this parameter was accepted but silently ignored, so changing
    it had no effect on the inner tuning loop.
    """
    print(f"\n[Nested CV] Starting {outer_splits}x{inner_splits} Nested Cross-Validation...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE)
    outer_scores = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"[Nested CV] Processing Outer Fold {fold + 1}/{outer_splits}...")
        X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        # Inner loop: tune models using Optuna on X_train_fold
        fold_model = train_knn_rf_ensemble(
            X_train_fold,
            y_train_fold,
            n_trials=n_trials,
            inner_splits=inner_splits,
            output_path=None,
        )

        # Evaluate on the holdout test fold
        preds = fold_model.predict(X_test_fold)
        score = f1_score(y_test_fold, preds, zero_division=0)
        outer_scores.append(score)
        print(f"[Nested CV] Outer Fold {fold + 1} F1 Score: {score:.4f}")

    mean_score = np.mean(outer_scores)
    std_score = np.std(outer_scores)
    print(f"[Nested CV] Completed. Unbiased F1 Score: {mean_score:.4f} \u00b1 {std_score:.4f}\n")
    return mean_score


def train_knn_rf_ensemble(
    X_train,
    y_train,
    n_trials: int = 15,
    inner_splits: int = 3,
    output_path: str | Path | None = "saved_models/knn_rf_ensemble_model.pkl",
):
    """
    Trains a Stacking Ensemble of KNN and Random Forest using Bayesian Optimization (Optuna).

    Args:
        X_train: The training data (customer session features).
        y_train: The correct answers for the training data (did they buy or not).
        n_trials: How many Optuna trials to run for hyperparameter optimization.
            Internally, half of this budget (minimum 2) is used as pure random
            exploration (n_startup_trials) before TPE's model-guided search
            takes over, so very small values (e.g. 2-4) will barely exercise
            TPE at all — prefer at least 8-10 where runtime allows.
        inner_splits: Number of folds used by the inner cross-validation loop
            that scores each Optuna trial. Exposed as a parameter (rather than
            hardcoded) so evaluate_with_nested_cv's inner_splits argument
            actually controls it.
        output_path: Where to save the finished model file. Set to None to skip saving.

    Returns:
        The final trained, stacked, and calibrated ensemble model.
    """
    if X_train is None or len(X_train) == 0:
        raise ValueError("[train_knn_rf_ensemble] X_train is empty — cannot train on no data.")
    if y_train is None or y_train.nunique() < 2:
        raise ValueError("[train_knn_rf_ensemble] y_train must contain at least 2 classes.")

    expected_columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    missing_columns = [col for col in expected_columns if col not in X_train.columns]
    if missing_columns:
        raise ValueError(
            f"[train_knn_rf_ensemble] X_train is missing expected column(s): {missing_columns}"
        )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)

    # ---- Step 1: Optimize KNN Pipeline with PCA using Optuna -----------------
    # [Advanced Technique 2: Bayesian Optimization (Optuna)]
    # Instead of blindly trying every single combination like a brute-force attack (GridSearch),
    # Optuna uses probability to "guess" the best settings, getting smarter with every single trial.
    def objective_knn(trial):
        n_neighbors = trial.suggest_int("n_neighbors", 3, 51, step=2)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        p = trial.suggest_int("p", 1, 2)
        n_components = trial.suggest_int("pca__n_components", 5, 20)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numerical=True)),
                ("smote", get_smote()),
                # [Advanced Technique 3: Dimensionality Reduction (PCA)]
                # KNN gets very confused when there are too many columns ("Curse of Dimensionality").
                # PCA solves this by compressing the columns down into the most mathematically important summaries.
                ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
                ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)),
            ]
        )
        scores = cross_val_score(pipeline, X_train, y_train, cv=inner_cv, scoring="f1", n_jobs=-1)
        return scores.mean()

    # ---- Step 2: Optimize Random Forest using Optuna -------------------------
    def objective_rf(trial):
        n_estimators = trial.suggest_categorical("n_estimators", [50, 100, 150])
        max_depth = trial.suggest_categorical("max_depth", [8, 12, None])

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(scale_numerical=True)),
                ("smote", get_smote()),
                ("rf", RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )),
            ]
        )
        scores = cross_val_score(pipeline, X_train, y_train, cv=inner_cv, scoring="f1", n_jobs=-1)
        return scores.mean()

    # FIX: both studies now use a seeded TPESampler, matching the random_state=42
    # discipline used everywhere else in this codebase (train/test split, SMOTE,
    # PCA, RandomForestClassifier). Without this, re-running the script could
    # select different "best" hyperparameters each time.
    #
    # FIX (n_startup_trials): TPESampler defaults to n_startup_trials=10 — i.e.
    # pure random sampling for the first 10 trials of any study, with the actual
    # TPE model only kicking in afterwards. With n_trials at or below 10 (as in
    # the original version of this script), EVERY trial was random sampling in
    # disguise — Optuna never got to exercise the "Bayesian" part of Bayesian
    # optimization. Scaling n_startup_trials to half of whatever trial budget is
    # requested guarantees a genuine random-exploration phase followed by a
    # genuine model-guided exploitation phase, at any budget size, so the
    # smaller nested-CV inner budget and the larger final-model budget both
    # actually use TPE rather than being random search wearing an Optuna label.
    n_startup_trials = max(2, n_trials // 2)

    print(
        f"[train_knn_rf_ensemble] Starting Optuna tuning for KNN "
        f"({n_trials} trials, {n_startup_trials} random-startup + "
        f"{n_trials - n_startup_trials} TPE-guided)..."
    )
    knn_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=n_startup_trials),
    )
    knn_study.optimize(objective_knn, n_trials=n_trials)
    print(f"[train_knn_rf_ensemble] Best KNN settings: {knn_study.best_params}")

    print(
        f"[train_knn_rf_ensemble] Starting Optuna tuning for Random Forest "
        f"({n_trials} trials, {n_startup_trials} random-startup + "
        f"{n_trials - n_startup_trials} TPE-guided)..."
    )
    rf_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=n_startup_trials),
    )
    rf_study.optimize(objective_rf, n_trials=n_trials)
    print(f"[train_knn_rf_ensemble] Best RF settings: {rf_study.best_params}")

    # Build the best estimators based on Optuna results.
    # FIX: these pipelines are intentionally left UNFITTED here. StackingClassifier
    # clones every estimator it's given and fits the clone itself (both to build
    # each base learner's final fitted copy, and internally via cross-validation
    # to generate the out-of-fold meta-features used to train the meta-learner).
    # Calling .fit() on these pipelines before handing them to StackingClassifier
    # was previously redundant — that fitted state was thrown away and refit
    # from scratch anyway, so those two extra fits (potentially the slowest step
    # for KNN with a large K) were pure wasted computation.
    best_knn = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numerical=True)),
            ("smote", get_smote()),
            ("pca", PCA(n_components=knn_study.best_params["pca__n_components"], random_state=RANDOM_STATE)),
            ("knn", KNeighborsClassifier(
                n_neighbors=knn_study.best_params["n_neighbors"],
                weights=knn_study.best_params["weights"],
                p=knn_study.best_params["p"],
            )),
        ]
    )

    best_rf = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numerical=True)),
            ("smote", get_smote()),
            ("rf", RandomForestClassifier(
                n_estimators=rf_study.best_params["n_estimators"],
                max_depth=rf_study.best_params["max_depth"],
                random_state=RANDOM_STATE,
                n_jobs=-1
            )),
        ]
    )

    X_train, y_train = remove_outliers_iqr_train(X_train, y_train)

    # ---- Step 3: Train the final combined model (Stacking) ------------
    # [Advanced Technique 4: Stacking Ensemble / Meta-Learning]
    # Instead of just taking a fixed average of the KNN and RF votes, we train a
    # third "Manager" model (LogisticRegression) to watch them and dynamically learn
    # who to trust and when to trust them based on their past predictions.
    ensemble = StackingClassifier(
        estimators=[("knn", best_knn), ("rf", best_rf)],
        final_estimator=LogisticRegression(),
        cv=3,
        n_jobs=-1,
    )

    # ---- Step 4: Mathematical Probability Calibration -------------------
    print("[train_knn_rf_ensemble] Applying Platt Scaling (Sigmoid Calibration)...")
    calibrated_ensemble = CalibratedClassifierCV(
        estimator=ensemble, method="sigmoid", cv=3
    )
    calibrated_ensemble.fit(X_train, y_train)

    if output_path:
        try:
            save_model(calibrated_ensemble, output_path)
        except Exception as e:
            print(f"[train_knn_rf_ensemble] Warning: failed to save model to {output_path}: {e}")

    return calibrated_ensemble


if __name__ == "__main__":
    df = preprocess_data()
    X_train, X_test, y_train, y_test = split_dataset(df)

    # 1. Advanced Evaluation: Nested Cross-Validation (Unbiased Estimate)
    # FIX: raised from n_trials=5 to n_trials=8. At 5 trials, n_startup_trials
    # (see train_knn_rf_ensemble) would round down to 2, leaving only 3
    # TPE-guided trials per study — workable, but 8 gives a slightly more
    # meaningful split (4 random + 4 TPE-guided) without materially changing
    # runtime, since this loop already repeats per outer fold.
    print("\n========================================================")
    print("PHASE 1: RIGOROUS NESTED CROSS-VALIDATION EVALUATION")
    print("========================================================")
    evaluate_with_nested_cv(X_train, y_train, n_trials=8, outer_splits=3, inner_splits=3)

    # 2. Train the Final Production Model on the full training set
    # FIX: raised from n_trials=10 to n_trials=20, so this study runs a genuine
    # 10 random-startup + 10 TPE-guided trials, rather than 10 trials that were
    # entirely inside Optuna's default random-startup phase.
    print("\n========================================================")
    print("PHASE 2: TRAINING FINAL PRODUCTION STACKING ENSEMBLE")
    print("========================================================")
    model = train_knn_rf_ensemble(
        X_train, y_train, n_trials=20, output_path="saved_models/knn_rf_ensemble_model.pkl"
    )

    print("\n========================================================")
    print("PHASE 3: HOLDOUT TEST SET PERFORMANCE")
    print("========================================================")
    metrics = evaluate_model(model, X_test, y_test)
    print_metrics("Final Stacking Ensemble", metrics)

    # FIX: re-enabled (was commented out in the version you sent, which would
    # have silently skipped generating the SHAP explanation charts referenced
    # in the report). Remove this block again if disabling SHAP for this
    # version was actually intentional.
    print("\n[SHAP] Generating explanation charts for the final model...")
    try:
        generate_shap_explanation(
            model=model,
            X_test=X_test,
            save_dir="report_assets/plots",
            prefix="knn_rf_",
            show=False,
        )
    except Exception as e:
        print(f"[SHAP] Skip generating explanation charts: {e}")

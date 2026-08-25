"""
xgboost_model.py
-----------------
XGBoost trained on a SMOTE-resampled pipeline, with hyperparameters
chosen by Particle Swarm Optimization (PSO) instead of hand-picking them.

Pipeline:  encode categorical features -> SMOTE -> XGBoost

Why PSO: a swarm of candidate hyperparameter sets ("particles") moves
through the search space. Each particle is pulled toward (a) the best
point it has personally found, and (b) the best point the whole swarm
has found. Over a few iterations the swarm concentrates on promising
hyperparameters, using far fewer model fits than a full grid search.

Each candidate is scored by 5-fold cross-validated PR-AUC (Average
Precision), which is the appropriate metric here since Revenue is
imbalanced (~15.5% Purchase). SMOTE is applied inside each training
fold only — the validation fold is never resampled — so there is no
data leakage into the score.
"""

import sys
from pathlib import Path

import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_preprocessing import (
    TrainFittedDataCleaner,
    build_preprocessor,
    get_smote,
    preprocess_data,
)
from src.utils import (
    evaluate_model,
    generate_shap_explanation,
    print_metrics,
    save_model,
    split_dataset,
)

# Hyperparameters PSO searches over: name -> (low, high, is_integer)
SEARCH_SPACE = {
    "n_estimators": (200, 800, True),
    "max_depth": (3, 12, True),
    "learning_rate": (0.005, 0.10, False),
    "subsample": (0.60, 1.00, False),
    "colsample_bytree": (0.60, 1.00, False),
    "min_child_weight": (1, 10, True),
    "gamma": (0.00, 0.50, False),
    "reg_lambda": (0.10, 10.00, False),
}


def _decode(position):
    """Turn one PSO position vector into a dict of XGBoost hyperparameters."""
    params = {}
    for value, (name, (low, high, is_int)) in zip(position, SEARCH_SPACE.items()):
        value = np.clip(value, low, high)
        params[name] = int(round(value)) if is_int else float(value)
    return params


def _build_pipeline(params, preprocessor):
    """The one pipeline used everywhere: encode -> SMOTE -> XGBoost."""
    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        **params,
    )
    return Pipeline(
        [
            ("cleaner", TrainFittedDataCleaner()),
            ("preprocessor", clone(preprocessor)),
            ("smote", get_smote()),
            ("xgb", xgb),
        ]
    )


def _fitness(position, X_train, y_train, preprocessor, n_splits=5):
    """Mean cross-validated PR-AUC for one candidate hyperparameter set."""
    params = _decode(position)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        pipeline = _build_pipeline(params, preprocessor)
        pipeline.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
        val_proba = pipeline.predict_proba(X_train.iloc[val_idx])[:, 1]
        scores.append(average_precision_score(y_train.iloc[val_idx], val_proba))
    return float(np.mean(scores))


def pso_search(
    X_train,
    y_train,
    preprocessor,
    n_particles=10,
    n_iterations=10,
    w=0.6,
    c1=1.5,
    c2=1.5,
    random_state=42,
):
    """
    Run PSO and return the best hyperparameters found.

    Returns a dict: {"best_params": ..., "best_score": ..., "history": [...]}
    `history` is the best CV PR-AUC seen after each iteration (useful for a
    convergence plot in the report).
    """
    rng = np.random.default_rng(random_state)
    low = np.array([v[0] for v in SEARCH_SPACE.values()])
    high = np.array([v[1] for v in SEARCH_SPACE.values()])
    n_dims = len(SEARCH_SPACE)

    position = rng.uniform(low, high, size=(n_particles, n_dims))
    velocity = rng.uniform(-1, 1, size=(n_particles, n_dims)) * (high - low) * 0.1

    pbest_position = position.copy()
    pbest_score = np.array(
        [_fitness(p, X_train, y_train, preprocessor) for p in position]
    )
    best_idx = int(np.argmax(pbest_score))
    gbest_position, gbest_score = pbest_position[best_idx].copy(), float(
        pbest_score[best_idx]
    )
    history = [gbest_score]
    print(f"[PSO] init  best PR-AUC={gbest_score:.4f}")

    for i in range(n_iterations):
        r1 = rng.uniform(0, 1, position.shape)
        r2 = rng.uniform(0, 1, position.shape)
        velocity = (
            w * velocity
            + c1 * r1 * (pbest_position - position)
            + c2 * r2 * (gbest_position - position)
        )
        position = np.clip(position + velocity, low, high)

        score = np.array(
            [_fitness(p, X_train, y_train, preprocessor) for p in position]
        )
        improved = score > pbest_score
        pbest_position[improved] = position[improved]
        pbest_score[improved] = score[improved]

        if pbest_score.max() > gbest_score:
            best_idx = int(np.argmax(pbest_score))
            gbest_position, gbest_score = pbest_position[best_idx].copy(), float(
                pbest_score[best_idx]
            )

        history.append(gbest_score)
        print(f"[PSO] iter {i + 1}/{n_iterations}  best PR-AUC={gbest_score:.4f}")

    return {
        "best_params": _decode(gbest_position),
        "best_score": gbest_score,
        "history": history,
    }


def train_xgboost(
    X_train, y_train, output_path="saved_models/xgboost_pso.pkl", **pso_kwargs
):
    """Search hyperparameters with PSO, then fit the final pipeline on the full training set."""
    preprocessor = build_preprocessor(scale_numerical=False)

    result = pso_search(X_train, y_train, preprocessor, **pso_kwargs)
    print(f"\nBest CV PR-AUC : {result['best_score']:.4f}")
    print(f"Best params    : {result['best_params']}")

    model = _build_pipeline(result["best_params"], preprocessor)
    model.fit(X_train, y_train)
    model.leakage_safe_protocol_ = "fixed-split-v1"

    if output_path:
        save_model(model, output_path)

    return model, result


def threshold_scan(
    model, X_test, y_test, thresholds=(0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
):
    """
    Print Precision/Recall/F1 at a few candidate probability cutoffs.
    Diagnostic only — pick the final threshold on a validation set, not
    the test set, if you actually change the deployed cutoff.
    """
    proba = model.predict_proba(X_test)[:, 1]
    print(f"\n{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}")
    for t in thresholds:
        pred = (proba >= t).astype(int)
        print(
            f"{t:<12.2f}"
            f"{precision_score(y_test, pred, zero_division=0):<12.4f}"
            f"{recall_score(y_test, pred, zero_division=0):<12.4f}"
            f"{f1_score(y_test, pred, zero_division=0):<12.4f}"
        )


if __name__ == "__main__":
    # XGBoost intentionally uses the shared raw/prepared split with no IQR,
    # no Z-score filtering, and no numerical scaling.
    df = preprocess_data()
    X_train, X_test, y_train, y_test = split_dataset(df)

    model, pso_result = train_xgboost(X_train, y_train, n_particles=10, n_iterations=10)

    metrics = evaluate_model(model, X_test, y_test)
    print_metrics("XGBoost (SMOTE + PSO)", metrics)

    threshold_scan(model, X_test, y_test)

    try:
        generate_shap_explanation(
            model,
            X_test,
            save_dir="report_assets/plots",
            prefix="xgboost_pso_",
            show=False,
        )
    except Exception as e:
        print(f"[SHAP] Skipped: {e}")

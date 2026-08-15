import sys
from pathlib import Path

# Add project root directory to sys.path so 'src' can be imported when running script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from src.data_preprocessing import build_preprocessor, get_smote, preprocess_data
from src.utils import (
    evaluate_model,
    generate_shap_explanation,
    print_metrics,
    save_model,
)


def train_knn_rf_ensemble(
    X_train,
    y_train,
    knn_param_grid: dict | None = None,
    rf_param_grid: dict | None = None,
    weight_candidates: list[int] | None = None,
    output_path: str | Path = "saved_models/knn_rf_ensemble_model.pkl",
):
    """
    Train the KNN module as a soft-voting ensemble of K-Nearest Neighbors
    (instance-based, local decision boundary) and Random Forest (bagged,
    tree-based, global decision boundary). KNN alone underperforms on this
    dataset (CV F1 ~ 0.55); combining it with a tuned Random Forest lifts
    CV F1 to ~0.67-0.68, which is why this ensemble is used as the KNN
    module's final model rather than plain KNN.

    Both base models are tuned independently with their own GridSearchCV
    inside this function (KNN is tuned here directly, not sourced from a
    separate script), and the soft-voting weight given to Random Forest
    (relative to KNN's fixed weight of 1) is then chosen empirically:
    out-of-fold predicted probabilities are computed once for each tuned
    base model via cross_val_predict, then every candidate weight is scored
    analytically against those out-of-fold probabilities to find the weight
    that maximizes F1, without refitting either model per candidate weight.

    Args:
        X_train: Training features (raw DataFrame, preprocessed inside the Pipeline).
        y_train: Training target (0/1).
        knn_param_grid: Hyperparameter grid for KNN's GridSearchCV. Defaults
            to a grid over n_neighbors (odd values 3-41, avoiding tie votes),
            weights, and the Minkowski distance order p.
        rf_param_grid: Hyperparameter grid for Random Forest's GridSearchCV.
            Defaults to a grid over n_estimators and max_depth.
        weight_candidates: RF soft-voting weights to test (KNN weight fixed
            at 1). Defaults to [1..20]; scores plateau well before 20 since a
            large RF weight makes the ensemble converge toward Random
            Forest's predictions alone.
        output_path: Where to save the trained ensemble (.pkl). Pass None to skip saving.

    Returns:
        The fitted VotingClassifier ensemble, using the best-found KNN and
        Random Forest hyperparameters and the empirically best voting weight.

    Raises:
        ValueError: If X_train/y_train are empty or y_train has fewer than 2 classes.
        RuntimeError: If either GridSearchCV fails to fit.
    """
    if X_train is None or len(X_train) == 0:
        raise ValueError(
            "[train_knn_rf_ensemble] X_train is empty — cannot train on no data."
        )
    if y_train is None or y_train.nunique() < 2:
        raise ValueError(
            "[train_knn_rf_ensemble] y_train must contain at least 2 classes."
        )

    if knn_param_grid is None:
        # n_neighbors widened to 61 after testing showed solo KNN's F1 keeps
        # climbing slowly even past 41 (0.545 at n=41 up to 0.560 at n=101).
        # That's a real trend, not noise, but very slow/diminishing (+0.016
        # F1 across 60 more neighbors) — and since the final ensemble weights
        # Random Forest far more heavily than KNN (see weight_candidates
        # below), squeezing out KNN's last few decimal points here has little
        # effect on the ensemble's final score. 61 captures most of the
        # practical gain without chasing marginal returns on the weaker,
        # down-weighted base model.
        knn_param_grid = {
            "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 25, 31, 41, 51, 61],
            "knn__weights": ["uniform", "distance"],
            "knn__p": [1, 2],
        }
    if rf_param_grid is None:
        rf_param_grid = {
            "rf__n_estimators": [200, 300],
            "rf__max_depth": [8, 12, None],
        }
    if weight_candidates is None:
        weight_candidates = list(range(1, 21))

    knn_pipeline_template = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numerical=True)),
            ("smote", get_smote()),
            ("knn", KNeighborsClassifier()),
        ]
    )

    rf_pipeline_template = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numerical=True)),
            ("smote", get_smote()),
            ("rf", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )

    # --- Tune KNN -------------------------------------------------------
    knn_search = GridSearchCV(
        estimator=knn_pipeline_template,
        param_grid=knn_param_grid,
        scoring="f1",
        cv=5,
        verbose=1,
        n_jobs=-1,
    )
    try:
        knn_search.fit(X_train, y_train)
    except Exception as e:
        raise RuntimeError(
            f"[train_knn_rf_ensemble] KNN GridSearchCV failed to fit: {e}"
        ) from e

    best_knn = knn_search.best_estimator_
    print(f"\n[train_knn_rf_ensemble] Best KNN params: {knn_search.best_params_}")

    # --- Tune Random Forest ----------------------------------------------
    rf_search = GridSearchCV(
        estimator=rf_pipeline_template,
        param_grid=rf_param_grid,
        scoring="f1",
        cv=5,
        verbose=1,
        n_jobs=-1,
    )
    try:
        rf_search.fit(X_train, y_train)
    except Exception as e:
        raise RuntimeError(
            f"[train_knn_rf_ensemble] Random Forest GridSearchCV failed to fit: {e}"
        ) from e

    best_rf = rf_search.best_estimator_
    print(f"[train_knn_rf_ensemble] Best RF params: {rf_search.best_params_}")

    # --- Empirically select the soft-voting weight ------------------------
    # Get out-of-fold predicted probabilities for each tuned base model once,
    # then score every candidate weight analytically against them. This
    # avoids refitting the ensemble once per candidate weight.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    knn_oof_proba = cross_val_predict(
        best_knn, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    rf_oof_proba = cross_val_predict(
        best_rf, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]

    best_weight, best_f1 = weight_candidates[0], -1.0
    for w in weight_candidates:
        combined_proba = (knn_oof_proba + w * rf_oof_proba) / (1 + w)
        combined_pred = (combined_proba >= 0.5).astype(int)
        f1 = f1_score(y_train, combined_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_weight = f1, w

    print(
        f"[train_knn_rf_ensemble] Best RF soft-voting weight: {best_weight} "
        f"(out-of-fold F1: {best_f1:.4f})"
    )

    # --- Fit the final ensemble on the full training set -------------------
    # best_knn and best_rf are already fitted on the full X_train by
    # GridSearchCV (refit=True by default); VotingClassifier clones and
    # refits them internally when .fit() is called below.
    ensemble = VotingClassifier(
        estimators=[("knn", best_knn), ("rf", best_rf)],
        voting="soft",
        weights=[1, best_weight],
    )
    ensemble.fit(X_train, y_train)

    if output_path:
        try:
            save_model(ensemble, output_path)
        except Exception as e:
            print(
                f"[train_knn_rf_ensemble] Warning: failed to save model to {output_path}: {e}"
            )

    return ensemble


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = preprocess_data(transform=False)
    model = train_knn_rf_ensemble(
        X_train, y_train, output_path="saved_models/knn_rf_ensemble_model.pkl"
    )
    metrics = evaluate_model(model, X_test, y_test)
    print_metrics("KNN + Random Forest Ensemble", metrics)

    # Generate and save SHAP explanations
    print("\n[SHAP] Generating SHAP explanations for KNN + RF Ensemble...")
    try:
        generate_shap_explanation(
            model=model,
            X_test=X_test,
            save_dir="report_assets/plots",
            prefix="knn_rf_",
            show=False,
        )
    except Exception as e:
        print(f"[SHAP] Skipped SHAP generation: {e}")

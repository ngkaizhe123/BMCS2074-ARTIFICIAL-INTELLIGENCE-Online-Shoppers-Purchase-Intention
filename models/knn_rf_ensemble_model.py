import sys
from pathlib import Path

# Add project root directory to sys.path so 'src' can be imported when running script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from src.data_preprocessing import build_preprocessor, get_smote, preprocess_data
from src.utils import evaluate_model, generate_shap_explanation, print_metrics, save_model


def train_knn_rf_ensemble(
    X_train,
    y_train,
    knn_params: dict | None = None,
    rf_weight: int = 3,
    output_path: str | Path = "saved_models/knn_rf_ensemble_model.pkl",
):
    """
    Bonus/extension module: soft-voting ensemble combining K-Nearest Neighbors
    (instance-based, local decision boundary) with Random Forest (bagged,
    tree-based, global decision boundary). The two algorithms make different
    kinds of errors, which is the theoretical motivation for ensembling them.

    Note: empirical testing on this dataset (see report Discussion) shows this
    ensemble matches, but does not exceed, a standalone tuned Random Forest.
    It is included to demonstrate ensemble methodology and its trade-offs,
    not to replace the individual KNN model used for the 3-way comparison.

    Args:
        X_train: Training features (raw DataFrame, preprocessed inside the Pipeline).
        y_train: Training target (0/1).
        knn_params: Best KNN hyperparameters found by train_knn's GridSearchCV
            (e.g. {"n_neighbors": 21, "weights": "distance", "p": 1}).
            Defaults to that best-found configuration if not provided.
        rf_weight: Soft-voting weight given to Random Forest relative to KNN's
            weight of 1. Higher values lean the ensemble more toward Random
            Forest's predictions (empirically the stronger of the two models).
        output_path: Where to save the trained ensemble (.pkl). Pass None to skip saving.

    Returns:
        The fitted VotingClassifier ensemble.

    Raises:
        ValueError: If X_train/y_train are empty or y_train has fewer than 2 classes.
        RuntimeError: If GridSearchCV fails to fit the Random Forest branch.
    """
    if X_train is None or len(X_train) == 0:
        raise ValueError("[train_knn] X_train is empty — cannot train on no data.")
    if y_train is None or y_train.nunique() < 2:
        raise ValueError("[train_knn] y_train must contain at least 2 classes.")

    if knn_params is None:
        knn_params = {"n_neighbors": 21, "weights": "distance", "p": 1}

    knn_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numerical=True)),
            ("smote", get_smote()),
            ("knn", KNeighborsClassifier(**knn_params)),
        ]
    )

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numerical=True)),
            ("smote", get_smote()),
            ("rf", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )

    # Light tuning for the Random Forest branch only; KNN keeps its
    # already-tuned parameters from train_knn() to avoid re-tuning both
    # models inside a combinatorial grid, which would be very slow.
    rf_param_grid = {
        "rf__n_estimators": [200, 300],
        "rf__max_depth": [8, 12, None],
    }

    rf_search = GridSearchCV(
        estimator=rf_pipeline,
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

    print(f"\n[train_knn_rf_ensemble] Best RF params: {rf_search.best_params_}")

    knn_pipeline.fit(X_train, y_train)

    ensemble = VotingClassifier(
        estimators=[
            ("knn", knn_pipeline),
            ("rf", rf_search.best_estimator_),
        ],
        voting="soft",
        weights=[1, rf_weight],
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

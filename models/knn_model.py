import sys
from pathlib import Path

# Add project root directory to sys.path so 'src' can be imported when running script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from src.data_preprocessing import build_preprocessor, get_smote, preprocess_data
from src.utils import evaluate_model, print_metrics, save_model


def train_knn(
    X_train,
    y_train,
    use_smote: bool = True,
    output_path: str | Path = "saved_models/knn_model.pkl",
):
    """Train KNN Classifier with Pipeline & Grid Search, and save trained model."""
    preprocessor = build_preprocessor(scale_numerical=True)
    knn = KNeighborsClassifier()

    if use_smote:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", get_smote()),
                ("knn", knn),
            ]
        )
    else:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("knn", knn),
            ]
        )

    param_grid = {
        "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=5,
        verbose=2,
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print(f"\n[train_knn] Best KNN params: {grid_search.best_params_}")
    print(f"[train_knn] Best CV F1 score: {grid_search.best_score_:.4f}")

    if output_path:
        save_model(best_model, output_path)

    return best_model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = preprocess_data(transform=False)
    model = train_knn(X_train, y_train, output_path="saved_models/knn_model.pkl")
    metrics = evaluate_model(model, X_test, y_test)
    print_metrics("KNN Classifier", metrics)

import sys
from pathlib import Path

# Add project root directory to sys.path so 'src' can be imported when running script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from src.data_preprocessing import build_preprocessor, preprocess_data
from src.utils import evaluate_model, print_metrics, save_model


def train_svm(
    X_train,
    y_train,
    use_smote: bool = True,
    output_path: str | Path = "saved_models/svm_model.pkl",
):
    """Train SVM Classifier with Pipeline & Grid Search, and save trained model."""
    preprocessor = build_preprocessor(scale_numerical=True)
    svm = SVC(probability=True, random_state=42)

    if use_smote:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("svm", svm),
            ]
        )
    else:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("svm", svm),
            ]
        )

    param_grid = {
        "svm__C": [0.1, 1, 10],
        "svm__kernel": ["rbf", "linear"],
        "svm__gamma": ["scale", "auto"],
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

    print(f"\n[train_svm] Best SVM params: {grid_search.best_params_}")
    print(f"[train_svm] Best CV F1 score: {grid_search.best_score_:.4f}")

    if output_path:
        save_model(best_model, output_path)

    return best_model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = preprocess_data(transform=False)
    model = train_svm(X_train, y_train, output_path="saved_models/svm_model.pkl")
    metrics = evaluate_model(model, X_test, y_test)
    print_metrics("SVM Classifier", metrics)

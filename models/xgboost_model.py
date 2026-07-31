import sys
from pathlib import Path

# Add project root directory to sys.path so 'src' can be imported when running script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from imblearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from src.data_preprocessing import build_preprocessor, get_smote, preprocess_data
from src.utils import evaluate_model, print_metrics, save_model


def train_xgboost(
    X_train,
    y_train,
    use_smote: bool = True,
    output_path: str | Path = "saved_models/xgboost_model.pkl",
):
    """
    Train XGBoost model using imblearn Pipeline and save trained model.

    Pipeline steps:
      1. preprocessor (OneHotEncoder for categorical, passthrough for numerical)
      2. SMOTE (resamples training folds in CV only, avoiding data leakage)
      3. XGBClassifier
    """
    preprocessor = build_preprocessor(scale_numerical=False)

    xgb = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )

    if use_smote:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", get_smote()),
                ("xgb", xgb),
            ]
        )
        param_dist = {
            "xgb__n_estimators": [100, 200, 300],
            "xgb__max_depth": [3, 5, 7],
            "xgb__learning_rate": [0.01, 0.05, 0.1],
            "xgb__subsample": [0.8, 1.0],
            "xgb__colsample_bytree": [0.8, 1.0],
            "xgb__min_child_weight": [1, 3, 5],
            "xgb__gamma": [0, 0.1, 0.3],
        }
    else:
        # Scale pos weight method (alternative to SMOTE)
        scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
        xgb.set_params(scale_pos_weight=scale_pos_weight)

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("xgb", xgb),
            ]
        )
        param_dist = {
            "xgb__n_estimators": [100, 200, 300],
            "xgb__max_depth": [3, 5, 7],
            "xgb__learning_rate": [0.01, 0.05, 0.1],
            "xgb__subsample": [0.8, 1.0],
            "xgb__colsample_bytree": [0.8, 1.0],
            "xgb__min_child_weight": [1, 3, 5],
            "xgb__gamma": [0, 0.1, 0.3],
        }

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=30,
        scoring="f1",
        cv=5,
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    print("\n[train_xgboost] Best Parameters:", random_search.best_params_)
    print(f"[train_xgboost] Best CV F1 Score: {random_search.best_score_:.4f}")

    if output_path:
        save_model(best_model, output_path)

    return best_model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _ = preprocess_data(transform=False)
    model = train_xgboost(
        X_train, y_train, output_path="saved_models/xgboost_model.pkl"
    )
    metrics = evaluate_model(model, X_test, y_test)
    print_metrics("XGBoost Classifier", metrics)

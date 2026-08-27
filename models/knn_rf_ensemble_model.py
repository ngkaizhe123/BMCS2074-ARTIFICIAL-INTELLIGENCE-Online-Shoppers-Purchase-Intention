import sys
from pathlib import Path

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
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

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
    save_metrics,
)


def train_knn_rf_ensemble(
    X_train,
    y_train,
    knn_param_grid: dict | None = None,
    rf_param_grid: dict | None = None,
    output_path: str | Path = "saved_models/knn_rf_ensemble_model.pkl",
):
    """
    This is the project's KNN + Random Forest ensemble model. On its own,
    KNN (K-Nearest Neighbors) doesn't predict very well on this dataset.
    So instead of using KNN by itself, this function combines it with a
    Random Forest model using a Stacking Classifier — a "Manager" model
    (Logistic Regression) that learns how to optimally combine both
    models' predictions. This is much more advanced than simple voting.

    Both models are automatically tuned to find their best settings using
    GridSearchCV. Then a Stacking Classifier learns how to combine them,
    followed by probability calibration and optimal threshold tuning.

    Args:
        X_train: The training data (customer session features).
        y_train: The correct answers for the training data (did they buy or not).
        knn_param_grid: Which KNN settings to try out. If not given, a sensible
            default set of options is used.
        rf_param_grid: Which Random Forest settings to try out. If not given,
            a sensible default set of options is used.

        output_path: Where to save the finished model file. Set to None to skip saving.

    Returns:
        The final trained, stacked, calibrated ensemble model with optimal threshold.

    Raises:
        ValueError: If the training data is missing, missing expected columns,
            or if the target column contains values other than 0/1.
        RuntimeError: If training either model fails.
    """
    # ---- Basic safety checks before we start training -------------------
    if X_train is None or len(X_train) == 0:
        raise ValueError(
            "[train_knn_rf_ensemble] X_train is empty — cannot train on no data."
        )
    if y_train is None or y_train.nunique() < 2:
        raise ValueError(
            "[train_knn_rf_ensemble] y_train must contain at least 2 classes."
        )

    # Check that the data actually has all the columns this model expects
    # (e.g. "Month", "PageValues", etc). If a column is missing, this stops
    # training immediately with a clear message, instead of letting it fail
    # later with a confusing error buried inside the preprocessing step.
    expected_columns = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    missing_columns = [col for col in expected_columns if col not in X_train.columns]
    if missing_columns:
        raise ValueError(
            "[train_knn_rf_ensemble] X_train is missing expected column(s): "
            f"{missing_columns}. Expected columns: {expected_columns}."
        )

    # Check that the target column only contains 0 and 1 (the model expects
    # "did they purchase or not" as 0/1, not True/False, "Yes"/"No", or
    # anything else).
    allowed_labels = {0, 1}
    actual_labels = set(y_train.unique())
    if not actual_labels.issubset(allowed_labels):
        raise ValueError(
            "[train_knn_rf_ensemble] y_train must only contain 0 and 1 "
            f"(found: {sorted(actual_labels)}). Convert your target column to "
            "0/1 (e.g. using .astype(int)) before calling this function."
        )

    # Default settings to try during tuning, if the caller didn't provide their own.
    if knn_param_grid is None:
        knn_param_grid = {
            "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 25, 31, 41, 51, 61],
            "knn__weights": ["uniform", "distance"],
            "knn__p": [1, 2],
        }
    if rf_param_grid is None:
        rf_param_grid = {
            "rf__n_estimators": [100, 150],
            "rf__max_depth": [8, 12, None],
        }

    # A "recipe" for how KNN should process the data before predicting:
    # 1. Convert raw columns into a numeric format the model can use.
    # 2. Balance out the data (SMOTE), since far fewer customers buy than don't.
    # 3. Run the KNN model itself.
    knn_pipeline_template = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numerical=True)),
            ("smote", get_smote()),
            ("knn", KNeighborsClassifier()),
        ]
    )

    # Same idea, but for Random Forest.
    rf_pipeline_template = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numerical=True)),
            ("smote", get_smote()),
            ("rf", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )

    # Outlier handle
    X_train, y_train = remove_outliers_iqr_train(X_train, y_train)

    # ---- Step 1: Find the best settings for KNN --------------------------
    # This tries every combination in knn_param_grid and keeps whichever
    # combination performs best.
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

    print(
        f"[train_knn_rf_ensemble] Best KNN settings found: "
        f"{knn_search.best_params_}"
    )

    # ---- Step 2: Find the best settings for Random Forest -----------------
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

    print(
        f"[train_knn_rf_ensemble] Best Random Forest settings found: "
        f"{rf_search.best_params_}"
    )

    # ---- Step 3: Build the Stacking Ensemble (Meta-Learning) ---------------
    # [Advanced Technique: Stacking Classifier / Meta-Learning]
    # Instead of just taking a fixed weighted average of the KNN and RF votes,
    # we train a third "Manager" model (Logistic Regression) that watches
    # KNN and RF make predictions and LEARNS when to trust each one.
    # This is much smarter than a static weight.
    ensemble = StackingClassifier(
        estimators=[("knn", best_knn), ("rf", best_rf)],
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1,
    )

    # ---- Step 4: Mathematical Probability Calibration -------------------
    # Platt Scaling (Sigmoid Calibration) fits a logistic regression on top
    # of the ensemble's outputs to convert them into smooth, continuous
    # probabilities. Without this, KNN and RF can output extreme values
    # like 1.0 or 0.0 which aren't true probabilities.
    print(
        "[train_knn_rf_ensemble] Applying Platt Scaling (Sigmoid Calibration) with 5-fold CV..."
    )
    calibrated_ensemble = CalibratedClassifierCV(
        estimator=ensemble, method="sigmoid", cv=5
    )
    calibrated_ensemble.fit(X_train, y_train)

    # ---- Step 5: Find the Optimal Decision Threshold --------------------
    # [Advanced Technique: Optimal Threshold Tuning]
    # Instead of using the default 0.5 cutoff, we scan every threshold
    # from 0.01 to 0.99 and pick the one that gives the best F1 score.
    # This compensates for the imbalanced dataset (far more non-buyers
    # than buyers), where 0.5 may not be the best decision boundary.
    print("[train_knn_rf_ensemble] Finding optimal decision threshold...")
    train_proba = calibrated_ensemble.predict_proba(X_train)[:, 1]

    best_threshold, best_f1_thresh = 0.5, -1.0
    for t in np.arange(0.01, 1.0, 0.01):
        preds = (train_proba >= t).astype(int)
        f1 = f1_score(y_train, preds, zero_division=0)
        if f1 > best_f1_thresh:
            best_f1_thresh, best_threshold = f1, round(float(t), 2)

    calibrated_ensemble.optimal_threshold_ = best_threshold
    print(
        f"[train_knn_rf_ensemble] Optimal threshold: {best_threshold:.4f} "
        f"(F1 = {best_f1_thresh:.4f}, vs default 0.5)"
    )

    # ---- Step 6: Save the finished model so it can be reused later --------
    if output_path:
        try:
            save_model(calibrated_ensemble, output_path)
        except Exception as e:
            print(
                f"[train_knn_rf_ensemble] Warning: failed to save model to {output_path}: {e}"
            )

    return calibrated_ensemble


if __name__ == "__main__":
    # This block only runs when this file is executed directly
    # (e.g. "python models/knn_rf_ensemble_model.py"), not when it's
    # imported by another file like app.py.

    # Load data and split into train/test sets.
    # preprocess_data() cleans the raw CSV (dedup, impute, group rare categories, etc.).
    # split_dataset() performs the stratified train/test split (in utils.py).
    df = preprocess_data()
    X_train, X_test, y_train, y_test = split_dataset(df)

    # Train the model and save it to disk.
    model = train_knn_rf_ensemble(
        X_train, y_train, output_path="saved_models/knn_rf_ensemble_model.pkl"
    )

    # Print out how well the model performed on data it has never seen.
    metrics = evaluate_model(model, X_test, y_test)
    print_metrics("KNN + Random Forest Ensemble", metrics)

    metrics_output_path = project_root / "report_assets" / "metrics.json"
    save_metrics("Knn Rf Ensemble Model", "knn", metrics, metrics_output_path)

    # Generate charts explaining which features most influenced the
    # model's predictions (used in the report/dashboard).
    print("\n[SHAP] Generating explanation charts for KNN + RF Ensemble...")
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

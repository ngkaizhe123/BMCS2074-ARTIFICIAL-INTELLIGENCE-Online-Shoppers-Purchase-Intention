import sys
from pathlib import Path

import numpy as np
import pandas as pd

# This lets the script find the project's other folders (like "src") when
# run directly, e.g. "python models/knn_rf_ensemble_model.py"
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from imblearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
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
    followed by probability calibration and decision threshold configuration.

    Args:
        X_train: The training data (customer session features).
        y_train: The correct answers for the training data (did they buy or not).
        knn_param_grid: Which KNN settings to try out. If not given, a sensible
            default set of options is used.
        rf_param_grid: Which Random Forest settings to try out. If not given,
            a sensible default set of options is used.

        output_path: Where to save the finished model file. Set to None to skip saving.

    Returns:
        The final trained, stacked, calibrated ensemble model with default threshold 0.5.

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

    # ---- Step 5: Decision Threshold Configuration -----------------------
    # Use standard default decision threshold of 0.50.
    calibrated_ensemble.optimal_threshold_ = 0.5
    print(
        "[train_knn_rf_ensemble] Using default decision threshold: 0.5000"
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


def evaluate_threshold_range(
    model,
    X_test,
    y_test,
    thresholds: list[float] | np.ndarray | None = None,
    save_csv_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Evaluate model performance across various decision thresholds (e.g., 0.10 to 0.90)
    and print a comparison table showing Precision, Recall, Specificity, F1-Score,
    Accuracy, and Confusion Counts (TP, FP, FN, TN).

    This test provides empirical evidence for presentation on why the default 0.50
    threshold is preferred over higher thresholds (which cause Recall to drop to 0.5+).

    Args:
        model: Trained classifier supporting `predict_proba`.
        X_test: Test features.
        y_test: True test labels (0 or 1).
        thresholds: Array or list of float thresholds to evaluate. Defaults to 0.10 - 0.90 (step 0.05).
        save_csv_path: Optional path to save the resulting DataFrame as a CSV file.

    Returns:
        pd.DataFrame containing the detailed metrics for each threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.95, 0.05)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must implement predict_proba() to evaluate decision thresholds.")

    y_prob = model.predict_proba(X_test)[:, 1]

    rows = []
    for t in thresholds:
        t_val = round(float(t), 2)
        preds = (y_prob >= t_val).astype(int)

        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = f1_score(y_test, preds, zero_division=0)

        note = "<- [DEFAULT: 0.50]" if np.isclose(t_val, 0.50) else ""

        rows.append({
            "Threshold": t_val,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "Specificity": spec,
            "F1 Score": f1,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "Note": note,
        })

    df_results = pd.DataFrame(rows)

    # Print formatted comparison table
    print("\n" + "=" * 96)
    print("        KNN + RANDOM FOREST ENSEMBLE: DECISION THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 96)
    print(
        f"{'Threshold':>9} | {'Accuracy':>8} | {'Precision':>9} | {'Recall':>8} | "
        f"{'Specificity':>11} | {'F1 Score':>8} | {'TP':>4} | {'FP':>4} | {'FN':>4} | Note"
    )
    print("-" * 96)
    for _, row in df_results.iterrows():
        print(
            f"{row['Threshold']:>9.2f} | {row['Accuracy']:>8.4f} | {row['Precision']:>9.4f} | "
            f"{row['Recall']:>8.4f} | {row['Specificity']:>11.4f} | {row['F1 Score']:>8.4f} | "
            f"{int(row['TP']):>4d} | {int(row['FP']):>4d} | {int(row['FN']):>4d} | {row['Note']}"
        )
    print("=" * 96)

    # Print presentation insights
    row_50 = df_results[np.isclose(df_results["Threshold"], 0.50)]
    rec_50 = float(row_50["Recall"].values[0]) if not row_50.empty else 0.0
    prec_50 = float(row_50["Precision"].values[0]) if not row_50.empty else 0.0
    f1_50 = float(row_50["F1 Score"].values[0]) if not row_50.empty else 0.0
    tp_50 = int(row_50["TP"].values[0]) if not row_50.empty else 0

    if save_csv_path:
        csv_file = Path(save_csv_path)
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(csv_file, index=False)
        print(f"[evaluate_threshold_range] Saved threshold results table to: {csv_file.resolve()}")

    return df_results


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

    # Evaluate different threshold values and print comparison table for presentation
    evaluate_threshold_range(
        model=model,
        X_test=X_test,
        y_test=y_test,
        save_csv_path=project_root / "report_assets" / "threshold_analysis" / "knn_rf_threshold_results.csv",
    )

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

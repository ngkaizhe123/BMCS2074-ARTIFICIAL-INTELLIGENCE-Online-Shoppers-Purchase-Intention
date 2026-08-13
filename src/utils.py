"""
utils.py
--------
Utility functions for dataset loading, saving processed outputs,
model evaluation, metrics reporting, model persistence, and SHAP explanations.
"""

from __future__ import annotations

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split


def load_raw_dataset(
    filepath: str | Path = "data/raw/online_shoppers_intention.csv",
) -> pd.DataFrame:
    """Load raw dataset CSV file."""
    path = Path(filepath)
    print(f"[load_raw_dataset] Loading dataset from {path}...")
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")
    return pd.read_csv(path)


def split_dataset(
    df: pd.DataFrame,
    target: str = "Revenue",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split dataset into features X and target y, and then into train/test splits."""
    X = df.drop(columns=[target]) if target in df.columns else df
    y = df[target].astype(int) if target in df.columns else None

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def save_cleaned_dataset(
    df: pd.DataFrame,
    filepath: str | Path = "data/processed/cleaned_online_shoppers_intention.csv",
) -> None:
    """
    Save the cleaned dataset to a single CSV file.
    Creates parent directories if they do not exist.
    """
    path = Path(filepath)
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path, index=False)
    print(f"[save_cleaned_dataset] Cleaned dataset saved to: {path.resolve()} (Shape: {df.shape})")


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a trained model and return a dictionary of evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(
            y_test, y_pred, zero_division=0
        ),
    }

    return metrics


def print_metrics(model_name: str, metrics: dict) -> None:
    """Print evaluation metrics in a formatted terminal table."""
    print("=" * 50)
    print(f" Model Evaluation Metrics: {model_name}")
    print("=" * 50)
    print(f"Accuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1 Score : {metrics['F1']:.4f}")
    if metrics.get("AUC") is not None:
        print(f"AUC      : {metrics['AUC']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["Confusion Matrix"])

    print("\nClassification Report:")
    print(metrics["Classification Report"])


def plot_confusion_matrix(model, X_test, y_test):
    """Plot confusion matrix chart."""
    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot(cmap="Blues")

    plt.title("Confusion Matrix")
    return plt.gcf()


def plot_roc_curve(model, X_test, y_test):
    """Plot ROC curve chart."""
    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc_score = roc_auc_score(y_test, probabilities)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], "--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")

    plt.legend()
    plt.grid(True)

    return plt.gcf()


def save_model(model, output_path: str | Path) -> None:
    """Save trained model/pipeline to file, creating parent directories if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[save_model] Model saved successfully to {path.resolve()}")


def load_model(filepath: str | Path):
    """Load trained model/pipeline from file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path.resolve()}")
    return joblib.load(path)


def generate_shap_explanation(
    model,
    X_test: pd.DataFrame,
    max_display: int = 15,
    save_dir: str | Path | None = None,
    prefix: str = "",
    show: bool = True,
):
    """
    Generate SHAP plots (Beeswarm, Feature Importance Bar, Waterfall) for a trained Pipeline or Model.

    Parameters
    ----------
    model : Trained Pipeline or Estimator.
    X_test : pandas DataFrame of raw testing features.
    max_display : Maximum number of top features to display in plots.
    save_dir : Directory path to save PNG figures (optional).
    prefix : Optional filename prefix (e.g., 'xgboost_', 'knn_', 'svm_').
    show : Whether to call plt.show() for figures.

    Returns
    -------
    (explainer, shap_values, figures_dict)
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "The 'shap' library is required for SHAP explanations. "
            "Please install it using: pip install shap"
        )

    # 1. Extract preprocessor and final estimator from Pipeline if applicable
    if hasattr(model, "named_steps"):
        if "preprocessor" in model.named_steps:
            preprocessor = model.named_steps["preprocessor"]
            X_test_transformed = preprocessor.transform(X_test)
            if hasattr(preprocessor, "get_feature_names_out"):
                feature_names = preprocessor.get_feature_names_out()
            else:
                feature_names = [f"feature_{i}" for i in range(X_test_transformed.shape[1])]
        else:
            X_test_transformed = X_test
            feature_names = getattr(X_test, "columns", [f"feature_{i}" for i in range(X_test.shape[1])])

        estimator = model.steps[-1][1]
    else:
        estimator = model
        X_test_transformed = X_test
        feature_names = getattr(X_test, "columns", [f"feature_{i}" for i in range(X_test.shape[1])])

    # Convert sparse matrix to dense array if necessary
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    feature_names = [str(name) for name in feature_names]

    # 2. Select appropriate Explainer
    estimator_name = estimator.__class__.__name__.lower()
    if "xgb" in estimator_name or "forest" in estimator_name or "tree" in estimator_name:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer(X_test_transformed)
    else:
        # KernelExplainer for non-tree models (e.g. KNN, SVM)
        background = shap.sample(X_test_transformed, min(50, len(X_test_transformed)))
        predict_fn = estimator.predict_proba if hasattr(estimator, "predict_proba") else estimator.predict
        raw_shap_values = shap.KernelExplainer(predict_fn, background).shap_values(
            X_test_transformed[: min(100, len(X_test_transformed))]
        )
        explainer = shap.KernelExplainer(predict_fn, background)

        if isinstance(raw_shap_values, list) and len(raw_shap_values) == 2:
            val = raw_shap_values[1]
            base_val = (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )
        else:
            val = raw_shap_values
            base_val = explainer.expected_value

        shap_values = shap.Explanation(
            values=val,
            base_values=base_val,
            data=X_test_transformed[: min(100, len(X_test_transformed))],
            feature_names=feature_names,
        )

    if hasattr(shap_values, "feature_names") and (
        shap_values.feature_names is None or len(shap_values.feature_names) == 0
    ):
        shap_values.feature_names = feature_names

    figures = {}

    def _save_or_show(fig, filename):
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            out_file = save_path / f"{prefix}{filename}"
            fig.savefig(out_file, dpi=150, bbox_inches="tight")
            print(f"[generate_shap_explanation] Saved: {out_file.resolve()}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    model_label = prefix.rstrip("_").upper() if prefix else estimator.__class__.__name__

    # Plot 1: Beeswarm Plot
    fig_bee = plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title(f"SHAP Beeswarm Plot ({model_label})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig_bee, "shap_beeswarm.png")
    figures["beeswarm"] = fig_bee

    # Plot 2: Bar Plot (Importance)
    fig_bar = plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title(f"SHAP Feature Importance ({model_label})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig_bar, "shap_feature_importance.png")
    figures["bar"] = fig_bar

    # Plot 3: Single Sample Waterfall Plot
    fig_waterfall = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], max_display=min(10, max_display), show=False)
    plt.title(f"SHAP Waterfall Plot ({model_label} Sample #0)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig_waterfall, "shap_waterfall.png")
    figures["waterfall"] = fig_waterfall

    return explainer, shap_values, figures


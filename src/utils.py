"""
utils.py
--------
Utility functions for dataset loading, saving processed outputs,
model evaluation, metrics reporting, model persistence, and SHAP explanations.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
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
    """Perform a stratified train/test split on the dataset."""
    X = df.drop(columns=[target]) if target in df.columns else df
    y = df[target].astype(int) if target in df.columns else None
    if y is None:
        raise ValueError(
            f"Target column '{target}' is required for a stratified split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[split_dataset] Total observations: {len(df):,}")
    print(f"[split_dataset] Train: {len(X_train):,}; Test: {len(X_test):,}")
    print(
        f"[split_dataset] Train class distribution: {y_train.value_counts().sort_index().to_dict()}"
    )
    print(
        f"[split_dataset] Test class distribution: {y_test.value_counts().sort_index().to_dict()}"
    )
    return X_train, X_test, y_train, y_test


def save_cleaned_dataset(
    df: pd.DataFrame,
    filepath: str | Path = "data/processed/cleaned_online_shoppers_intention.csv",
) -> None:
    """Save the cleaned dataset to a single CSV file."""
    path = Path(filepath)
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path, index=False)
    print(
        f"[save_cleaned_dataset] Cleaned dataset saved to: {path.resolve()} (Shape: {df.shape})"
    )


def evaluate_model(model, X_test, y_test, threshold: float | None = None) -> dict:
    """Evaluate a trained model and return a dictionary of evaluation metrics.

    Decision threshold resolution (this is the fix for the
    terminal-vs-metrics.json mismatch):
      1. If `threshold` is passed explicitly, use it.
      2. Else if the model itself has an `.optimal_threshold_` attribute
         (set by a training script right before save_model(), e.g.
         svm_model.py after its OOF threshold scan), use that.
      3. Else fall back to 0.5 (equivalent to plain model.predict()).

    Why this matters: re-evaluating a saved model later (e.g. from
    model_visualize.py, after the process that trained it has already
    exited) now automatically uses the SAME threshold the model was
    tuned and originally reported with, instead of silently re-scoring
    everything at the default 0.5 and producing different numbers than
    what the training script printed and saved to metrics.json.
    """
    if threshold is None:
        threshold = getattr(model, "optimal_threshold_", 0.5)

    y_prob = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )
    y_pred = (
        (y_prob >= threshold).astype(int)
        if y_prob is not None
        else model.predict(X_test)
    )

    metrics = {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred, zero_division=0),
    }

    return metrics


def print_metrics(model_name: str, metrics: dict) -> None:
    """Print evaluation metrics in a formatted terminal table."""
    print("=" * 50)
    print(f" Model Evaluation Metrics: {model_name}")
    print("=" * 50)
    if metrics.get("Threshold") is not None:
        print(f"Threshold: {metrics['Threshold']:.4f}")
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


def save_metrics(model_name: str, stem: str, metrics: dict, output_path: Path):
    """Persist a model's metrics dict to the shared metrics.json file.

    Reads the existing file, updates only this model_name's entry, and
    writes the merged result back -- other models' entries are preserved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = metrics.get("Confusion Matrix")
    if hasattr(cm, "tolist"):
        cm = cm.tolist()

    serializable_metrics = {
        "stem": stem,
        "Threshold": (
            float(metrics["Threshold"]) if metrics.get("Threshold") is not None else 0.5
        ),
        "Accuracy": float(metrics.get("Accuracy", 0.0)),
        "Precision": float(metrics.get("Precision", 0.0)),
        "Recall": float(metrics.get("Recall", 0.0)),
        "F1 Score": float(metrics.get("F1 Score", metrics.get("F1", 0.0))),
        "AUC": float(metrics["AUC"]) if metrics.get("AUC") is not None else None,
        "Confusion Matrix": cm,
        "Classification Report": metrics.get("Classification Report", ""),
    }

    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                all_metrics = json.load(f)
        except json.JSONDecodeError:
            all_metrics = {}
    else:
        all_metrics = {}

    all_metrics[model_name] = serializable_metrics

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Metrics for {model_name} saved to {output_path}")


def plot_confusion_matrix(model, X_test, y_test, threshold: float | None = None):
    """Plot confusion matrix chart, using the model's tuned threshold if set."""
    if threshold is None:
        threshold = getattr(model, "optimal_threshold_", 0.5)
    if hasattr(model, "predict_proba"):
        predictions = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    else:
        predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot(cmap="Blues")

    plt.title("Confusion Matrix")
    return plt.gcf()


def plot_roc_curve(model, X_test, y_test):
    """Plot ROC curve chart. ROC-AUC is threshold-independent."""
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


def save_model(model, output_path: str | Path, compress: int = 3) -> None:
    """Save trained model/pipeline to file, creating parent directories if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=compress)
    print(
        f"[save_model] Model saved successfully to {path.resolve()} (compress={compress})"
    )


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
    """Generate SHAP plots (Beeswarm, Feature Importance Bar, Waterfall) for a trained Pipeline or Model."""
    is_imblearn_or_sklearn_pipeline = hasattr(model, "named_steps")

    if is_imblearn_or_sklearn_pipeline and "preprocessor" in model.named_steps:
        preprocessor = model.named_steps["preprocessor"]
        X_for_preprocessor = (
            model.named_steps["cleaner"].transform(X_test)
            if "cleaner" in model.named_steps
            else X_test
        )
        X_transformed = preprocessor.transform(X_for_preprocessor)
        if hasattr(X_transformed, "toarray"):
            X_transformed = X_transformed.toarray()
        feature_names = (
            [str(n) for n in preprocessor.get_feature_names_out()]
            if hasattr(preprocessor, "get_feature_names_out")
            else [f"feature_{i}" for i in range(X_transformed.shape[1])]
        )
        estimator = model.steps[-1][1]

        estimator_name = estimator.__class__.__name__.lower()
        if (
            "xgb" in estimator_name
            or "randomforest" in estimator_name
            or "decisiontree" in estimator_name
        ):
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer(X_transformed)
        else:
            n_bg = min(50, len(X_transformed))
            n_ex = min(100, len(X_transformed))
            idx = np.random.RandomState(42).choice(
                len(X_transformed), n_bg, replace=False
            )
            background = X_transformed[idx]
            X_explain = X_transformed[:n_ex]

            predict_fn = (
                estimator.predict_proba
                if hasattr(estimator, "predict_proba")
                else estimator.predict
            )
            explainer = shap.KernelExplainer(predict_fn, background)
            raw = explainer.shap_values(X_explain)

            if isinstance(raw, list) and len(raw) == 2:
                val = raw[1]
                base_val = (
                    explainer.expected_value[1]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value
                )
            elif isinstance(raw, np.ndarray) and raw.ndim == 3:
                val = raw[:, :, 1]
                base_val = (
                    explainer.expected_value[1]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value
                )
            else:
                val = raw
                base_val = explainer.expected_value

            shap_values = shap.Explanation(
                values=val,
                base_values=base_val,
                data=X_explain,
                feature_names=feature_names,
            )

    else:
        col_names = (
            list(X_test.columns)
            if isinstance(X_test, pd.DataFrame)
            else [f"feature_{i}" for i in range(X_test.shape[1])]
        )
        feature_names = col_names
        n_bg = min(50, len(X_test))
        n_ex = min(100, len(X_test))
        background = (
            X_test.sample(n=n_bg, random_state=42)
            if isinstance(X_test, pd.DataFrame)
            else X_test[:n_bg]
        )
        X_explain = (
            X_test.iloc[:n_ex] if isinstance(X_test, pd.DataFrame) else X_test[:n_ex]
        )
        _base_predict = (
            model.predict_proba if hasattr(model, "predict_proba") else model.predict
        )

        def _predict_with_df(data):
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data, columns=col_names)
            return _base_predict(data)

        explainer = shap.KernelExplainer(_predict_with_df, background)
        raw = explainer.shap_values(X_explain)

        if isinstance(raw, list) and len(raw) == 2:
            val = raw[1]
            base_val = (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )
        elif isinstance(raw, np.ndarray) and raw.ndim == 3:
            val = raw[:, :, 1]
            base_val = (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )
        else:
            val = raw
            base_val = explainer.expected_value

        data_array = (
            X_explain.values if isinstance(X_explain, pd.DataFrame) else X_explain
        )

        shap_values = shap.Explanation(
            values=val,
            base_values=base_val,
            data=data_array,
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

    fig_bee = plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title(f"SHAP Beeswarm Plot ({model_label})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig_bee, "shap_beeswarm.png")
    figures["beeswarm"] = fig_bee

    fig_bar = plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title(
        f"SHAP Feature Importance ({model_label})", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    _save_or_show(fig_bar, "shap_feature_importance.png")
    figures["bar"] = fig_bar

    shap.plots.waterfall(shap_values[0], max_display=min(10, max_display), show=False)
    fig_waterfall = plt.gcf()
    fig_waterfall.set_size_inches(10, 6)
    _save_or_show(fig_waterfall, "shap_waterfall.png")
    figures["waterfall"] = fig_waterfall

    return explainer, shap_values, figures

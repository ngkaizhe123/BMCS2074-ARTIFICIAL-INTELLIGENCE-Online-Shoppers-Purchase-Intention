"""
utils.py
--------
Utility functions for dataset loading, saving processed outputs,
model evaluation, metrics reporting, model persistence, and SHAP explanations.
"""

from __future__ import annotations

import json
import hashlib
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
    split_path: str | Path | None = None,
):
    """Load or create the project's one fixed, stratified raw holdout split.

    The manifest stores row positions after deterministic pre-split preparation.
    Every training script and visualization calls this function, so a holdout is
    never regenerated from model-specific (for example IQR-filtered) data.
    """
    X = df.drop(columns=[target]) if target in df.columns else df
    y = df[target].astype(int) if target in df.columns else None
    if y is None:
        raise ValueError(
            f"Target column '{target}' is required for a stratified split."
        )

    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()
    path = (
        Path(split_path)
        if split_path is not None
        else Path(__file__).resolve().parent.parent
        / "data"
        / "processed"
        / "fixed_train_test_split.json"
    )
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError(
                "The persisted split does not match the current prepared dataset. "
                "Deliberately remove the split manifest and retrain every model."
            )
        train_idx = np.asarray(manifest["train_indices"], dtype=int)
        test_idx = np.asarray(manifest["test_indices"], dtype=int)
        print(f"[split_dataset] Loaded fixed split from {path}.")
    else:
        all_idx = np.arange(len(df))
        train_idx, test_idx = train_test_split(
            all_idx, test_size=test_size, random_state=random_state, stratify=y
        )
        manifest = {
            "fingerprint": fingerprint,
            "n_observations": len(df),
            "test_size": test_size,
            "random_state": random_state,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[split_dataset] Created fixed split manifest at {path}.")

    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    print(f"[split_dataset] Raw/prepared observations: {len(df):,}")
    print(
        f"[split_dataset] Train: {len(X_train):,}; Test: {len(X_test):,} (test rows are untouched)"
    )
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
    """
    Save the cleaned dataset to a single CSV file.
    Creates parent directories if they do not exist.
    """
    path = Path(filepath)
    path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(path, index=False)
    print(
        f"[save_cleaned_dataset] Cleaned dataset saved to: {path.resolve()} (Shape: {df.shape})"
    )


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
        "Classification Report": classification_report(y_test, y_pred, zero_division=0),
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


def save_metrics(model_name: str, stem: str, metrics: dict, output_path: Path):
    """Persist a model's metrics dict to the shared metrics.json file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert non-serializable objects (like numpy arrays)
    serializable_metrics = {"stem": stem}
    for k, v in metrics.items():
        if isinstance(v, np.ndarray):
            serializable_metrics[k] = v.tolist()
        else:
            serializable_metrics[k] = v

    # Ensure both "F1" and "F1 Score" keys exist for compatibility
    if "F1" in metrics and "F1 Score" not in serializable_metrics:
        serializable_metrics["F1 Score"] = metrics["F1"]
    elif "F1 Score" in metrics and "F1" not in serializable_metrics:
        serializable_metrics["F1"] = metrics["F1 Score"]

    if output_path.exists():
        try:
            with open(output_path, "r") as f:
                all_metrics = json.load(f)
        except json.JSONDecodeError:
            all_metrics = {}
    else:
        all_metrics = {}

    all_metrics[model_name] = serializable_metrics

    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Metrics for {model_name} saved to {output_path}")


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


def save_model(model, output_path: str | Path, compress: int = 3) -> None:
    """Save trained model/pipeline to file, creating parent directories if needed.

    Args:
        model: The trained model or pipeline to save.
        output_path: Destination file path for the .pkl file.
        compress: joblib compression level 0-9 (0 = none, 3 = good balance of
            size vs speed, 9 = maximum compression). Defaults to 3, which
            typically reduces file size by 3-5x with negligible load overhead.
    """
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

    # Ensure custom model classes are available in sys.modules['__main__']
    # to support models saved when a script was executed directly as __main__
    try:
        import sys

        main_mod = sys.modules.get("__main__")
        if main_mod is not None:
            from models.fsvm import FuzzySVM

            if not hasattr(main_mod, "FuzzySVM"):
                setattr(main_mod, "FuzzySVM", FuzzySVM)
    except Exception:
        pass

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

    # 1. Determine if the model contains ColumnTransformers that need
    #    DataFrame column names (e.g. VotingClassifier with sub-Pipelines).
    #    KernelExplainer internally converts all data to numpy arrays before
    #    calling predict_proba, so we wrap predict_fn to restore column names.
    is_imblearn_or_sklearn_pipeline = hasattr(model, "named_steps")

    if is_imblearn_or_sklearn_pipeline and "preprocessor" in model.named_steps:
        # Standard Pipeline: pre-transform X_test and explain in feature space
        preprocessor = model.named_steps["preprocessor"]
        # Keep SHAP aligned with the fitted inference route.  The cleaner is
        # a row-preserving transformer; any outlier sampler is intentionally
        # skipped at inference by imblearn Pipeline.
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
            # Fast TreeExplainer path
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer(X_transformed)
        else:
            # KernelExplainer on already-transformed numpy arrays: safe, no DataFrame needed
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
                print("Detected 3D SHAP output, extracting positive class")

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

        # --- Key fix: wrap predict_proba to restore DataFrame column names ---

        _base_predict = (
            model.predict_proba if hasattr(model, "predict_proba") else model.predict
        )

        def _predict_with_df(data):
            """Always convert input to a DataFrame with correct column names."""

            if not isinstance(data, pd.DataFrame):

                data = pd.DataFrame(data, columns=col_names)

            return _base_predict(data)

        explainer = shap.KernelExplainer(_predict_with_df, background)

        raw = explainer.shap_values(X_explain)

        print("RAW TYPE:", type(raw))

        if isinstance(raw, np.ndarray):

            print("RAW SHAPE:", raw.shape)

        elif isinstance(raw, list):

            print("LIST LENGTH:", len(raw))

            for i, arr in enumerate(raw):

                print(f"class {i} shape:", np.array(arr).shape)

        # -------------------------------------------------------------

        if isinstance(raw, list) and len(raw) == 2:

            val = raw[1]

            base_val = (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )

        elif isinstance(raw, np.ndarray) and raw.ndim == 3:

            print(
                "[SHAP] Detected 3D SHAP output (samples, features, classes), extracting positive class (Class 1)"
            )

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
    plt.title(
        f"SHAP Feature Importance ({model_label})", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    _save_or_show(fig_bar, "shap_feature_importance.png")
    figures["bar"] = fig_bar

    # Plot 3: Single Sample Waterfall Plot
    # shap.plots.waterfall() creates its own figure internally;
    # use plt.gcf() to retrieve it after the call.
    shap.plots.waterfall(shap_values[0], max_display=min(10, max_display), show=False)
    fig_waterfall = plt.gcf()
    fig_waterfall.set_size_inches(10, 6)
    _save_or_show(fig_waterfall, "shap_waterfall.png")
    figures["waterfall"] = fig_waterfall

    return explainer, shap_values, figures

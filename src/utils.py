from pathlib import Path
import joblib
import matplotlib.pyplot as plt
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


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model and return a dictionary of metrics."""
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


def print_metrics(model_name: str, metrics: dict):
    """Print evaluation metrics in a readable format."""
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


def save_model(model, output_path: str | Path):
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

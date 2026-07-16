import joblib
import matplotlib.pyplot as plt

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


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained binary classification model.

    Parameters
    ----------
    model
        Trained machine learning model.

    X_test
        Testing features.

    y_test
        Testing labels.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred),
    }

    return metrics


def print_metrics(metrics):
    print("=" * 50)

    print(f"Accuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1 Score : {metrics['F1']:.4f}")
    print(f"AUC      : {metrics['AUC']:.4f}")

    print("\nConfusion Matrix")
    print(metrics["Confusion Matrix"])

    print("\nClassification Report")
    print(metrics["Classification Report"])


def plot_confusion_matrix(model, X_test, y_test):

    predictions = model.predict(X_test)
    cm = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()

    plt.title("Confusion Matrix")

    return plt.gcf()


def plot_roc_curve(model, X_test, y_test):

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
    plt.grid()

    return plt.gcf()  # get current figure size


# Save trained model
def save_model(model, filepath):
    joblib.dump(model, filepath)


# Load a trained model
def load_model(filepath):
    return joblib.load(filepath)

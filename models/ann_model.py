import os

import joblib
from sklearn.neural_network import MLPClassifier

from src.data_preprocessing import get_cleaned_data
from src.utils import print_evaluation_report


def run_ann():
    X_train_scaled, X_test_scaled, y_train, y_test = get_cleaned_data(
        filepath="data/customer_purchase_data.csv"
    )

    print("[INFO] Building ANN (MLPClassifier) model...")
    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42,
    )

    print("[INFO] Start training ANN model...")
    model.fit(X_train_scaled, y_train)

    print("[INFO] Model trained successfully, running final testing...")
    y_pred = model.predict(X_test_scaled)

    print_evaluation_report(y_test, y_pred, "ANN (MLPClassifier)")

    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(model, "saved_models/ann_model.pkl")
    print("[SUCCESS] ANN model saved successfully at saved_models/ann_model.pkl")


if __name__ == "__main__":
    run_ann()

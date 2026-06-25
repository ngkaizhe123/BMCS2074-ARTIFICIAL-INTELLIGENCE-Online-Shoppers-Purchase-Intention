import os

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_cleaned_data(filepath="data/customer_purchase_data.csv"):
    print(" Fetching and cleaning data ...")
    df = pd.read_csv(filepath)

    df = pd.get_dummies(df, columns=["ProductCategory"], drop_first=False)

    X = df.drop("PurchaseStatus", axis=1)
    y = df["PurchaseStatus"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs("saved_models", exist_ok=True)

    joblib.dump(scaler, "saved_models/scaler.pkl")
    print("[SUCCESS] Scaler saved at saved_models/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test

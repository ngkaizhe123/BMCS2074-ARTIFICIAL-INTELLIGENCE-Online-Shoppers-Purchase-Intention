import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def preprocess_data(filepath="../data/raw/online_shoppers_intention.csv"):
    # Load raw dataset
    df = pd.read_csv(filepath)

    # Convert target to integer
    # False -> 0
    # True  -> 1
    df["Revenue"] = df["Revenue"].astype(int)

    X = df.drop("Revenue", axis=1)
    y = df["Revenue"]

    categorical_features = ["Month", "VisitorType", "Weekend"]
    numerical_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "num",
                StandardScaler(),
                numerical_features,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    )

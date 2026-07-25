import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES = ["Month", "VisitorType", "Weekend"]

NUMERICAL_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

TARGET = "Revenue"


# ---------------------------------------------------------------------------
# Step 1 – Load
# ---------------------------------------------------------------------------


def load_data(
    filepath: str = "../data/raw/online_shoppers_intention.csv",
) -> pd.DataFrame:
    """Load raw dataset CSV."""
    df = pd.read_csv(filepath)
    print(f"[load_data] Loaded {df.shape[0]} rows × {df.shape[1]} columns.")
    return df


# ---------------------------------------------------------------------------
# Step 2 – Handle Missing Values
# ---------------------------------------------------------------------------


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
    - Numerical columns -> median
    - Categorical columns -> mode
    """
    missing_before = df.isnull().sum().sum()

    for col in NUMERICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)

    missing_after = df.isnull().sum().sum()
    print(
        f"[handle_missing_values] Missing values: {missing_before} -> {missing_after}"
    )
    return df


# ---------------------------------------------------------------------------
# Step 3 – Remove Outliers
# ---------------------------------------------------------------------------


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor: float = 1.5,
) -> pd.DataFrame:
    """Remove outliers using IQR method."""
    if columns is None:
        columns = [c for c in NUMERICAL_FEATURES if c in df.columns]

    before = len(df)
    mask = pd.Series(True, index=df.index)

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        col_mask = df[col].between(lower, upper)
        mask &= col_mask

    df = df[mask].reset_index(drop=True)
    print(
        f"[remove_outliers_iqr] Rows: {before} -> {len(df)} (removed {before - len(df)})"
    )
    return df


def remove_outliers_zscore(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Remove outliers using Z-score method."""
    if columns is None:
        columns = [c for c in NUMERICAL_FEATURES if c in df.columns]

    before = len(df)
    z_scores = np.abs(stats.zscore(df[columns], nan_policy="omit"))
    mask = (z_scores < threshold).all(axis=1)

    df = df[mask].reset_index(drop=True)
    print(
        f"[remove_outliers_zscore] Rows: {before} -> {len(df)} (removed {before - len(df)})"
    )
    return df


# ---------------------------------------------------------------------------
# Step 4 – Encode Target
# ---------------------------------------------------------------------------


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean/string Revenue column to integer (0 or 1)."""
    df[TARGET] = df[TARGET].astype(int)
    return df


# ---------------------------------------------------------------------------
# Step 5 – Build Preprocessing ColumnTransformer
# ---------------------------------------------------------------------------


def build_preprocessor(scale_numerical: bool = False) -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer:
    - OneHotEncoder for Categorical features
    - StandardScaler for Numerical features (if scale_numerical=True, e.g., KNN/SVM)
      or passthrough (if scale_numerical=False, e.g., XGBoost)
    """
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    if scale_numerical:
        num_transformer = StandardScaler()
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", cat_transformer, CATEGORICAL_FEATURES),
                ("num", num_transformer, NUMERICAL_FEATURES),
            ],
            verbose_feature_names_out=False,
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", cat_transformer, CATEGORICAL_FEATURES),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

    return preprocessor


# ---------------------------------------------------------------------------
# Step 6 – Full Data Pipeline (Compatible with All Models)
# ---------------------------------------------------------------------------


def preprocess_data(
    filepath: str = "data/raw/online_shoppers_intention.csv",
    outlier_method: str = "none",
    test_size: float = 0.2,
    random_state: int = 42,
    transform: bool = False,
    scale_numerical: bool = False,
) -> tuple:
    """
    Load, clean, split data.

    Parameters
    ----------
    filepath        : Path to dataset CSV.
    outlier_method  : 'none' | 'iqr' | 'zscore'
    test_size       : Split ratio for testing set.
    random_state    : Seed for reproducibility.
    transform       : If True, returns transformed numpy matrices & fitted preprocessor.
                      If False, returns raw DataFrames X_train, X_test (ideal for imblearn Pipeline).
    scale_numerical : Whether to apply StandardScaler to numerical features.

    Returns
    -------
    (X_train, X_test, y_train, y_test, preprocessor)
    """
    # 1. Load
    df = load_data(filepath)

    # 2. Clean missing values
    df = handle_missing_values(df)

    # 3. Outliers
    if outlier_method == "iqr":
        df = remove_outliers_iqr(df)
    elif outlier_method == "zscore":
        df = remove_outliers_zscore(df)

    # 4. Target encoding
    df = encode_target(df)

    # 5. Feature / Target split
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    # 6. Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(scale_numerical=scale_numerical)

    if transform:
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        return X_train_processed, X_test_processed, y_train, y_test, preprocessor

    return X_train, X_test, y_train, y_test, preprocessor

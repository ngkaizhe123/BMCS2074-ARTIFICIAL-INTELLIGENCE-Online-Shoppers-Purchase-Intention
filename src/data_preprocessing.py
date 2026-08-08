"""
data_preprocessing.py
----------------------
Cleaning, splitting, and preprocessing pipeline for the
Online Shoppers Intention dataset.

Pipeline overview
-----------------
 1. load_data               – Read raw CSV
 2. remove_duplicates       – Drop 125 exact-duplicate rows (found via EDA)
 3. handle_missing_values   – Median/mode imputation (safeguard; raw data has none)
 4. remove_outliers_iqr     – IQR method (for KNN / SVM that are distance-sensitive)
 5. remove_outliers_zscore  – Z-score method (alternative to IQR)
 6. encode_target           – Boolean Revenue → int (0 / 1)
 7. run_preprocessing_pipeline – Steps 2-6 assembled; also called by preprocess_data()
 8. build_preprocessor      – sklearn ColumnTransformer (OHE + optional StandardScaler)
 9. get_smote               – SMOTE instance for training pipelines
10. preprocess_data         – Full pipeline: clean → split → (optional transform)

Saving cleaned data (run as script)
------------------------------------
    python src/data_preprocessing.py
    → writes  data/processed/cleaned_online_shoppers_intention.csv
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from pathlib import Path
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES = ["Month", "VisitorType", "Weekend"]

# All numeric-typed columns (used for encoding / scaling). This includes the
# 4 ordinal/ID-coded columns (OperatingSystems, Browser, Region, TrafficType)
# since scaling them alongside true numerics is a common, harmless choice.
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

# Columns that are genuinely continuous and safe to run outlier-detection on.
# Deliberately EXCLUDES:
#   - OperatingSystems / Browser / Region / TrafficType: these are categorical
#     ID codes stored as integers, not continuous measurements. IQR/Z-score
#     bounds on an ID code are meaningless (e.g. Browser has Q1==Q3==2, so
#     IQR==0 and *any* other browser code gets flagged as an "outlier").
#   - Informational / Informational_Duration / PageValues / SpecialDay: these
#     are heavily zero-inflated (Q1==Q3==0 for most sessions), so IQR==0 and
#     ANY nonzero value gets flagged as an outlier. PageValues in particular
#     is the single strongest predictor of Revenue (corr ~0.49) — dropping
#     "outliers" here means deleting almost every row that actually converts.
CONTINUOUS_FEATURES_FOR_OUTLIERS = [
    "Administrative",
    "Administrative_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
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
    print(f"[load_data] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.")
    return df


# ---------------------------------------------------------------------------
# Step 2 – Remove Duplicates
# ---------------------------------------------------------------------------


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact-duplicate rows identified during EDA.

    EDA finding → Preprocessing action
    ------------------------------------
    The raw dataset contains 125 duplicated rows (~1.0% of data).
    Keeping duplicates can introduce bias because the same session is
    counted multiple times, inflating model confidence for those patterns.

    Action: drop all but the first occurrence of each duplicate.
    """
    n_before = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    n_after = len(df_clean)
    print(
        f"[remove_duplicates] Rows: {n_before:,} -> {n_after:,} "
        f"(removed {n_before - n_after} duplicates)"
    )
    return df_clean


# ---------------------------------------------------------------------------
# Step 3 – Handle Missing Values
# ---------------------------------------------------------------------------


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
    - Numerical columns -> median
    - Categorical columns -> mode

    EDA finding → Preprocessing action
    ------------------------------------
    EDA confirmed the raw dataset has 0 missing values.
    This step is kept as a safeguard so the pipeline remains robust if the
    dataset is ever updated or partially filled with NaN during merges.
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
# Step 4 – Remove Outliers
# ---------------------------------------------------------------------------


def _report_removal(df_before: pd.DataFrame, mask: pd.Series, method: str) -> None:
    """Shared reporting/safety-check for outlier removal, incl. class balance."""
    before = len(df_before)
    after = int(mask.sum())
    removed = before - after
    pct_removed = removed / before * 100 if before else 0.0
    print(
        f"[{method}] Rows: {before:,} -> {after:,} "
        f"(removed {removed}, {pct_removed:.1f}%)"
    )

    if TARGET in df_before.columns:
        for cls in sorted(df_before[TARGET].unique()):
            cls_before = (df_before[TARGET] == cls).sum()
            cls_after = (df_before.loc[mask, TARGET] == cls).sum()
            cls_pct_removed = (
                (cls_before - cls_after) / cls_before * 100 if cls_before else 0.0
            )
            print(
                f"    class {cls}: {cls_before} -> {cls_after} "
                f"(removed {cls_pct_removed:.1f}%)"
            )

    if pct_removed > 20:
        print(
            f"    [WARNING] {method} removed more than 20% of rows. "
            "Consider using CONTINUOUS_FEATURES_FOR_OUTLIERS-only columns, "
            "a larger factor/threshold, or skipping outlier removal "
            "entirely for tree-based models (RF/XGBoost), which are "
            "robust to outliers by design."
        )


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Remove outliers using the IQR method.

    EDA finding → Preprocessing action
    ------------------------------------
    Box plots in EDA showed extreme right-skew and long tails in
    Administrative, Administrative_Duration, ProductRelated,
    ProductRelated_Duration, BounceRates, and ExitRates.
    Distance-sensitive models (KNN, SVM) are strongly affected by these
    extreme values, so outlier removal is recommended for those pipelines.
    Tree-based models (XGBoost) are robust to outliers by design;
    outlier_method='none' is the default for XGBoost.

    NOTE: bounds are intersected (AND) across all `columns`, so the more
    columns you pass, the more rows get dropped. By default this only runs
    on CONTINUOUS_FEATURES_FOR_OUTLIERS — genuinely continuous, non-zero-
    inflated columns — to avoid silently deleting most of the dataset.
    Pass `columns` explicitly if you want a different set, but check the
    per-column IQR first (a column with IQR==0 will flag almost everything
    as an outlier).
    """
    if columns is None:
        columns = CONTINUOUS_FEATURES_FOR_OUTLIERS
    columns = [c for c in columns if c in df.columns]

    mask = pd.Series(True, index=df.index)
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            print(
                f"    [skip] '{col}' has IQR==0 (Q1==Q3=={Q1}); skipping to "
                "avoid flagging every nonzero value as an outlier."
            )
            continue
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        mask &= df[col].between(lower, upper)

    _report_removal(df, mask, "remove_outliers_iqr")
    return df[mask].reset_index(drop=True)


def remove_outliers_zscore(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Remove outliers using the Z-score method.

    EDA finding → Preprocessing action
    ------------------------------------
    Statistical summary in EDA shows high skewness and kurtosis for several
    continuous features (see print_statistical_summary). Z-score method is
    an alternative to IQR, typically removing rows where any feature deviates
    more than `threshold` standard deviations from the mean.

    By default this only runs on CONTINUOUS_FEATURES_FOR_OUTLIERS (see note
    on remove_outliers_iqr) — running Z-score on categorical ID codes like
    Browser/OperatingSystems is not meaningful.
    """
    if columns is None:
        columns = CONTINUOUS_FEATURES_FOR_OUTLIERS
    columns = [c for c in columns if c in df.columns]

    z_scores = np.abs(stats.zscore(df[columns], nan_policy="omit"))
    mask = (z_scores < threshold).all(axis=1)
    mask = pd.Series(mask, index=df.index)

    _report_removal(df, mask, "remove_outliers_zscore")
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 5 – Encode Target
# ---------------------------------------------------------------------------


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert boolean/string Revenue column to integer (0 or 1).

    EDA finding → Preprocessing action
    ------------------------------------
    EDA confirmed Revenue is a boolean column (True/False).
    scikit-learn and XGBoost expect integer labels; this step ensures
    True → 1 (purchase) and False → 0 (no purchase) for all models.
    """
    df = df.copy()
    df[TARGET] = df[TARGET].astype(int)
    return df


# ---------------------------------------------------------------------------
# Step 6 – Unified Cleaning Pipeline (used for CSV export & preprocess_data)
# ---------------------------------------------------------------------------


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    outlier_method: str = "none",
) -> pd.DataFrame:
    """
    Run the full data cleaning pipeline on a raw DataFrame and return the
    cleaned dataset (no train/test split, no sklearn transformers applied).

    This function is called both when:
      a) Running this script directly to save cleaned_online_shoppers_intention.csv
      b) Inside preprocess_data() before the train/test split

    Steps
    -----
    1. remove_duplicates       – Drop 125 duplicate rows found in EDA
    2. handle_missing_values   – Median/mode imputation (safeguard)
    3. remove_outliers_*       – Optional IQR or Z-score outlier removal
    4. encode_target           – Revenue bool → int
    """
    print(f"\n[run_preprocessing_pipeline] Starting. Input shape: {df.shape}")

    df = remove_duplicates(df)
    df = handle_missing_values(df)

    if outlier_method == "iqr":
        df = remove_outliers_iqr(df)
    elif outlier_method == "zscore":
        df = remove_outliers_zscore(df)
    elif outlier_method != "none":
        raise ValueError(
            f"Unknown outlier_method '{outlier_method}'. "
            "Use 'none', 'iqr', or 'zscore'."
        )

    df = encode_target(df)

    print(f"[run_preprocessing_pipeline] Done. Output shape: {df.shape}\n")
    return df


# ---------------------------------------------------------------------------
# Step 7 – Build Preprocessing ColumnTransformer
# ---------------------------------------------------------------------------


def build_preprocessor(scale_numerical: bool = False) -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer:
    - OneHotEncoder for Categorical features
    - StandardScaler for Numerical features (if scale_numerical=True, e.g., KNN/SVM)
      or passthrough (if scale_numerical=False, e.g., XGBoost)

    EDA finding → Preprocessing action
    ------------------------------------
    EDA showed that Month, VisitorType, and Weekend are nominal categorical
    variables with no inherent numeric ordering. OneHotEncoding is applied to
    convert them to binary indicator columns without imposing false ordinality.

    Numerical features are standardised only for distance-based models
    (KNN, SVM) where feature scale directly affects distance computation.
    XGBoost is scale-invariant (uses split thresholds), so scaling is skipped.
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
# Step 8 – Shared SMOTE Instance (used inside each model's Pipeline)
# ---------------------------------------------------------------------------


def get_smote(random_state: int = 42) -> SMOTE:
    """
    Return a configured SMOTE instance for oversampling the minority class.

    EDA finding → Preprocessing action
    ------------------------------------
    EDA revealed a severe class imbalance: ~84.5% No Purchase vs ~15.5%
    Purchase (ratio ≈ 5.4:1). Training directly on this imbalanced data
    biases the model towards predicting "No Purchase" for everything.
    SMOTE synthetically oversamples the minority class (Purchase=1) in the
    training fold only (inside the imblearn Pipeline), preventing data leakage.

    Must be used as a step inside an imblearn Pipeline (not applied once
    upfront) so resampling only happens on training folds during CV.
    """
    return SMOTE(random_state=random_state)


# ---------------------------------------------------------------------------
# Step 9 – Full Data Pipeline (Compatible with All Models)
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

    # 2–5. Clean (duplicates → missing values → outliers → encode target)
    df = run_preprocessing_pipeline(df, outlier_method=outlier_method)

    # 6. Feature / Target split
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    # 7. Train / Test split
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


# ---------------------------------------------------------------------------
# Run directly → save cleaned CSV to data/processed/
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Define paths
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    input_path = project_root / "data" / "raw" / "online_shoppers_intention.csv"
    output_dir = project_root / "data" / "processed"
    output_path = output_dir / "cleaned_online_shoppers_intention.csv"

    print(f"Loading raw data from {input_path}...")
    try:
        raw_df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {input_path}")
        sys.exit(1)

    # 2. Run cleaning pipeline (no outlier removal by default — keep full dataset
    #    for EDA reference; models apply their own outlier handling)
    clean_df = run_preprocessing_pipeline(raw_df, outlier_method="none")

    # 3. Save the cleaned dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving cleaned data to {output_path}...")
    clean_df.to_csv(output_path, index=False)
    print(f"[OK] Cleaned data saved! Shape: {clean_df.shape}")
    print(f"   Columns: {list(clean_df.columns)}")

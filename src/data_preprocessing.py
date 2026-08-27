"""
data_preprocessing.py
----------------------
Cleaning, splitting, and preprocessing pipeline for the
Online Shoppers Intention dataset.

Pre-split (deterministic or low-risk aggregate statistics)
------------------------------------------------------------
 1. load_data                    - Read raw CSV
 2. remove_duplicates            - Drop ~125 exact-duplicate rows (EDA finding)
 3. validate_business_rules      - Report domain-level data integrity violations
 4. fix_duration_consistency     - Fix page-count <-> duration mismatches
 5. cap_duration_outliers        - Cap durations to fixed DURATION_CAPS constants
 6. clip_rates                   - Enforce BounceRates / ExitRates within [0, 1]
 7. validate_categorical_values  - Replace invalid category values with the mode
 8. group_rare_categories        - Group rare ID categories into 'Other_xxx'
 9. drop_special_day             - Remove SpecialDay column
10. handle_missing_value         - Median/mode imputation (safeguard; no-op today)
11. encode_target                - Boolean Revenue -> int (0 / 1)
12. prepare_dataset_for_split    - Assembles steps 2-11; called by preprocess_data()
13. preprocess_data              - Entry point: load -> clean -> return df

Why these are safe to run BEFORE the split (see the leakage-risk table
from an earlier discussion): handle_missing_value and
validate_categorical_values are currently no-ops on this dataset (0
missing values, 0 invalid categories) -- they only exist as safeguards.
fix_duration_consistency and group_rare_categories compute an aggregate
statistic (a median duration, a category frequency count) over 12,000+
rows, not the target label -- the difference between "computed on the
full dataset" vs. "computed on the 80% training split" is negligible in
practice. The ONE step with a real leakage risk is outlier removal
(it deletes rows), which is why it is NOT in this file's pre-split
pipeline -- see remove_outliers_iqr_train() below, which must be called
explicitly on (X_train, y_train) only, after split_dataset().

Saving cleaned data (run as script)
------------------------------------
    python src/data_preprocessing.py
    -> writes  data/processed/cleaned_online_shoppers_intention.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORICAL_FEATURES = [
    "Month",
    "VisitorType",
    "Weekend",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

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
]

CONTINUOUS_FEATURES_FOR_OUTLIERS = [
    "Administrative",
    "Administrative_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
]

TARGET = "Revenue"

PAGE_DURATION_PAIRS = [
    ("Administrative", "Administrative_Duration"),
    ("Informational", "Informational_Duration"),
    ("ProductRelated", "ProductRelated_Duration"),
]

DURATION_CAPS = {
    "Administrative_Duration": 3600,
    "Informational_Duration": 3600,
    "ProductRelated_Duration": 36000,
}

VALID_MONTHS = {"Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
VALID_VISITOR_TYPES = {"Returning_Visitor", "New_Visitor", "Other"}
VALID_SPECIAL_DAY = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}


def load_data(
    filepath: str = "../data/raw/online_shoppers_intention.csv",
) -> pd.DataFrame:
    """Load raw dataset CSV."""
    df = pd.read_csv(filepath)
    print(f"[load_data] Loaded {df.shape[0]:,} rows x {df.shape[1]} columns.")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact-duplicate rows (125 found in EDA, ~1.0% of data)."""
    n_before = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    n_after = len(df_clean)
    print(
        f"[remove_duplicates] Rows: {n_before:,} -> {n_after:,} (removed {n_before - n_after} duplicates)"
    )
    return df_clean


def validate_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Check domain-level data integrity and report violations (no changes made here)."""
    n = len(df)
    print(f"\n[validate_business_rules] Checking {n:,} rows...")
    total_violations = 0

    for pages_col, dur_col in PAGE_DURATION_PAIRS:
        if pages_col not in df.columns or dur_col not in df.columns:
            continue
        dur_without_pages = ((df[pages_col] == 0) & (df[dur_col] > 0)).sum()
        pages_without_dur = ((df[pages_col] > 0) & (df[dur_col] == 0)).sum()
        if dur_without_pages > 0:
            print(
                f"    [WARN] {dur_without_pages} rows have {pages_col}==0 but {dur_col}>0"
            )
            total_violations += dur_without_pages
        if pages_without_dur > 0:
            print(
                f"    [WARN] {pages_without_dur} rows have {pages_col}>0 but {dur_col}==0"
            )
            total_violations += pages_without_dur

    if "BounceRates" in df.columns and "ExitRates" in df.columns:
        bounce_gt_exit = (df["BounceRates"] > df["ExitRates"]).sum()
        if bounce_gt_exit > 0:
            print(f"    [WARN] {bounce_gt_exit} rows have BounceRates > ExitRates")
            total_violations += bounce_gt_exit

    if "PageValues" in df.columns:
        neg_pv = (df["PageValues"] < 0).sum()
        if neg_pv > 0:
            print(f"    [WARN] {neg_pv} rows have negative PageValues")
            total_violations += neg_pv

    if "SpecialDay" in df.columns:
        invalid_sd = (~df["SpecialDay"].isin(VALID_SPECIAL_DAY)).sum()
        if invalid_sd > 0:
            print(f"    [WARN] {invalid_sd} rows have invalid SpecialDay values")
            total_violations += invalid_sd

    if total_violations == 0:
        print("    [OK] All business rules passed - no violations found.")
    else:
        print(f"    [TOTAL] {total_violations} violations detected across all rules.")
    return df


def fix_duration_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix page-count <-> duration mismatches (both directions):
      - pages == 0 but duration > 0  -> set duration to 0 (fixed rule)
      - pages > 0  but duration == 0 -> fill with the median of nonzero
        durations for that column.

    The second case uses a median computed over the full dataset. At
    12,000+ rows, a median computed on the full dataset vs. only the 80%
    training split differs negligibly -- this is the "low risk"
    aggregate-statistic case, not a target-based computation.
    """
    df = df.copy()
    total_fixed = 0

    for pages_col, dur_col in PAGE_DURATION_PAIRS:
        if pages_col not in df.columns or dur_col not in df.columns:
            continue

        mask_dur_no_pages = (df[pages_col] == 0) & (df[dur_col] > 0)
        n_fix1 = int(mask_dur_no_pages.sum())
        if n_fix1 > 0:
            df.loc[mask_dur_no_pages, dur_col] = 0
            print(
                f"    [{dur_col}] Set {n_fix1} rows to 0 (duration>0 but {pages_col}==0)"
            )
            total_fixed += n_fix1

        mask_pages_no_dur = (df[pages_col] > 0) & (df[dur_col] == 0)
        n_fix2 = int(mask_pages_no_dur.sum())
        if n_fix2 > 0:
            nonzero_median = df.loc[df[dur_col] > 0, dur_col].median()
            df.loc[mask_pages_no_dur, dur_col] = nonzero_median
            print(
                f"    [{dur_col}] Filled {n_fix2} rows with median {nonzero_median:.1f} ({pages_col}>0 but duration==0)"
            )
            total_fixed += n_fix2

    print(f"[fix_duration_consistency] Total rows fixed: {total_fixed}")
    return df


def cap_duration_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap extreme durations at fixed, business-defined limits (see DURATION_CAPS)."""
    df = df.copy()
    total_capped = 0
    for col, cap in DURATION_CAPS.items():
        if col not in df.columns:
            continue
        n_over = int((df[col] > cap).sum())
        if n_over > 0:
            df[col] = df[col].clip(upper=cap)
            print(
                f"    [{col}] Capped {n_over} rows at {cap:,} seconds ({cap / 3600:.0f} hr)"
            )
            total_capped += n_over
    print(f"[cap_duration_outliers] Total values capped: {total_capped}")
    return df


def clip_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce BounceRates/ExitRates in [0, 1], and BounceRates <= ExitRates."""
    df = df.copy()
    n_clipped = 0
    for rate_col in ["BounceRates", "ExitRates"]:
        if rate_col not in df.columns:
            continue
        out_of_range = int(((df[rate_col] < 0) | (df[rate_col] > 1)).sum())
        if out_of_range > 0:
            df[rate_col] = df[rate_col].clip(0, 1)
            print(f"    [{rate_col}] Clipped {out_of_range} values to [0, 1]")
            n_clipped += out_of_range

    if "BounceRates" in df.columns and "ExitRates" in df.columns:
        invalid_mask = df["BounceRates"] > df["ExitRates"]
        n_invalid = int(invalid_mask.sum())
        if n_invalid > 0:
            df.loc[invalid_mask, "BounceRates"] = df.loc[invalid_mask, "ExitRates"]
            print(
                f"    [BounceRates] Corrected {n_invalid} rows where BounceRates > ExitRates"
            )
            n_clipped += n_invalid

    print(f"[clip_rates] Total corrections: {n_clipped}")
    return df


def validate_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace invalid Month/VisitorType values with the column mode; coerce
    Weekend to bool. EDA found 0 invalid values, so this is currently a
    no-op safeguard.
    """
    df = df.copy()
    total_fixed = 0

    for col, valid_set in {
        "Month": VALID_MONTHS,
        "VisitorType": VALID_VISITOR_TYPES,
    }.items():
        if col not in df.columns:
            continue
        invalid_mask = ~df[col].isin(valid_set)
        n_invalid = int(invalid_mask.sum())
        if n_invalid > 0:
            mode_val = df.loc[~invalid_mask, col].mode()[0]
            df.loc[invalid_mask, col] = mode_val
            total_fixed += n_invalid

    if "Weekend" in df.columns:
        df["Weekend"] = df["Weekend"].astype(bool)

    print(f"[validate_categorical_values] Total corrections: {total_fixed}")
    return df


def group_rare_categories(df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    """
    Group rare ID-coded categories (count < threshold) into 'Other_xxx',
    to reduce dimensionality before One-Hot Encoding.

    Groups by FEATURE FREQUENCY, not by the target label, so it does not
    leak Revenue information. Without this step, Browser/TrafficType/
    Region/OperatingSystems get one-hot encoded with every single rare ID
    as its own column -- exactly the dimensionality blow-up this function
    exists to prevent.
    """
    df = df.copy()
    for col in ["OperatingSystems", "Browser", "TrafficType", "Region"]:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        counts = df[col].value_counts()
        rare_cats = counts[counts < threshold].index
        if len(rare_cats) > 0:
            print(
                f"    [{col}] Grouping {len(rare_cats)} rare categories into 'Other_{col}'."
            )
            df.loc[df[col].isin(rare_cats), col] = f"Other_{col}"
    return df


def drop_special_day(df: pd.DataFrame) -> pd.DataFrame:
    """Drop SpecialDay: weakest correlation with Revenue (r=-0.08) and CV
    PR-AUC does not drop when it's removed. Deterministic column drop --
    safe to run before the split."""
    if "SpecialDay" in df.columns:
        print("    [drop_special_day] Dropping 'SpecialDay' column.")
        df = df.drop(columns=["SpecialDay"])
    return df


def handle_missing_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values if any are present. Checks first and only
    imputes when actually necessary, so the log clearly shows "we
    checked and there was nothing to do" rather than silently looping
    over columns that never had anything to fill.
    """
    df = df.copy()
    missing_count = int(df.isnull().sum().sum())

    if missing_count == 0:
        print("[handle_missing_value] No missing values found.")
        return df

    print(
        f"[handle_missing_value] {missing_count} missing values found. Applying imputation..."
    )
    for col in NUMERICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])

    remaining_missing = int(df.isnull().sum().sum())
    if remaining_missing == 0:
        print("[handle_missing_value] All missing values handled.")
    else:
        print(
            f"[handle_missing_value] Warning: {remaining_missing} missing values remain."
        )
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean Revenue column to integer (0 or 1)."""
    df = df.copy()
    df[TARGET] = df[TARGET].astype(int)
    return df


def prepare_dataset_for_split(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run every cleaning step that is safe to apply BEFORE the train/test
    split. Outlier removal is intentionally NOT included here -- see
    remove_outliers_iqr_train() below.
    """
    print(f"\n[prepare_dataset_for_split] Starting. Input shape: {df.shape}")
    df = remove_duplicates(df)
    df = validate_business_rules(df)
    df = fix_duration_consistency(df)
    df = cap_duration_outliers(df)
    df = clip_rates(df)
    df = validate_categorical_values(df)
    df = group_rare_categories(df)
    df = drop_special_day(df)
    df = handle_missing_value(df)
    df = encode_target(df)
    print(f"[prepare_dataset_for_split] Done. Output shape: {df.shape}\n")
    return df


def run_preprocessing_pipeline(
    df: pd.DataFrame, outlier_method: str = "none"
) -> pd.DataFrame:
    """Entry point used both for the standalone CSV-export script and by
    preprocess_data(). outlier_method must be 'none' here."""
    if outlier_method != "none":
        raise ValueError(
            "Outlier removal cannot be applied before the train/test split. "
            "Call remove_outliers_iqr_train(X_train, y_train) after "
            "split_dataset() instead."
        )
    return prepare_dataset_for_split(df)


def remove_outliers_iqr_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    columns: list[str] | None = None,
    factor: float = 1.5,
):
    """
    Remove outlier rows using the IQR method.

    IMPORTANT: call this AFTER split_dataset(), on (X_train, y_train)
    only. Never on the full dataset or X_test -- it deletes rows.

    Returns
    -------
    (X_train_clean, y_train_clean)
    """
    columns = columns or CONTINUOUS_FEATURES_FOR_OUTLIERS
    mask = pd.Series(True, index=X_train.index)

    for col in columns:
        if col not in X_train.columns:
            continue
        q1 = X_train[col].quantile(0.25)
        q3 = X_train[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            print(f"    [skip] '{col}' has IQR==0; skipping.")
            continue
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        mask &= X_train[col].between(lower, upper)

    X_train_clean = X_train.loc[mask].copy()
    y_train_clean = y_train.loc[mask].copy()

    n_before, n_after = len(X_train), len(X_train_clean)
    pct_removed = (n_before - n_after) / n_before * 100 if n_before else 0.0
    print(
        f"[remove_outliers_iqr_train] Rows: {n_before:,} -> {n_after:,} ({pct_removed:.1f}% removed)"
    )
    if pct_removed > 20:
        print("    [WARNING] Removed more than 20% of rows.")

    return X_train_clean, y_train_clean


def build_preprocessor(scale_numerical: bool = False) -> ColumnTransformer:
    """OneHotEncoder for categorical features; StandardScaler for numerical
    features only when scale_numerical=True (KNN/SVM)."""
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    if scale_numerical:
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", cat_transformer, CATEGORICAL_FEATURES),
                ("num", StandardScaler(), NUMERICAL_FEATURES),
            ],
            verbose_feature_names_out=False,
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[("cat", cat_transformer, CATEGORICAL_FEATURES)],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
    return preprocessor


def get_smote(random_state: int = 42) -> SMOTE:
    """SMOTE oversamples the minority (Purchase) class. Must be used as a
    step inside an imblearn Pipeline (fit on training folds only)."""
    return SMOTE(random_state=random_state)


def preprocess_data(
    filepath: str = "data/raw/online_shoppers_intention.csv",
    outlier_method: str = "none",
) -> pd.DataFrame:
    """Load and clean data. Returns the cleaned DataFrame -- call
    split_dataset() on the result, then (optionally)
    remove_outliers_iqr_train() on X_train only."""
    df = load_data(filepath)
    df = run_preprocessing_pipeline(df, outlier_method=outlier_method)
    return df


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    input_path = project_root / "data" / "raw" / "online_shoppers_intention.csv"

    print(f"Loading raw data from {input_path}...")
    try:
        raw_df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {input_path}")
        sys.exit(1)

    clean_df = run_preprocessing_pipeline(raw_df, outlier_method="none")

    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cleaned_online_shoppers_intention.csv"
    clean_df.to_csv(output_path, index=False)
    print(f"[OK] Cleaned data saved! Shape: {clean_df.shape}")
    print(f"   Columns: {list(clean_df.columns)}")

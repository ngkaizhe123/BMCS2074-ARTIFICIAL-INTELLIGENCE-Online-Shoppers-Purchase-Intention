"""
data_preprocessing.py
----------------------
Cleaning, splitting, and preprocessing pipeline for the
Online Shoppers Intention dataset.

Pre-split (deterministic — no learned statistics)
--------------------------------------------------
 1. load_data                   – Read raw CSV
 2. remove_duplicates           – Drop ~125 exact-duplicate rows (EDA finding)
 3. validate_business_rules     – Report domain-level data integrity violations
 4. cap_duration_outliers       – Cap durations to fixed DURATION_CAPS constants
 5. clip_rates                  – Enforce BounceRates / ExitRates within [0, 1]
 6. drop_special_day            – Remove SpecialDay column
 7. encode_target               – Boolean Revenue → int (0 / 1)
 8. prepare_dataset_for_split   – Assembles steps 2-7; called by preprocess_data()
 9. preprocess_data             – Entry point: load → deterministic clean → return df

Post-split (fitted on training data only — inside sklearn/imblearn Pipeline)
-----------------------------------------------------------------------------
10. TrainFittedDataCleaner      – Learns medians/modes/rare-cats in fit(); applies
                                  stored values in transform() — safe for test set.
11. TrainingOutlierFilter       – IQR / Z-score row removal in fit_resample();
                                  imblearn skips it at predict time automatically.
12. build_preprocessor          – ColumnTransformer: OHE + optional StandardScaler
13. get_smote                   – SMOTE instance for use inside an imblearn Pipeline

Saving cleaned data (run as script)
-------------------------------------
    python src/data_preprocessing.py
    → writes  data/processed/cleaned_online_shoppers_intention.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
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

# All numeric-typed columns (used for scaling).
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

# Numerical features selected for outlier detection based on EDA.
# These include count/duration/rate variables with meaningful skewness
# and long-tailed distributions. Zero-inflated features are excluded
# because conventional IQR filtering can incorrectly remove valid
# non-zero observations.

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

# Page-count ↔ duration column pairs used by business-rule validation.
# Each tuple maps (page_count_column, duration_column).
PAGE_DURATION_PAIRS = [
    ("Administrative", "Administrative_Duration"),
    ("Informational", "Informational_Duration"),
    ("ProductRelated", "ProductRelated_Duration"),
]

# Business-meaningful upper limits for session durations (in seconds).
# Sessions exceeding these caps likely represent idle tabs, bots, or
# tracking errors rather than genuine user engagement.
DURATION_CAPS = {
    "Administrative_Duration": 3600,  # 1 hour
    "Informational_Duration": 3600,  # 1 hour
    "ProductRelated_Duration": 36000,  # 10 hours
}

# Valid values for categorical columns in the Online Shoppers dataset.
# NOTE: Jan and Apr are absent from the raw dataset.
VALID_MONTHS = {"Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"}
VALID_VISITOR_TYPES = {"Returning_Visitor", "New_Visitor", "Other"}
VALID_SPECIAL_DAY = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}


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
# Step 4 – Remove Outliers  (handled by TrainingOutlierFilter inside Pipeline)
# ---------------------------------------------------------------------------


# Outlier removal is handled exclusively by TrainingOutlierFilter (see below),
# which is an imblearn sampler fitted inside the model pipeline on training
# data only.  Standalone IQR/Z-score functions were removed to prevent
# accidental pre-split leakage.


# ---------------------------------------------------------------------------
# Step 5 – Encode Target
# ---------------------------------------------------------------------------


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert boolean Revenue column to integer (0 or 1).

    EDA confirmed Revenue is a boolean column (True/False).
    True → 1 (purchase) and False → 0 (no purchase).
    This is deterministic and safe to apply before the train/test split.
    """
    df = df.copy()
    df[TARGET] = df[TARGET].astype(int)
    return df


# ---------------------------------------------------------------------------
# Step 6 – Business Rule Validation (flag logical data-quality issues)
# ---------------------------------------------------------------------------


def validate_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check domain-level data integrity and report violations.

    Business rules validated
    -------------------------
    1. Duration without pages: if pages_viewed == 0, duration should be 0.
       A non-zero duration with zero page views indicates a tracking error
       (the user never visited that section, so no time should be recorded).
    2. Pages without duration: if pages_viewed > 0, duration should be > 0.
       Visiting pages with exactly 0 seconds recorded is suspicious and may
       indicate bot traffic or incomplete tracking.
    3. BounceRates <= ExitRates: by Google Analytics definition, bounce rate
       is a special case of exit rate (single-page sessions). A session
       cannot have BounceRate > ExitRate.
    4. Non-negative PageValues: PageValues represents the average monetary
       value of pages visited before a transaction — negative values are
       not meaningful.
    5. Valid SpecialDay values: the dataset encodes proximity to a special
       day using fixed intervals {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}. Values
       outside this set suggest data corruption.

    This function REPORTS violations via print statements but does NOT
    drop rows or modify data — the downstream fix functions handle that.
    """
    n = len(df)
    print(f"\n[validate_business_rules] Checking {n:,} rows...")
    total_violations = 0

    # Rule 1 & 2: Page-count ↔ duration consistency
    for pages_col, dur_col in PAGE_DURATION_PAIRS:
        if pages_col not in df.columns or dur_col not in df.columns:
            continue
        dur_without_pages = ((df[pages_col] == 0) & (df[dur_col] > 0)).sum()
        pages_without_dur = ((df[pages_col] > 0) & (df[dur_col] == 0)).sum()
        if dur_without_pages > 0:
            print(
                f"    [WARN] {dur_without_pages} rows have {pages_col}==0 "
                f"but {dur_col}>0 (duration without page visits)"
            )
            total_violations += dur_without_pages
        if pages_without_dur > 0:
            print(
                f"    [WARN] {pages_without_dur} rows have {pages_col}>0 "
                f"but {dur_col}==0 (page visits without duration)"
            )
            total_violations += pages_without_dur

    # Rule 3: BounceRates <= ExitRates
    if "BounceRates" in df.columns and "ExitRates" in df.columns:
        bounce_gt_exit = (df["BounceRates"] > df["ExitRates"]).sum()
        if bounce_gt_exit > 0:
            print(
                f"    [WARN] {bounce_gt_exit} rows have BounceRates > ExitRates "
                "(violates GA definition)"
            )
            total_violations += bounce_gt_exit

    # Rule 4: Non-negative PageValues
    if "PageValues" in df.columns:
        neg_pv = (df["PageValues"] < 0).sum()
        if neg_pv > 0:
            print(f"    [WARN] {neg_pv} rows have negative PageValues")
            total_violations += neg_pv

    # Rule 5: Valid SpecialDay values
    if "SpecialDay" in df.columns:
        invalid_sd = (~df["SpecialDay"].isin(VALID_SPECIAL_DAY)).sum()
        if invalid_sd > 0:
            print(
                f"    [WARN] {invalid_sd} rows have SpecialDay values "
                f"outside {sorted(VALID_SPECIAL_DAY)}"
            )
            total_violations += invalid_sd

    if total_violations == 0:
        print("    [OK] All business rules passed — no violations found.")
    else:
        print(f"    [TOTAL] {total_violations} violations detected across all rules.")

    return df


# fix_duration_consistency (median-learned) is handled by
# TrainFittedDataCleaner.transform() inside the model pipeline.


# ---------------------------------------------------------------------------
# Step 8 – Cap Duration Outliers (business-meaningful upper limits)
# ---------------------------------------------------------------------------


def cap_duration_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap extreme session durations at business-meaningful upper limits.

    Business logic
    ---------------
    Unlike statistical outlier removal (IQR / Z-score) which drops entire
    rows, capping *preserves* the session while limiting the impact of
    extreme values that almost certainly represent idle tabs, bots, or
    tracking errors rather than genuine browsing.

    Caps applied (see DURATION_CAPS constant):
        - Administrative_Duration  → 3 600 s  (1 hour)
        - Informational_Duration   → 3 600 s  (1 hour)
        - ProductRelated_Duration  → 36 000 s (10 hours)

    These thresholds are deliberately generous so that only truly
    unreasonable values are affected. For example, a 10-hour product
    browsing session is already extreme but plausible for comparison
    shopping; beyond that is almost certainly an abandoned tab.

    This step is especially valuable for distance-based models (KNN, SVM)
    where a single extreme value can distort the entire distance space.
    """
    df = df.copy()
    total_capped = 0

    for col, cap in DURATION_CAPS.items():
        if col not in df.columns:
            continue
        n_over = (df[col] > cap).sum()
        if n_over > 0:
            df[col] = df[col].clip(upper=cap)
            print(
                f"    [{col}] Capped {n_over} rows at {cap:,} seconds "
                f"({cap / 3600:.0f} hr)"
            )
            total_capped += n_over

    print(f"[cap_duration_outliers] Total values capped: {total_capped}")
    return df


# ---------------------------------------------------------------------------
# Step 9 – Clip Rates (enforce [0, 1] bounds and BounceRates ≤ ExitRates)
# ---------------------------------------------------------------------------


def clip_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce valid boundaries on BounceRates and ExitRates.

    Business logic
    ---------------
    1. Both rates must lie in [0, 1] — they represent proportions.
    2. BounceRates ≤ ExitRates — by Google Analytics definition, a "bounce"
       is a single-page session exit. Every bounce is an exit, so the bounce
       rate for a page can never exceed its exit rate.

    When BounceRates > ExitRates, this function sets BounceRates = ExitRates
    (choosing the more reliable metric as the anchor, since ExitRates is
    computed from a larger sample of pageviews).
    """
    df = df.copy()
    n_clipped = 0

    for rate_col in ["BounceRates", "ExitRates"]:
        if rate_col not in df.columns:
            continue
        out_of_range = ((df[rate_col] < 0) | (df[rate_col] > 1)).sum()
        if out_of_range > 0:
            df[rate_col] = df[rate_col].clip(0, 1)
            print(f"    [{rate_col}] Clipped {out_of_range} values to [0, 1]")
            n_clipped += out_of_range

    # Fix BounceRates > ExitRates
    if "BounceRates" in df.columns and "ExitRates" in df.columns:
        invalid_mask = df["BounceRates"] > df["ExitRates"]
        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            df.loc[invalid_mask, "BounceRates"] = df.loc[invalid_mask, "ExitRates"]
            print(
                f"    [BounceRates] Corrected {n_invalid} rows where "
                "BounceRates > ExitRates (set BounceRates = ExitRates)"
            )
            n_clipped += n_invalid

    print(f"[clip_rates] Total corrections: {n_clipped}")
    return df


# validate_categorical_values and group_rare_categories (mode/frequency-learned)
# are handled by TrainFittedDataCleaner.fit() / .transform() inside the model
# pipeline.  Standalone versions were removed to prevent accidental pre-split leakage.


def drop_special_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop SpecialDay column from the dataset.

    SpecialDay encodes proximity to a commercial holiday as a fixed ordinal
    (0.0 – 1.0).  It is dropped because it is heavily zero-inflated and
    adds little predictive signal once other session features are present.
    This is a deterministic column drop — safe to run before the split.
    """
    if "SpecialDay" in df.columns:
        print("    [drop_special_day] Dropping 'SpecialDay' column.")
        df = df.drop(columns=["SpecialDay"])
    return df


# ---------------------------------------------------------------------------
# Step 11 – Unified Cleaning Pipeline (used for CSV export & preprocess_data)
# ---------------------------------------------------------------------------


def prepare_dataset_for_split(df: pd.DataFrame) -> pd.DataFrame:
    """Apply only deterministic, non-data-learned preparation before splitting.

    Anything that learns from feature distributions belongs in a fitted pipeline,
    not here.  This is intentionally the only preparation allowed before the
    train/test split is made.
    """
    print(f"\n[prepare_dataset_for_split] Starting. Input shape: {df.shape}")
    df = remove_duplicates(df)
    df = validate_business_rules(df)

    # These rules use fixed, domain-defined constants only.
    df = df.copy()
    for pages_col, dur_col in PAGE_DURATION_PAIRS:
        if pages_col in df.columns and dur_col in df.columns:
            df.loc[(df[pages_col] == 0) & (df[dur_col] > 0), dur_col] = 0
    df = cap_duration_outliers(df)
    df = clip_rates(df)
    df = drop_special_day(df)
    df = encode_target(df)
    print(f"[prepare_dataset_for_split] Done. Output shape: {df.shape}\n")
    return df


class TrainFittedDataCleaner(BaseEstimator, TransformerMixin):
    """Data-dependent cleaning fitted on a training partition only.

    The transformer learns imputation values, duration medians, categorical
    replacement modes, and rare-category maps in ``fit``.  ``transform``
    applies those already-fitted values to either training or test data and
    never removes rows.
    """

    def __init__(self, rare_threshold: int = 10):
        self.rare_threshold = rare_threshold

    def fit(self, X, y=None):
        X = X.copy()
        self.numeric_medians_ = {
            c: X[c].median() for c in NUMERICAL_FEATURES if c in X.columns
        }
        self.categorical_modes_ = {
            c: X[c].mode().iloc[0]
            for c in CATEGORICAL_FEATURES
            if c in X.columns and not X[c].mode().empty
        }
        self.nonzero_duration_medians_ = {}
        for pages_col, dur_col in PAGE_DURATION_PAIRS:
            if pages_col in X.columns and dur_col in X.columns:
                values = X.loc[X[dur_col] > 0, dur_col]
                self.nonzero_duration_medians_[dur_col] = (
                    values.median() if not values.empty else 0.0
                )
        self.rare_categories_ = {}
        for col in ["OperatingSystems", "Browser", "TrafficType", "Region"]:
            if col in X.columns:
                counts = X[col].astype(str).value_counts()
                self.rare_categories_[col] = set(
                    counts[counts < self.rare_threshold].index
                )
        return self

    def transform(self, X):
        X = X.copy()
        for col, value in self.numeric_medians_.items():
            if col in X.columns:
                X[col] = X[col].fillna(value)
        for col, value in self.categorical_modes_.items():
            if col in X.columns:
                X[col] = X[col].fillna(value)

        for pages_col, dur_col in PAGE_DURATION_PAIRS:
            if pages_col not in X.columns or dur_col not in X.columns:
                continue
            X.loc[(X[pages_col] == 0) & (X[dur_col] > 0), dur_col] = 0
            missing_duration = (X[pages_col] > 0) & (X[dur_col] == 0)
            fallback = self.nonzero_duration_medians_.get(dur_col, 0.0)
            X.loc[missing_duration, dur_col] = fallback

        valid_sets = {"Month": VALID_MONTHS, "VisitorType": VALID_VISITOR_TYPES}
        for col, valid in valid_sets.items():
            if col in X.columns:
                invalid = ~X[col].isin(valid)
                if invalid.any():
                    X.loc[invalid, col] = self.categorical_modes_[col]
        if "Weekend" in X.columns:
            X["Weekend"] = X["Weekend"].astype(bool)

        for col, rare_values in self.rare_categories_.items():
            if col in X.columns:
                X[col] = X[col].astype(str)
                X.loc[X[col].isin(rare_values), col] = f"Other_{col}"
        return X


class TrainingOutlierFilter(BaseEstimator):
    """imblearn sampler that removes rows only while a pipeline is fitted.

    Its ``fit_resample`` learns bounds and returns filtered training rows.
    imblearn deliberately skips samplers during ``predict``/``predict_proba``;
    test observations therefore cannot be removed.
    """

    def __init__(
        self,
        method: str = "iqr",
        columns=None,
        factor: float = 1.5,
        threshold: float = 3.0,
    ):
        self.method = method
        self.columns = columns
        self.factor = factor
        self.threshold = threshold

    def fit_resample(self, X, y):
        if self.method not in {"iqr", "zscore"}:
            raise ValueError("method must be 'iqr' or 'zscore'")
        columns = self.columns or CONTINUOUS_FEATURES_FOR_OUTLIERS
        self.columns_ = [c for c in columns if c in X.columns]
        mask = pd.Series(True, index=X.index)
        self.bounds_ = {}
        for col in self.columns_:
            if self.method == "iqr":
                q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
                spread = q3 - q1
                if spread == 0:
                    continue
                lower, upper = q1 - self.factor * spread, q3 + self.factor * spread
                self.bounds_[col] = (lower, upper)
                mask &= X[col].between(lower, upper)
            else:
                mean, std = X[col].mean(), X[col].std()
                if std == 0 or pd.isna(std):
                    continue
                self.bounds_[col] = (mean, std)
                mask &= (X[col] - mean).abs() < self.threshold * std
        self.n_samples_before_ = len(X)
        self.n_samples_after_ = int(mask.sum())
        self.n_samples_removed_ = self.n_samples_before_ - self.n_samples_after_
        return X.loc[mask], (
            y.loc[mask] if hasattr(y, "loc") else np.asarray(y)[mask.to_numpy()]
        )


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    outlier_method: str = "none",
) -> pd.DataFrame:
    """
    Apply only deterministic, pre-split preparation to a raw DataFrame.

    This function is called both when:
      a) Running this script directly to save cleaned_online_shoppers_intention.csv
      b) Inside preprocess_data() before the train/test split

    Steps applied (all use fixed constants — no learned statistics)
    ---------------------------------------------------------------
    1. remove_duplicates       – Drop exact-duplicate rows (~125 found in EDA)
    2. validate_business_rules – Report domain-level data-quality violations
    3. duration zero-fix       – Set duration=0 where pages_count==0 (constant rule)
    4. cap_duration_outliers   – Clip durations to DURATION_CAPS constants
    5. clip_rates              – Enforce BounceRates / ExitRates in [0, 1]
    6. drop_special_day        – Remove SpecialDay column
    7. encode_target           – Revenue bool → int

    Steps deferred to TrainFittedDataCleaner / TrainingOutlierFilter (inside Pipeline)
    -----------------------------------------------------------------------------------
    - Missing value imputation    (learns median / mode from training data)
    - Duration consistency fix    (learns nonzero median from training data)
    - Categorical validation       (learns mode from training data)
    - Rare category grouping       (learns frequency counts from training data)
    - IQR / Z-score outlier removal (learns Q1/Q3 or mean/std from training data)
    """
    print(f"\n[run_preprocessing_pipeline] Starting. Input shape: {df.shape}")

    if outlier_method != "none":
        raise ValueError(
            "Outlier removal must be a TrainingOutlierFilter inside a training "
            "pipeline. It cannot be applied before the train/test split."
        )
    return prepare_dataset_for_split(df)


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
) -> pd.DataFrame:
    """
    Load and clean data.

    Parameters
    ----------
    filepath        : Path to dataset CSV.
    outlier_method  : 'none' | 'iqr' | 'zscore'

    Returns
    -------
    pd.DataFrame
        The cleaned dataset.
    """
    # 1. Load
    df = load_data(filepath)

    # Only deterministic preparation is allowed before the train/test split.
    df = run_preprocessing_pipeline(df, outlier_method=outlier_method)

    return df


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

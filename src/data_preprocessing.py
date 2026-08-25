"""
data_preprocessing.py
----------------------
Cleaning, splitting, and preprocessing pipeline for the
Online Shoppers Intention dataset.

Pipeline overview
-----------------
 1. load_data                    – Read raw CSV
 2. remove_duplicates            – Drop 125 exact-duplicate rows (found via EDA)
 3. handle_missing_values        – Median/mode imputation (safeguard; raw data has none)
 4. remove_outliers_iqr          – IQR method (for KNN / SVM that are distance-sensitive)
 5. remove_outliers_zscore       – Z-score method (alternative to IQR)
 6. encode_target                – Boolean Revenue → int (0 / 1)
 7. validate_business_rules      – Domain-level data integrity checks
 8. fix_duration_consistency     – Fix page-count ↔ duration mismatches
 9. cap_duration_outliers        – Cap extreme durations to business-meaningful limits
10. clip_rates                   – Enforce BounceRates / ExitRates within [0, 1]
11. validate_categorical_values  – Ensure categorical columns have valid values
12. run_preprocessing_pipeline   – Steps 2-11 assembled; also called by preprocess_data()
13. build_preprocessor           – sklearn ColumnTransformer (OHE + optional StandardScaler)
14. get_smote                    – SMOTE instance for training pipelines
15. preprocess_data              – Full pipeline: clean → split → (optional transform)

Saving cleaned data (run as script)
------------------------------------
    python src/data_preprocessing.py
    → writes  data/processed/cleaned_online_shoppers_intention.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy import stats
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

    n_total = len(df)
    print(f"\n[remove_outliers_iqr] IQR diagnostics (factor={factor}):")
    print(
        f"    {'Column':<26}{'Q1':>12}{'Median':>12}{'Q3':>12}{'IQR':>12}"
        f"{'Lower':>14}{'Upper':>14}{'Flagged':>12}{'% Flagged':>12}"
    )
    print("    " + "-" * 126)

    mask = pd.Series(True, index=df.index)
    for col in columns:
        Q1 = df[col].quantile(0.25)
        med = df[col].quantile(0.50)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        if IQR == 0:
            print(
                f"    {col:<26}{Q1:>12.3f}{med:>12.3f}{Q3:>12.3f}{IQR:>12.3f}"
                f"{'--':>14}{'--':>14}{'SKIPPED':>12}{'--':>12}"
            )
            print(
                f"        [skip] IQR==0 (Q1==Q3=={Q1}); skipping to avoid "
                "flagging every nonzero value as an outlier."
            )
            continue

        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        col_mask = df[col].between(lower, upper)
        n_flagged = int((~col_mask).sum())
        pct_flagged = n_flagged / n_total * 100 if n_total else 0.0

        print(
            f"    {col:<26}{Q1:>12.3f}{med:>12.3f}{Q3:>12.3f}{IQR:>12.3f}"
            f"{lower:>14.3f}{upper:>14.3f}{n_flagged:>12,}{pct_flagged:>11.2f}%"
        )

        mask &= col_mask

    print("    " + "-" * 126)
    n_flagged_any = int((~mask).sum())
    pct_flagged_any = n_flagged_any / n_total * 100 if n_total else 0.0
    print(
        f"    [combined] Rows flagged by >=1 column: {n_flagged_any:,} "
        f"({pct_flagged_any:.2f}% of {n_total:,})\n"
    )

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


# ---------------------------------------------------------------------------
# Step 7 – Fix Duration Consistency
# ---------------------------------------------------------------------------


def fix_duration_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix cross-feature inconsistencies between page-count and duration columns.

    Business logic
    ---------------
    - If pages_viewed == 0 but duration > 0:
        → Set duration to 0 (can't spend time on pages never visited).
    - If pages_viewed > 0 but duration == 0:
        → Replace duration with the median of non-zero durations for that
          feature (a visit must take *some* time; 0 is a tracking gap).

    These inconsistencies are likely caused by incomplete tracking or
    session-timeout quirks in Google Analytics. Leaving them unfixed
    introduces noise: e.g. a session with 10 product pages but 0 seconds
    would mislead the model into thinking heavy browsing = instant action.
    """
    df = df.copy()
    total_fixed = 0

    for pages_col, dur_col in PAGE_DURATION_PAIRS:
        if pages_col not in df.columns or dur_col not in df.columns:
            continue

        # Case 1: duration > 0 but no pages visited → zero out duration
        mask_dur_no_pages = (df[pages_col] == 0) & (df[dur_col] > 0)
        n_fix1 = mask_dur_no_pages.sum()
        if n_fix1 > 0:
            df.loc[mask_dur_no_pages, dur_col] = 0
            print(
                f"    [{dur_col}] Set {n_fix1} rows to 0 "
                f"(had duration > 0 but {pages_col} == 0)"
            )
            total_fixed += n_fix1

        # Case 2: pages > 0 but duration == 0 → fill with median of non-zero
        mask_pages_no_dur = (df[pages_col] > 0) & (df[dur_col] == 0)
        n_fix2 = mask_pages_no_dur.sum()
        if n_fix2 > 0:
            nonzero_median = df.loc[df[dur_col] > 0, dur_col].median()
            df.loc[mask_pages_no_dur, dur_col] = nonzero_median
            print(
                f"    [{dur_col}] Filled {n_fix2} rows with median {nonzero_median:.1f} "
                f"(had {pages_col} > 0 but duration == 0)"
            )
            total_fixed += n_fix2

    print(f"[fix_duration_consistency] Total rows fixed: {total_fixed}")
    return df


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


# ---------------------------------------------------------------------------
# Step 10 – Validate Categorical Values
# ---------------------------------------------------------------------------


def validate_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure categorical columns contain only valid values.

    Business logic
    ---------------
    - Month: the raw dataset contains 10 months (Jan and Apr are absent).
      Any value outside the known set is replaced with the column mode.
    - VisitorType: must be one of Returning_Visitor, New_Visitor, Other.
    - Weekend: must be boolean (True/False).

    Invalid categories can arise from manual data entry (e.g. the Live
    Prediction form), data merges, or encoding errors. Replacing with
    mode is safe because these columns have dominant categories
    (e.g. ~85% Returning_Visitor) and a single misfire won't bias the model.
    """
    df = df.copy()
    total_fixed = 0

    validations = {
        "Month": VALID_MONTHS,
        "VisitorType": VALID_VISITOR_TYPES,
    }

    for col, valid_set in validations.items():
        if col not in df.columns:
            continue
        invalid_mask = ~df[col].isin(valid_set)
        n_invalid = invalid_mask.sum()
        if n_invalid > 0:
            mode_val = df.loc[~invalid_mask, col].mode()[0]
            invalid_values = df.loc[invalid_mask, col].unique().tolist()
            df.loc[invalid_mask, col] = mode_val
            print(
                f"    [{col}] Replaced {n_invalid} invalid values "
                f"{invalid_values} with mode '{mode_val}'"
            )
            total_fixed += n_invalid

    # Weekend: coerce to boolean
    if "Weekend" in df.columns:
        try:
            df["Weekend"] = df["Weekend"].astype(bool)
        except (ValueError, TypeError):
            n_bad = df["Weekend"].apply(lambda x: x not in (True, False, 0, 1)).sum()
            if n_bad > 0:
                print(f"    [Weekend] {n_bad} non-boolean values coerced")
                total_fixed += n_bad

    if total_fixed == 0:
        print("[validate_categorical_values] All categories valid.")
    else:
        print(f"[validate_categorical_values] Total corrections: {total_fixed}")

    return df


def drop_special_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop SpecialDay column from the dataset as requested.
    """
    if "SpecialDay" in df.columns:
        print("    [drop_special_day] Dropping 'SpecialDay' column.")
        df = df.drop(columns=["SpecialDay"])
    return df


def group_rare_categories(
    df: pd.DataFrame,
    threshold: int = 10,
    rare_map: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """
    Group rare categories into an 'Other_xxx' category.
    This applies to integer ID-coded categorical features (OperatingSystems, Browser,
    TrafficType, Region) to reduce noise and dimensionality before One-Hot Encoding.
    """
    df = df.copy()
    cols_to_group = ["OperatingSystems", "Browser", "TrafficType", "Region"]
    for col in cols_to_group:
        if col in df.columns:
            # Convert column to string so we can mix original IDs with 'Other_xxx'
            df[col] = df[col].astype(str)
            if rare_map and col in rare_map:
                rare_cats = rare_map[col]
            elif len(df) >= threshold:
                counts = df[col].value_counts()
                rare_cats = counts[counts < threshold].index.tolist()
            else:
                rare_cats = []

            if len(rare_cats) > 0:
                print(
                    f"    [{col}] Grouping {len(rare_cats)} rare categories into 'Other_{col}'."
                )
                df.loc[df[col].isin(rare_cats), col] = f"Other_{col}"
    return df


# ---------------------------------------------------------------------------
# Step 11 – Unified Cleaning Pipeline (used for CSV export & preprocess_data)
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
    1. remove_duplicates            – Drop 125 duplicate rows found in EDA
    2. handle_missing_values        – Median/mode imputation (safeguard)
    3. validate_business_rules      – Flag domain-level data-quality issues
    4. fix_duration_consistency     – Fix page-count ↔ duration mismatches
    5. cap_duration_outliers        – Cap extreme durations to business limits
    6. clip_rates                   – Enforce BounceRates/ExitRates in [0,1]
    7. validate_categorical_values  – Replace invalid category values
    8. group_rare_categories        – Group rare ID categories into 'Other_xxx'
    9. remove_outliers_*            – Optional IQR or Z-score outlier removal
    10. encode_target               – Revenue bool → int
    """
    print(f"\n[run_preprocessing_pipeline] Starting. Input shape: {df.shape}")

    # --- Technical cleaning ---
    df = remove_duplicates(df)
    df = handle_missing_values(df)

    # --- Business rule validation & fixes ---
    df = validate_business_rules(df)
    df = fix_duration_consistency(df)
    df = cap_duration_outliers(df)
    df = clip_rates(df)
    df = validate_categorical_values(df)
    df = group_rare_categories(df)
    df = drop_special_day(df)

    # --- Statistical outlier removal (optional) ---
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

    # 2–5. Clean (duplicates → missing values → outliers → encode target)
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

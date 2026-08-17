"""
eda.py
------
Exploratory Data Analysis for the Online Shoppers Intention dataset.

Sections
--------
 1.  Dataset Overview & Quality Check
 2.  Target Distribution & Imbalance
 3.  Numerical Feature Distributions
 4.  Box Plots (Outlier Inspection)
 5.  Categorical Feature Distributions
 6.  Correlation Heatmap
 7.  PageValues Deep-Dive
 8.  BounceRates / ExitRates Analysis
 9.  Monthly Purchase Rate
10.  Visitor Type Analysis
11.  Weekend Effect
12.  Feature Interaction Analysis
13.  Statistical Summary Table
14.  Key Findings Summary

Usage (from project root)
--------------------------
    from src.eda import run_eda
    df = run_eda(save_dir="report_assets/plots")
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

PALETTE_BINARY = ["#4C72B0", "#DD8452"]  # blue = 0 / No, orange = 1 / Yes
SNS_STYLE = "whitegrid"

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
]

CATEGORICAL_FEATURES = ["Month", "VisitorType", "Weekend"]
ORDINAL_FEATURES = ["OperatingSystems", "Browser", "Region", "TrafficType"]

MONTH_ORDER = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _save_show(fig: plt.Figure, name: str, save_dir: str | None, show: bool) -> None:
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ===================================================================
# 1. Dataset Overview & Quality Check
# ===================================================================


# Checks raw dataset shape, dtypes, duplicate rows, and missing values.
# Duplicates found here (125 rows) -> handled by remove_duplicates() in data_preprocessing.py.
# Missing values confirmed as 0 -> handle_missing_values() is kept in the pipeline as a safeguard.
# Revenue dtype is bool -> encode_target() converts it to int (0/1) before model training.
def plot_dataset_overview(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    print("=" * 70)
    print("1. DATASET OVERVIEW & QUALITY CHECK")
    print("=" * 70)
    print(f"  Shape            : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory usage     : {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    print(f"  Duplicated rows  : {df.duplicated().sum()}")
    missing = df.isnull().sum()
    n_missing = missing[missing > 0]
    print(
        f"  Missing values   : {n_missing.sum()} "
        f"({n_missing.to_dict() if len(n_missing) else 'None'})"
    )
    print(f"\n  Data Types:")
    for col in df.columns:
        print(f"    {col:<30s} {str(df[col].dtype)}")
    print(f"\n  Numerical Summary:")
    print(df[NUMERICAL_FEATURES].describe().round(3).to_string())


# ===================================================================
# 2. Target Distribution & Imbalance
# ===================================================================


# Shows the class imbalance: ~84.5% No Purchase vs ~15.5% Purchase (ratio ~5.4:1).
# -> get_smote() in data_preprocessing.py oversamples the minority class (Purchase=1)
#    inside the imblearn Pipeline, so resampling only happens on training folds (no data leakage).
# -> train_test_split uses stratify=y to preserve this ratio across train/test splits.
def plot_target_distribution(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    counts = df["Revenue"].value_counts().sort_index()
    labels = ["No Purchase (0)", "Purchase (1)"]
    pcts = counts / counts.sum() * 100
    ratio = counts[0] / counts[1]

    sns.set_style(SNS_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Target Variable – Revenue", fontsize=14, fontweight="bold")

    # Bar
    bars = axes[0].bar(labels, counts.values, color=PALETTE_BINARY, edgecolor="w")
    axes[0].set_title("Count")
    axes[0].set_ylabel("Sessions")
    for b, c, p in zip(bars, counts.values, pcts.values):
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 40,
            f"{c:,}\n({p:.1f}%)",
            ha="center",
            fontsize=10,
        )

    # Pie
    axes[1].pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=PALETTE_BINARY,
        startangle=90,
        wedgeprops={"edgecolor": "w"},
    )
    axes[1].set_title("Proportion")

    plt.tight_layout()
    _save_show(fig, "01_target_distribution", save_dir, show)

    print(
        f"\n  Target: No Purchase = {counts[0]:,} ({pcts[0]:.1f}%), "
        f"Purchase = {counts[1]:,} ({pcts[1]:.1f}%)"
    )
    print(f"  Imbalance ratio ~ {ratio:.2f}:1")


# ===================================================================
# 3. Numerical Feature Distributions
# ===================================================================


# Histograms reveal heavy right-skew and long tails across most numerical features.
# -> For distance-sensitive models (KNN, SVM): remove_outliers_iqr() or remove_outliers_zscore()
#    should be used (outlier_method='iqr' or 'zscore' in preprocess_data()).
# -> For tree-based models (XGBoost): robust to skew by design; outlier_method='none' (default).
# -> StandardScaler in build_preprocessor(scale_numerical=True) is applied for KNN/SVM
#    to normalise magnitudes before distance computation.
def plot_numerical_distributions(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    sns.set_style(SNS_STYLE)
    n_cols = 2
    n_rows = int(np.ceil(len(NUMERICAL_FEATURES) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(NUMERICAL_FEATURES):
        ax = axes[i]
        for rev, color, lbl in zip([0, 1], PALETTE_BINARY, ["No Purchase", "Purchase"]):
            s = df.loc[df["Revenue"] == rev, col].dropna()
            ax.hist(s, bins=40, alpha=0.5, color=color, label=lbl, density=True)
            s.plot.kde(ax=ax, color=color, lw=1.5)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Numerical Features by Revenue", fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save_show(fig, "02_numerical_distributions", save_dir, show)


# ===================================================================
# 4. Box Plots
# ===================================================================


# Box plots show extreme outliers in: Administrative, Administrative_Duration,
# ProductRelated, ProductRelated_Duration, BounceRates, ExitRates.
# -> These 6 columns form CONTINUOUS_FEATURES_FOR_OUTLIERS in data_preprocessing.py;
#    only these are used for IQR/Z-score outlier removal.
# Informational*, PageValues, and SpecialDay are intentionally EXCLUDED from outlier removal
# because their IQR == 0 (zero-inflated): applying IQR would flag nearly all non-zero
# values as outliers, deleting the majority of the purchasing sessions.
def plot_boxplots(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    sns.set_style(SNS_STYLE)
    n_cols = 2
    n_rows = int(np.ceil(len(NUMERICAL_FEATURES) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(NUMERICAL_FEATURES):
        ax = axes[i]
        sns.boxplot(data=df, x="Revenue", y=col, ax=ax, palette=PALETTE_BINARY)
        ax.set_title(col, fontsize=10)
        ax.set_xticklabels(["No", "Yes"])
        ax.set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Box Plots by Revenue", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_show(fig, "03_boxplots", save_dir, show)


# ===================================================================
# 5. Categorical Feature Distributions
# ===================================================================


# Stacked bar charts show purchase rate by each categorical feature.
# Month, VisitorType, and Weekend are nominal (no numeric ordering)
# -> OneHotEncoder is applied to CATEGORICAL_FEATURES in build_preprocessor().
# OperatingSystems, Browser, Region, TrafficType are integer-coded IDs with no
# meaningful numeric scale -> treated as NUMERICAL_FEATURES (passthrough for XGBoost,
# StandardScaler for KNN/SVM).
def plot_categorical_distributions(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    sns.set_style(SNS_STYLE)
    all_cats = CATEGORICAL_FEATURES + ORDINAL_FEATURES
    n_cols = 2
    n_rows = int(np.ceil(len(all_cats) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(all_cats):
        ax = axes[i]
        ct = pd.crosstab(df[col], df["Revenue"], normalize="index") * 100
        ct.columns = ["No Purchase", "Purchase"]
        ct.plot(kind="bar", stacked=True, ax=ax, color=PALETTE_BINARY, edgecolor="w")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(fontsize=7)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Purchase Rate by Category", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_show(fig, "04_categorical_distributions", save_dir, show)


# ===================================================================
# 6. Correlation Heatmap
# ===================================================================


# Correlation heatmap identifies which features are most predictive of Revenue.
# PageValues has the strongest positive correlation (~0.49) with Revenue
# -> PageValues is intentionally excluded from CONTINUOUS_FEATURES_FOR_OUTLIERS
#    so that outlier removal does not destroy the most informative feature.
# BounceRates and ExitRates are highly correlated with each other (~0.91)
# -> both columns are kept; no dimensionality reduction is applied at this stage.
def plot_correlation_heatmap(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    cols = NUMERICAL_FEATURES + ["Revenue"]
    corr = df[cols].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_show(fig, "05_correlation_heatmap", save_dir, show)

    # Print top correlations with Revenue
    rev_corr = corr["Revenue"].drop("Revenue").sort_values(ascending=False)
    print("\n  Correlation with Revenue (top 5):")
    for feat, val in rev_corr.head(5).items():
        print(f"    {feat:<30s} {val:+.4f}")


# ===================================================================
# 7. PageValues Deep-Dive
# ===================================================================


# Deep-dive shows PageValues is zero in ~75% of sessions (zero-inflated).
# The IQR of PageValues equals 0 across this distribution, so IQR-based outlier
# removal would incorrectly flag ALL non-zero PageValues as outliers.
# -> PageValues is excluded from CONTINUOUS_FEATURES_FOR_OUTLIERS in data_preprocessing.py.
# Sessions with PageValues > 0 have a ~49% purchase rate vs ~2% when PageValues == 0,
# confirming it is the single most important feature and must not be removed.
def plot_pagevalues_deep_dive(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    sns.set_style(SNS_STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PageValues – Deep-Dive", fontsize=14, fontweight="bold")

    # (a) Violin
    sns.violinplot(
        data=df,
        x="Revenue",
        y="PageValues",
        ax=axes[0, 0],
        palette=PALETTE_BINARY,
        inner="quartile",
    )
    axes[0, 0].set_xticklabels(["No", "Yes"])
    axes[0, 0].set_title("Distribution by Revenue")

    # (b) Zero vs Non-zero
    pv_zero = (df["PageValues"] == 0).mean() * 100
    pv_nz_rate = df.loc[df["PageValues"] > 0, "Revenue"].mean() * 100
    pv_z_rate = df.loc[df["PageValues"] == 0, "Revenue"].mean() * 100

    axes[0, 1].axis("off")
    txt = (
        f"PageValues = 0 : {pv_zero:.1f}% of sessions\n"
        f"  → Purchase rate: {pv_z_rate:.2f}%\n\n"
        f"PageValues > 0 : {100 - pv_zero:.1f}% of sessions\n"
        f"  → Purchase rate: {pv_nz_rate:.2f}%\n\n"
        f"⚡ PageValues > 0 is a strong\n"
        f"   signal for purchase."
    )
    axes[0, 1].text(
        0.05,
        0.5,
        txt,
        fontsize=11,
        va="center",
        bbox=dict(boxstyle="round", fc="#f5f5f5"),
    )

    # (c) Histogram of PageValues > 0
    pv_pos = df.loc[df["PageValues"] > 0, "PageValues"]
    axes[1, 0].hist(pv_pos, bins=50, color=PALETTE_BINARY[1], alpha=0.7, edgecolor="w")
    axes[1, 0].set_title("PageValues > 0 Distribution")
    axes[1, 0].set_xlabel("PageValues")

    # (d) PageValues vs ExitRates scatter (sampled)
    sample = df.sample(min(3000, len(df)), random_state=42)
    axes[1, 1].scatter(
        sample["PageValues"],
        sample["ExitRates"],
        c=sample["Revenue"],
        cmap="coolwarm",
        alpha=0.3,
        s=10,
    )
    axes[1, 1].set_xlabel("PageValues")
    axes[1, 1].set_ylabel("ExitRates")
    axes[1, 1].set_title("PageValues vs ExitRates")

    plt.tight_layout()
    _save_show(fig, "06_pagevalues_deep_dive", save_dir, show)


# ===================================================================
# 8. BounceRates / ExitRates Analysis
# ===================================================================


# BounceRates and ExitRates both appear in CONTINUOUS_FEATURES_FOR_OUTLIERS
# and are included in IQR/Z-score outlier removal for KNN/SVM pipelines.
# Their high mutual correlation (~0.91) is visualised here to confirm both
# carry overlapping but complementary signal — neither is dropped.
def plot_bounce_exit_analysis(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    sns.set_style(SNS_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("BounceRates & ExitRates", fontsize=14, fontweight="bold")

    # Scatter BounceRates vs ExitRates
    sample = df.sample(min(4000, len(df)), random_state=42)
    axes[0].scatter(
        sample["BounceRates"],
        sample["ExitRates"],
        c=sample["Revenue"],
        cmap="coolwarm",
        alpha=0.3,
        s=8,
    )
    axes[0].set_xlabel("BounceRates")
    axes[0].set_ylabel("ExitRates")
    axes[0].set_title("BounceRates vs ExitRates")

    # Box: BounceRates by Revenue
    sns.boxplot(
        data=df, x="Revenue", y="BounceRates", ax=axes[1], palette=PALETTE_BINARY
    )
    axes[1].set_xticklabels(["No", "Yes"])
    axes[1].set_title("BounceRates by Revenue")

    # Box: ExitRates by Revenue
    sns.boxplot(data=df, x="Revenue", y="ExitRates", ax=axes[2], palette=PALETTE_BINARY)
    axes[2].set_xticklabels(["No", "Yes"])
    axes[2].set_title("ExitRates by Revenue")

    plt.tight_layout()
    _save_show(fig, "07_bounce_exit_analysis", save_dir, show)


# ===================================================================
# 9. Monthly Purchase Rate
# ===================================================================


# Month is a string categorical with no inherent numeric ordering.
# -> Included in CATEGORICAL_FEATURES and OneHotEncoded in build_preprocessor().
# Monthly variation in purchase rate confirms Month carries predictive value
# and should not be dropped or ordinally encoded.
def plot_monthly_purchase_rate(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    month_rate = (
        df.groupby("Month")["Revenue"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "rate", "count": "n"})
    )
    month_rate = month_rate.reindex([m for m in MONTH_ORDER if m in month_rate.index])

    sns.set_style(SNS_STYLE)
    fig, ax1 = plt.subplots(figsize=(13, 5))

    bars = ax1.bar(
        month_rate.index,
        month_rate["rate"] * 100,
        color=PALETTE_BINARY[1],
        alpha=0.85,
        edgecolor="w",
    )
    ax1.set_ylabel("Purchase Rate (%)", color=PALETTE_BINARY[1])
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.tick_params(axis="y", labelcolor=PALETTE_BINARY[1])
    ax1.set_title("Monthly Sessions & Purchase Rate", fontsize=13, fontweight="bold")

    ax2 = ax1.twinx()
    ax2.plot(
        month_rate.index,
        month_rate["n"],
        color=PALETTE_BINARY[0],
        marker="o",
        lw=2,
        label="Sessions",
    )
    ax2.set_ylabel("Sessions", color=PALETTE_BINARY[0])
    ax2.tick_params(axis="y", labelcolor=PALETTE_BINARY[0])

    for b, r in zip(bars, month_rate["rate"]):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.3,
            f"{r * 100:.1f}%",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    _save_show(fig, "08_monthly_purchase_rate", save_dir, show)


# ===================================================================
# 10. Visitor Type Analysis
# ===================================================================


# VisitorType is a nominal categorical with 3 values: Returning_Visitor,
# New_Visitor, Other — no meaningful numeric ordering exists.
# -> Included in CATEGORICAL_FEATURES and OneHotEncoded in build_preprocessor().
def plot_visitor_type(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    vt = (
        df.groupby("VisitorType")["Revenue"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "purchases", "count": "total", "mean": "rate"})
        .sort_values("total", ascending=False)
    )

    sns.set_style(SNS_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Visitor Type Analysis", fontsize=14, fontweight="bold")

    x = np.arange(len(vt))
    w = 0.35
    axes[0].bar(x - w / 2, vt["total"], w, label="Total", color=PALETTE_BINARY[0])
    axes[0].bar(
        x + w / 2, vt["purchases"], w, label="Purchases", color=PALETTE_BINARY[1]
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(vt.index)
    axes[0].set_title("Sessions vs Purchases")
    axes[0].legend()

    axes[1].bar(vt.index, vt["rate"] * 100, color=PALETTE_BINARY[1], alpha=0.8)
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].set_title("Purchase Rate by Visitor Type")
    for i, (idx, r) in enumerate(vt["rate"].items()):
        axes[1].text(i, r * 100 + 0.3, f"{r * 100:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    _save_show(fig, "09_visitor_type", save_dir, show)


# ===================================================================
# 11. Weekend Effect
# ===================================================================


# Weekend is a boolean feature (True/False) that shows a small but measurable
# difference in purchase rate between weekday and weekend sessions.
# -> Included in CATEGORICAL_FEATURES and OneHotEncoded in build_preprocessor().
# Treating it as numeric would imply False < True ordering, which OneHotEncoding avoids.
def plot_weekend_effect(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    wk = df.groupby("Weekend")["Revenue"].agg(["mean", "count"])
    wk.index = ["Weekday", "Weekend"]

    sns.set_style(SNS_STYLE)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(wk.index, wk["mean"] * 100, color=PALETTE_BINARY, edgecolor="w")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Purchase Rate – Weekday vs Weekend", fontsize=12, fontweight="bold")
    ax.set_ylabel("Purchase Rate (%)")
    for b, (idx, r) in zip(bars, wk.iterrows()):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.2,
            f"{r['mean'] * 100:.2f}%\n(n={r['count']:,})",
            ha="center",
            fontsize=10,
        )
    plt.tight_layout()
    _save_show(fig, "10_weekend_effect", save_dir, show)


# ===================================================================
# 12. Feature Interaction Analysis
# ===================================================================


def plot_feature_interactions(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    sns.set_style(SNS_STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Feature Interactions", fontsize=14, fontweight="bold")

    # (a) ProductRelated_Duration vs PageValues
    sample = df.sample(min(3000, len(df)), random_state=42)
    axes[0, 0].scatter(
        sample["ProductRelated_Duration"],
        sample["PageValues"],
        c=sample["Revenue"],
        cmap="coolwarm",
        alpha=0.3,
        s=8,
    )
    axes[0, 0].set_xlabel("ProductRelated_Duration")
    axes[0, 0].set_ylabel("PageValues")
    axes[0, 0].set_title("Duration vs PageValues")

    # (b) Month × VisitorType purchase rate heatmap
    month_vt = df.groupby(["Month", "VisitorType"])["Revenue"].mean().unstack()
    month_vt = month_vt.reindex([m for m in MONTH_ORDER if m in month_vt.index])
    sns.heatmap(
        month_vt * 100,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=axes[0, 1],
        annot_kws={"size": 7},
    )
    axes[0, 1].set_title("Purchase Rate (%) – Month × VisitorType")

    # (c) BounceRates vs PageValues
    axes[1, 0].scatter(
        sample["BounceRates"],
        sample["PageValues"],
        c=sample["Revenue"],
        cmap="coolwarm",
        alpha=0.3,
        s=8,
    )
    axes[1, 0].set_xlabel("BounceRates")
    axes[1, 0].set_ylabel("PageValues")
    axes[1, 0].set_title("BounceRates vs PageValues")

    # (d) Weekend × VisitorType
    wk_vt = df.groupby(["Weekend", "VisitorType"])["Revenue"].mean().unstack()
    wk_vt.index = ["Weekday", "Weekend"]
    sns.heatmap(
        wk_vt * 100,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=axes[1, 1],
        annot_kws={"size": 9},
    )
    axes[1, 1].set_title("Purchase Rate (%) – Weekend × VisitorType")

    plt.tight_layout()
    _save_show(fig, "11_feature_interactions", save_dir, show)


# ===================================================================
# 13. Statistical Summary Table
# ===================================================================


# Statistical summary (mean, median, std, skewness, kurtosis) by Revenue class
# quantifies the distributional differences that justify preprocessing choices:
# -> High skewness/kurtosis in continuous features -> supports outlier removal for KNN/SVM.
# -> Mann-Whitney U p-values confirm which features are statistically significant
#    predictors of Revenue — informing feature selection decisions.
def print_statistical_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("13. STATISTICAL SUMMARY BY REVENUE")
    print("=" * 70)

    summary_rows = []
    for col in NUMERICAL_FEATURES:
        for rev, lbl in zip([0, 1], ["No Purchase", "Purchase"]):
            s = df.loc[df["Revenue"] == rev, col]
            summary_rows.append(
                {
                    "Feature": col,
                    "Revenue": lbl,
                    "Mean": f"{s.mean():.3f}",
                    "Median": f"{s.median():.3f}",
                    "Std": f"{s.std():.3f}",
                    "Skew": f"{s.skew():.3f}",
                    "Kurtosis": f"{s.kurtosis():.3f}",
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # Mann-Whitney U test for each numerical feature
    print("\n  Mann-Whitney U test (Revenue 0 vs 1):")
    for col in NUMERICAL_FEATURES:
        g0 = df.loc[df["Revenue"] == 0, col]
        g1 = df.loc[df["Revenue"] == 1, col]
        stat, pval = stats.mannwhitneyu(g0, g1, alternative="two-sided")
        sig = (
            "***"
            if pval < 0.001
            else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
        )
        print(f"    {col:<30s} p={pval:.2e} {sig}")


# ===================================================================
# 14. Key Findings Summary
# ===================================================================


def print_key_findings(df: pd.DataFrame) -> None:
    rev_counts = df["Revenue"].value_counts()
    pv_zero_pct = (df["PageValues"] == 0).mean() * 100
    pv_pos_rate = df.loc[df["PageValues"] > 0, "Revenue"].mean() * 100
    pv_zero_rate = df.loc[df["PageValues"] == 0, "Revenue"].mean() * 100

    month_rate = df.groupby("Month")["Revenue"].mean()
    best_month = month_rate.idxmax()
    worst_month = month_rate.idxmin()

    vt_rate = df.groupby("VisitorType")["Revenue"].mean()
    best_vt = vt_rate.idxmax()

    wk_rate = df.groupby("Weekend")["Revenue"].mean()
    wk_day = wk_rate[False] * 100
    wk_end = wk_rate[True] * 100

    print("\n" + "=" * 70)
    print("14. KEY FINDINGS")
    print("=" * 70)
    findings = [
        f"- Dataset has {df.shape[0]:,} sessions, {df.shape[1]} features, "
        f"no missing values.",
        f"- Target is imbalanced: {rev_counts[0]:,} No vs {rev_counts[1]:,} Yes "
        f"(ratio ~ {rev_counts[0] / rev_counts[1]:.1f}:1).",
        f"- PageValues = 0 in {pv_zero_pct:.1f}% of sessions; "
        f"purchase rate {pv_zero_rate:.2f}% vs {pv_pos_rate:.2f}% when > 0.",
        f"- Best month: {best_month} ({month_rate[best_month] * 100:.1f}%), "
        f"worst: {worst_month} ({month_rate[worst_month] * 100:.1f}%).",
        f"- Best visitor type: {best_vt} "
        f"({vt_rate[best_vt] * 100:.1f}% purchase rate).",
        f"- Weekday rate: {wk_day:.2f}%, Weekend rate: {wk_end:.2f}%.",
        "- PageValues is the strongest single predictor of Revenue.",
        "- BounceRates and ExitRates are negatively correlated with Revenue.",
        "- ProductRelated_Duration and PageValues show clear separation "
        "between classes.",
    ]
    for f in findings:
        print(f"  {f}")


# ===================================================================
# Master: run_eda
# ===================================================================


def run_eda(
    filepath: str = "data/raw/online_shoppers_intention.csv",
    save_dir: str | None = "report_assets/plots/eda",
    show: bool = False,
) -> pd.DataFrame:
    """
    Run the complete EDA pipeline.

    Parameters
    ----------
    filepath : Path to the raw CSV.
    save_dir : Directory for PNG plots (None to skip saving).
    show     : Whether to call plt.show() interactively.

    Returns
    -------
    The loaded DataFrame with Revenue encoded as int.
    """
    # EDA is run on the RAW dataset (before any preprocessing steps).
    # Revenue is temporarily encoded to int here for plotting purposes only
    # (e.g., colour-coding plots by class). The actual encoding for model
    # training is performed by encode_target() in data_preprocessing.py.
    p = Path(filepath)
    if not p.is_absolute():
        p = project_root / p

    df = pd.read_csv(p)
    df["Revenue"] = df["Revenue"].astype(int)  # for visualisation only

    print(f"\n[run_eda] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.\n")

    plot_dataset_overview(df, save_dir, show)
    plot_target_distribution(df, save_dir, show)
    plot_numerical_distributions(df, save_dir, show)
    plot_boxplots(df, save_dir, show)
    plot_categorical_distributions(df, save_dir, show)
    plot_correlation_heatmap(df, save_dir, show)
    plot_pagevalues_deep_dive(df, save_dir, show)
    plot_bounce_exit_analysis(df, save_dir, show)
    plot_monthly_purchase_rate(df, save_dir, show)
    plot_visitor_type(df, save_dir, show)
    plot_weekend_effect(df, save_dir, show)
    plot_feature_interactions(df, save_dir, show)
    print_statistical_summary(df)
    print_key_findings(df)

    print(f"\n[run_eda] Plots saved to: {save_dir or '(not saved)'}")
    return df


if __name__ == "__main__":
    data_path = project_root / "data" / "raw" / "online_shoppers_intention.csv"
    output_dir = project_root / "report_assets" / "plots" / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_eda(filepath=data_path, save_dir=str(output_dir), show=False)

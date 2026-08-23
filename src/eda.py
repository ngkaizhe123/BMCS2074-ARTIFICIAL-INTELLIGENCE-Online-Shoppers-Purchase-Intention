"""
eda.py
------
Exploratory Data Analysis (EDA) for Online Shoppers Purchasing Intention dataset.

Sections:
 1. Dataset Overview & Quality Check
 2. Target Distribution & Imbalance
 3. Numerical Feature Distributions (per feature)
 4. Box Plots & Outlier Inspection (per feature)
 5. Categorical Feature Purchase Rates
 6. Categorical Frequency & Low-Count Category Inspection (per feature)
 7. Correlation Heatmap
 8. PageValues Deep-Dive
 9. BounceRates & ExitRates Analysis
10. Monthly Purchase Rate Analysis
11. Visitor Type Analysis
12. Weekend Effect Analysis
13. Feature Interaction Analysis
14. Statistical Summary & Hypothesis Testing
15. Key Findings Summary

Usage:
    from src.eda import run_eda
    df = run_eda(save_dir="report_assets/plots/eda")
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
# Shared Visual Style & Feature Definitions
# ---------------------------------------------------------------------------

PALETTE_BINARY = ["#4C72B0", "#DD8452"]  # blue = No Purchase (0), orange = Purchase (1)
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


def plot_dataset_overview(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    """
    Inspect raw dataset dimensions, data types, missing values, and duplicate records.
    """
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
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    print(f"\n  Numerical Summary:")
    print(df[num_cols].describe().round(3).to_string())


# ===================================================================
# 2. Target Distribution & Imbalance
# ===================================================================


def plot_target_distribution(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    """
    Examine the class distribution of the target variable (Revenue) to evaluate class imbalance.
    """
    counts = df["Revenue"].value_counts().sort_index()
    labels = ["No Purchase (0)", "Purchase (1)"]
    pcts = counts / counts.sum() * 100
    ratio = counts[0] / counts[1]

    sns.set_style(SNS_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        "Target Variable Distribution (Revenue)", fontsize=14, fontweight="bold"
    )

    # Bar chart
    bars = axes[0].bar(labels, counts.values, color=PALETTE_BINARY, edgecolor="w")
    axes[0].set_title("Session Counts by Class")
    axes[0].set_ylabel("Sessions")

    for b, c, p in zip(bars, counts.values, pcts.values):
        axes[0].text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 40,
            f"{c:,}\n({p:.1f}%)",
            ha="center",
            fontsize=10,
        )

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=PALETTE_BINARY,
        startangle=90,
        wedgeprops={"edgecolor": "w"},
    )
    axes[1].set_title("Class Proportion")

    plt.tight_layout()
    _save_show(fig, "01_target_distribution", save_dir, show)

    print(
        f"\n  Target: No Purchase = {counts[0]:,} ({pcts[0]:.1f}%), "
        f"Purchase = {counts[1]:,} ({pcts[1]:.1f}%)"
    )
    print(f"  Imbalance ratio ~ {ratio:.2f}:1")


# ===================================================================
# 3. Numerical Feature Distributions (Per Feature)
# ===================================================================


def plot_numerical_distributions(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    """
    Analyze distribution shape, density, and skewness for each numerical feature across target classes.
    """
    sns.set_style(SNS_STYLE)
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]

    for col in num_cols:
        fig, ax = plt.subplots(figsize=(9, 5))
        for rev, color, lbl in zip([0, 1], PALETTE_BINARY, ["No Purchase", "Purchase"]):
            s = df.loc[df["Revenue"] == rev, col].dropna()
            ax.hist(s, bins=40, alpha=0.45, color=color, label=lbl, density=True)
            if s.std() > 0:
                s.plot.kde(ax=ax, color=color, lw=1.8)
        skew_val = df[col].skew()
        ax.set_title(f"{col} – Distribution by Revenue", fontsize=12, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend(fontsize=10)
        ax.text(
            0.97,
            0.95,
            f"Skewness: {skew_val:+.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="#f5f5f5", alpha=0.85),
        )
        plt.tight_layout()
        _save_show(fig, f"02_dist_{col}", save_dir, show)


# ===================================================================
# 4. Box Plots & Outlier Inspection (Per Feature)
# ===================================================================


def plot_boxplots(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    """
    Inspect feature spread, dispersion, and extreme outliers for each numerical feature.
    """
    sns.set_style(SNS_STYLE)
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]

    for col in num_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=df, x="Revenue", y=col, ax=ax, palette=PALETTE_BINARY)
        ax.set_title(
            f"{col} – Outlier Inspection by Revenue", fontsize=12, fontweight="bold"
        )
        ax.set_xticklabels(["No Purchase", "Purchase"])
        ax.set_xlabel("")
        ax.set_ylabel(col)
        plt.tight_layout()
        _save_show(fig, f"03_boxplot_{col}", save_dir, show)


# ===================================================================
# 5. Categorical Feature Distributions & Purchase Rates
# ===================================================================


def plot_categorical_distributions(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    """
    Evaluate conversion rates across categorical variables using stacked proportions.
    """
    sns.set_style(SNS_STYLE)
    all_cats = CATEGORICAL_FEATURES + ORDINAL_FEATURES
    n_cols = 2
    n_rows = int(np.ceil(len(all_cats) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(all_cats):
        if col not in df.columns:
            continue
        ax = axes[i]
        ct = pd.crosstab(df[col], df["Revenue"], normalize="index") * 100
        ct.columns = ["No Purchase", "Purchase"]
        ct.plot(kind="bar", stacked=True, ax=ax, color=PALETTE_BINARY, edgecolor="w")
        ax.set_title(f"Purchase Rate by {col}", fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(fontsize=7)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Conversion Rate Proportions across Categories",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    _save_show(fig, "04_categorical_purchase_rate", save_dir, show)


# ===================================================================
# 6. Categorical Frequency & Low-Count Category Inspection (Per Feature)
# ===================================================================


def plot_ordinal_rare_categories(
    df: pd.DataFrame,
    save_dir: str | None = None,
    show: bool = True,
    threshold: int = 10,
) -> None:
    """
    Examine category frequency distributions for ID-coded categorical features
    to identify low-frequency categories (count < threshold).
    """
    sns.set_style(SNS_STYLE)

    for col in ORDINAL_FEATURES:
        if col not in df.columns:
            continue
        fig, ax = plt.subplots(figsize=(9, 5))
        counts = df[col].value_counts().sort_index()
        colors = [
            "#E74C3C" if c < threshold else PALETTE_BINARY[0] for c in counts.values
        ]
        bars = ax.bar(
            counts.index.astype(str), counts.values, color=colors, edgecolor="w"
        )
        ax.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            lw=1.5,
            label=f"Low Count Threshold (< {threshold})",
        )
        n_rare = (counts < threshold).sum()
        n_total = len(counts)
        ax.set_title(
            f"{col} – Category Frequencies ({n_rare} / {n_total} categories < {threshold} samples)",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel(f"{col} Category ID")
        ax.set_ylabel("Sample Count")
        ax.legend(fontsize=9)

        for bar, count in zip(bars, counts.values):
            if count < threshold:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts.values) * 0.01,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#E74C3C",
                    fontweight="bold",
                )

        plt.tight_layout()
        _save_show(fig, f"05_count_{col}", save_dir, show)

    print(f"\n  Category counts below threshold ({threshold}):")
    for col in ORDINAL_FEATURES:
        if col in df.columns:
            counts = df[col].value_counts()
            rare = counts[counts < threshold]
            print(f"    {col:<20s}: {len(rare)} categories with count < {threshold}")


# ===================================================================
# 7. Correlation Heatmap
# ===================================================================


def plot_correlation_heatmap(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    """
    Evaluate linear relationships and collinearity among all numeric features and the target variable.
    EDA runs on the raw dataset, so ORDINAL_FEATURES are still integers and included here.
    """
    # Include both continuous numerical features AND integer-coded ordinal features
    all_numeric = NUMERICAL_FEATURES + ORDINAL_FEATURES
    cols = [c for c in all_numeric if c in df.columns] + ["Revenue"]
    corr = df[cols].corr()

    n = len(cols)
    fig_size = max(10, n)  # scale figure with number of features
    fig, ax = plt.subplots(figsize=(fig_size, fig_size - 1))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title(
        "Feature Correlation Matrix (All Numeric Features)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    _save_show(fig, "06_correlation_heatmap", save_dir, show)

    rev_corr = corr["Revenue"].drop("Revenue").sort_values(ascending=False)
    print("\n  Top correlations with Revenue:")
    for feat, val in rev_corr.head(5).items():
        print(f"    {feat:<30s} {val:+.4f}")


# ===================================================================
# 8. PageValues Deep-Dive
# ===================================================================


def plot_pagevalues_deep_dive(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    """
    Perform a detailed analysis of PageValues due to its zero-inflated distribution and high predictive power.
    """
    sns.set_style(SNS_STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("PageValues Feature Analysis", fontsize=14, fontweight="bold")

    # Violin plot
    sns.violinplot(
        data=df,
        x="Revenue",
        y="PageValues",
        ax=axes[0, 0],
        palette=PALETTE_BINARY,
        inner="quartile",
    )
    axes[0, 0].set_xticklabels(["No Purchase", "Purchase"])
    axes[0, 0].set_title("Distribution by Revenue Class")

    # Zero vs Non-zero summary
    pv_zero = (df["PageValues"] == 0).mean() * 100
    pv_nz_rate = df.loc[df["PageValues"] > 0, "Revenue"].mean() * 100
    pv_z_rate = df.loc[df["PageValues"] == 0, "Revenue"].mean() * 100

    axes[0, 1].axis("off")
    txt = (
        f"PageValues = 0 : {pv_zero:.1f}% of sessions\n"
        f"  → Conversion rate: {pv_z_rate:.2f}%\n\n"
        f"PageValues > 0 : {100 - pv_zero:.1f}% of sessions\n"
        f"  → Conversion rate: {pv_nz_rate:.2f}%\n\n"
        f"Key Insight: Non-zero PageValues strongly\n"
        f"correlates with purchase completion."
    )
    axes[0, 1].text(
        0.05,
        0.5,
        txt,
        fontsize=11,
        va="center",
        bbox=dict(boxstyle="round", fc="#f5f5f5"),
    )

    # Non-zero distribution
    pv_pos = df.loc[df["PageValues"] > 0, "PageValues"]
    axes[1, 0].hist(pv_pos, bins=50, color=PALETTE_BINARY[1], alpha=0.7, edgecolor="w")
    axes[1, 0].set_title("Distribution of Non-Zero PageValues")
    axes[1, 0].set_xlabel("PageValues")
    axes[1, 0].set_ylabel("Frequency")

    # Scatter vs ExitRates
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
    axes[1, 1].set_title("PageValues vs. ExitRates")

    plt.tight_layout()
    _save_show(fig, "07_pagevalues_deep_dive", save_dir, show)


# ===================================================================
# 9. BounceRates & ExitRates Analysis
# ===================================================================


def plot_bounce_exit_analysis(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    """
    Examine the relationship and distributions of BounceRates and ExitRates with respect to user purchase behavior.
    """
    sns.set_style(SNS_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("BounceRates & ExitRates Relationship", fontsize=14, fontweight="bold")

    # Scatter
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
    axes[0].set_title("BounceRates vs. ExitRates Scatter")

    # Boxplot BounceRates
    sns.boxplot(
        data=df, x="Revenue", y="BounceRates", ax=axes[1], palette=PALETTE_BINARY
    )
    axes[1].set_xticklabels(["No Purchase", "Purchase"])
    axes[1].set_title("BounceRates by Target Class")

    # Boxplot ExitRates
    sns.boxplot(data=df, x="Revenue", y="ExitRates", ax=axes[2], palette=PALETTE_BINARY)
    axes[2].set_xticklabels(["No Purchase", "Purchase"])
    axes[2].set_title("ExitRates by Target Class")

    plt.tight_layout()
    _save_show(fig, "08_bounce_exit_analysis", save_dir, show)


# ===================================================================
# 10. Monthly Purchase Rate Analysis
# ===================================================================


def plot_monthly_purchase_rate(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    """
    Analyze seasonality by examining monthly traffic volume and conversion rates.
    """
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
    ax1.set_ylabel("Conversion Rate (%)", color=PALETTE_BINARY[1])
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.tick_params(axis="y", labelcolor=PALETTE_BINARY[1])
    ax1.set_title(
        "Monthly Traffic Volume & Conversion Rate", fontsize=13, fontweight="bold"
    )

    ax2 = ax1.twinx()
    ax2.plot(
        month_rate.index,
        month_rate["n"],
        color=PALETTE_BINARY[0],
        marker="o",
        lw=2,
        label="Sessions",
    )
    ax2.set_ylabel("Total Sessions", color=PALETTE_BINARY[0])
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
    _save_show(fig, "09_monthly_purchase_rate", save_dir, show)


# ===================================================================
# 11. Visitor Type Analysis
# ===================================================================


def plot_visitor_type(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    """
    Compare total session volume and conversion rates across different visitor segments.
    """
    vt = (
        df.groupby("VisitorType")["Revenue"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "purchases", "count": "total", "mean": "rate"})
        .sort_values("total", ascending=False)
    )

    sns.set_style(SNS_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Visitor Type Segment Analysis", fontsize=14, fontweight="bold")

    x = np.arange(len(vt))
    w = 0.35
    axes[0].bar(
        x - w / 2, vt["total"], w, label="Total Sessions", color=PALETTE_BINARY[0]
    )
    axes[0].bar(
        x + w / 2, vt["purchases"], w, label="Purchases", color=PALETTE_BINARY[1]
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(vt.index)
    axes[0].set_title("Sessions vs. Conversions")
    axes[0].legend()

    axes[1].bar(vt.index, vt["rate"] * 100, color=PALETTE_BINARY[1], alpha=0.8)
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].set_title("Conversion Rate by Visitor Segment")
    for i, (idx, r) in enumerate(vt["rate"].items()):
        axes[1].text(i, r * 100 + 0.3, f"{r * 100:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    _save_show(fig, "10_visitor_type_analysis", save_dir, show)


# ===================================================================
# 12. Weekend Effect Analysis
# ===================================================================


def plot_weekend_effect(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    """
    Evaluate purchase rate differences between weekday and weekend browsing sessions.
    """
    wk = df.groupby("Weekend")["Revenue"].agg(["mean", "count"])
    wk.index = ["Weekday", "Weekend"]

    sns.set_style(SNS_STYLE)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(wk.index, wk["mean"] * 100, color=PALETTE_BINARY, edgecolor="w")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(
        "Conversion Rate – Weekday vs. Weekend", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Conversion Rate (%)")
    for b, (idx, r) in zip(bars, wk.iterrows()):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.2,
            f"{r['mean'] * 100:.2f}%\n(n={r['count']:,})",
            ha="center",
            fontsize=10,
        )
    plt.tight_layout()
    _save_show(fig, "11_weekend_effect", save_dir, show)


# ===================================================================
# 13. Feature Interaction Analysis
# ===================================================================


def plot_feature_interactions(
    df: pd.DataFrame, save_dir: str | None = None, show: bool = True
) -> None:
    """
    Explore multi-feature interactions and joint impacts on conversion probability.
    """
    sns.set_style(SNS_STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Feature Interaction Analysis", fontsize=14, fontweight="bold")

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
    axes[0, 0].set_title("Duration vs. PageValues")

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
    axes[0, 1].set_title("Conversion Rate (%) – Month × VisitorType")

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
    axes[1, 0].set_title("BounceRates vs. PageValues")

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
    axes[1, 1].set_title("Conversion Rate (%) – Weekend × VisitorType")

    plt.tight_layout()
    _save_show(fig, "12_feature_interactions", save_dir, show)


# ===================================================================
# 14. Statistical Summary Table & Hypothesis Testing
# ===================================================================


def print_statistical_summary(df: pd.DataFrame) -> None:
    """
    Compute descriptive statistical metrics and execute non-parametric hypothesis tests (Mann-Whitney U).
    """
    print("\n" + "=" * 70)
    print("14. STATISTICAL SUMMARY BY TARGET CLASS")
    print("=" * 70)

    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    summary_rows = []
    for col in num_cols:
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

    print("\n  Mann-Whitney U Test Results (Revenue 0 vs 1):")
    for col in num_cols:
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
# 15. Key Findings Summary
# ===================================================================


def print_key_findings(df: pd.DataFrame) -> None:
    """
    Print high-level exploratory findings.
    """
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
    print("15. KEY FINDINGS SUMMARY")
    print("=" * 70)
    findings = [
        f"- Dataset contains {df.shape[0]:,} sessions and {df.shape[1]} features.",
        f"- Target variable is imbalanced: {rev_counts[0]:,} No Purchase vs. {rev_counts[1]:,} Purchase (ratio ~{rev_counts[0] / rev_counts[1]:.1f}:1).",
        f"- PageValues = 0 in {pv_zero_pct:.1f}% of sessions; conversion rate is {pv_zero_rate:.2f}% when zero vs. {pv_pos_rate:.2f}% when > 0.",
        f"- Highest conversion month: {best_month} ({month_rate[best_month] * 100:.1f}%), lowest: {worst_month} ({month_rate[worst_month] * 100:.1f}%).",
        f"- Highest converting visitor segment: {best_vt} ({vt_rate[best_vt] * 100:.1f}% conversion rate).",
        f"- Weekday conversion rate: {wk_day:.2f}%, Weekend conversion rate: {wk_end:.2f}%.",
        "- Numerical features display significant positive skewness and long tails.",
        "- Several ID-coded categorical features contain low-frequency categories (< 10 samples).",
    ]
    for f in findings:
        print(f"  {f}")


# ===================================================================
# Master Execution Pipeline
# ===================================================================


def run_eda(
    filepath: str = "data/raw/online_shoppers_intention.csv",
    save_dir: str | None = "report_assets/plots/eda",
    show: bool = False,
) -> pd.DataFrame:
    """
    Execute complete EDA pipeline and export generated plots.
    """
    p = Path(filepath)
    if not p.is_absolute():
        p = project_root / p

    df = pd.read_csv(p)
    df["Revenue"] = df["Revenue"].astype(int)

    print(f"\n[run_eda] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.\n")

    plot_dataset_overview(df, save_dir, show)
    plot_target_distribution(df, save_dir, show)
    plot_numerical_distributions(df, save_dir, show)
    plot_boxplots(df, save_dir, show)
    plot_categorical_distributions(df, save_dir, show)
    plot_ordinal_rare_categories(df, save_dir, show)
    plot_correlation_heatmap(df, save_dir, show)
    plot_pagevalues_deep_dive(df, save_dir, show)
    plot_bounce_exit_analysis(df, save_dir, show)
    plot_monthly_purchase_rate(df, save_dir, show)
    plot_visitor_type(df, save_dir, show)
    plot_weekend_effect(df, save_dir, show)
    plot_feature_interactions(df, save_dir, show)
    print_statistical_summary(df)
    print_key_findings(df)

    print(f"\n[run_eda] All plots saved to: {save_dir or '(not saved)'}")
    return df


if __name__ == "__main__":
    data_path = project_root / "data" / "raw" / "online_shoppers_intention.csv"
    output_dir = project_root / "report_assets" / "plots" / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_eda(filepath=data_path, save_dir=str(output_dir), show=False)

"""
eda.py
------
Exploratory Data Analysis for the Online Shoppers Intention dataset.

All plotting functions accept an optional `save_dir` argument.
When provided, each figure is saved to `save_dir/<plot_name>.png`
in addition to being displayed (or closed silently if show=False).

Usage (from project root)
--------------------------
    from src.eda import run_eda
    run_eda(save_dir="report_assets/plots")
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

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

PALETTE_BINARY = ["#4C72B0", "#DD8452"]  # blue = 0, orange = 1
SNS_STYLE = "whitegrid"

# Columns referenced throughout
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


# ---------------------------------------------------------------------------
# Helper: save / show figure
# ---------------------------------------------------------------------------


def _save_show(fig: plt.Figure, name: str, save_dir: str | None, show: bool) -> None:
    """Save figure to *save_dir* and optionally display it."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# 1.  Dataset Overview
# ---------------------------------------------------------------------------


def plot_dataset_overview(
    df: pd.DataFrame,
    save_dir: str | None = None,
    show: bool = True,
) -> None:
    """Print a concise overview: shape, dtypes, missing values."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape      : {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
    print(f"\nData Types :\n{df.dtypes.to_string()}")
    missing = df.isnull().sum()
    print(f"\nMissing Values :\n{missing[missing > 0].to_string() or 'None'}")
    print(f"\nNumerical Summary :\n{df.describe().to_string()}")


# ---------------------------------------------------------------------------
# 2.  Target Distribution
# ---------------------------------------------------------------------------


def plot_target_distribution(
    df: pd.DataFrame,
    save_dir: str | None = None,
    show: bool = True,
) -> None:
    """Bar + pie chart for Revenue (target) distribution."""
    counts = df["Revenue"].value_counts().sort_index()
    labels = ["No Purchase (0)", "Purchase (1)"]
    pcts = counts / counts.sum() * 100

    sns.set_style(SNS_STYLE)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Target Variable Distribution – Revenue", fontsize=14, fontweight="bold"
    )

    # --- Bar chart ---
    bars = axes[0].bar(labels, counts.values, color=PALETTE_BINARY, edgecolor="white")
    axes[0].set_title("Count")
    axes[0].set_ylabel("Number of Sessions")
    for bar, count, pct in zip(bars, counts.values, pcts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 50,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # --- Pie chart ---
    axes[1].pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=PALETTE_BINARY,
        startangle=90,
        wedgeprops={"edgecolor": "white"},
    )
    axes[1].set_title("Proportion")

    plt.tight_layout()
    _save_show(fig, "01_target_distribution", save_dir, show)


# ---------------------------------------------------------------------------
# 3.  Numerical Feature Distributions
# ---------------------------------------------------------------------------


def plot_numerical_distributions(
    df: pd.DataFrame,
    save_dir: str | None = None,
    show: bool = True,
) -> None:
    """Histograms (with KDE) for every numerical feature, split by Revenue."""
    sns.set_style(SNS_STYLE)
    n_cols = 2
    n_rows = int(np.ceil(len(NUMERICAL_FEATURES) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(NUMERICAL_FEATURES):
        ax = axes[i]
        for rev, color in zip([0, 1], PALETTE_BINARY):
            subset = df[df["Revenue"] == rev][col].dropna()
            label = "No Purchase" if rev == 0 else "Purchase"
            ax.hist(subset, bins=40, alpha=0.55, color=color, label=label, density=True)
            subset.plot.kde(ax=ax, color=color, linewidth=1.5)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.legend(fontsize=8)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Numerical Feature Distributions by Revenue",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    _save_show(fig, "02_numerical_distributions", save_dir, show)


# ---------------------------------------------------------------------------
# 4.  Box Plots (Outlier Inspection)
# ---------------------------------------------------------------------------


def plot_boxplots(
    df: pd.DataFrame,
    save_dir: str | None = None,
    show: bool = True,
) -> None:
    """Box plots for numerical features to visualise spread and outliers."""
    sns.set_style(SNS_STYLE)
    n_cols = 2
    n_rows = int(np.ceil(len(NUMERICAL_FEATURES) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(NUMERICAL_FEATURES):
        ax = axes[i]
        sns.boxplot(
            data=df,
            x="Revenue",
            y=col,
            ax=ax,
            palette=PALETTE_BINARY,
        )
        ax.set_title(col, fontsize=10)
        ax.set_xticklabels(["No Purchase", "Purchase"])
        ax.set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Box Plots – Numerical Features by Revenue",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    _save_show(fig, "03_boxplots_numerical", save_dir, show)


# ---------------------------------------------------------------------------
# 5.  Categorical Feature Distributions
# ---------------------------------------------------------------------------


def plot_categorical_distributions(
    df: pd.DataFrame,
    save_dir: str | None = None,
    show: bool = True,
) -> None:
    """Stacked / grouped bar charts for each categorical feature vs Revenue."""
    sns.set_style(SNS_STYLE)
    all_cats = CATEGORICAL_FEATURES + ORDINAL_FEATURES
    n_cols = 2
    n_rows = int(np.ceil(len(all_cats) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(all_cats):
        ax = axes[i]
        crosstab = pd.crosstab(df[col], df["Revenue"], normalize="index") * 100
        crosstab.columns = ["No Purchase", "Purchase"]
        crosstab.plot(
            kind="bar", stacked=True, ax=ax, color=PALETTE_BINARY, edgecolor="white"
        )
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Categorical Feature Purchase Rate", fontsize=14, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    _save_show(fig, "04_categorical_distributions", save_dir, show)


# ---------------------------------------------------------------------------
# 6.  Correlation Heatmap
# ---------------------------------------------------------------------------


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_dir: str | None = None,
    show: bool = True,
) -> None:
    """Pearson correlation heatmap for numerical features + target."""
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
    ax.set_title(
        "Correlation Heatmap – Numerical Features", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    _save_show(fig, "05_correlation_heatmap", save_dir, show)


# ---------------------------------------------------------------------------
# 7.  PageValues vs Revenue  (scatter / violin)
# ---------------------------------------------------------------------------


def plot_pagevalues_vs_revenue(
    df: pd.DataFrame,
    save_dir: str | None = None,
    show: bool = True,
) -> None:
    """
    PageValues is typically the strongest predictor of purchase.
    Show a violin plot to highlight the difference in distribution.
    """
    sns.set_style(SNS_STYLE)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.violinplot(
        data=df,
        x="Revenue",
        y="PageValues",
        ax=ax,
        palette=PALETTE_BINARY,
        inner="quartile",
    )
    ax.set_xticklabels(["No Purchase (0)", "Purchase (1)"])
    ax.set_title("PageValues Distribution by Revenue", fontsize=12, fontweight="bold")
    ax.set_xlabel("Revenue")
    ax.set_ylabel("PageValues")
    plt.tight_layout()
    _save_show(fig, "06_pagevalues_vs_revenue", save_dir, show)


# ---------------------------------------------------------------------------
# 8.  Monthly Purchase Rate
# ---------------------------------------------------------------------------


def plot_monthly_purchase_rate(
    df: pd.DataFrame,
    save_dir: str | None = None,
    show: bool = True,
) -> None:
    """Bar chart showing purchase rate (%) per month."""
    month_order = [
        "Feb",
        "Mar",
        "May",
        "June",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    month_rate = (
        df.groupby("Month")["Revenue"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "purchase_rate", "count": "sessions"})
    )
    month_rate = month_rate.reindex([m for m in month_order if m in month_rate.index])

    sns.set_style(SNS_STYLE)
    fig, ax1 = plt.subplots(figsize=(12, 5))

    bars = ax1.bar(
        month_rate.index,
        month_rate["purchase_rate"] * 100,
        color=PALETTE_BINARY[1],
        alpha=0.8,
        edgecolor="white",
    )
    ax1.set_ylabel("Purchase Rate (%)", color=PALETTE_BINARY[1])
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.tick_params(axis="y", labelcolor=PALETTE_BINARY[1])
    ax1.set_title(
        "Monthly Session Count & Purchase Rate", fontsize=12, fontweight="bold"
    )

    # Overlay session count on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(
        month_rate.index,
        month_rate["sessions"],
        color=PALETTE_BINARY[0],
        marker="o",
        linewidth=2,
        label="Sessions",
    )
    ax2.set_ylabel("Number of Sessions", color=PALETTE_BINARY[0])
    ax2.tick_params(axis="y", labelcolor=PALETTE_BINARY[0])

    for bar, rate in zip(bars, month_rate["purchase_rate"]):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{rate * 100:.1f}%",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    _save_show(fig, "07_monthly_purchase_rate", save_dir, show)


# ---------------------------------------------------------------------------
# 9.  Visitor Type Distribution
# ---------------------------------------------------------------------------


def plot_visitor_type(
    df: pd.DataFrame,
    save_dir: str | None = None,
    show: bool = True,
) -> None:
    """Grouped bar: visitor type count and purchase rate."""
    vtype = (
        df.groupby("VisitorType")["Revenue"]
        .agg(["sum", "count", "mean"])
        .rename(columns={"sum": "purchases", "count": "total", "mean": "rate"})
        .sort_values("total", ascending=False)
    )

    sns.set_style(SNS_STYLE)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(vtype))
    width = 0.35

    ax.bar(
        x - width / 2,
        vtype["total"],
        width,
        label="Total Sessions",
        color=PALETTE_BINARY[0],
    )
    ax.bar(
        x + width / 2,
        vtype["purchases"],
        width,
        label="Purchases",
        color=PALETTE_BINARY[1],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(vtype.index)
    ax.set_ylabel("Count")
    ax.set_title("Visitor Type – Sessions vs Purchases", fontsize=12, fontweight="bold")
    ax.legend()

    for xi, rate in zip(x, vtype["rate"]):
        ax.text(
            xi + width / 2,
            vtype.loc[vtype.index[xi == x], "purchases"].values[0] + 20,
            f"{rate * 100:.1f}%",
            ha="center",
            fontsize=9,
            color="black",
        )

    plt.tight_layout()
    _save_show(fig, "08_visitor_type", save_dir, show)


# ---------------------------------------------------------------------------
# 10.  Weekend Effect
# ---------------------------------------------------------------------------


def plot_weekend_effect(
    df: pd.DataFrame,
    save_dir: str | None = None,
    show: bool = True,
) -> None:
    """Compare purchase rates on weekdays vs weekends."""
    weekend = df.groupby("Weekend")["Revenue"].agg(["mean", "count"])
    weekend.index = ["Weekday", "Weekend"]

    sns.set_style(SNS_STYLE)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        weekend.index, weekend["mean"] * 100, color=PALETTE_BINARY, edgecolor="white"
    )
    ax.set_ylabel("Purchase Rate (%)")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Purchase Rate – Weekday vs Weekend", fontsize=12, fontweight="bold")

    for bar, (idx, row) in zip(bars, weekend.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{row['mean'] * 100:.2f}%\n(n={row['count']:,})",
            ha="center",
            fontsize=10,
        )

    plt.tight_layout()
    _save_show(fig, "09_weekend_effect", save_dir, show)


# ---------------------------------------------------------------------------
# Master function: run_eda
# ---------------------------------------------------------------------------


def run_eda(
    filepath: str = "../../data/raw/online_shoppers_intention.csv",
    save_dir: str | None = "report_assets/plots",
    show: bool = False,
) -> pd.DataFrame:
    """
    Run the complete EDA pipeline.

    Parameters
    ----------
    filepath : Path to the raw CSV.
    save_dir : Directory where PNG plots are saved.
               Pass ``None`` to skip saving.
    show     : Whether to call ``plt.show()`` for each figure.
               Set to ``True`` when running interactively in a notebook.

    Returns
    -------
    The loaded DataFrame (for further exploration in a notebook).
    """
    df = pd.read_csv(filepath)
    df["Revenue"] = df["Revenue"].astype(int)

    print(f"\n[run_eda] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.\n")

    plot_dataset_overview(df, save_dir=save_dir, show=show)
    plot_target_distribution(df, save_dir=save_dir, show=show)
    plot_numerical_distributions(df, save_dir=save_dir, show=show)
    plot_boxplots(df, save_dir=save_dir, show=show)
    plot_categorical_distributions(df, save_dir=save_dir, show=show)
    plot_correlation_heatmap(df, save_dir=save_dir, show=show)
    plot_pagevalues_vs_revenue(df, save_dir=save_dir, show=show)
    plot_monthly_purchase_rate(df, save_dir=save_dir, show=show)
    plot_visitor_type(df, save_dir=save_dir, show=show)
    plot_weekend_effect(df, save_dir=save_dir, show=show)

    print("\n[run_eda] All plots saved to:", save_dir or "(not saved)")
    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    data_path = project_root / "data" / "raw" / "online_shoppers_intention.csv"
    output_dir = project_root / "report_assets" / "plots"
    output_dir.mkdir(exist_ok=True)

    run_eda(filepath=data_path, save_dir=output_dir, show=False)

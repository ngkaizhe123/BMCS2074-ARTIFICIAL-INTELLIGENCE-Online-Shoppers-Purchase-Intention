import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import streamlit as st
from scipy import stats as scipy_stats

from src.eda import (
    CATEGORICAL_FEATURES,
    MONTH_ORDER,
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
)
from src.ui_theme import apply_theme

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Data Exploration", page_icon="📊", layout="wide")
apply_theme()

st.title("📊 Data Exploration")
st.caption(
    "Mirrors the analysis in `src/eda.py` — every number and chart here is "
    "computed live from the raw dataset, and links back to why the "
    "preprocessing pipeline (`src/data_preprocessing.py`) makes the choices it does."
)
st.markdown("---")

DATA_PATH = project_root / "data" / "raw" / "online_shoppers_intention.csv"
PLOT_DIR = project_root / "report_assets" / "plots" / "eda"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Revenue"] = df["Revenue"].astype(int)
    return df


@st.cache_data
def compute_stats(df: pd.DataFrame):
    """All the live numbers used across the sections below, computed once
    and cached — mirrors the calculations printed by src/eda.py."""
    corr = df[NUMERICAL_FEATURES + ["Revenue"]].corr()
    top_corr = corr["Revenue"].drop("Revenue").sort_values(ascending=False)

    pv_zero_pct = (df["PageValues"] == 0).mean() * 100
    pv_zero_rate = df.loc[df["PageValues"] == 0, "Revenue"].mean() * 100
    pv_pos_rate = df.loc[df["PageValues"] > 0, "Revenue"].mean() * 100

    be_corr = df["BounceRates"].corr(df["ExitRates"])

    month_rate = df.groupby("Month")["Revenue"].mean()
    month_rate = month_rate.reindex([m for m in MONTH_ORDER if m in month_rate.index])
    best_month, worst_month = month_rate.idxmax(), month_rate.idxmin()

    vt_rate = df.groupby("VisitorType")["Revenue"].mean().sort_values(ascending=False)

    wk_rate = df.groupby("Weekend")["Revenue"].mean()

    summary_rows = []
    for col in NUMERICAL_FEATURES:
        for rev, lbl in zip([0, 1], ["No Purchase", "Purchase"]):
            s = df.loc[df["Revenue"] == rev, col]
            summary_rows.append(
                {
                    "Feature": col,
                    "Revenue": lbl,
                    "Mean": round(s.mean(), 3),
                    "Median": round(s.median(), 3),
                    "Std": round(s.std(), 3),
                    "Skew": round(s.skew(), 3),
                    "Kurtosis": round(s.kurtosis(), 3),
                }
            )
    summary_df = pd.DataFrame(summary_rows)

    mw_rows = []
    for col in NUMERICAL_FEATURES:
        g0 = df.loc[df["Revenue"] == 0, col]
        g1 = df.loc[df["Revenue"] == 1, col]
        _, pval = scipy_stats.mannwhitneyu(g0, g1, alternative="two-sided")
        sig = (
            "***"
            if pval < 0.001
            else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        )
        mw_rows.append({"Feature": col, "p-value": pval, "Significant?": sig})
    mw_df = pd.DataFrame(mw_rows)

    return {
        "top_corr": top_corr,
        "pv_zero_pct": pv_zero_pct,
        "pv_zero_rate": pv_zero_rate,
        "pv_pos_rate": pv_pos_rate,
        "be_corr": be_corr,
        "best_month": best_month,
        "worst_month": worst_month,
        "month_rate": month_rate,
        "vt_rate": vt_rate,
        "wk_rate": wk_rate,
        "summary_df": summary_df,
        "mw_df": mw_df,
    }


def show_plot(filename: str, title: str, caption: str = "", expanded: bool = False):
    """Load a plot saved by src/eda.py's run_eda(), with a friendly
    fallback if it hasn't been generated yet."""
    path = PLOT_DIR / filename
    with st.expander(f"🖼️ {title}", expanded=expanded):
        if path.exists():
            st.image(str(path), width="stretch")
            if caption:
                st.caption(caption)
        else:
            st.info(
                f"Plot not found. Run `python -m src.eda` (or `run_eda()`) "
                f"to generate `{filename}`."
            )


def chip(label: str):
    st.markdown(f'<span class="section-chip">{label}</span>', unsafe_allow_html=True)


df = load_data()
stats_ = compute_stats(df)

# ═══════════════════════════════════════════════════════════════════════
# 1. Dataset Overview & Quality Check
# ═══════════════════════════════════════════════════════════════════════
st.header("1. Dataset Overview & Quality Check")
chip("SHAPE · DTYPES · DUPLICATES · MISSING VALUES")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{df.shape[0]:,}")
c2.metric("Columns", df.shape[1])
c3.metric("Missing Values", df.isnull().sum().sum())
c4.metric("Duplicated Rows", df.duplicated().sum())
c5.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024:.0f} KB")

st.info(
    "ℹ️ **What this means for preprocessing:** the "
    f"**{df.duplicated().sum()} duplicate rows** found here are dropped by "
    "`drop_duplicate_rows()` before the train/test split, to avoid the same "
    "session leaking into both sides. Missing values are confirmed at 0, but "
    "`handle_missing_values()` stays in the pipeline as a safeguard for future "
    "data pulls. `Revenue` loads as a boolean — `encode_target()` converts it "
    "to 0/1 before training."
)

col_left, col_right = st.columns(2)
with col_left:
    st.subheader("First 10 Rows")
    st.dataframe(df.head(10), width="stretch")
with col_right:
    st.subheader("Data Types")
    st.dataframe(
        df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Type"}),
        width="stretch",
        height=390,
    )

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# 2. Target Distribution & Imbalance
# ═══════════════════════════════════════════════════════════════════════
st.header("2. Target Variable – Revenue")
chip("CLASS IMBALANCE")

counts = df["Revenue"].value_counts().sort_index()
labels = {0: "No Purchase (0)", 1: "Purchase (1)"}
ratio = counts[0] / counts[1]

m1, m2, m3 = st.columns(3)
m1.metric(labels[0], f"{counts[0]:,}", f"{counts[0] / counts.sum() * 100:.1f}%")
m2.metric(labels[1], f"{counts[1]:,}", f"{counts[1] / counts.sum() * 100:.1f}%")
m3.metric("Imbalance Ratio", f"{ratio:.2f} : 1")

st.bar_chart(counts.rename(index=labels))
st.info(
    "ℹ️ **What this means for preprocessing:** `train_test_split(..., stratify=y)` "
    "preserves this ~5.5:1 ratio across train/test, and `get_smote()` oversamples "
    "the minority (Purchase) class **inside** the imblearn Pipeline — so resampling "
    "only ever touches training folds, never the validation/test data."
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# 3. Numerical Feature Distributions
# ═══════════════════════════════════════════════════════════════════════
st.header("3. Numerical Feature Distributions")
chip("SKEW & SCALE")

st.dataframe(df[NUMERICAL_FEATURES].describe().T.round(3), width="stretch")
show_plot(
    "02_numerical_distributions.png",
    "Histograms by Revenue",
    caption="Heavy right-skew and long tails across most features.",
)
st.info(
    "ℹ️ **What this means for preprocessing:** tree-based models (XGBoost) are "
    "robust to this skew by design (`outlier_method='none'`), but distance-based "
    "models (KNN, SVM) need `StandardScaler` (`scale_numerical=True`) to stop "
    "large-magnitude features like `ProductRelated_Duration` from dominating "
    "the distance calculation."
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# 4. Box Plots (Outlier Inspection)
# ═══════════════════════════════════════════════════════════════════════
st.header("4. Box Plots — Outlier Inspection")
chip("WHICH COLUMNS ARE SAFE TO OUTLIER-FILTER")

show_plot(
    "03_boxplots.png",
    "Box Plots by Revenue",
    caption="Administrative, Administrative_Duration, ProductRelated, "
    "ProductRelated_Duration, BounceRates, and ExitRates show the clearest outliers.",
)
st.warning(
    "⚠️ **Why `Informational*`, `PageValues`, and `SpecialDay` are excluded from "
    "outlier removal:** all three are zero-inflated — their IQR is 0. Running IQR "
    "filtering on them would flag almost every non-zero value as an outlier, "
    "deleting the majority of purchasing sessions in the process. Only the 6 "
    "columns above are used for IQR/Z-score removal."
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# 5. Categorical & Ordinal Feature Distributions
# ═══════════════════════════════════════════════════════════════════════
st.header("5. Categorical & Ordinal Feature Distributions")
chip("ENCODING STRATEGY")

st.markdown(
    "**Nominal categories** (`Month`, `VisitorType`, `Weekend`) — one-hot encoded:"
)
for col_name in CATEGORICAL_FEATURES:
    with st.expander(f"📂 {col_name}"):
        vc = df[col_name].value_counts()
        st.bar_chart(vc)

st.markdown(
    "**Integer-coded IDs** (`OperatingSystems`, `Browser`, `Region`, `TrafficType`) "
    "— treated as numerical (passthrough for XGBoost, scaled for KNN/SVM), *not* "
    "one-hot encoded, since they're high-cardinality IDs rather than a handful of "
    "nominal labels:"
)
for col_name in ORDINAL_FEATURES:
    with st.expander(f"📂 {col_name}"):
        vc = df[col_name].value_counts().sort_index()
        st.bar_chart(vc)

show_plot(
    "04_categorical_distributions.png",
    "Purchase Rate by Category (all 7 fields)",
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# 6. Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════
st.header("6. Correlation Heatmap")
chip("WHAT DRIVES REVENUE")

show_plot("05_correlation_heatmap.png", "Correlation Heatmap", expanded=False)

st.subheader("Top correlations with Revenue")
top_corr_df = stats_["top_corr"].head(5).reset_index()
top_corr_df.columns = ["Feature", "Correlation"]
st.dataframe(top_corr_df.style.format({"Correlation": "{:+.4f}"}), width="stretch")

st.info(
    f"ℹ️ **PageValues is the standout** at r = {stats_['top_corr'].iloc[0]:+.3f} — "
    "this is exactly why it's kept out of outlier removal (see Section 4): "
    "stripping it out would gut the most informative feature in the dataset. "
    f"BounceRates and ExitRates correlate with each other at r = {stats_['be_corr']:.3f} "
    "(see Section 8) but both are kept since they carry complementary signal."
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# 7. PageValues Deep-Dive
# ═══════════════════════════════════════════════════════════════════════
st.header("7. PageValues Deep-Dive")
chip("THE SINGLE STRONGEST PREDICTOR")

p1, p2, p3 = st.columns(3)
p1.metric("Sessions with PageValues = 0", f"{stats_['pv_zero_pct']:.1f}%")
p2.metric("Purchase rate when = 0", f"{stats_['pv_zero_rate']:.2f}%")
p3.metric(
    "Purchase rate when > 0", f"{stats_['pv_pos_rate']:.2f}%", delta="strong signal"
)

show_plot(
    "06_pagevalues_deep_dive.png",
    "PageValues — 4-panel Deep-Dive",
    caption="Violin distribution, zero-inflation breakdown, non-zero histogram, and PageValues vs ExitRates.",
)
st.success(
    f"✅ A session with **any** PageValues > 0 is roughly "
    f"**{stats_['pv_pos_rate'] / max(stats_['pv_zero_rate'], 0.01):.0f}x** more likely "
    "to convert than one with PageValues = 0. This single feature carries most of the "
    "model's discriminative power."
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# 8. BounceRates / ExitRates Analysis
# ═══════════════════════════════════════════════════════════════════════
st.header("8. BounceRates / ExitRates Analysis")
chip(f"CORRELATED AT r = {stats_['be_corr']:.2f}")

show_plot("07_bounce_exit_analysis.png", "BounceRates & ExitRates")
st.info(
    "ℹ️ Both are included in `CONTINUOUS_FEATURES_FOR_OUTLIERS` for KNN/SVM "
    "pipelines (`outlier_method='iqr'` or `'zscore'`) — their high mutual "
    "correlation means they carry overlapping but not identical signal, so "
    "both are kept rather than dropping one."
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# 9. Monthly Purchase Rate
# ═══════════════════════════════════════════════════════════════════════
st.header("9. Monthly Purchase Rate")
chip("SEASONALITY")

mo1, mo2 = st.columns(2)
mo1.metric(
    "Best month",
    stats_["best_month"],
    f"{stats_['month_rate'][stats_['best_month']] * 100:.1f}% purchase rate",
)
mo2.metric(
    "Worst month",
    stats_["worst_month"],
    f"{stats_['month_rate'][stats_['worst_month']] * 100:.1f}% purchase rate",
    delta_color="inverse",
)
show_plot("08_monthly_purchase_rate.png", "Monthly Sessions & Purchase Rate")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# 10. Visitor Type Analysis
# ═══════════════════════════════════════════════════════════════════════
st.header("10. Visitor Type Analysis")
chip("NEW VS RETURNING")

vt_cols = st.columns(len(stats_["vt_rate"]))
for col, (vt_name, rate) in zip(vt_cols, stats_["vt_rate"].items()):
    col.metric(vt_name, f"{rate * 100:.1f}%")
show_plot("09_visitor_type.png", "Sessions, Purchases & Rate by Visitor Type")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# 11. Weekend Effect
# ═══════════════════════════════════════════════════════════════════════
st.header("11. Weekend Effect")
chip("WEEKDAY VS WEEKEND")

w1, w2 = st.columns(2)
w1.metric("Weekday purchase rate", f"{stats_['wk_rate'][False] * 100:.2f}%")
w2.metric("Weekend purchase rate", f"{stats_['wk_rate'][True] * 100:.2f}%")
show_plot("10_weekend_effect.png", "Purchase Rate – Weekday vs Weekend")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# 12. Feature Interaction Analysis
# ═══════════════════════════════════════════════════════════════════════
st.header("12. Feature Interaction Analysis")
chip("COMBINED EFFECTS")

show_plot(
    "11_feature_interactions.png",
    "Duration×PageValues, Month×VisitorType, BounceRates×PageValues, Weekend×VisitorType",
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# 13. Statistical Summary Table
# ═══════════════════════════════════════════════════════════════════════
st.header("13. Statistical Summary Table")
chip("MEAN · MEDIAN · SKEW · KURTOSIS · SIGNIFICANCE")

with st.expander("📐 Full summary (mean / median / std / skew / kurtosis by Revenue)"):
    st.dataframe(stats_["summary_df"], width="stretch", height=420)

st.subheader(
    "Mann-Whitney U test — is each feature significantly different by Revenue?"
)
mw_display = stats_["mw_df"].copy()
mw_display["p-value"] = mw_display["p-value"].apply(lambda v: f"{v:.2e}")
st.dataframe(mw_display, width="stretch")
st.caption(
    "`***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` not significant. "
    "All 10 numerical features differ significantly between Purchase and "
    "No-Purchase sessions — none are dropped on statistical grounds."
)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════
# 14. Key Findings Summary
# ═══════════════════════════════════════════════════════════════════════
st.header("14. Key Findings")
chip("TL;DR")

findings = [
    f"Dataset has **{df.shape[0]:,}** sessions, **{df.shape[1]}** features, no missing values.",
    f"Target is imbalanced: **{counts[0]:,}** No vs **{counts[1]:,}** Yes (ratio ~ **{ratio:.1f}:1**).",
    f"PageValues = 0 in **{stats_['pv_zero_pct']:.1f}%** of sessions; purchase rate "
    f"**{stats_['pv_zero_rate']:.2f}%** vs **{stats_['pv_pos_rate']:.2f}%** when > 0.",
    f"Best month: **{stats_['best_month']}** ({stats_['month_rate'][stats_['best_month']] * 100:.1f}%), "
    f"worst: **{stats_['worst_month']}** ({stats_['month_rate'][stats_['worst_month']] * 100:.1f}%).",
    f"Best visitor type: **{stats_['vt_rate'].index[0]}** ({stats_['vt_rate'].iloc[0] * 100:.1f}% purchase rate).",
    f"Weekday rate: **{stats_['wk_rate'][False] * 100:.2f}%**, weekend rate: **{stats_['wk_rate'][True] * 100:.2f}%**.",
    "**PageValues is the strongest single predictor of Revenue.**",
    "BounceRates and ExitRates are negatively correlated with Revenue.",
    "ProductRelated_Duration and PageValues show clear separation between classes.",
]
st.markdown(
    '<div class="app-card">' + "".join(f"<p>• {f}</p>" for f in findings) + "</div>",
    unsafe_allow_html=True,
)

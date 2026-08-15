import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd

from src.data_preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Data Exploration", page_icon="📊", layout="wide")
st.title("📊 Data Exploration")
st.markdown("---")

# ── Load dataset ────────────────────────────────────────────────────────────
DATA_PATH = project_root / "data" / "raw" / "online_shoppers_intention.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Revenue"] = df["Revenue"].astype(int)
    return df


df = load_data()

# ── 1. Dataset Overview ────────────────────────────────────────────────────
st.header("1. Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Rows", f"{df.shape[0]:,}")
col2.metric("Total Columns", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())

st.subheader("First 10 Rows")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("Data Types")
st.dataframe(
    df.dtypes.reset_index().rename(columns={"index": "Column", 0: "Type"}),
    use_container_width=True,
)

# ── 2. Target Distribution ─────────────────────────────────────────────────
st.header("2. Target Variable – Revenue")
counts = df["Revenue"].value_counts().sort_index()
labels = {0: "No Purchase (0)", 1: "Purchase (1)"}

col1, col2 = st.columns(2)
with col1:
    st.subheader("Value Counts")
    display_counts = counts.rename(index=labels)
    st.bar_chart(display_counts)
with col2:
    st.subheader("Proportion")
    for idx, cnt in counts.items():
        pct = cnt / counts.sum() * 100
        st.write(f"**{labels[idx]}**: {cnt:,} ({pct:.1f}%)")
    st.info(
        f"⚠️ Class imbalance ratio: "
        f"**1 : {counts[0] / counts[1]:.1f}** (No Purchase : Purchase)"
    )

# ── 3. Numerical Statistics ────────────────────────────────────────────────
st.header("3. Numerical Feature Statistics")
st.dataframe(df[NUMERICAL_FEATURES].describe().T, use_container_width=True)

# ── 4. Categorical Feature Value Counts ────────────────────────────────────
st.header("4. Categorical Feature Distributions")
for col_name in CATEGORICAL_FEATURES:
    with st.expander(f"📂 {col_name}"):
        vc = df[col_name].value_counts()
        st.bar_chart(vc)

# ── 5. Saved EDA Plots ─────────────────────────────────────────────────────
PLOT_DIR = project_root / "report_assets" / "plots" / "eda"

if PLOT_DIR.exists():
    png_files = sorted(PLOT_DIR.glob("*.png"))
    if png_files:
        st.header("5. EDA Plots")
        st.caption(f"Auto-loaded from `report_assets/plots/` ({len(png_files)} plots)")
        for img_path in png_files:
            nice_name = img_path.stem.replace("_", " ").title()
            with st.expander(f"🖼️ {nice_name}", expanded=False):
                st.image(str(img_path), use_container_width=True)

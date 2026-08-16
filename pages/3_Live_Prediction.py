import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import joblib

from src.data_preprocessing import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Live Prediction", page_icon="🔮", layout="wide")
st.title("🔮 Live Prediction")
st.markdown("---")

# ── Auto-detect saved models ───────────────────────────────────────────────
SAVED_DIR = project_root / "saved_models"


@st.cache_resource
def discover_models():
    """Scan saved_models/ for .pkl model files."""
    models = {}
    if SAVED_DIR.exists():
        for pkl in sorted(SAVED_DIR.glob("*.pkl")):
            skip_names = {"preprocessor", "scaler"}
            stem = pkl.stem.lower()
            if any(s in stem for s in skip_names):
                continue
            try:
                model = joblib.load(pkl)
                nice_name = pkl.stem.replace("_", " ").title()
                models[nice_name] = model
            except Exception as e:
                st.warning(f"⚠️ Failed to load `{pkl.name}`: {e}")
    return models


models = discover_models()

if not models:
    st.error(
        "❌ No trained models found in `saved_models/`. "
        "Please train at least one model first."
    )
    st.stop()

# ── Model selector ─────────────────────────────────────────────────────────
model_name = st.sidebar.selectbox(
    "Select a model for prediction",
    list(models.keys()),
)
selected_model = models[model_name]
st.sidebar.success(f"Using: **{model_name}**")

# ── Month & VisitorType value options ──────────────────────────────────────
MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "June",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]

# ── Input form ──────────────────────────────────────────────────────────────
st.markdown("### Enter customer browsing session data below:")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Page Metrics")
    administrative = st.number_input("Administrative", min_value=0, value=0)
    administrative_duration = st.number_input(
        "Administrative Duration", min_value=0.0, value=0.0, format="%.2f"
    )
    informational = st.number_input("Informational", min_value=0, value=0)
    informational_duration = st.number_input(
        "Informational Duration", min_value=0.0, value=0.0, format="%.2f"
    )
    product_related = st.number_input("Product Related", min_value=0, value=1)
    product_related_duration = st.number_input(
        "Product Related Duration", min_value=0.0, value=0.0, format="%.2f"
    )

    st.subheader("Engagement Metrics")
    bounce_rates = st.number_input(
        "Bounce Rates", min_value=0.0, max_value=1.0, value=0.02, format="%.4f"
    )
    exit_rates = st.number_input(
        "Exit Rates", min_value=0.0, max_value=1.0, value=0.05, format="%.4f"
    )
    page_values = st.number_input(
        "Page Values", min_value=0.0, value=0.0, format="%.2f"
    )

with col_right:
    st.subheader("Session Context")
    special_day = st.slider(
        "Special Day (proximity to a special date)", 0.0, 1.0, 0.0, 0.2
    )
    month = st.selectbox("Month", MONTHS, index=1)
    operating_system = st.selectbox(
        "Operating System", options=[1, 2, 3, 4, 5, 6, 7, 8], index=1
    )
    browser = st.selectbox(
        "Browser", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], index=1
    )
    region = st.selectbox("Region", options=[1, 2, 3, 4, 5, 6, 7, 8, 9], index=0)
    traffic_type = st.selectbox(
        "Traffic Type",
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        index=1,
    )
    visitor_type = st.selectbox("Visitor Type", VISITOR_TYPES, index=0)
    weekend = st.checkbox("Is Weekend?", value=False)

# ── Build input DataFrame ──────────────────────────────────────────────────
st.markdown("---")

if st.button("🚀 Predict Purchase Intention", type="primary", use_container_width=True):
    input_data = pd.DataFrame(
        [
            {
                "Administrative": administrative,
                "Administrative_Duration": administrative_duration,
                "Informational": informational,
                "Informational_Duration": informational_duration,
                "ProductRelated": product_related,
                "ProductRelated_Duration": product_related_duration,
                "BounceRates": bounce_rates,
                "ExitRates": exit_rates,
                "PageValues": page_values,
                "SpecialDay": special_day,
                "Month": month,
                "OperatingSystems": operating_system,
                "Browser": browser,
                "Region": region,
                "TrafficType": traffic_type,
                "VisitorType": visitor_type,
                "Weekend": weekend,
            }
        ]
    )

    st.subheader("📋 Input Summary")
    st.dataframe(input_data, use_container_width=True)

    try:
        prediction = selected_model.predict(input_data)[0]
        proba = (
            selected_model.predict_proba(input_data)[0]
            if hasattr(selected_model, "predict_proba")
            else None
        )

        st.markdown("---")

        if prediction == 1:
            st.success("### ✅ Prediction: **WILL PURCHASE**")
        else:
            st.error("### ❌ Prediction: **WILL NOT PURCHASE**")

        if proba is not None:
            col1, col2 = st.columns(2)
            col1.metric("No Purchase Probability", f"{proba[0] * 100:.1f}%")
            col2.metric("Purchase Probability", f"{proba[1] * 100:.1f}%")

            st.progress(float(proba[1]))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

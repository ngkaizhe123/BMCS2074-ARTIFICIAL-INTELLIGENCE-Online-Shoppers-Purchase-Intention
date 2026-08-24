import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import joblib
import pandas as pd
import streamlit as st
import time

from src.data_preprocessing import preprocess_data
from src.utils import load_model
from src.ui_theme import (
    apply_theme,
    confidence_label,
    model_icon,
    navigation_breadcrumb,
    page_loading_animation,
    probability_meter,
    verdict_banner,
)

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(page_title="Live Prediction", page_icon="🔮", layout="wide")
apply_theme()

# ── Page loading animation (only on first load) ─────────────────────────
if "lp_loaded" not in st.session_state:
    page_loading_animation(
        "🔮",
        "Live Prediction",
        "Loading trained models for inference...",
        duration=1.2,
    )
    st.session_state["lp_loaded"] = True

# ── Navigation breadcrumb ────────────────────────────────────────────────
navigation_breadcrumb("Live Prediction")

st.title("🔮 Live Prediction")
st.caption(
    "Fill in a browsing session's details — or load an example below — "
    "to see whether the model predicts a purchase."
)
st.markdown("---")

SAVED_DIR = project_root / "saved_models"


# ── Auto-detect saved models ────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
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
                model = load_model(pkl)
                nice_name = pkl.stem.replace("_", " ").title()
                models[nice_name] = {"model": model, "stem": pkl.stem}
            except Exception as e:
                st.warning(f"⚠️ Failed to load `{pkl.name}`: {e}")
    return models


@st.cache_data
def get_sample_pool():
    """Real test-set rows (features + true label), used by the
    'random real session' quick-fill button below."""
    df = preprocess_data(
        filepath=str(project_root / "data" / "raw" / "online_shoppers_intention.csv"),
    )
    from src.utils import split_dataset

    X_train, X_test, y_train, y_test = split_dataset(df)
    pool = X_test.copy()
    pool["Revenue"] = y_test.values
    return pool


models = discover_models()

if not models:
    st.error(
        "❌ No trained models found in `saved_models/`. "
        "Please train at least one model first."
    )
    st.stop()

# ── Sidebar: model picker ───────────────────────────────────────────────
st.sidebar.header("Prediction Mode")
prediction_mode = st.sidebar.radio(
    "Select mode", ["Single Model", "Compare All Models"]
)

st.sidebar.header("Models")
if prediction_mode == "Single Model":
    model_name = st.sidebar.selectbox("Choose a trained model", list(models.keys()))
    selected_info = models[model_name]
    selected_model = selected_info["model"]
    icon = model_icon(selected_info["stem"])
    st.sidebar.markdown(
        f'<span class="model-badge">{icon} Active: {model_name}</span>',
        unsafe_allow_html=True,
    )
    if not hasattr(selected_model, "predict_proba"):
        st.sidebar.caption(
            "ℹ️ This model only returns a hard class label, no probability."
        )
else:
    st.sidebar.info(f"🔮 Will predict using all {len(models)} models.")

# ── Static option lists ─────────────────────────────────────────────────
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

# ── Presets ──────────────────────────────────────────────────────────────
# Defaults match the original form's starting values.
DEFAULT_EXAMPLE = {
    "administrative": 0,
    "administrative_duration": 0.0,
    "informational": 0,
    "informational_duration": 0.0,
    "product_related": 1,
    "product_related_duration": 0.0,
    "bounce_rates": 0.02,
    "exit_rates": 0.05,
    "page_values": 0.0,
    "month": "Feb",
    "operating_system": 2,
    "browser": 2,
    "region": 1,
    "traffic_type": 2,
    "visitor_type": "Returning_Visitor",
    "weekend": False,
}

# Hand-built from the EDA findings: PageValues is the strongest signal,
# November has the highest purchase rate, New_Visitor converts better.
HIGH_INTENT_EXAMPLE = {
    "administrative": 2,
    "administrative_duration": 45.0,
    "informational": 1,
    "informational_duration": 20.0,
    "product_related": 45,
    "product_related_duration": 1200.0,
    "bounce_rates": 0.001,
    "exit_rates": 0.01,
    "page_values": 35.0,
    "month": "Nov",
    "operating_system": 2,
    "browser": 2,
    "region": 1,
    "traffic_type": 2,
    "visitor_type": "New_Visitor",
    "weekend": True,
}

LOW_INTENT_EXAMPLE = {
    "administrative": 0,
    "administrative_duration": 0.0,
    "informational": 0,
    "informational_duration": 0.0,
    "product_related": 2,
    "product_related_duration": 15.0,
    "bounce_rates": 0.2,
    "exit_rates": 0.2,
    "page_values": 0.0,
    "month": "May",
    "operating_system": 2,
    "browser": 2,
    "region": 1,
    "traffic_type": 2,
    "visitor_type": "Returning_Visitor",
    "weekend": False,
}

# Seed session_state on first load so widgets have sensible defaults
# without needing value=/index= (which can conflict with key= updates).
for _k, _v in DEFAULT_EXAMPLE.items():
    st.session_state.setdefault(_k, _v)


def load_preset(preset: dict) -> None:
    st.session_state.pop("_last_random_truth", None)
    for k, v in preset.items():
        st.session_state[k] = v


def load_random_real_session() -> None:
    pool = get_sample_pool()
    row = pool.sample(1).iloc[0]
    preset = {
        "administrative": int(row["Administrative"]),
        "administrative_duration": float(row["Administrative_Duration"]),
        "informational": int(row["Informational"]),
        "informational_duration": float(row["Informational_Duration"]),
        "product_related": int(row["ProductRelated"]),
        "product_related_duration": float(row["ProductRelated_Duration"]),
        "bounce_rates": float(row["BounceRates"]),
        "exit_rates": float(row["ExitRates"]),
        "page_values": float(row["PageValues"]),
        "month": str(row["Month"]),
        "operating_system": int(row["OperatingSystems"]),
        "browser": int(row["Browser"]),
        "region": int(row["Region"]),
        "traffic_type": int(row["TrafficType"]),
        "visitor_type": str(row["VisitorType"]),
        "weekend": bool(row["Weekend"]),
    }
    for k, v in preset.items():
        st.session_state[k] = v
    st.session_state["_last_random_truth"] = int(row["Revenue"])


st.markdown("#### ⚡ Quick fill")
c1, c2, c3, c4 = st.columns(4)
c1.button("🟢 High-intent example", on_click=load_preset, args=(HIGH_INTENT_EXAMPLE,))
c2.button("🔴 Low-intent example", on_click=load_preset, args=(LOW_INTENT_EXAMPLE,))
c3.button("🎲 Random real session", on_click=load_random_real_session)
c4.button("↩️ Reset form", on_click=load_preset, args=(DEFAULT_EXAMPLE,))

if "_last_random_truth" in st.session_state:
    truth = st.session_state["_last_random_truth"]
    truth_label = "Purchase" if truth == 1 else "No Purchase"
    st.caption(
        f"📌 Loaded a real session from the test set — its actual recorded "
        f"outcome was **{truth_label}**. Submit below to see if the model agrees."
    )

st.markdown("---")

# ── Input form (single rerun on submit, instead of one per field) ──────
with st.form("prediction_form"):
    tab1, tab2, tab3 = st.tabs(
        ["🛒 Shopping Behaviour", "📊 Engagement Signals", "🧭 Session Context"]
    )

    with tab1:
        st.markdown(
            '<span class="section-chip">PAGES VIEWED & TIME SPENT</span>',
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            administrative = st.number_input(
                "Administrative pages viewed",
                min_value=0,
                step=1,
                key="administrative",
                help="Account-related pages visited, e.g. order history, account settings.",
            )
            administrative_duration = st.number_input(
                "Time on admin pages (sec)",
                min_value=0.0,
                step=1.0,
                format="%.2f",
                key="administrative_duration",
            )
        with col2:
            informational = st.number_input(
                "Informational pages viewed",
                min_value=0,
                step=1,
                key="informational",
                help="Pages like 'About Us', shipping policy, FAQ.",
            )
            informational_duration = st.number_input(
                "Time on informational pages (sec)",
                min_value=0.0,
                step=1.0,
                format="%.2f",
                key="informational_duration",
            )
        with col3:
            product_related = st.number_input(
                "Product pages viewed",
                min_value=0,
                step=1,
                key="product_related",
                help="Number of actual product pages browsed — usually the strongest volume signal.",
            )
            product_related_duration = st.number_input(
                "Time on product pages (sec)",
                min_value=0.0,
                step=1.0,
                format="%.2f",
                key="product_related_duration",
            )

    with tab2:
        st.markdown(
            '<span class="section-chip">HOW ENGAGED WAS THE VISITOR</span>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            bounce_rates = st.number_input(
                "Bounce rate",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.4f",
                key="bounce_rates",
                help="Share of visitors who left immediately after viewing only this page. Lower = more engaged.",
            )
            exit_rates = st.number_input(
                "Exit rate",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                format="%.4f",
                key="exit_rates",
                help="Share of pageviews to this page that were the last one in the session. Lower = more engaged.",
            )
        with col2:
            page_values = st.number_input(
                "Page value",
                min_value=0.0,
                step=1.0,
                format="%.2f",
                key="page_values",
                help="⭐ The strongest predictor in this dataset — the average value "
                "(from Google Analytics) of the pages visited before a purchase.",
            )

    with tab3:
        st.markdown(
            '<span class="section-chip">WHEN & HOW THEY ARRIVED</span>',
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            month = st.selectbox("Month", MONTHS, key="month")
            visitor_type = st.selectbox(
                "Visitor type", VISITOR_TYPES, key="visitor_type"
            )
            weekend = st.checkbox("Session on a weekend?", key="weekend")
        with col2:
            operating_system = st.selectbox(
                "Operating system (code)",
                options=list(range(1, 9)),
                key="operating_system",
            )
            browser = st.selectbox(
                "Browser (code)", options=list(range(1, 14)), key="browser"
            )
        with col3:
            region = st.selectbox(
                "Region (code)", options=list(range(1, 10)), key="region"
            )
            traffic_type = st.selectbox(
                "Traffic type (code)", options=list(range(1, 21)), key="traffic_type"
            )

    submitted = st.form_submit_button(
        "🚀 Predict Purchase Intention", type="primary", width="stretch"
    )

# ── Prediction (runs once, only on submit) ──────────────────────────────
if submitted:
    # ── Advanced Loading Animation ──
    with st.spinner("🤖 AI is analyzing the session data..."):
        time.sleep(1.5)  # Add dramatic suspense for the prediction

    warnings = []
    if product_related == 0 and product_related_duration > 0:
        warnings.append(
            "Product pages viewed is 0 but time on product pages is > 0 — double-check these values."
        )
    if administrative == 0 and administrative_duration > 0:
        warnings.append(
            "Administrative pages viewed is 0 but time on admin pages is > 0 — double-check these values."
        )
    if informational == 0 and informational_duration > 0:
        warnings.append(
            "Informational pages viewed is 0 but time on informational pages is > 0 — double-check these values."
        )
    for w in warnings:
        st.warning(f"⚠️ {w}")

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
                "Month": str(month),
                "OperatingSystems": str(operating_system),
                "Browser": str(browser),
                "Region": str(region),
                "TrafficType": str(traffic_type),
                "VisitorType": str(visitor_type),
                "Weekend": bool(weekend),
            }
        ]
    )

    try:
        if prediction_mode == "Single Model":
            prediction = selected_model.predict(input_data)[0]
            proba = (
                selected_model.predict_proba(input_data)[0]
                if hasattr(selected_model, "predict_proba")
                else None
            )

            st.markdown("---")
            purchase_proba = float(proba[1]) if proba is not None else float(prediction)
            confidence_pct = (
                purchase_proba * 100 if prediction == 1 else (1 - purchase_proba) * 100
            )

            # Trigger Success/Failure animations based on prediction
            if prediction == 1:
                st.balloons()
            else:
                st.snow()

            verdict_banner(bool(prediction == 1), confidence_pct)

            if proba is not None:
                probability_meter(purchase_proba)
                m1, m2, m3 = st.columns(3)
                m1.metric("No Purchase probability", f"{proba[0] * 100:.1f}%")
                m2.metric("Purchase probability", f"{proba[1] * 100:.1f}%")
                m3.metric("Confidence", confidence_label(purchase_proba))
            else:
                st.info(
                    "This model only exposes a hard class prediction, not a probability estimate."
                )

            if "_last_random_truth" in st.session_state:
                truth = st.session_state["_last_random_truth"]
                truth_label = "Purchase" if truth == 1 else "No Purchase"
                if truth == prediction:
                    st.success(f"✅ Matches the real recorded outcome ({truth_label}).")
                else:
                    st.error(
                        f"❌ Model disagreed with the real recorded outcome ({truth_label})."
                    )
        else:
            st.markdown("---")
            st.subheader("🤖 Multiple Model Predictions")

            if "_last_random_truth" in st.session_state:
                truth = st.session_state["_last_random_truth"]
                truth_label = "Purchase" if truth == 1 else "No Purchase"
                st.info(f"📌 Actual recorded outcome: **{truth_label}**")

            # ── Collect all predictions first ────────────────────────────
            all_results = []
            for name, info in models.items():
                model = info["model"]
                try:
                    pred = model.predict(input_data)[0]
                    prob = (
                        model.predict_proba(input_data)[0]
                        if hasattr(model, "predict_proba")
                        else None
                    )
                    pur_prob = float(prob[1]) if prob is not None else float(pred)
                    all_results.append(
                        {
                            "name": name,
                            "icon": model_icon(info["stem"]),
                            "pred": pred,
                            "prob": prob,
                            "pur_prob": pur_prob,
                            "error": None,
                        }
                    )
                except Exception as e:
                    all_results.append(
                        {
                            "name": name,
                            "icon": model_icon(info["stem"]),
                            "pred": None,
                            "prob": None,
                            "pur_prob": None,
                            "error": str(e),
                        }
                    )

            # ── Determine consensus and trigger correct animation ────────
            valid_preds = [r["pred"] for r in all_results if r["pred"] is not None]
            purchase_count = sum(1 for p in valid_preds if p == 1)
            total_count = len(valid_preds)

            if total_count > 0:
                from src.ui_theme import multi_model_verdict_banner

                if purchase_count == total_count:
                    # All models agree: Purchase → balloons
                    st.balloons()
                elif purchase_count == 0:
                    # All models agree: No Purchase → snow
                    st.snow()
                # Mixed results → no balloons/snow, just the amber banner

                multi_model_verdict_banner(purchase_count, total_count)

            # ── Display per-model cards ──────────────────────────────────
            cols = st.columns(len(all_results))
            for col, result in zip(cols, all_results):
                with col:
                    st.markdown(f"**{result['icon']} {result['name']}**")

                    if result["error"]:
                        st.error(f"Error: {result['error']}")
                        continue

                    if result["pred"] == 1:
                        st.success("🛒 Purchase")
                    else:
                        st.error("❌ No Purchase")

                    if result["prob"] is not None:
                        st.caption(
                            f"Purchase Probability: {result['pur_prob']*100:.1f}%"
                        )
                        probability_meter(result["pur_prob"])
                    else:
                        st.caption("No probability available")

        with st.expander("📋 Input summary", expanded=False):
            st.dataframe(input_data, width="stretch")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

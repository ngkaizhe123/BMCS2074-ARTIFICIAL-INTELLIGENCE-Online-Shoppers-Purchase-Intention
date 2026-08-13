import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import joblib

from src.data_preprocessing import preprocess_data
from src.utils import evaluate_model

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")
st.title("📈 Model Comparison")
st.markdown("---")

# ── Auto-detect saved models ───────────────────────────────────────────────
SAVED_DIR = project_root / "saved_models"
PLOT_DIR = project_root / "report_assets" / "plots"


@st.cache_resource
def discover_models():
    """Scan saved_models/ for .pkl files and load them."""
    models = {}
    if SAVED_DIR.exists():
        for pkl in sorted(SAVED_DIR.glob("*.pkl")):
            # Skip non-model files (e.g. preprocessor.pkl, scaler.pkl)
            skip_names = {"preprocessor", "scaler"}
            stem = pkl.stem.lower()
            if any(s in stem for s in skip_names):
                continue
            try:
                model = joblib.load(pkl)
                nice_name = pkl.stem.replace("_", " ").title()
                models[nice_name] = {"path": pkl, "model": model, "stem": pkl.stem.split("_")[0]}
            except Exception as e:
                st.warning(f"⚠️ Failed to load `{pkl.name}`: {e}")
    return models


@st.cache_data
def get_test_data():
    """Load and split data (returns raw DataFrames for pipeline models)."""
    X_train, X_test, y_train, y_test, _ = preprocess_data(
        filepath=str(project_root / "data" / "raw" / "online_shoppers_intention.csv"),
        transform=False,
    )
    return X_test, y_test


models = discover_models()

if not models:
    st.error(
        "❌ No trained models found in `saved_models/`. "
        "Please train at least one model first."
    )
    st.stop()

st.success(f"✅ Found **{len(models)}** trained model(s): {', '.join(models.keys())}")

# ── Evaluate all models on the same test set ───────────────────────────────
X_test, y_test = get_test_data()

results = []
detail_metrics = {}

for name, info in models.items():
    model = info["model"]
    try:
        metrics = evaluate_model(model, X_test, y_test)
        results.append(
            {
                "Model": name,
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1 Score": metrics["F1"],
                "AUC": metrics["AUC"] if metrics["AUC"] is not None else "N/A",
            }
        )
        detail_metrics[name] = {"metrics": metrics, "stem": info["stem"]}
    except Exception as e:
        st.warning(f"⚠️ Could not evaluate **{name}**: {e}")

# ── Summary Table ──────────────────────────────────────────────────────────
if results:
    st.header("Performance Summary")
    summary_df = pd.DataFrame(results).set_index("Model")
    st.dataframe(
        summary_df.style.format(
            {c: "{:.4f}" for c in summary_df.columns if c != "AUC"}
        ).highlight_max(axis=0, color="#d4edda"),
        use_container_width=True,
    )

    # ── Bar chart comparison ───────────────────────────────────────────────
    st.header("Metric Comparison")
    chart_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
    chart_df = summary_df[chart_cols]
    st.bar_chart(chart_df)

    # ── Per-model details & SHAP Explanations ───────────────────────────────
    st.header("Detailed Reports & SHAP Interpretability")
    for name, item in detail_metrics.items():
        metrics = item["metrics"]
        stem = item["stem"]

        with st.expander(f"📄 {name} Details & Interpretability", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Confusion Matrix")
                cm = metrics["Confusion Matrix"]
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual: No", "Actual: Yes"],
                    columns=["Pred: No", "Pred: Yes"],
                )
                st.dataframe(cm_df, use_container_width=True)
            with col2:
                st.subheader("Classification Report")
                st.text(metrics["Classification Report"])

            # Check if SHAP plots exist for this model
            if PLOT_DIR.exists():
                shap_plots = list(PLOT_DIR.glob(f"{stem}_shap_*.png"))
                if shap_plots:
                    st.markdown("---")
                    st.subheader(f"🧠 SHAP Model Explanations ({name})")
                    for img_file in sorted(shap_plots):
                        plot_title = img_file.stem.replace("_", " ").title()
                        st.image(str(img_file), caption=plot_title, use_container_width=True)

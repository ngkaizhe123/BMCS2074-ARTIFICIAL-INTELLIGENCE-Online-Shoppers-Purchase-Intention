import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import json

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Model Comparison", page_icon="📈", layout="wide")
st.title("📈 Model Comparison")
st.markdown("---")

# ── Auto-detect saved models ───────────────────────────────────────────────
PLOT_DIR = project_root / "report_assets" / "plots"
METRICS_PATH = project_root / "report_assets" / "metrics.json"

if not METRICS_PATH.exists():
    st.error(
        "❌ `metrics.json` not found.\n\n"
        "Please run `python src/model_visualize.py` first to evaluate models and generate metrics."
    )
    st.stop()

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    all_metrics = json.load(f)

if not all_metrics:
    st.warning("No model metrics found in `metrics.json`.")
    st.stop()

st.success(f"✅ Loaded metrics for **{len(all_metrics)}** trained model(s): {', '.join(all_metrics.keys())}")

results = []
detail_metrics = {}

for name, metrics in all_metrics.items():
    results.append(
        {
            "Model": name,
            "Accuracy": metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1 Score": metrics["F1 Score"],
            "AUC": metrics["AUC"] if metrics["AUC"] is not None else "N/A",
        }
    )
    detail_metrics[name] = {"metrics": metrics, "stem": metrics["stem"]}

# ── Summary Table ──────────────────────────────────────────────────────────
# ── Summary Table ──────────────────────────────────────────────────────────
if results:
    st.header("🏆 Performance Summary")
    summary_df = pd.DataFrame(results).set_index("Model")
    
    # Custom styling
    styled_df = (
        summary_df.style
        .format({c: "{:.4f}" for c in summary_df.columns if c != "AUC"})
        .background_gradient(cmap="Blues", axis=0)
    )
    st.dataframe(styled_df, use_container_width=True)

    # ── Bar chart comparison ───────────────────────────────────────────────
    st.header("📊 Metric Comparison")
    
    metrics_plot_path = PLOT_DIR / "model_comparison_metrics.png"
    auc_plot_path = PLOT_DIR / "model_comparison_auc.png"
    
    col_metric, col_auc = st.columns(2)
    with col_metric:
        if metrics_plot_path.exists():
            st.image(str(metrics_plot_path), use_container_width=True)
        else:
            # Fallback to streamlit native chart
            chart_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
            chart_df = summary_df[chart_cols]
            st.bar_chart(chart_df)
            
    with col_auc:
        if auc_plot_path.exists():
            st.image(str(auc_plot_path), use_container_width=True)

    # ── Per-model details & SHAP Explanations ───────────────────────────────
    st.header("🔬 Detailed Reports & Interpretability")
    
    model_names = list(detail_metrics.keys())
    tabs = st.tabs([f"🤖 {name}" for name in model_names])
    
    for tab, (name, item) in zip(tabs, detail_metrics.items()):
        with tab:
            metrics = item["metrics"]
            stem = item["stem"]

            st.markdown(f"### Performance Metrics: **{name}**")
            
            col_cm, col_roc = st.columns(2)
            with col_cm:
                cm_plot_path = PLOT_DIR / f"confusion_matrix_{stem}.png"
                if cm_plot_path.exists():
                    st.image(str(cm_plot_path), use_container_width=True)
                else:
                    st.markdown("**Confusion Matrix**")
                    cm = metrics["Confusion Matrix"]
                    cm_df = pd.DataFrame(
                        cm,
                        index=["Actual: No", "Actual: Yes"],
                        columns=["Pred: No", "Pred: Yes"],
                    )
                    st.dataframe(cm_df, use_container_width=True)
            
            with col_roc:
                roc_plot_path = PLOT_DIR / f"roc_curve_{stem}.png"
                if roc_plot_path.exists():
                    st.image(str(roc_plot_path), use_container_width=True)

            st.markdown("**Classification Report**")
            st.code(metrics["Classification Report"], language="text")

            # Check if SHAP plots exist for this model (using glob with wildcard after stem to catch knn_rf etc)
            if PLOT_DIR.exists():
                shap_plots = list(PLOT_DIR.glob(f"{stem}*_shap_*.png"))
                if shap_plots:
                    st.divider()
                    st.markdown("### 🧠 SHAP Model Explanations")
                    
                    beeswarm = next((p for p in shap_plots if "beeswarm" in p.name.lower()), None)
                    importance = next((p for p in shap_plots if "importance" in p.name.lower()), None)
                    waterfall = next((p for p in shap_plots if "waterfall" in p.name.lower()), None)
                    
                    c1, c2 = st.columns(2)
                    if beeswarm:
                        with c1:
                            st.markdown("##### Beeswarm Plot (Global Impact)")
                            st.image(str(beeswarm), use_container_width=True)
                    if importance:
                        with c2:
                            st.markdown("##### Feature Importance (Mean |SHAP|)")
                            st.image(str(importance), use_container_width=True)
                            
                    if waterfall:
                        st.markdown("##### Waterfall Plot (Local Explanation - Sample #0)")
                        c3, c4, c5 = st.columns([1, 2, 1])
                        with c4:
                            st.image(str(waterfall), use_container_width=True)

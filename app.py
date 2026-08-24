import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
from models import FuzzySVM, HybridKernelSVC
from src.ui_theme import apply_theme, navigation_breadcrumb, page_loading_animation

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Online Shopper Purchase Intention",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

# ── Page loading animation (only on first load) ─────────────────────────────
if "home_loaded" not in st.session_state:
    page_loading_animation(
        "🛍️",
        "Welcome to Online Shopper Purchase Intention",
        "Initializing the system...",
        duration=1.5,
    )
    st.session_state["home_loaded"] = True

# ── Navigation breadcrumb ────────────────────────────────────────────────────
navigation_breadcrumb("Home")

# ── Main landing page ──────────────────────────────────────────────────────
st.title("🛍️ Online Shopper Purchase Intention System")
st.markdown("---")

st.markdown("""
### 📌 Project Overview
This platform demonstrates the use of **Supervised Machine Learning** to predict 
whether an online shopper will make a purchase, based on their browsing behaviour 
(page views, bounce rate, session duration, visitor type, etc.).

### 🎯 Objectives
* **Binary Classification** – Predict purchase intent (`0` = No Purchase, `1` = Purchase).
* **Multi-Model Comparison** – Train and compare several ML algorithms and evaluate 
  their performance on the same dataset.
* **Imbalanced Data Handling** – Apply **SMOTE** oversampling inside an `imblearn` 
  Pipeline to avoid data leakage during cross-validation.

---

### 🚀 Navigation Guide
Use the **sidebar** on the left to explore:

| Page | Description |
|------|-------------|
| 📊 **Data Exploration** | View dataset structure, distributions, and EDA plots |
| 📈 **Model Comparison** | Compare evaluation metrics across all trained models |
| 🔮 **Live Prediction** | Enter simulated user data and get a real-time purchase prediction |

---

### 👨‍💻 Development Team
* **NG KAI ZHE** – XGBoost model & Streamlit architecture
* **TAN YONG SHENG** – SVM model
* **YAU SOON HAN** – KNN model & feature engineering
""")

# ── Sidebar: show detected models ──────────────────────────────────────────
saved_dir = project_root / "saved_models"
pkl_files = sorted(saved_dir.glob("*.pkl")) if saved_dir.exists() else []

st.sidebar.markdown("### Detected Models")
if pkl_files:
    for p in pkl_files:
        st.sidebar.success(f"✅  `{p.name}`")
else:
    st.sidebar.warning("No `.pkl` models found in `saved_models/`.")

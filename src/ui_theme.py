"""
ui_theme.py
-----------
Shared visual theme and reusable UI building blocks for the Streamlit app,
so Data Exploration / Model Comparison / Live Prediction all share one
consistent look instead of each relying on Streamlit's bare defaults.

Usage in any page, right after st.set_page_config(...):

    from src.ui_theme import apply_theme
    apply_theme()
"""

import streamlit as st

# Palette matches src/eda.py's PALETTE_BINARY so plots and UI agree:
# blue = "No Purchase", orange = "Purchase".
PRIMARY = "#4C72B0"
ACCENT = "#DD8452"
SUCCESS = "#0E9F6E"
DANGER = "#E02424"
MUTED = "#6B7280"
BORDER = "#E5E7EB"


def apply_theme() -> None:
    """Inject shared CSS. Safe to call once at the top of every page."""
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 2rem; max-width: 1200px; }}

        /* Generic card */
        .app-card {{
            background: #FFFFFF;
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 1.1rem 1.4rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }}

        /* Small pill/badge */
        .badge {{
            display: inline-block;
            border-radius: 999px;
            padding: 0.15rem 0.7rem;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge-primary {{ background: {PRIMARY}1F; color: {PRIMARY}; }}
        .badge-accent  {{ background: {ACCENT}26; color: #9A4A20; }}

        /* Section label above a group of inputs */
        .section-chip {{
            display: inline-block;
            background: #EEF2FF;
            color: #4338CA;
            border-radius: 999px;
            padding: 0.15rem 0.7rem;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            margin-bottom: 0.6rem;
        }}

        /* Prediction verdict banners */
        .verdict-yes, .verdict-no {{
            border-radius: 14px;
            padding: 1.4rem 1rem;
            text-align: center;
            margin-bottom: 0.75rem;
        }}
        .verdict-yes {{
            background: linear-gradient(135deg, #ECFDF5, #D1FAE5);
            border: 1px solid #6EE7B7;
        }}
        .verdict-no {{
            background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
            border: 1px solid #FCA5A5;
        }}
        .verdict-title {{ font-size: 1.55rem; font-weight: 800; margin: 0; }}
        .verdict-sub {{ color: {MUTED}; margin-top: 0.2rem; font-size: 0.95rem; }}

        /* Make preset / quick-fill buttons full width & rounded */
        div[data-testid="stHorizontalBlock"] .stButton button {{
            width: 100%;
            border-radius: 10px;
        }}

        /* Sidebar model badge */
        .model-badge {{
            display: inline-block;
            background: {PRIMARY}1F;
            color: {PRIMARY};
            border-radius: 8px;
            padding: 0.25rem 0.6rem;
            font-size: 0.8rem;
            font-weight: 700;
            margin-top: 0.3rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def verdict_banner(will_purchase: bool, confidence_pct: float) -> None:
    """Big colored verdict banner shown after a prediction."""
    if will_purchase:
        st.markdown(
            f"""
            <div class="verdict-yes">
                <p class="verdict-title">✅ Likely to Purchase</p>
                <p class="verdict-sub">Model confidence: {confidence_pct:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="verdict-no">
                <p class="verdict-title">🚫 Unlikely to Purchase</p>
                <p class="verdict-sub">Model confidence: {confidence_pct:.1f}%</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def probability_meter(proba_purchase: float) -> None:
    """Horizontal gradient meter (blue=No Purchase -> orange=Purchase)
    with a marker at the predicted purchase probability. Pure HTML/CSS,
    no extra plotting dependency needed."""
    pct = max(0.0, min(1.0, proba_purchase)) * 100
    st.markdown(
        f"""
        <div style="margin: 0.4rem 0 1.1rem 0;">
          <div style="display:flex; justify-content:space-between; font-size:0.78rem;
                      color:{MUTED}; margin-bottom:4px;">
            <span>0% · No Purchase</span><span>50%</span><span>100% · Purchase</span>
          </div>
          <div style="position:relative; height:14px; border-radius:999px;
                      background: linear-gradient(90deg, {PRIMARY} 0%, #E5E7EB 50%, {ACCENT} 100%);">
            <div style="position:absolute; left:{pct}%; top:-7px; transform:translateX(-50%);
                        width:0; height:0; border-left:8px solid transparent;
                        border-right:8px solid transparent; border-top:11px solid #111827;"></div>
          </div>
          <div style="text-align:center; margin-top:10px; font-size:1.4rem; font-weight:800;">
            {pct:.1f}%
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_label(proba_purchase: float) -> str:
    """Qualitative label for how far the prediction sits from the 0.5
    decision boundary — helps a non-technical viewer read the number."""
    dist = abs(proba_purchase - 0.5)
    if dist >= 0.35:
        return "Very High"
    if dist >= 0.20:
        return "High"
    if dist >= 0.08:
        return "Moderate"
    return "Low (borderline case)"


def model_icon(stem: str) -> str:
    """Best-effort icon based on the model file's name."""
    s = stem.lower()
    if "xgb" in s or "xgboost" in s or "boost" in s:
        return "🌳"
    if "svm" in s or "svc" in s:
        return "📐"
    if "knn" in s or "neighbor" in s:
        return "📍"
    if "forest" in s or "rf" in s:
        return "🌲"
    return "🤖"

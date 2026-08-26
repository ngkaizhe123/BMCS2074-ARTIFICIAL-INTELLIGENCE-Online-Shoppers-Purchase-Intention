"""
ui_theme.py
-----------
Shared visual theme and reusable UI building blocks for the Streamlit app,
so Data Exploration / Model Comparison / Live Prediction all share one
consistent look instead of each relying on Streamlit's bare defaults.

Usage in any page, right after st.set_page_config(...):

    from src.ui_theme import apply_theme, page_loading_animation
    apply_theme()
    page_loading_animation("📊", "Data Exploration", "Loading dataset and charts...")
"""

import time
import streamlit as st

# Palette matches src/eda.py's PALETTE_BINARY so plots and UI agree:
# blue = "No Purchase", orange = "Purchase".
PRIMARY = "#4C72B0"
ACCENT = "#DD8452"
SUCCESS = "#0E9F6E"
DANGER = "#E02424"
MUTED = "#6B7280"
BORDER = "#E5E7EB"

# ── Page registry for navigation breadcrumb ──────────────────────────────
PAGE_REGISTRY = {
    "Home": {"icon": "🛍️", "color": "#6366F1"},
    "Data Exploration": {"icon": "📊", "color": "#8B5CF6"},
    "Model Comparison": {"icon": "📈", "color": "#EC4899"},
    "Live Prediction": {"icon": "🔮", "color": "#F59E0B"},
}


def apply_theme() -> None:
    """Inject shared CSS. Safe to call once at the top of every page."""
    st.markdown(
        f"""
        <style>
        /* ═══════════════════════════════════════════════════════════════
           GLOBAL LAYOUT
           ═══════════════════════════════════════════════════════════════ */
        .block-container {{ padding-top: 3.5rem; max-width: 1200px; }}

        /* ═══════════════════════════════════════════════════════════════
           1. ANIMATION KEYFRAMES
           ═══════════════════════════════════════════════════════════════ */
        @keyframes fadeInUp {{
            0%   {{ opacity: 0; transform: translateY(20px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes fadeInLeft {{
            0%   {{ opacity: 0; transform: translateX(-30px); }}
            100% {{ opacity: 1; transform: translateX(0); }}
        }}
        @keyframes fadeInRight {{
            0%   {{ opacity: 0; transform: translateX(30px); }}
            100% {{ opacity: 1; transform: translateX(0); }}
        }}
        @keyframes fadeInScale {{
            0%   {{ opacity: 0; transform: scale(0.85); }}
            100% {{ opacity: 1; transform: scale(1); }}
        }}
        @keyframes popIn {{
            0%   {{ opacity: 0; transform: scale(0.9) translateY(20px); }}
            70%  {{ transform: scale(1.03); }}
            100% {{ opacity: 1; transform: scale(1) translateY(0); }}
        }}
        @keyframes pulseGlow {{
            0%   {{ box-shadow: 0 0 0 0 rgba(14, 159, 110, 0.5); }}
            70%  {{ box-shadow: 0 0 0 15px rgba(14, 159, 110, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(14, 159, 110, 0); }}
        }}
        @keyframes shakeError {{
            0%, 100% {{ transform: translateX(0); }}
            10%, 30%, 50%, 70%, 90% {{ transform: translateX(-4px); }}
            20%, 40%, 60%, 80% {{ transform: translateX(4px); }}
        }}
        @keyframes shimmer {{
            0%   {{ background-position: -200% 0; }}
            100% {{ background-position: 200% 0; }}
        }}
        @keyframes slideDown {{
            0%   {{ opacity: 0; transform: translateY(-10px); max-height: 0; }}
            100% {{ opacity: 1; transform: translateY(0); max-height: 500px; }}
        }}
        @keyframes gradientShift {{
            0%   {{ background-position: 0% 50%; }}
            50%  {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        @keyframes float {{
            0%, 100% {{ transform: translateY(0px); }}
            50%      {{ transform: translateY(-6px); }}
        }}
        @keyframes countUp {{
            0%   {{ opacity: 0; transform: translateY(10px); }}
            100% {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes borderGlow {{
            0%, 100% {{ border-color: rgba(99, 102, 241, 0.3); }}
            50%      {{ border-color: rgba(99, 102, 241, 0.8); }}
        }}
        @keyframes fillProgress {{
            from {{ left: 0%; }}
            to   {{ left: var(--target-pct); }}
        }}
        @keyframes typewriter {{
            from {{ width: 0; }}
            to   {{ width: 100%; }}
        }}

        /* ═══════════════════════════════════════════════════════════════
           2. PAGE TRANSITION — CASCADE FADE-IN
           ═══════════════════════════════════════════════════════════════ */
        .main > div > div > div > div {{
            animation: fadeInUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        }}
        /* Stagger child blocks for a cascade waterfall effect */
        .main .stVerticalBlock > div:nth-child(1)  {{ animation-delay: 0.05s; }}
        .main .stVerticalBlock > div:nth-child(2)  {{ animation-delay: 0.10s; }}
        .main .stVerticalBlock > div:nth-child(3)  {{ animation-delay: 0.15s; }}
        .main .stVerticalBlock > div:nth-child(4)  {{ animation-delay: 0.20s; }}
        .main .stVerticalBlock > div:nth-child(5)  {{ animation-delay: 0.25s; }}
        .main .stVerticalBlock > div:nth-child(6)  {{ animation-delay: 0.30s; }}
        .main .stVerticalBlock > div:nth-child(7)  {{ animation-delay: 0.35s; }}
        .main .stVerticalBlock > div:nth-child(8)  {{ animation-delay: 0.40s; }}
        .main .stVerticalBlock > div:nth-child(9)  {{ animation-delay: 0.45s; }}
        .main .stVerticalBlock > div:nth-child(10) {{ animation-delay: 0.50s; }}

        /* ═══════════════════════════════════════════════════════════════
           3. NAVIGATION BREADCRUMB BAR
           ═══════════════════════════════════════════════════════════════ */
        .nav-breadcrumb {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 0.6rem 1rem;
            background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
            border: 1px solid #e0e7ff;
            border-radius: 12px;
            margin-bottom: 1rem;
            animation: fadeInLeft 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
            overflow: hidden;
        }}
        .nav-crumb {{
            font-size: 0.82rem;
            color: {MUTED};
            transition: color 0.2s ease;
        }}
        .nav-crumb a {{
            color: {PRIMARY};
            text-decoration: none;
            font-weight: 500;
        }}
        .nav-crumb a:hover {{
            text-decoration: underline;
        }}
        .nav-crumb-active {{
            font-size: 0.82rem;
            font-weight: 700;
            color: #4338CA;
            background: #EEF2FF;
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            animation: popIn 0.5s ease both 0.3s;
        }}
        .nav-separator {{
            color: #C7D2FE;
            font-size: 0.7rem;
        }}

        /* ═══════════════════════════════════════════════════════════════
           4. PAGE LOADING OVERLAY
           ═══════════════════════════════════════════════════════════════ */
        .page-loader {{
            text-align: center;
            padding: 3rem 1rem;
            animation: fadeInScale 0.6s ease both;
        }}
        .page-loader-icon {{
            font-size: 3.5rem;
            animation: float 2s ease-in-out infinite;
            display: block;
            margin-bottom: 1rem;
        }}
        .page-loader-title {{
            font-size: 1.3rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 0.4rem;
        }}
        .page-loader-sub {{
            font-size: 0.9rem;
            color: {MUTED};
        }}
        .page-loader-bar {{
            width: 200px;
            height: 4px;
            border-radius: 999px;
            margin: 1.2rem auto 0 auto;
            background: linear-gradient(90deg, {PRIMARY}, {ACCENT}, {PRIMARY});
            background-size: 200% auto;
            animation: shimmer 1.5s linear infinite;
        }}

        /* ═══════════════════════════════════════════════════════════════
           5. BUTTON ANIMATIONS
           ═══════════════════════════════════════════════════════════════ */
        .stButton > button {{
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: relative;
            overflow: hidden;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px) scale(1.02) !important;
            box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.15),
                        0 4px 10px -3px rgba(0, 0, 0, 0.08) !important;
            filter: brightness(1.05) !important;
        }}
        .stButton > button:active {{
            transform: translateY(1px) scale(0.98) !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        }}
        /* Ripple effect */
        .stButton > button::after {{
            content: '';
            position: absolute;
            top: 50%; left: 50%;
            width: 120%; height: 120%;
            background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
            border-radius: 50%;
            transform: translate(-50%, -50%) scale(0);
            opacity: 0;
            transition: transform 0.5s ease-out, opacity 0.5s ease-out;
        }}
        .stButton > button:active::after {{
            transform: translate(-50%, -50%) scale(2.5);
            opacity: 1;
            transition: 0s;
        }}

        /* ═══════════════════════════════════════════════════════════════
           6. INPUT & SELECTION BOX ANIMATIONS
           ═══════════════════════════════════════════════════════════════ */
        div[data-testid="stNumberInput"] input,
        div[data-baseweb="select"] > div {{
            transition: all 0.3s ease !important;
            border: 1px solid {BORDER} !important;
        }}
        div[data-testid="stNumberInput"] input:focus,
        div[data-baseweb="select"] > div:focus-within {{
            transform: scale(1.01);
            box-shadow: 0 0 0 3px rgba(76, 114, 176, 0.15),
                        0 4px 12px rgba(76, 114, 176, 0.08) !important;
            border-color: {PRIMARY} !important;
        }}
        /* Checkbox animation */
        div[data-testid="stCheckbox"] label {{
            transition: all 0.2s ease !important;
        }}
        div[data-testid="stCheckbox"] label:hover {{
            transform: translateX(2px);
        }}
        /* Slider thumb glow */
        div[data-testid="stSlider"] > div {{
            transition: all 0.3s ease !important;
        }}

        /* ═══════════════════════════════════════════════════════════════
           7. TABS ANIMATION
           ═══════════════════════════════════════════════════════════════ */
        button[data-baseweb="tab"] {{
            transition: all 0.3s ease !important;
        }}
        button[data-baseweb="tab"]:hover {{
            background-color: rgba(76, 114, 176, 0.06) !important;
            transform: translateY(-2px);
        }}
        /* Tab content slide-in */
        div[data-baseweb="tab-panel"] {{
            animation: slideDown 0.4s ease both;
        }}

        /* ═══════════════════════════════════════════════════════════════
           8. METRIC CARDS ANIMATION
           ═══════════════════════════════════════════════════════════════ */
        div[data-testid="stMetric"] {{
            animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) both;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border-radius: 8px;
            padding: 0.3rem;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-3px) scale(1.02);
            box-shadow: 0 8px 20px -5px rgba(0,0,0,0.1);
        }}
        /* Stagger metrics in a row */
        div[data-testid="stHorizontalBlock"] > div:nth-child(1) div[data-testid="stMetric"] {{ animation-delay: 0.1s; }}
        div[data-testid="stHorizontalBlock"] > div:nth-child(2) div[data-testid="stMetric"] {{ animation-delay: 0.2s; }}
        div[data-testid="stHorizontalBlock"] > div:nth-child(3) div[data-testid="stMetric"] {{ animation-delay: 0.3s; }}
        div[data-testid="stHorizontalBlock"] > div:nth-child(4) div[data-testid="stMetric"] {{ animation-delay: 0.4s; }}
        div[data-testid="stHorizontalBlock"] > div:nth-child(5) div[data-testid="stMetric"] {{ animation-delay: 0.5s; }}

        /* ═══════════════════════════════════════════════════════════════
           9. EXPANDER ANIMATION
           ═══════════════════════════════════════════════════════════════ */
        details[data-testid="stExpander"] {{
            transition: all 0.3s ease !important;
            border-radius: 10px !important;
        }}
        details[data-testid="stExpander"]:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        }}
        details[data-testid="stExpander"][open] > div {{
            animation: slideDown 0.4s ease both;
        }}

        /* ═══════════════════════════════════════════════════════════════
           10. SIDEBAR ANIMATIONS
           ═══════════════════════════════════════════════════════════════ */
        section[data-testid="stSidebar"] {{
            transition: all 0.3s ease;
        }}
        section[data-testid="stSidebar"] .stRadio label {{
            transition: all 0.2s ease !important;
            border-radius: 6px;
            padding: 2px 4px;
        }}
        section[data-testid="stSidebar"] .stRadio label:hover {{
            background: rgba(76, 114, 176, 0.08);
            transform: translateX(3px);
        }}
        /* Sidebar nav links */
        section[data-testid="stSidebar"] a {{
            transition: all 0.3s ease !important;
        }}
        section[data-testid="stSidebar"] a:hover {{
            transform: translateX(4px) !important;
            filter: brightness(1.1) !important;
        }}

        /* ═══════════════════════════════════════════════════════════════
           11. DATAFRAME TABLE ANIMATION
           ═══════════════════════════════════════════════════════════════ */
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {{
            animation: fadeInUp 0.5s ease both;
            transition: box-shadow 0.3s ease;
        }}
        div[data-testid="stDataFrame"]:hover,
        div[data-testid="stTable"]:hover {{
            box-shadow: 0 8px 20px -5px rgba(0,0,0,0.08);
        }}

        /* ═══════════════════════════════════════════════════════════════
           12. IMAGE/PLOT ANIMATION
           ═══════════════════════════════════════════════════════════════ */
        div[data-testid="stImage"] {{
            animation: fadeInScale 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
            transition: transform 0.4s ease, box-shadow 0.4s ease;
            border-radius: 8px;
            overflow: hidden;
        }}
        div[data-testid="stImage"]:hover {{
            transform: scale(1.01);
            box-shadow: 0 10px 30px -10px rgba(0,0,0,0.15);
        }}

        /* ═══════════════════════════════════════════════════════════════
           13. CUSTOM COMPONENTS
           ═══════════════════════════════════════════════════════════════ */
        /* Generic card */
        .app-card {{
            background: #FFFFFF;
            border: 1px solid {BORDER};
            border-radius: 12px;
            padding: 1.1rem 1.4rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: fadeInUp 0.6s ease both;
        }}
        .app-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1);
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
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .section-chip:hover {{
            transform: scale(1.05);
            box-shadow: 0 2px 8px rgba(67, 56, 202, 0.15);
        }}

        /* Prediction verdict banners — animated! */
        .verdict-yes, .verdict-no, .verdict-mixed {{
            border-radius: 14px;
            padding: 1.4rem 1rem;
            text-align: center;
            margin-bottom: 0.75rem;
            transform-origin: center;
        }}
        .verdict-yes {{
            background: linear-gradient(135deg, #ECFDF5, #D1FAE5);
            border: 1px solid #6EE7B7;
            animation: popIn 0.7s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards,
                       pulseGlow 2.5s infinite 0.8s;
        }}
        .verdict-no {{
            background: linear-gradient(135deg, #FEF2F2, #FEE2E2);
            border: 1px solid #FCA5A5;
            animation: popIn 0.7s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards,
                       shakeError 0.6s ease-in-out 0.2s;
        }}
        .verdict-mixed {{
            background: linear-gradient(135deg, #FFFBEB, #FEF3C7);
            border: 1px solid #FCD34D;
            animation: popIn 0.7s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards,
                       borderGlow 2s ease-in-out infinite;
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
            animation: popIn 0.5s ease forwards;
        }}

        /* ═══════════════════════════════════════════════════════════════
           14. SECTION HEADER ANIMATION
           ═══════════════════════════════════════════════════════════════ */
        .animated-header {{
            animation: fadeInLeft 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
            position: relative;
            padding-left: 12px;
        }}
        .animated-header::before {{
            content: '';
            position: absolute;
            left: 0;
            top: 10%;
            height: 80%;
            width: 4px;
            border-radius: 999px;
            background: linear-gradient(180deg, {PRIMARY}, {ACCENT});
            animation: fadeInUp 0.6s ease both 0.3s;
        }}

        /* ═══════════════════════════════════════════════════════════════
           15. SUCCESS / ERROR / WARNING ALERT ANIMATION
           ═══════════════════════════════════════════════════════════════ */
        div[data-testid="stAlert"] {{
            animation: fadeInLeft 0.4s ease both;
            transition: transform 0.2s ease;
        }}
        div[data-testid="stAlert"]:hover {{
            transform: translateX(3px);
        }}

        /* ═══════════════════════════════════════════════════════════════
           16. HORIZONTAL DIVIDER ANIMATION
           ═══════════════════════════════════════════════════════════════ */
        hr {{
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, {BORDER}, transparent);
            animation: fadeInUp 0.5s ease both;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# NAVIGATION BREADCRUMB
# ─────────────────────────────────────────────────────────────────────────
def navigation_breadcrumb(current_page: str) -> None:
    """Show a breadcrumb bar at the top of the page indicating current
    location within the app, with all pages listed."""
    crumbs = []
    for page_name, info in PAGE_REGISTRY.items():
        if page_name == current_page:
            crumbs.append(
                f'<span class="nav-crumb-active">{info["icon"]} {page_name}</span>'
            )
        else:
            crumbs.append(f'<span class="nav-crumb">{info["icon"]} {page_name}</span>')

    separator = ' <span class="nav-separator">›</span> '
    st.markdown(
        f'<div class="nav-breadcrumb">{separator.join(crumbs)}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# PAGE LOADING ANIMATION
# ─────────────────────────────────────────────────────────────────────────
def page_loading_animation(
    icon: str, title: str, subtitle: str, duration: float = 1.2
) -> None:
    """Show a beautiful full-width loading animation when a page first opens,
    then automatically clear it.  Uses st.empty() so the placeholder
    vanishes once loading completes."""
    placeholder = st.empty()
    placeholder.markdown(
        f"""
        <div class="page-loader">
            <span class="page-loader-icon">{icon}</span>
            <div class="page-loader-title">{title}</div>
            <div class="page-loader-sub">{subtitle}</div>
            <div class="page-loader-bar"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(duration)
    placeholder.empty()


# ─────────────────────────────────────────────────────────────────────────
# ANIMATED SECTION HEADER
# ─────────────────────────────────────────────────────────────────────────
def animated_header(text: str, level: int = 2) -> None:
    """Render a section header with a left accent bar and slide-in animation."""
    tag = f"h{level}"
    st.markdown(
        f'<div class="animated-header"><{tag} style="margin:0;">{text}</{tag}></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# VERDICT BANNER
# ─────────────────────────────────────────────────────────────────────────
def verdict_banner(
    will_purchase: bool, confidence_pct: float, threshold: float = 0.5
) -> None:
    """Big colored verdict banner shown after a prediction."""
    threshold_note = (
        f" · Decision Cutoff: {threshold * 100:.1f}%"
        if abs(threshold - 0.5) > 0.001
        else ""
    )
    if will_purchase:
        st.markdown(
            f"""
            <div class="verdict-yes">
                <p class="verdict-title">✅ Likely to Purchase</p>
                <p class="verdict-sub">Model confidence: {confidence_pct:.1f}%{threshold_note}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="verdict-no">
                <p class="verdict-title">🚫 Unlikely to Purchase</p>
                <p class="verdict-sub">Model confidence: {confidence_pct:.1f}%{threshold_note}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def multi_model_verdict_banner(purchase_count: int, total_count: int) -> None:
    """Verdict banner for Compare All Models mode.
    Shows Purchase / No Purchase / Mixed banner based on model consensus."""
    if purchase_count == total_count:
        # All models agree: Purchase
        st.markdown(
            f"""
            <div class="verdict-yes">
                <p class="verdict-title">✅ All {total_count} Models Agree: Likely to Purchase</p>
                <p class="verdict-sub">Unanimous consensus across all models</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif purchase_count == 0:
        # All models agree: No Purchase
        st.markdown(
            f"""
            <div class="verdict-no">
                <p class="verdict-title">🚫 All {total_count} Models Agree: Unlikely to Purchase</p>
                <p class="verdict-sub">Unanimous consensus across all models</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Models disagree
        no_purchase_count = total_count - purchase_count
        st.markdown(
            f"""
            <div class="verdict-mixed">
                <p class="verdict-title">⚖️ Models Disagree</p>
                <p class="verdict-sub">
                    {purchase_count} model(s) predict Purchase · 
                    {no_purchase_count} model(s) predict No Purchase
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────
# PROBABILITY METER
# ─────────────────────────────────────────────────────────────────────────
def probability_meter(proba_purchase: float, threshold: float = 0.5) -> None:
    """Horizontal gradient meter (blue=No Purchase -> orange=Purchase)
    with a marker at the predicted purchase probability and a clear threshold cutoff indicator."""
    pct = max(0.0, min(1.0, proba_purchase)) * 100
    thr_pct = max(0.0, min(1.0, threshold)) * 100
    is_custom_threshold = abs(threshold - 0.5) > 0.001

    cutoff_label = f"Cutoff: {thr_pct:.1f}%" if is_custom_threshold else "Cutoff:50.0%"
    cutoff_color = (
        "#374151"
        if not is_custom_threshold
        else ("#1E40AF" if thr_pct < 50 else "#9A3412")
    )
    badge_html = (
        '<span style="background:#EEF2FF; color:#4338CA; padding:2px 6px; border-radius:4px; font-weight:600;">Calibrated</span>'
        if is_custom_threshold
        else '<span style="background:#F3F4F6; color:#6B7280; padding:2px 6px; border-radius:4px; font-weight:500;">Standard</span>'
    )
    status_label = "&ge; Cutoff" if pct >= thr_pct else "&lt; Cutoff"

    html = (
        f'<div style="margin: 0.4rem 0 1.1rem 0;">'
        f'<div style="display:flex; justify-content:space-between; align-items:flex-end; font-size:0.72rem; color:{MUTED}; margin-bottom:6px; line-height:1.2;">'
        f'<div style="flex:1; text-align:left; white-space:normal; padding-right:2px;">0%<br><span style="font-size:0.65rem; color:{MUTED};">No Purchase</span></div>'
        f'<div style="flex:1; text-align:center; font-weight:700; color:{cutoff_color}; font-size:0.73rem; white-space:nowrap; padding:0 2px;">{cutoff_label}</div>'
        f'<div style="flex:1; text-align:right; white-space:normal; padding-left:2px;">100%<br><span style="font-size:0.65rem; color:{MUTED};">Purchase</span></div>'
        f'</div>'
        f'<div style="position:relative; height:14px; border-radius:999px; background: linear-gradient(90deg, {PRIMARY} 0%, #E5E7EB 50%, {ACCENT} 100%);">'
        f'<div style="position:absolute; left:{thr_pct}%; top:-3px; bottom:-3px; width:2px; background:#1F2937; border-left:1px dashed #FFFFFF; z-index:2;" title="Decision Cutoff: {thr_pct:.1f}%"></div>'
        f'<div style="position:absolute; left:0%; top:-7px; transform:translateX(-50%); width:0; height:0; border-left:8px solid transparent; border-right:8px solid transparent; border-top:11px solid #111827; z-index:3; --target-pct:{pct}%; animation: fillProgress 1.2s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;"></div>'
        f'</div>'
        f'<div style="display:flex; justify-content:space-between; align-items:center; margin-top:8px;">'
        f'<div style="font-size:0.72rem; color:{MUTED};">{badge_html}</div>'
        f'<div style="font-size:1.2rem; font-weight:700; animation: countUp 0.8s ease both 0.4s;">{pct:.1f}%</div>'
        f'<div style="font-size:0.72rem; color:{MUTED}; font-weight:600;">{status_label}</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────
# UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────
def confidence_label(proba_purchase: float, threshold: float = 0.5) -> str:
    """Qualitative label for how far the prediction sits from the
    decision boundary — helps a non-technical viewer read the number."""
    dist = abs(proba_purchase - threshold)
    if dist >= 0.30:
        return "Very High"
    if dist >= 0.15:
        return "High"
    if dist >= 0.05:
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

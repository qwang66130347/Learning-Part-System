"""
shared_ui.py
————————————
Modern Enterprise Analytics Design System for the DQN English Learning Dashboard.
Provides:
  - Global CSS styles (clean, white cards, soft shadows, rounded corners, blue theme)
  - Unified sidebar navigation
  - Modern page header renderer
  - Reusable KPI metric card HTML generator
  - Standardized Plotly layout / theme applicator
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# COLOR PALETTE (From prompt design system)
# ---------------------------------------------------------------------------
PRIMARY_BLUE  = "#1565C0"
DARK_BLUE     = "#0D47A1"
ACCENT_BLUE   = "#42A5F5"
BG_LIGHT      = "#F5F7FA"
CARD_BG       = "#FFFFFF"
BORDER_COLOR  = "#E5E7EB"
TEXT_PRIMARY  = "#1F2937"
TEXT_MUTED    = "#6B7280"

__all__ = [
    "PRIMARY_BLUE",
    "DARK_BLUE",
    "ACCENT_BLUE",
    "BG_LIGHT",
    "CARD_BG",
    "BORDER_COLOR",
    "TEXT_PRIMARY",
    "TEXT_MUTED",
    "ACTIVITY_COLORS",
    "AGENT_COLORS",
    "DESIGN_CSS",
    "render_sidebar",
    "render_page_header",
    "render_section_header",
    "render_kpi_card",
    "apply_plotly_theme",
]

# Activity Color Mapping
ACTIVITY_COLORS = {
    "Listening":  "#1565C0",  # Primary Blue
    "Reading":    "#42A5F5",  # Accent Blue
    "Vocabulary": "#10B981",  # Emerald Green
    "Grammar":    "#F59E0B",  # Amber
}

AGENT_COLORS = {
    "DQN":           "#1565C0",
    "Weakest-Skill": "#10B981",
    "Random":        "#F59E0B",
}

# ---------------------------------------------------------------------------
# GLOBAL CSS
# ---------------------------------------------------------------------------
DESIGN_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Base page setup */
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: {TEXT_PRIMARY};
}}

.stApp {{
    background-color: {BG_LIGHT};
}}

.main .block-container {{
    padding-top: 1.25rem;
    padding-bottom: 2.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1400px;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0D2D5E 0%, #081B3A 100%);
    color: #FFFFFF;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}}

section[data-testid="stSidebar"] * {{
    color: #FFFFFF !important;
}}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stNumberInput label {{
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #E2E8F0 !important;
}}

/* Sidebar navigation links */
[data-testid="stSidebarNav"] {{
    display: none;
}}

.sidebar-nav-item {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-radius: 8px;
    color: #E2E8F0 !important;
    text-decoration: none;
    font-size: 0.9rem;
    font-weight: 500;
    margin-bottom: 4px;
    transition: all 0.15s ease;
}}

.sidebar-nav-item:hover {{
    background: rgba(255, 255, 255, 0.1);
    color: #FFFFFF !important;
}}

.sidebar-nav-item.active {{
    background: {PRIMARY_BLUE};
    color: #FFFFFF !important;
    font-weight: 600;
    box-shadow: 0 2px 6px rgba(21, 101, 192, 0.4);
}}

/* ── Modern Page Header Banner ── */
.dashboard-header {{
    background: linear-gradient(135deg, {PRIMARY_BLUE} 0%, {DARK_BLUE} 100%);
    border-radius: 16px;
    padding: 24px 30px;
    color: #FFFFFF;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 14px rgba(13, 71, 161, 0.15);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 16px;
}}

.dashboard-header-left h1 {{
    margin: 0;
    font-size: 1.75rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    line-height: 1.2;
}}

.dashboard-header-left p {{
    margin: 6px 0 0 0;
    font-size: 0.92rem;
    color: #E0E7FF;
    font-weight: 400;
}}

.dashboard-header-badges {{
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}}

.header-pill {{
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.25);
    padding: 6px 14px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    color: #FFFFFF;
    text-transform: uppercase;
}}

/* ── Enterprise Cards ── */
.analytics-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 14px;
    padding: 22px 24px;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.04);
    margin-bottom: 1.25rem;
    transition: all 0.2s ease;
}}

.analytics-card:hover {{
    box-shadow: 0 8px 16px -4px rgba(15, 23, 42, 0.06), 0 2px 4px -2px rgba(15, 23, 42, 0.04);
}}

/* ── KPI Metrics Grid ── */
.kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 14px;
    margin-bottom: 1.5rem;
}}

.kpi-metric-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER_COLOR};
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    position: relative;
    overflow: hidden;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}

.kpi-metric-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 12px -2px rgba(0, 0, 0, 0.06);
}}

.kpi-metric-card::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: {PRIMARY_BLUE};
    border-radius: 4px 0 0 4px;
}}

.kpi-card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}}

.kpi-label {{
    font-size: 0.72rem;
    font-weight: 600;
    color: {TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

.kpi-icon {{
    font-size: 1.1rem;
    opacity: 0.75;
}}

.kpi-value {{
    font-size: 1.7rem;
    font-weight: 700;
    color: #111827;
    line-height: 1.15;
    margin-bottom: 4px;
    letter-spacing: -0.02em;
}}

.kpi-footer {{
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    color: {TEXT_MUTED};
}}

.delta-badge {{
    display: inline-flex;
    align-items: center;
    padding: 1px 6px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.7rem;
}}

.delta-positive {{
    background: #DCFCE7;
    color: #15803D;
}}

.delta-neutral {{
    background: #F1F5F9;
    color: #475569;
}}

.delta-negative {{
    background: #FEE2E2;
    color: #B91C1C;
}}

/* ── Section Titles ── */
.section-heading {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 1.5rem;
    margin-bottom: 0.85rem;
    padding-bottom: 6px;
    border-bottom: 2px solid #EEF2F6;
}}

.section-title-text {{
    font-size: 1.15rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    display: flex;
    align-items: center;
    gap: 8px;
}}

.section-title-text::before {{
    content: '';
    display: inline-block;
    width: 6px;
    height: 18px;
    background: {PRIMARY_BLUE};
    border-radius: 3px;
}}

.section-subtitle {{
    font-size: 0.82rem;
    color: {TEXT_MUTED};
    margin-top: 2px;
}}

/* ── Status Badges ── */
.badge {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}}
.badge-ok   {{ background: #DEF7EC; color: #03543F; border: 1px solid #BCF0DA; }}
.badge-warn {{ background: #FEF08A; color: #854D0E; border: 1px solid #FDE047; }}
.badge-err  {{ background: #FDE8E8; color: #9B1C1C; border: 1px solid #F8B4B4; }}
.badge-info {{ background: #E1EFFE; color: #1E429F; border: 1px solid #B4C6FC; }}

/* ── Button styling ── */
.stButton > button {{
    background-color: {PRIMARY_BLUE};
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.25rem;
    font-weight: 600;
    font-size: 0.88rem;
    transition: all 0.15s ease;
    box-shadow: 0 1px 3px rgba(21, 101, 192, 0.2);
}}

.stButton > button:hover {{
    background-color: {DARK_BLUE};
    box-shadow: 0 4px 8px rgba(13, 71, 161, 0.3);
    transform: translateY(-1px);
}}

.stButton > button:active {{
    transform: translateY(0);
}}

/* Streamlit Native Tabs Enhancement */
.stTabs [data-baseweb="tab-list"] {{
    gap: 8px;
    border-bottom: 1px solid {BORDER_COLOR};
    padding-bottom: 2px;
    margin-bottom: 1rem;
}}

.stTabs [data-baseweb="tab"] {{
    background-color: #FFFFFF;
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px 8px 0 0;
    padding: 8px 18px;
    font-weight: 600;
    font-size: 0.88rem;
    color: {TEXT_MUTED} !important;
    transition: all 0.15s;
}}

.stTabs [aria-selected="true"] {{
    background-color: {PRIMARY_BLUE} !important;
    color: #FFFFFF !important;
    border-color: {PRIMARY_BLUE} !important;
}}

/* Streamlit Metric Overrides */
div[data-testid="stMetricValue"] {{
    color: {TEXT_PRIMARY} !important;
    font-weight: 700 !important;
    font-size: 1.6rem !important;
}}
div[data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED} !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

/* Dataframe Styling */
.stDataFrame {{
    border: 1px solid {BORDER_COLOR};
    border-radius: 10px;
    overflow: hidden;
}}
</style>
"""


# ---------------------------------------------------------------------------
# SIDEBAR RENDERER
# ---------------------------------------------------------------------------
def render_sidebar():
    """Render the consistent, enterprise-grade dark-blue navigation sidebar."""
    with st.sidebar:
        # Brand Header
        st.markdown(f"""
        <div style="padding: 10px 0 20px 0; border-bottom: 1px solid rgba(255,255,255,0.12); margin-bottom: 16px;">
          <div style="display: flex; align-items: center; gap: 10px;">
            <div style="background: {PRIMARY_BLUE}; width: 38px; height: 38px; border-radius: 10px;
                        display: flex; align-items: center; justify-content: center; font-size: 1.25rem;">
              🎓
            </div>
            <div>
              <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.01em;">
                DQN Analytics
              </div>
              <div style="font-size: 0.7rem; color: #93C5FD; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">
                Personalized English RL
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='font-size:0.7rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;'>Navigation</div>", unsafe_allow_html=True)
        st.page_link("app.py",                               label="🏠  Dashboard Home")
        st.page_link("pages/1_Learner_Simulation.py",        label="👤  Learner Simulation")
        st.page_link("pages/2_Recommended_Learning_Path.py", label="🗺️  Recommended Path")
        st.page_link("pages/3_Model_Evaluation.py",          label="📊  Model Evaluation")
        st.page_link("pages/4_Training_Analytics.py",        label="📈  Training Analytics")

        st.markdown("---")
        st.markdown("<div style='font-size:0.7rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;'>Research Specs</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
                    border-radius: 10px; padding: 12px 14px; font-size: 0.78rem; line-height: 1.6;">
          <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="color:#94A3B8;">Algorithm:</span>
            <span style="font-weight:600; color:#93C5FD;">Deep Q-Network</span>
          </div>
          <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="color:#94A3B8;">State / Action:</span>
            <span style="font-weight:600; color:#FFFFFF;">11 dims / 4 acts</span>
          </div>
          <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="color:#94A3B8;">Seeds:</span>
            <span style="font-weight:600; color:#FFFFFF;">42, 43, 44, 45, 46</span>
          </div>
          <div style="display:flex; justify-content:space-between;">
            <span style="color:#94A3B8;">Checkpoints:</span>
            <span style="font-weight:600; color:#34D399;">5 / 5 Verified</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:20px; font-size:0.68rem; color:#94A3B8; text-align:center;'>Academic Research Suite<br>Simulated Learner Evaluation</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE HEADER RENDERER
# ---------------------------------------------------------------------------
def render_page_header(title: str, subtitle: str, badge_text: str = "Multi-Seed DQN", icon: str = "🎓"):
    """Render a modern banner header for pages."""
    st.markdown(f"""
    <div class="dashboard-header">
      <div class="dashboard-header-left">
        <h1>{icon} {title}</h1>
        <p>{subtitle}</p>
      </div>
      <div class="dashboard-header-badges">
        <span class="header-pill">RL ENGINE</span>
        <span class="header-pill">{badge_text}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SECTION TITLE RENDERER
# ---------------------------------------------------------------------------
def render_section_header(title: str, subtitle: str = ""):
    """Render a section header with clean indicator and optional subtitle."""
    sub_html = f"<div class='section-subtitle'>{subtitle}</div>" if subtitle else ""
    st.markdown(f"""
    <div class="section-heading">
      <div>
        <div class="section-title-text">{title}</div>
        {sub_html}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# KPI METRIC CARD GENERATOR
# ---------------------------------------------------------------------------
def render_kpi_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_type: str = "positive",  # positive, neutral, negative
    icon: str = "📈",
    footer_text: str = "",
) -> str:
    """Generate HTML for a single modern KPI metric card."""
    delta_html = ""
    if delta:
        delta_class = f"delta-{delta_type}"
        delta_html = f"<span class='delta-badge {delta_class}'>{delta}</span>"

    footer_content = f"{delta_html} <span>{footer_text}</span>" if (delta_html or footer_text) else ""

    return f"""
    <div class="kpi-metric-card">
      <div class="kpi-card-header">
        <span class="kpi-label">{label}</span>
        <span class="kpi-icon">{icon}</span>
      </div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-footer">{footer_content}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# PLOTLY THEME APPLICATOR
# ---------------------------------------------------------------------------
def apply_plotly_theme(
    fig: go.Figure,
    height: int = 340,
    title: str | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    show_legend: bool = True,
    legend_orientation: str = "h",
) -> go.Figure:
    """
    Apply a consistent modern white enterprise theme to any Plotly figure.
    """
    fig.update_layout(
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=12, color=TEXT_PRIMARY),
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        height=height,
        margin=dict(l=20, r=20, t=40 if title else 20, b=30),
        title=dict(
            text=f"<b>{title}</b>" if title else None,
            font=dict(size=14, color=TEXT_PRIMARY),
            x=0.01,
            y=0.96,
        ) if title else None,
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font_size=12,
            font_family="Inter, sans-serif",
            bordercolor=BORDER_COLOR,
        ),
        legend=dict(
            orientation=legend_orientation,
            yanchor="bottom",
            y=-0.25 if legend_orientation == "h" else 1.0,
            xanchor="center" if legend_orientation == "h" else "left",
            x=0.5 if legend_orientation == "h" else 1.02,
            font=dict(size=11, color=TEXT_MUTED),
            bgcolor="rgba(255,255,255,0.7)",
        ) if show_legend else dict(visible=False),
    )

    if x_title is not None:
        fig.update_xaxes(title_text=x_title, title_font=dict(size=11, color=TEXT_MUTED))
    if y_title is not None:
        fig.update_yaxes(title_text=y_title, title_font=dict(size=11, color=TEXT_MUTED))

    fig.update_xaxes(
        showgrid=True,
        gridcolor="#F3F4F6",
        linecolor=BORDER_COLOR,
        tickfont=dict(size=11, color=TEXT_MUTED),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#F3F4F6",
        linecolor=BORDER_COLOR,
        zerolinecolor=BORDER_COLOR,
        tickfont=dict(size=11, color=TEXT_MUTED),
    )

    return fig

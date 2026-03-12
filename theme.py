"""
utils/theme.py
──────────────
Shared Plotly theme, CSS injection, and UI helper components.
"""

import streamlit as st
import plotly.graph_objects as go

# ── Brand palette ──────────────────────────────────────────────────────────
CAREEM_GREEN  = "#1DB954"
CAREEM_TEAL   = "#00B4AB"
CAREEM_DARK   = "#0D1B2A"
CAREEM_NAVY   = "#112240"
CAREEM_GOLD   = "#F5A623"
CAREEM_RED    = "#E84545"
CAREEM_TEXT   = "#CCD6F6"
CAREEM_MUTED  = "#8892B0"

PALETTE = [
    "#1DB954", "#00B4AB", "#F5A623", "#E84545",
    "#A78BFA", "#60A5FA", "#34D399", "#FB923C",
    "#F472B6", "#94A3B8",
]


# ── CSS ────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

:root {
    --green:  #1DB954;
    --teal:   #00B4AB;
    --dark:   #0D1B2A;
    --navy:   #112240;
    --gold:   #F5A623;
    --red:    #E84545;
    --text:   #CCD6F6;
    --muted:  #8892B0;
    --border: rgba(0,180,171,0.2);
    --card:   rgba(17,34,64,0.95);
}

html, body, [class*="css"]           { font-family:'DM Sans',sans-serif; background:#0D1B2A !important; color:#CCD6F6 !important; }
.main                                { background:#0D1B2A !important; }
section[data-testid="stSidebar"]     { background:linear-gradient(180deg,#0D1B2A 0%,#112240 60%,#0D1B2A 100%) !important; border-right:1px solid var(--border); }
[data-testid="stSidebar"] *          { color:#CCD6F6 !important; }
.stRadio label                       { color:#CCD6F6 !important; font-size:0.85rem !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color:#CCD6F6 !important; }

/* KPI card */
.kpi-card { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1.1rem 1.3rem; position:relative; overflow:hidden; }
.kpi-card::before { content:''; position:absolute; top:0; left:0; width:3px; height:100%; background:linear-gradient(180deg,#1DB954,#00B4AB); border-radius:3px 0 0 3px; }
.kpi-label { font-size:0.68rem; font-weight:600; color:var(--muted); letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.3rem; }
.kpi-value { font-family:'Syne',sans-serif; font-size:1.9rem; font-weight:800; color:#fff; line-height:1; }
.kpi-delta { font-size:0.7rem; color:var(--green); margin-top:0.3rem; }

/* Section header */
.section-header { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#fff; margin-bottom:0.2rem; }
.section-sub    { font-size:0.82rem; color:var(--muted); margin-bottom:1.2rem; }
.divider        { height:1px; background:var(--border); margin:1.4rem 0; }

/* Insight / warning boxes */
.insight-box { background:rgba(29,185,84,0.07); border-left:3px solid var(--green); border-radius:0 8px 8px 0; padding:0.65rem 1rem; font-size:0.81rem; color:var(--text); margin-top:0.4rem; line-height:1.55; }
.warn-box    { background:rgba(245,166,35,0.08); border-left:3px solid var(--gold);  border-radius:0 8px 8px 0; padding:0.65rem 1rem; font-size:0.81rem; color:var(--text); margin-top:0.4rem; line-height:1.55; }
.insight-box strong { color:var(--green); }
.warn-box    strong { color:var(--gold);  }

/* Bias badges */
.badge      { display:inline-block; padding:0.2rem 0.7rem; border-radius:20px; font-size:0.7rem; font-weight:700; letter-spacing:0.06em; }
.badge-high { background:rgba(232,69,69,0.18);  color:#E84545; border:1px solid rgba(232,69,69,0.4); }
.badge-med  { background:rgba(245,166,35,0.18); color:#F5A623; border:1px solid rgba(245,166,35,0.4); }
.badge-low  { background:rgba(29,185,84,0.18);  color:#1DB954; border:1px solid rgba(29,185,84,0.4); }

/* Metric mini-card */
.mini-card      { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:0.9rem 1rem; text-align:center; }
.mini-card .val { font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:800; color:#fff; }
.mini-card .lbl { font-size:0.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; }

/* Streamlit element overrides */
.stDataFrame                              { background:var(--card) !important; }
[data-testid="stMetricValue"]             { font-family:'Syne',sans-serif !important; font-weight:800; color:#fff !important; }
[data-testid="stMetricLabel"]             { color:var(--muted) !important; font-size:0.7rem !important; }
[data-testid="stMetricDelta"]             { color:var(--green) !important; }
.stTabs [data-baseweb="tab"]              { color:var(--muted) !important; font-family:'DM Sans',sans-serif !important; }
.stTabs [aria-selected="true"]            { color:var(--green) !important; border-bottom-color:var(--green) !important; }
h1,h2,h3,h4                              { font-family:'Syne',sans-serif !important; color:#fff !important; }
p, li                                     { color:var(--text) !important; }
.stAlert                                  { background:rgba(17,34,64,0.9) !important; border-color:var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ── Plotly theme helpers ───────────────────────────────────────────────────
def apply_theme(fig, height: int = None):
    """Apply Careem dark theme to any Plotly figure."""
    updates = dict(
        paper_bgcolor="rgba(13,27,42,0)",
        plot_bgcolor="rgba(17,34,64,0.7)",
        font=dict(color=CAREEM_TEXT, family="DM Sans, sans-serif", size=12),
        legend=dict(
            bgcolor="rgba(13,27,42,0.85)",
            bordercolor="rgba(0,180,171,0.3)",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    if height:
        updates["height"] = height
    fig.update_layout(**updates)
    fig.update_xaxes(gridcolor="rgba(0,180,171,0.10)", zerolinecolor="rgba(0,180,171,0.15)", tickfont=dict(color=CAREEM_MUTED))
    fig.update_yaxes(gridcolor="rgba(0,180,171,0.10)", zerolinecolor="rgba(0,180,171,0.15)", tickfont=dict(color=CAREEM_MUTED))
    return fig


# ── UI component helpers ───────────────────────────────────────────────────
def kpi_card(label: str, value: str, delta: str = "", icon: str = ""):
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{icon}&nbsp;{label}</div>
        <div class="kpi-value">{value}</div>
        {"<div class='kpi-delta'>"+delta+"</div>" if delta else ""}
    </div>""", unsafe_allow_html=True)


def insight_box(text: str, warn: bool = False):
    cls = "warn-box" if warn else "insight-box"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def section_header(title: str, subtitle: str = ""):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def divider():
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

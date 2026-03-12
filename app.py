"""
app.py
───────
Careem UAE — Surge Pricing Fairness Analytics Dashboard
University Data Analytics Project

Run locally:
    streamlit run app.py

Deploy:
    Push app.py + all subfolders to GitHub and connect to Streamlit Cloud.
"""

import streamlit as st

# ── Page configuration (MUST be the very first Streamlit call) ─────────────
st.set_page_config(
    page_title="Careem UAE | Surge Pricing Fairness Analytics",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ────────────────────────────────────────────────────────────────
from data.generate_data import generate_dataset
from utils.theme import inject_css

import pages.home               as pg_home
import pages.dataset_overview   as pg_dataset
import pages.eda                as pg_eda
import pages.classification     as pg_classification
import pages.clustering         as pg_clustering
import pages.association_rules  as pg_association
import pages.regression         as pg_regression
import pages.bias_detection     as pg_bias

# ── Global CSS ─────────────────────────────────────────────────────────────
inject_css()

# ── Dataset (cached) ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Generating synthetic dataset…")
def load_data():
    return generate_dataset(3000, seed=42)

df = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.2rem">
        <span style="font-size:1.6rem">🚖</span>
        <span style="font-family:Syne,sans-serif;font-size:1.35rem;font-weight:800;
              background:linear-gradient(135deg,#1DB954,#00B4AB);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            Careem UAE
        </span>
    </div>
    <div style="font-size:0.68rem;color:#8892B0;letter-spacing:0.14em;
         text-transform:uppercase;margin-bottom:1.5rem;">
        Surge Pricing Fairness Analytics
    </div>
    """, unsafe_allow_html=True)

    NAV_OPTIONS = [
        "🏠  Home",
        "📊  Dataset Overview",
        "🔍  EDA & Visualizations",
        "🤖  Classification Models",
        "🔵  Clustering Analysis",
        "🔗  Association Rule Mining",
        "📈  Regression Forecast",
        "⚖️  Bias Detection",
    ]
    page = st.radio("Navigate", NAV_OPTIONS, label_visibility="collapsed")

    st.markdown('<div style="height:1px;background:rgba(0,180,171,0.2);margin:1rem 0"></div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:0.7rem;color:#8892B0;line-height:1.8">
        <div style="color:#1DB954;font-weight:600;margin-bottom:0.2rem">Project</div>
        University Data Analytics<br>
        Surge Pricing Fairness Study
        <div style="color:#1DB954;font-weight:600;margin:0.6rem 0 0.2rem">Platform</div>
        Streamlit Cloud
        <div style="color:#1DB954;font-weight:600;margin:0.6rem 0 0.2rem">Dataset</div>
        3,000 Synthetic Ride Records<br>
        Dubai, UAE — 2024
        <div style="color:#1DB954;font-weight:600;margin:0.6rem 0 0.2rem">Models</div>
        LR · Decision Tree · RF · XGBoost<br>
        K-Means · Apriori · Ridge · Lasso
    </div>
    """, unsafe_allow_html=True)

# ── Page routing ───────────────────────────────────────────────────────────
if   page == "🏠  Home":                  pg_home.render(df)
elif page == "📊  Dataset Overview":      pg_dataset.render(df)
elif page == "🔍  EDA & Visualizations":  pg_eda.render(df)
elif page == "🤖  Classification Models": pg_classification.render(df)
elif page == "🔵  Clustering Analysis":   pg_clustering.render(df)
elif page == "🔗  Association Rule Mining": pg_association.render(df)
elif page == "📈  Regression Forecast":   pg_regression.render(df)
elif page == "⚖️  Bias Detection":        pg_bias.render(df)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#8892B0;font-size:0.68rem;
     padding:0.5rem 0 1rem;margin-top:2rem;
     border-top:1px solid rgba(0,180,171,0.1);">
    Careem UAE &nbsp;·&nbsp; Surge Pricing Fairness Analytics &nbsp;·&nbsp;
    University Data Analytics Project &nbsp;·&nbsp;
    Synthetic Dataset — 3,000 Records &nbsp;·&nbsp; Dubai 2024 &nbsp;·&nbsp;
    Built with Streamlit
</div>
""", unsafe_allow_html=True)

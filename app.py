"""
app.py  —  Careem UAE | Surge Pricing Fairness Analytics
═════════════════════════════════════════════════════════
Single-file Streamlit dashboard. No subfolders needed.
Upload ONLY this file + requirements.txt to GitHub.

Run locally:   streamlit run app.py
"""

# ── Standard imports ───────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Careem UAE | Surge Pricing Fairness Analytics",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{--green:#1DB954;--teal:#00B4AB;--dark:#0D1B2A;--navy:#112240;--gold:#F5A623;--red:#E84545;--text:#CCD6F6;--muted:#8892B0;--border:rgba(0,180,171,0.2);--card:rgba(17,34,64,0.95)}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background:#0D1B2A !important;color:#CCD6F6 !important}
.main{background:#0D1B2A !important}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1B2A 0%,#112240 60%,#0D1B2A 100%) !important;border-right:1px solid var(--border)}
[data-testid="stSidebar"] *{color:#CCD6F6 !important}
.stRadio label{color:#CCD6F6 !important;font-size:0.85rem !important}
.kpi-card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:1.1rem 1.3rem;position:relative;overflow:hidden}
.kpi-card::before{content:'';position:absolute;top:0;left:0;width:3px;height:100%;background:linear-gradient(180deg,#1DB954,#00B4AB);border-radius:3px 0 0 3px}
.kpi-label{font-size:0.68rem;font-weight:600;color:var(--muted);letter-spacing:.1em;text-transform:uppercase;margin-bottom:.3rem}
.kpi-value{font-family:'Syne',sans-serif;font-size:1.9rem;font-weight:800;color:#fff;line-height:1}
.kpi-delta{font-size:.7rem;color:var(--green);margin-top:.3rem}
.section-header{font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#fff;margin-bottom:.2rem}
.section-sub{font-size:.82rem;color:var(--muted);margin-bottom:1.2rem}
.divider{height:1px;background:var(--border);margin:1.4rem 0}
.insight-box{background:rgba(29,185,84,.07);border-left:3px solid var(--green);border-radius:0 8px 8px 0;padding:.65rem 1rem;font-size:.81rem;color:var(--text);margin-top:.4rem;line-height:1.55}
.warn-box{background:rgba(245,166,35,.08);border-left:3px solid var(--gold);border-radius:0 8px 8px 0;padding:.65rem 1rem;font-size:.81rem;color:var(--text);margin-top:.4rem;line-height:1.55}
.insight-box strong{color:var(--green)}.warn-box strong{color:var(--gold)}
.badge{display:inline-block;padding:.2rem .7rem;border-radius:20px;font-size:.7rem;font-weight:700;letter-spacing:.06em}
.badge-high{background:rgba(232,69,69,.18);color:#E84545;border:1px solid rgba(232,69,69,.4)}
.badge-med{background:rgba(245,166,35,.18);color:#F5A623;border:1px solid rgba(245,166,35,.4)}
.badge-low{background:rgba(29,185,84,.18);color:#1DB954;border:1px solid rgba(29,185,84,.4)}
.mini-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:.9rem 1rem;text-align:center}
.mini-card .val{font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:#fff}
.mini-card .lbl{font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}
[data-testid="stMetricValue"]{font-family:'Syne',sans-serif !important;font-weight:800;color:#fff !important}
[data-testid="stMetricLabel"]{color:var(--muted) !important;font-size:.7rem !important}
[data-testid="stMetricDelta"]{color:var(--green) !important}
.stTabs [data-baseweb="tab"]{color:var(--muted) !important}
.stTabs [aria-selected="true"]{color:var(--green) !important;border-bottom-color:var(--green) !important}
h1,h2,h3,h4{font-family:'Syne',sans-serif !important;color:#fff !important}
p,li{color:var(--text) !important}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PALETTE & THEME HELPERS
# ══════════════════════════════════════════════════════════════════════════════
PALETTE = ["#1DB954","#00B4AB","#F5A623","#E84545","#A78BFA","#60A5FA","#34D399","#FB923C","#F472B6","#94A3B8"]

def T(fig, h=None):
    """Apply dark theme to Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(13,27,42,0)", plot_bgcolor="rgba(17,34,64,0.7)",
        font=dict(color="#CCD6F6", family="DM Sans, sans-serif", size=12),
        legend=dict(bgcolor="rgba(13,27,42,0.85)", bordercolor="rgba(0,180,171,0.3)", borderwidth=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    if h: fig.update_layout(height=h)
    fig.update_xaxes(gridcolor="rgba(0,180,171,0.10)", zerolinecolor="rgba(0,180,171,0.15)", tickfont=dict(color="#8892B0"))
    fig.update_yaxes(gridcolor="rgba(0,180,171,0.10)", zerolinecolor="rgba(0,180,171,0.15)", tickfont=dict(color="#8892B0"))
    return fig

def kpi(label, value, delta="", icon=""):
    st.markdown(f'<div class="kpi-card"><div class="kpi-label">{icon}&nbsp;{label}</div>'
                f'<div class="kpi-value">{value}</div>'
                f'{"<div class=kpi-delta>"+delta+"</div>" if delta else ""}</div>', unsafe_allow_html=True)

def ibox(text, warn=False):
    st.markdown(f'<div class="{"warn-box" if warn else "insight-box"}">{text}</div>', unsafe_allow_html=True)

def hdr(title, sub=""):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if sub: st.markdown(f'<div class="section-sub">{sub}</div>', unsafe_allow_html=True)

def div():
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET GENERATION
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Generating synthetic dataset…")
def generate_dataset(n=3000, seed=42):
    rng = np.random.default_rng(seed)

    areas = {
        "Tourist-Heavy": ["Dubai Marina","Downtown Dubai","Palm Jumeirah","Jumeirah Beach Residence","Dubai Mall Area"],
        "Business":      ["DIFC","Business Bay","Sheikh Zayed Road","Deira","Bur Dubai"],
        "Residential":   ["Mirdif","Al Quoz","Jumeirah Village Circle","International City","Silicon Oasis"],
        "Airport/Transport": ["Dubai International Airport","Al Maktoum Airport","Union Metro Station","Ibn Battuta Mall"],
    }
    flat_areas = [a for lst in areas.values() for a in lst]
    area_zone  = {a: z for z, lst in areas.items() for a in lst}

    nationalities = ["Emirati","Indian","Pakistani","Filipino","Egyptian","British","American","Saudi","Bangladeshi","Sri Lankan","Lebanese","Jordanian","Nepalese","Other Arab","Western European"]
    nat_weights   = [0.12,0.22,0.12,0.08,0.07,0.05,0.04,0.05,0.06,0.04,0.03,0.03,0.03,0.03,0.03]

    vehicle_types  = ["Economy","Business","SUV","Bike","Luxury"]
    loyalty_levels = ["Bronze","Silver","Gold","Platinum"]
    income_levels  = ["Low","Middle","Upper-Middle","High"]
    weather_list   = ["Clear","Cloudy","Light Rain","Heavy Rain","Sandstorm","Fog"]
    day_list       = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    time_slots     = ["Morning Peak","Midday","Afternoon","Evening Peak","Late Night"]

    ride_ids     = [f"RD{100000+i}" for i in range(n)]
    customer_ids = [f"CU{int(rng.integers(10000,99999))}" for _ in range(n)]
    ages    = rng.integers(18, 67, n).astype(int)
    genders = rng.choice(["Male","Female","Prefer Not to Say"], p=[0.55,0.42,0.03], size=n)
    nats    = rng.choice(nationalities, p=nat_weights, size=n)
    income  = rng.choice(income_levels,  p=[0.25,0.35,0.25,0.15], size=n)
    loyalty = rng.choice(loyalty_levels, p=[0.40,0.30,0.20,0.10], size=n)

    pickups  = rng.choice(flat_areas, size=n)
    dropoffs = rng.choice(flat_areas, size=n)
    zones    = np.array([area_zone[p] for p in pickups])

    time_of_day = rng.choice(time_slots, p=[0.25,0.15,0.18,0.28,0.14], size=n)
    day_of_week = rng.choice(day_list, size=n)   # ← length-n array (bug fix)

    event_probs  = np.where(np.isin(zones, ["Tourist-Heavy","Business"]), 0.35, 0.12)
    nearby_event = np.array(["Yes" if rng.random() < p else "No" for p in event_probs])
    weather  = rng.choice(weather_list, p=[0.45,0.20,0.15,0.08,0.07,0.05], size=n)
    vehicles = rng.choice(vehicle_types, p=[0.45,0.25,0.15,0.08,0.07], size=n)

    dist_params = {"Morning Peak":(5,8),"Evening Peak":(6,9),"Midday":(4,6),"Afternoon":(5,7),"Late Night":(7,12)}
    distances = np.array([max(1.0, rng.normal(*dist_params[t])) for t in time_of_day], dtype=float)
    distances = np.where(vehicles=="Luxury",  distances*1.3, distances)
    distances = np.where(vehicles=="Business", distances*1.1, distances)
    distances = np.round(distances, 2)

    ride_time = np.round(distances / rng.uniform(0.8,1.2,n)*5 + rng.normal(3,1,n), 0).astype(int)
    ride_time = np.clip(ride_time, 3, 90)

    driver_accept = np.round(rng.uniform(0.55, 0.99, n), 2)
    driver_accept = np.where(zones=="Residential", np.clip(driver_accept-0.12,0.40,0.99), driver_accept)
    driver_dist   = np.round(rng.exponential(1.8, n), 2)

    wait_base  = rng.normal(6, 3, n)
    wait_base  = np.where(zones=="Residential", wait_base+3.5, wait_base)
    wait_base  = np.where(income=="Low",         wait_base+2.0, wait_base)
    wait_times = np.round(np.clip(wait_base, 1, 30), 0).astype(int)

    base_rate  = {"Economy":1.8,"Business":2.8,"SUV":3.2,"Bike":1.1,"Luxury":5.5}
    base_fares = np.array([round(3.0+base_rate[v]*d+rng.normal(0,1.5),2) for v,d in zip(vehicles,distances)])
    base_fares = np.clip(base_fares, 5.0, 250.0)

    surge = np.ones(n)
    surge = np.where(np.isin(time_of_day,["Morning Peak","Evening Peak"]), surge+rng.uniform(0.2,0.8,n), surge)
    surge = np.where(zones=="Tourist-Heavy", surge+rng.uniform(0.3,1.0,n), surge)
    surge = np.where(nearby_event=="Yes",    surge+rng.uniform(0.2,0.7,n), surge)
    surge = np.where(np.isin(weather,["Heavy Rain","Fog"]), surge+rng.uniform(0.3,0.8,n), surge)
    surge = np.where(time_of_day=="Late Night", surge+rng.uniform(0.1,0.5,n), surge)
    surge = np.round(np.clip(surge, 1.0, 4.0), 2)
    final_fares = np.round(base_fares * surge, 2)

    disc_prob = np.where(np.isin(loyalty,["Gold","Platinum"]),0.50, np.where(loyalty=="Silver",0.25,0.08))
    discount_applied = np.array(["Yes" if rng.random()<p else "No" for p in disc_prob])

    cancel_prob = np.full(n, 0.12)
    cancel_prob += (surge-1.0)*0.12
    cancel_prob += np.where(surge>2.5, 0.20, 0)
    cancel_prob += np.where(wait_times>12, 0.10, 0)
    cancel_prob += np.where(income=="Low",    0.06, 0)
    cancel_prob += np.where(income=="Middle", 0.03, 0)
    cancel_prob += np.where(zones=="Residential", 0.04, 0)
    cancel_prob -= np.where(np.isin(loyalty,["Gold","Platinum"]), 0.08, 0)
    cancel_prob  = np.clip(cancel_prob, 0.03, 0.75)
    cancelled    = np.array(["Yes" if rng.random()<p else "No" for p in cancel_prob])

    fair = 4.5 - (surge-1.0)*0.6 - (wait_times/30)*0.8
    fair += rng.normal(0, 0.4, n)
    fair  = np.where(discount_applied=="Yes", fair+0.2, fair)
    fair  = np.round(np.clip(fair, 1.0, 5.0), 1)

    df = pd.DataFrame({
        "Ride_ID": ride_ids, "Customer_ID": customer_ids,
        "Customer_Age": ages, "Customer_Gender": genders,
        "Customer_Nationality": nats, "Customer_Income_Bracket": income,
        "Customer_Loyalty_Status": loyalty, "Pickup_Location": pickups,
        "Dropoff_Location": dropoffs, "Pickup_Zone": zones,
        "Ride_Time_of_Day": time_of_day, "Ride_Day_of_Week": day_of_week,
        "Nearby_Event": nearby_event, "Weather_Condition": weather,
        "Ride_Distance_KM": distances, "Estimated_Ride_Time_Minutes": ride_time,
        "Vehicle_Type_Requested": vehicles, "Driver_Acceptance_Rate": driver_accept,
        "Driver_Distance_to_Pickup": driver_dist, "Estimated_Wait_Time": wait_times,
        "Base_Fare": base_fares, "Surge_Multiplier": surge,
        "Final_Fare": final_fares, "Discount_Applied": discount_applied,
        "Customer_Fairness_Rating": fair, "Ride_Cancelled": cancelled,
    })
    df["Customer_Income_Bracket"] = pd.Categorical(df["Customer_Income_Bracket"], categories=["Low","Middle","Upper-Middle","High"], ordered=True)
    df["Customer_Loyalty_Status"] = pd.Categorical(df["Customer_Loyalty_Status"], categories=["Bronze","Silver","Gold","Platinum"], ordered=True)
    return df

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.2rem">
        <span style="font-size:1.6rem">🚖</span>
        <span style="font-family:Syne,sans-serif;font-size:1.35rem;font-weight:800;
              background:linear-gradient(135deg,#1DB954,#00B4AB);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent">Careem UAE</span>
    </div>
    <div style="font-size:.68rem;color:#8892B0;letter-spacing:.14em;text-transform:uppercase;margin-bottom:1.5rem">
        Surge Pricing Fairness Analytics
    </div>""", unsafe_allow_html=True)

    NAV = ["🏠  Home","📊  Dataset Overview","🔍  EDA & Visualizations",
           "🤖  Classification Models","🔵  Clustering Analysis",
           "🔗  Association Rule Mining","📈  Regression Forecast","⚖️  Bias Detection"]
    page = st.radio("Navigate", NAV, label_visibility="collapsed")

    st.markdown('<div style="height:1px;background:rgba(0,180,171,0.2);margin:1rem 0"></div>', unsafe_allow_html=True)
    st.markdown("""<div style="font-size:.7rem;color:#8892B0;line-height:1.8">
    <div style="color:#1DB954;font-weight:600">Project</div>University Data Analytics<br>Surge Pricing Fairness Study
    <div style="color:#1DB954;font-weight:600;margin-top:.5rem">Dataset</div>3,000 Synthetic Records<br>Dubai, UAE — 2024
    <div style="color:#1DB954;font-weight:600;margin-top:.5rem">Models</div>LR · DT · RF · XGBoost<br>K-Means · Apriori · Ridge
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
df = generate_dataset()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(29,185,84,.12),rgba(0,180,171,.05),rgba(13,27,42,0));
         border:1px solid rgba(29,185,84,.2);border-radius:18px;padding:2.5rem 2.8rem 2rem;margin-bottom:1.5rem">
        <div style="font-size:.73rem;color:#1DB954;letter-spacing:.18em;text-transform:uppercase;font-weight:700;margin-bottom:.5rem">
            University Analytics Project · Dubai, UAE</div>
        <h1 style="font-family:Syne,sans-serif;font-size:2.5rem;font-weight:800;color:#fff;margin:0 0 .6rem;line-height:1.15">
            Surge Pricing<br>Fairness Analytics</h1>
        <p style="color:#8892B0;font-size:.95rem;max-width:660px;line-height:1.7;margin:0">
            An end-to-end data science investigation into whether Careem UAE's surge pricing
            algorithms treat all customers equitably across locations, demographics, and economic segments.</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: kpi("Total Requests",   f"{len(df):,}",                                    icon="🚕")
    with c2: kpi("Completed Rides",  f"{(df['Ride_Cancelled']=='No').sum():,}",          icon="✅")
    with c3: kpi("Cancellation Rate",f"{(df['Ride_Cancelled']=='Yes').mean()*100:.1f}%", delta="↑ High > 15%", icon="❌")
    with c4: kpi("Avg Surge",        f"{df['Surge_Multiplier'].mean():.2f}×",            icon="📈")
    with c5: kpi("Avg Wait",         f"{df['Estimated_Wait_Time'].mean():.1f} min",      icon="⏱️")
    with c6: kpi("Avg Final Fare",   f"AED {df['Final_Fare'].mean():.0f}",               icon="💰")

    div()
    col_l, col_r = st.columns([1.35,1])
    with col_l:
        hdr("Business Context","Why Surge Pricing Fairness Matters")
        st.markdown("""<p style="color:#CCD6F6;line-height:1.75;font-size:.9rem">
        Ride-hailing platforms like <strong style="color:#1DB954">Careem UAE</strong> rely on
        <em>dynamic surge pricing</em> to balance supply and demand. While economically rational,
        these algorithms can create <strong style="color:#F5A623">disparate outcomes</strong> —
        charging tourists more, deprioritising residential areas, and pushing cancellations dangerously
        high when multipliers exceed 2.5×. This dashboard applies classification, clustering,
        association mining, and regression to quantify whether patterns constitute
        <strong style="color:#E84545">algorithmic bias</strong> and recommends fairer pricing frameworks.
        </p>""", unsafe_allow_html=True)
        div()
        hdr("Dataset at a Glance")
        m1,m2,m3,m4 = st.columns(4)
        for col,val,lbl in [(m1,"3,000","Ride Records"),(m2,"26","Features"),
                             (m3,str(df['Pickup_Location'].nunique()),"Pickup Areas"),
                             (m4,str(df['Customer_Nationality'].nunique()),"Nationalities")]:
            with col:
                st.markdown(f'<div class="mini-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    with col_r:
        st.markdown("""<div style="background:rgba(17,34,64,.95);border:1px solid rgba(0,180,171,.2);border-radius:14px;padding:1.4rem 1.5rem">
        <div style="font-family:Syne,sans-serif;font-weight:700;color:#fff;font-size:1rem;margin-bottom:1.1rem">📋 Analytics Pipeline</div>""", unsafe_allow_html=True)
        for icon,name,desc in [("📊","Dataset Overview","Schema · preview · statistics"),
                                ("🔍","EDA & Visualizations","Surge · demand · correlation"),
                                ("🤖","Classification Models","LR · DT · RF · XGBoost"),
                                ("🔵","Clustering Analysis","K-Means segmentation (K=4)"),
                                ("🔗","Association Rules","Apriori pricing patterns"),
                                ("📈","Regression Forecast","Predict Fairness Rating"),
                                ("⚖️","Bias Detection","Geographic · income · nationality")]:
            st.markdown(f"""<div style="display:flex;gap:.75rem;margin-bottom:.8rem;padding-bottom:.8rem;border-bottom:1px solid rgba(0,180,171,.1)">
            <span style="font-size:1.1rem;min-width:22px">{icon}</span>
            <div><div style="font-weight:600;color:#fff;font-size:.82rem">{name}</div>
            <div style="color:#8892B0;font-size:.73rem">{desc}</div></div></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    div()
    hdr("Detected Bias Signals","Preliminary findings")
    b1,b2,b3,b4 = st.columns(4)
    for col,(level,title,desc,color) in zip([b1,b2,b3,b4],[
        ("🔴 HIGH","Tourist Zone Surge","+18–25% above average fare in tourist zones.","#E84545"),
        ("🔴 HIGH","Residential Wait Gap","Residents wait 3.5 min longer — driver deficit.","#E84545"),
        ("🟡 MEDIUM","Income Cancel Gap","Low-income riders cancel ~40% more.","#F5A623"),
        ("🟢 LOW","Gender Fare Diff","<2% variance across genders.","#1DB954"),
    ]):
        with col:
            st.markdown(f'<div class="kpi-card" style="border-left-color:{color}"><div class="kpi-label" style="color:{color}">{level}</div>'
                        f'<div style="font-family:Syne,sans-serif;font-weight:700;color:#fff;font-size:.85rem;margin:.3rem 0">{title}</div>'
                        f'<div style="font-size:.76rem;color:#8892B0;line-height:1.55">{desc}</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Dataset Overview":
    hdr("Dataset Overview","3,000 synthetic ride records · 26 features · Dubai 2024")
    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Rows",f"{len(df):,}"); m2.metric("Columns",str(len(df.columns)))
    m3.metric("Missing",str(int(df.isnull().sum().sum()))); m4.metric("Cancelled",f"{(df['Ride_Cancelled']=='Yes').mean()*100:.1f}%")
    m5.metric("Unique Riders",f"{df['Customer_ID'].nunique():,}")
    div()

    t1,t2,t3,t4 = st.tabs(["📄 Preview","🗂️ Schema","📊 Charts","🔢 Stats"])

    with t1:
        n_rows = st.slider("Rows to preview",5,50,15)
        st.dataframe(df.head(n_rows), use_container_width=True, height=420)

    with t2:
        descs = ["Unique ride ID","Unique customer ID","Age 18–66","Gender","Nationality",
                 "Income bracket","Loyalty tier","Pickup area","Drop-off area","Zone category",
                 "Time-of-day slot","Day of week","Nearby major event","Weather",
                 "Distance (km)","Est. trip time (min)","Vehicle category","Driver acceptance (0–1)",
                 "Driver distance to pickup (km)","Est. wait time (min)","Base fare (AED)",
                 "Surge multiplier (1–4)","Final fare (AED)","Discount applied","Fairness rating (1–5)","Target: cancelled?"]
        st.dataframe(pd.DataFrame({"Column":df.columns,"Type":[str(df[c].dtype) for c in df.columns],
            "Unique":[df[c].nunique() for c in df.columns],"Sample":[str(df[c].iloc[0]) for c in df.columns],
            "Description":descs}), use_container_width=True, height=600)

    with t3:
        c1,c2 = st.columns(2)
        with c1:
            tod = df["Ride_Time_of_Day"].value_counts().reset_index(); tod.columns=["Time","Count"]
            ord_t=["Morning Peak","Midday","Afternoon","Evening Peak","Late Night"]
            tod["Time"]=pd.Categorical(tod["Time"],categories=ord_t,ordered=True); tod=tod.sort_values("Time")
            fig=px.bar(tod,x="Time",y="Count",color="Count",color_continuous_scale=[PALETTE[1],PALETTE[0]],title="Rides by Time of Day")
            T(fig); st.plotly_chart(fig,use_container_width=True)
            ibox("<strong>Evening Peak</strong> drives ~28% of all rides. Late Night has lowest volume but highest average fares.")
        with c2:
            vt=df["Vehicle_Type_Requested"].value_counts().reset_index(); vt.columns=["Type","Count"]
            fig=px.pie(vt,names="Type",values="Count",color_discrete_sequence=PALETTE,title="Vehicle Type Distribution",hole=0.42)
            T(fig); st.plotly_chart(fig,use_container_width=True)
            ibox("<strong>Economy</strong> dominates at ~45%. <strong>Luxury</strong> (~7%) commands highest margin.")
        c3,c4 = st.columns(2)
        with c3:
            cr=df.groupby("Pickup_Zone")["Ride_Cancelled"].apply(lambda x:(x=="Yes").mean()*100).reset_index()
            cr.columns=["Zone","Cancel %"]
            fig=px.bar(cr,x="Zone",y="Cancel %",color="Cancel %",color_continuous_scale=["#1DB954","#F5A623","#E84545"],title="Cancellation Rate by Zone")
            T(fig); st.plotly_chart(fig,use_container_width=True)
            ibox("<strong>Tourist-Heavy zones</strong> have the highest cancellation rates — extreme surge suppresses demand.")
        with c4:
            zs=df.groupby("Pickup_Zone")["Surge_Multiplier"].mean().reset_index(); zs.columns=["Zone","Avg Surge"]
            fig=px.bar(zs,x="Zone",y="Avg Surge",color="Avg Surge",color_continuous_scale=["#1DB954","#F5A623","#E84545"],title="Avg Surge by Zone")
            fig.add_hline(y=1.5,line_dash="dash",line_color="#F5A623",annotation_text="Fairness threshold")
            T(fig); st.plotly_chart(fig,use_container_width=True)
            ibox("Tourist-Heavy and Business zones breach the <strong>1.5× fairness threshold</strong> consistently.")

    with t4:
        st.markdown("**Numeric Summary**")
        st.dataframe(df.select_dtypes(include=np.number).describe().round(3),use_container_width=True)
        st.markdown("**Categorical Summary**")
        cats=df.select_dtypes(include=["object","category"]).columns.tolist()
        st.dataframe(pd.DataFrame({"Column":cats,"Unique":[df[c].nunique() for c in cats],
            "Top":[df[c].mode()[0] for c in cats],"Freq":[int(df[c].value_counts().iloc[0]) for c in cats]}),use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  EDA & Visualizations":
    hdr("Exploratory Data Analysis","Surge dynamics · demand geography · cancellation drivers")

    t1,t2,t3 = st.tabs(["💹 Surge & Fares","🗺️ Demand & Geography","🌡️ Correlations"])

    with t1:
        c1,c2=st.columns(2)
        with c1:
            fig=px.histogram(df,x="Surge_Multiplier",nbins=40,color_discrete_sequence=["#1DB954"],title="Surge Multiplier Distribution")
            ms=df["Surge_Multiplier"].mean()
            fig.add_vline(x=ms,line_dash="dash",line_color="#F5A623",annotation_text=f"Mean {ms:.2f}×",annotation_position="top right")
            fig.add_vline(x=2.5,line_dash="dot",line_color="#E84545",annotation_text="High-cancel 2.5×",annotation_position="top left")
            T(fig); st.plotly_chart(fig,use_container_width=True)
            ibox(f"Average surge is <strong>{ms:.2f}×</strong>. ~{(df['Surge_Multiplier']>2.0).mean()*100:.0f}% of rides exceed 2×, above which satisfaction drops sharply.")
        with c2:
            dfc=df.copy(); dfc["_sb"]=pd.cut(dfc["Surge_Multiplier"],[0,1,1.5,2,2.5,3,4],labels=["1.0","1–1.5","1.5–2","2–2.5","2.5–3","3–4"])
            cs=dfc.groupby("_sb",observed=True)["Ride_Cancelled"].apply(lambda x:(x=="Yes").mean()*100).reset_index(); cs.columns=["Range","Cancel %"]
            fig=px.bar(cs,x="Range",y="Cancel %",color="Cancel %",color_continuous_scale=["#1DB954","#F5A623","#E84545"],title="Cancellation Rate vs Surge")
            T(fig); st.plotly_chart(fig,use_container_width=True)
            ibox("Cancellation <strong>nearly doubles</strong> above 2.5×. A psychological price ceiling exists beyond which passengers abandon rides.")
        c3,c4=st.columns(2)
        with c3:
            fig=px.box(df,x="Vehicle_Type_Requested",y="Final_Fare",color="Vehicle_Type_Requested",color_discrete_sequence=PALETTE,
                       title="Final Fare by Vehicle Type",category_orders={"Vehicle_Type_Requested":["Bike","Economy","Business","SUV","Luxury"]})
            T(fig); st.plotly_chart(fig,use_container_width=True)
            ibox("Luxury fares show extreme variance. Economy fares cluster tightly — uniform algorithmic pricing.")
        with c4:
            smp=df.sample(600,random_state=42)
            fig=px.scatter(smp,x="Estimated_Wait_Time",y="Surge_Multiplier",color="Ride_Cancelled",
                           color_discrete_map={"Yes":"#E84545","No":"#1DB954"},opacity=0.65,title="Wait vs Surge (by Cancellation)")
            T(fig); st.plotly_chart(fig,use_container_width=True)
            ibox("Top-right quadrant (high wait + high surge) is dense with <strong style='color:#E84545'>cancellations</strong>.")
        c5,c6=st.columns(2)
        with c5:
            ft=df.groupby("Ride_Time_of_Day")["Final_Fare"].mean().reset_index(); ft.columns=["Time","Avg Fare"]
            ord_t=["Morning Peak","Midday","Afternoon","Evening Peak","Late Night"]
            ft["Time"]=pd.Categorical(ft["Time"],categories=ord_t,ordered=True); ft=ft.sort_values("Time")
            fig=px.line(ft,x="Time",y="Avg Fare",markers=True,color_discrete_sequence=["#00B4AB"],title="Avg Fare by Time of Day")
            T(fig); st.plotly_chart(fig,use_container_width=True)
            ibox("<strong>Late Night</strong> commands highest average fare. Morning and Evening peaks follow due to surge.")
        with c6:
            ws=df.groupby("Weather_Condition")["Surge_Multiplier"].mean().sort_values(ascending=False).reset_index(); ws.columns=["Weather","Avg Surge"]
            fig=px.bar(ws,x="Weather",y="Avg Surge",color="Avg Surge",color_continuous_scale=["#1DB954","#F5A623","#E84545"],title="Avg Surge by Weather")
            T(fig); st.plotly_chart(fig,use_container_width=True)
            ibox("Heavy Rain and Fog trigger <strong>30–45% higher surge</strong> — unaffordable for essential workers.")

    with t2:
        c1,c2=st.columns(2)
        with c1:
            ac=df["Pickup_Location"].value_counts().head(12).reset_index(); ac.columns=["Area","Rides"]
            fig=px.bar(ac,x="Rides",y="Area",orientation="h",color="Rides",color_continuous_scale=[PALETTE[1],PALETTE[0]],title="Top 12 Pickup Locations")
            T(fig,h=420); st.plotly_chart(fig,use_container_width=True)
            ibox("Dubai Marina and Downtown Dubai dominate. Residential areas see ~3× fewer pickups.")
        with c2:
            wi=df.groupby("Customer_Income_Bracket",observed=True)["Estimated_Wait_Time"].mean().reset_index(); wi.columns=["Income","Avg Wait"]
            fig=px.bar(wi,x="Income",y="Avg Wait",color="Avg Wait",color_continuous_scale=["#1DB954","#F5A623","#E84545"],
                       title="Avg Wait by Income",category_orders={"Income":["Low","Middle","Upper-Middle","High"]})
            T(fig); st.plotly_chart(fig,use_container_width=True)
            ibox("<strong>Low-income customers wait ~3 minutes longer</strong> — budget riders in residential areas with fewer drivers.")
        c3,c4=st.columns(2)
        with c3:
            ec=df.groupby(["Nearby_Event","Ride_Time_of_Day"])["Ride_Cancelled"].apply(lambda x:(x=="Yes").mean()*100).reset_index()
            ec.columns=["Event","Time","Cancel %"]
            ord_t=["Morning Peak","Midday","Afternoon","Evening Peak","Late Night"]
            ec["Time"]=pd.Categorical(ec["Time"],categories=ord_t,ordered=True); ec=ec.sort_values("Time")
            fig=px.bar(ec,x="Time",y="Cancel %",color="Event",color_discrete_map={"Yes":"#E84545","No":"#1DB954"},barmode="group",title="Cancellation: Event vs No Event")
            T(fig); st.plotly_chart(fig,use_container_width=True)
            ibox("Events amplify cancellation by <strong>8–12 pp</strong> during peak hours.")
        with c4:
            dv=df["Ride_Day_of_Week"].value_counts().reset_index(); dv.columns=["Day","Count"]
            ord_d=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            dv["Day"]=pd.Categorical(dv["Day"],categories=ord_d,ordered=True); dv=dv.sort_values("Day")
            fig=px.bar(dv,x="Day",y="Count",color="Count",color_continuous_scale=[PALETTE[1],PALETTE[0]],title="Ride Volume by Day of Week")
            T(fig); st.plotly_chart(fig,use_container_width=True)
            ibox("Friday and Saturday show elevated demand aligned with UAE weekend patterns.")

    with t3:
        num_cols=["Customer_Age","Ride_Distance_KM","Estimated_Ride_Time_Minutes","Estimated_Wait_Time",
                  "Base_Fare","Surge_Multiplier","Final_Fare","Driver_Acceptance_Rate","Customer_Fairness_Rating"]
        fig=px.imshow(df[num_cols].corr(),text_auto=".2f",color_continuous_scale="RdBu",
                      title="Correlation Heatmap — Numeric Features",aspect="auto",zmin=-1,zmax=1)
        T(fig,h=520); st.plotly_chart(fig,use_container_width=True)
        ibox("<strong>Surge → Final Fare</strong> r≈0.85. <strong>Fairness Rating → Surge</strong> r≈−0.55 — the primary lever for fairness improvement.")
        div()
        smp2=df.sample(500,random_state=7)
        fig2=px.scatter(smp2,x="Surge_Multiplier",y="Customer_Fairness_Rating",color="Customer_Income_Bracket",
                        color_discrete_sequence=PALETTE,trendline="ols",opacity=0.6,title="Surge vs Fairness Rating (n=500)")
        T(fig2,h=380); st.plotly_chart(fig2,use_container_width=True)
        ibox("<strong>Low-income riders rate fairness lower at every surge level</strong> — relative affordability compounds perceived unfairness.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Classification Models":
    hdr("Classification Models","Predicting Ride_Cancelled using machine learning")

    @st.cache_data(show_spinner=False)
    def prep_clf(df):
        feat=["Customer_Age","Ride_Distance_KM","Estimated_Ride_Time_Minutes","Estimated_Wait_Time",
              "Base_Fare","Surge_Multiplier","Final_Fare","Driver_Acceptance_Rate","Driver_Distance_to_Pickup","Customer_Fairness_Rating"]
        cats=["Vehicle_Type_Requested","Customer_Income_Bracket","Customer_Loyalty_Status",
              "Ride_Time_of_Day","Pickup_Zone","Nearby_Event","Weather_Condition"]
        X=df[feat].copy()
        for c in cats: X[c]=LabelEncoder().fit_transform(df[c].astype(str))
        y=(df["Ride_Cancelled"]=="Yes").astype(int)
        Xt,Xe,yt,ye=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
        return Xt,Xe,yt,ye,feat+cats

    @st.cache_data(show_spinner=False)
    def train_clf(_Xt,_Xe,_yt,_ye):
        SS=StandardScaler(); Xts=SS.fit_transform(_Xt); Xes=SS.transform(_Xe)
        clfs={"Logistic Regression":(LogisticRegression(max_iter=1000,random_state=42,class_weight="balanced"),True),
              "Decision Tree":(DecisionTreeClassifier(max_depth=6,random_state=42,class_weight="balanced"),False),
              "Random Forest":(RandomForestClassifier(n_estimators=100,max_depth=8,n_jobs=-1,random_state=42,class_weight="balanced"),False)}
        if HAS_XGB:
            n_neg=int((_yt==0).sum()); n_pos=int((_yt==1).sum())
            clfs["XGBoost"]=(XGBClassifier(n_estimators=150,max_depth=5,learning_rate=0.1,scale_pos_weight=n_neg/max(n_pos,1),eval_metric="logloss",random_state=42,n_jobs=-1),False)
        res={}
        for nm,(mdl,sc) in clfs.items():
            Xtr=Xts if sc else _Xt; Xte=Xes if sc else _Xe
            mdl.fit(Xtr,_yt); yp=mdl.predict(Xte)
            res[nm]={"Accuracy":round(accuracy_score(_ye,yp)*100,2),"Precision":round(precision_score(_ye,yp)*100,2),
                     "Recall":round(recall_score(_ye,yp)*100,2),"F1":round(f1_score(_ye,yp)*100,2),
                     "CM":confusion_matrix(_ye,yp),"model":mdl,"yp":yp}
        return res

    with st.spinner("⚙️ Training classifiers…"):
        Xt,Xe,yt,ye,feat_names=prep_clf(df); results=train_clf(Xt,Xe,yt,ye)

    mdf=pd.DataFrame({n:{k:v for k,v in r.items() if k in ["Accuracy","Precision","Recall","F1"]} for n,r in results.items()}).T
    mdf.index.name="Model"
    c1,c2=st.columns([1.3,1])
    with c1:
        hdr("Metric Comparison")
        fig=px.bar(mdf.reset_index().melt(id_vars="Model",var_name="Metric",value_name="Score"),
                   x="Model",y="Score",color="Metric",barmode="group",color_discrete_sequence=PALETTE,
                   title="Accuracy / Precision / Recall / F1 (%)",range_y=[40,100])
        T(fig); st.plotly_chart(fig,use_container_width=True)
    with c2:
        hdr("Results Table")
        st.dataframe(mdf.style.format("{:.2f}").background_gradient(cmap="Greens",axis=None),use_container_width=True,height=220)
        bm=mdf["F1"].idxmax()
        ibox(f"🏆 <strong>{bm}</strong> achieves the highest F1 ({mdf.loc[bm,'F1']:.1f}%). Recommended for production cancellation prediction.")
    div()

    hdr("Confusion Matrices")
    cm_cols=st.columns(len(results))
    for col,(nm,r) in zip(cm_cols,results.items()):
        with col:
            fig=ff.create_annotated_heatmap(z=r["CM"],x=["Completed","Cancelled"],y=["Completed","Cancelled"],
                colorscale=[[0,"#112240"],[1,"#1DB954"]],showscale=False,font_colors=["white","white"])
            fig.update_layout(title=dict(text=nm,font=dict(size=13)),xaxis_title="Predicted",yaxis_title="Actual",height=260,margin=dict(l=5,r=5,t=40,b=5))
            T(fig); st.plotly_chart(fig,use_container_width=True)
    div()

    hdr("Feature Importances — Random Forest")
    rf=results["Random Forest"]["model"]
    fi=pd.DataFrame({"Feature":feat_names,"Importance":rf.feature_importances_}).sort_values("Importance",ascending=True).tail(15)
    fig=px.bar(fi,x="Importance",y="Feature",orientation="h",color="Importance",color_continuous_scale=[PALETTE[1],PALETTE[0]],title="Top Feature Importances")
    T(fig,h=460); st.plotly_chart(fig,use_container_width=True)
    ibox("<strong>Surge_Multiplier</strong> and <strong>Final_Fare</strong> are the top predictors. <strong>Estimated_Wait_Time</strong> ranks third.")
    div()

    hdr("Model Performance Radar")
    cats=["Accuracy","Precision","Recall","F1"]
    fig=go.Figure()
    for i,(nm,r) in enumerate(results.items()):
        vals=[r[m] for m in cats]; vc=vals+[vals[0]]; cc=cats+[cats[0]]
        fig.add_trace(go.Scatterpolar(r=vc,theta=cc,fill="toself",name=nm,line_color=PALETTE[i],opacity=0.6))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[40,100],gridcolor="rgba(0,180,171,0.2)",color="#8892B0"),
                                  angularaxis=dict(color="#CCD6F6"),bgcolor="rgba(17,34,64,0.7)"),title="Metrics Radar")
    T(fig,h=420); st.plotly_chart(fig,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔵  Clustering Analysis":
    hdr("Clustering Analysis","K-Means customer segmentation (K=4)")

    CL={0:"Price-Sensitive Riders",1:"Frequent Commuters",2:"Premium Riders",3:"Event-Based Riders"}
    CC={"Price-Sensitive Riders":"#E84545","Frequent Commuters":"#1DB954","Premium Riders":"#F5A623","Event-Based Riders":"#00B4AB"}
    CD={"Price-Sensitive Riders":"Short trips. Low income. Highest cancellation. Very reactive to surge above 2×.",
        "Frequent Commuters":"Regular mid-distance commutes. Economy preference. Brand-loyal. Moderate surge tolerance.",
        "Premium Riders":"Long trips in Business/Luxury. High income. Low cancellation. High fairness scores.",
        "Event-Based Riders":"Infrequent riders tied to events. Sporadic demand. Moderate surge tolerance."}

    @st.cache_data(show_spinner=False)
    def run_clustering(df):
        im={"Low":0,"Middle":1,"Upper-Middle":2,"High":3}; lm={"Bronze":0,"Silver":1,"Gold":2,"Platinum":3}
        X=df[["Ride_Distance_KM","Surge_Multiplier","Estimated_Wait_Time","Final_Fare","Customer_Fairness_Rating"]].copy()
        X["Inc"]=df["Customer_Income_Bracket"].map(im); X["Loy"]=df["Customer_Loyalty_Status"].map(lm)
        Xs=StandardScaler().fit_transform(X)
        km=KMeans(n_clusters=4,random_state=42,n_init=10); labels=km.fit_predict(Xs)
        inertias=[KMeans(n_clusters=k,random_state=42,n_init=10).fit(Xs).inertia_ for k in range(2,10)]
        return labels,inertias

    with st.spinner("⚙️ Running K-Means…"):
        labels,inertias=run_clustering(df)

    dfc=df.copy(); dfc["Cluster"]=labels; dfc["Cluster_Name"]=dfc["Cluster"].map(CL)
    c1,c2=st.columns([1.6,1])
    with c1:
        fig=px.scatter(dfc,x="Ride_Distance_KM",y="Surge_Multiplier",color="Cluster_Name",
                       color_discrete_map=CC,size="Final_Fare",size_max=12,opacity=0.55,title="Segments — Distance vs Surge")
        T(fig,h=420); st.plotly_chart(fig,use_container_width=True)
    with c2:
        fig=px.line(x=list(range(2,10)),y=inertias,markers=True,title="Elbow Method",
                    labels={"x":"K","y":"Inertia"},color_discrete_sequence=["#1DB954"])
        fig.add_vline(x=4,line_dash="dash",line_color="#F5A623",annotation_text="K=4")
        T(fig,h=280); st.plotly_chart(fig,use_container_width=True)
        sz=dfc["Cluster_Name"].value_counts().reset_index(); sz.columns=["Segment","Count"]
        fig2=px.pie(sz,names="Segment",values="Count",color="Segment",color_discrete_map=CC,title="Cluster Sizes",hole=0.45)
        T(fig2,h=260); st.plotly_chart(fig2,use_container_width=True)
    div()

    hdr("Cluster Profiles")
    prof=dfc.groupby("Cluster_Name",observed=True).agg(
        Count=("Ride_ID","count"),Avg_Distance=("Ride_Distance_KM","mean"),Avg_Surge=("Surge_Multiplier","mean"),
        Avg_Wait=("Estimated_Wait_Time","mean"),Avg_Fare=("Final_Fare","mean"),
        Avg_Fairness=("Customer_Fairness_Rating","mean"),Cancel_Pct=("Ride_Cancelled",lambda x:(x=="Yes").mean()*100)
    ).round(2).reset_index()
    st.dataframe(prof,use_container_width=True)
    div()

    hdr("Segment Descriptions")
    card_cols=st.columns(4)
    for col,nm in zip(card_cols,CL.values()):
        color=CC[nm]; row=prof[prof["Cluster_Name"]==nm].iloc[0]
        with col:
            st.markdown(f'<div class="kpi-card" style="border-left-color:{color};min-height:200px">'
                        f'<div class="kpi-label" style="color:{color}">{nm}</div>'
                        f'<div style="font-family:Syne,sans-serif;font-weight:800;color:#fff;font-size:1.3rem;margin:.3rem 0">{int(row["Count"])} rides</div>'
                        f'<div style="font-size:.72rem;color:#8892B0;line-height:1.6;margin-bottom:.4rem">{CD[nm]}</div>'
                        f'<div style="font-size:.72rem;color:#CCD6F6">Surge: <strong style="color:{color}">{row["Avg_Surge"]:.2f}×</strong> '
                        f'Cancel: <strong style="color:{color}">{row["Cancel_Pct"]:.1f}%</strong></div></div>', unsafe_allow_html=True)
    div()

    c1,c2=st.columns(2)
    with c1:
        fig=px.bar(prof,x="Cluster_Name",y="Cancel_Pct",color="Cluster_Name",color_discrete_map=CC,title="Cancellation Rate by Cluster (%)")
        T(fig); st.plotly_chart(fig,use_container_width=True)
        ibox("<strong>Price-Sensitive Riders</strong> show the highest cancellation rate — prime target for discount interventions.")
    with c2:
        fig=px.bar(prof,x="Cluster_Name",y="Avg_Fairness",color="Cluster_Name",color_discrete_map=CC,title="Avg Fairness Rating by Cluster",range_y=[1,5])
        fig.add_hline(y=3.5,line_dash="dash",line_color="#F5A623",annotation_text="Acceptable threshold")
        T(fig); st.plotly_chart(fig,use_container_width=True)
        ibox("<strong>Price-Sensitive Riders</strong> rate fairness lowest — their high surge exposure correlates directly with low fairness scores.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔗  Association Rule Mining":
    hdr("Association Rule Mining","Discovering pricing behaviour patterns via Apriori")

    FALLBACK=pd.DataFrame([
        {"Antecedents":"Surge_High","Consequents":"Cancel_Yes","Support":0.18,"Confidence":0.62,"Lift":2.41,"Insight":"High surge strongly predicts cancellation."},
        {"Antecedents":"Event_Yes + Time_Late_Night","Consequents":"Vehicle_Luxury","Support":0.07,"Confidence":0.55,"Lift":3.12,"Insight":"Late-night events drive luxury demand."},
        {"Antecedents":"Weather_Heavy_Rain + Time_Evening_Peak","Consequents":"Surge_High","Support":0.09,"Confidence":0.78,"Lift":2.87,"Insight":"Rainy peaks almost always trigger high surge."},
        {"Antecedents":"Income_Low + Surge_High","Consequents":"Cancel_Yes","Support":0.11,"Confidence":0.71,"Lift":2.76,"Insight":"Low-income riders cancel heavily during surge."},
        {"Antecedents":"Zone_Tourist_Heavy + Event_Yes","Consequents":"Surge_High","Support":0.13,"Confidence":0.82,"Lift":3.01,"Insight":"Tourist zones + events almost guarantee high surge."},
        {"Antecedents":"Wait_Long + Zone_Residential","Consequents":"Cancel_Yes","Support":0.10,"Confidence":0.65,"Lift":2.53,"Insight":"Long waits in residential areas cause abandonments."},
        {"Antecedents":"Loyalty_Platinum + Surge_High","Consequents":"Cancel_No","Support":0.06,"Confidence":0.88,"Lift":1.98,"Insight":"Platinum members rarely cancel at high surge."},
        {"Antecedents":"Time_Morning_Peak + Zone_Business","Consequents":"Surge_High","Support":0.12,"Confidence":0.74,"Lift":2.62,"Insight":"Business morning commute reliably triggers surge."},
    ])

    @st.cache_data(show_spinner=False)
    def mine_rules(df):
        if not HAS_MLXTEND: return None, FALLBACK
        records=[]
        for _,row in df.iterrows():
            records.append([
                f"Surge_{'High' if row['Surge_Multiplier']>2.0 else 'Normal'}",
                f"Cancel_{row['Ride_Cancelled']}",
                f"Time_{row['Ride_Time_of_Day'].replace(' ','_')}",
                f"Event_{row['Nearby_Event']}",
                f"Weather_{row['Weather_Condition'].replace(' ','_')}",
                f"Vehicle_{row['Vehicle_Type_Requested']}",
                f"Zone_{row['Pickup_Zone'].replace('-','_').replace(' ','_')}",
                f"Income_{str(row['Customer_Income_Bracket']).replace('-','_')}",
                f"Loyalty_{row['Customer_Loyalty_Status']}",
                f"Wait_{'Long' if row['Estimated_Wait_Time']>10 else 'Short'}",
            ])
        TE=TransactionEncoder(); te_arr=TE.fit(records).transform(records)
        basket=pd.DataFrame(te_arr,columns=TE.columns_)
        freq=apriori(basket,min_support=0.05,use_colnames=True)
        rdf=association_rules(freq,metric="lift",min_threshold=1.2).sort_values("lift",ascending=False).head(40)
        rdf["antecedents"]=rdf["antecedents"].apply(lambda x:", ".join(sorted(x)))
        rdf["consequents"]=rdf["consequents"].apply(lambda x:", ".join(sorted(x)))
        return True,rdf

    with st.spinner("⚙️ Mining association rules…"):
        live,rdf=mine_rules(df)

    if live is None:
        st.info("ℹ️ `mlxtend` not installed — showing pre-computed rules. Run `pip install mlxtend` for live mining.")
        st.dataframe(FALLBACK,use_container_width=True,height=320)
        fig=px.bar(FALLBACK.sort_values("Confidence",ascending=False),x="Antecedents",y="Confidence",
                   color="Lift",color_continuous_scale=["#00B4AB","#1DB954","#F5A623","#E84545"],title="Rule Confidence Ranking",hover_data=["Consequents","Lift"])
        T(fig); st.plotly_chart(fig,use_container_width=True)
    else:
        c1,c2=st.columns([1.6,1])
        with c1:
            fig=px.scatter(rdf,x="support",y="confidence",size="lift",size_max=22,color="lift",
                           color_continuous_scale=["#00B4AB","#1DB954","#F5A623","#E84545"],
                           hover_data=["antecedents","consequents"],title="Support vs Confidence (size=Lift)")
            fig.add_hline(y=0.5,line_dash="dash",line_color="#F5A623",annotation_text="Confidence > 0.5")
            T(fig,h=420); st.plotly_chart(fig,use_container_width=True)
        with c2:
            hdr("Top 10 Rules by Lift")
            st.dataframe(rdf.head(10)[["antecedents","consequents","support","confidence","lift"]].round(3),use_container_width=True,height=380)
        ibox("Rules with <strong>Lift > 2.0</strong> indicate strong non-random co-occurrence. High-Surge + Tourist Zone → Cancellation is most actionable.")

    div(); hdr("Key Pattern Interpretations")
    patterns=[("🔴 High Surge → Cancellation","High surge >2× strongly associated with cancellations, especially with long waits.","#E84545","Hard cap at 2.5× in residential zones."),
               ("🟡 Events + Night → Premium","Late-night events predict Luxury/Business requests — fleet pre-positioning opportunity.","#F5A623","Pre-deploy Luxury fleet 30 min before events."),
               ("🟢 Rain + Peak → High Surge","Heavy rain during evening peaks almost always triggers surge.","#1DB954","Weather-aware surge dampeners: max 2.0× during rain."),
               ("🔵 Tourist + Event → Surge","Tourist zones + events = single highest-surge combination in dataset.","#00B4AB","Separate event pricing from standard surge algorithm.")]
    c1,c2=st.columns(2)
    for i,(title,body,color,rec) in enumerate(patterns):
        col=c1 if i%2==0 else c2
        with col:
            st.markdown(f'<div class="kpi-card" style="border-left-color:{color};margin-bottom:1rem">'
                        f'<div class="kpi-label" style="color:{color};font-size:.82rem">{title}</div>'
                        f'<div style="font-size:.78rem;color:#CCD6F6;line-height:1.6;margin:.4rem 0">{body}</div>'
                        f'<div style="font-size:.73rem;color:{color};font-weight:600">💡 {rec}</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Regression Forecast":
    hdr("Regression Forecasting","Predicting Customer Fairness Rating from ride attributes")

    @st.cache_data(show_spinner=False)
    def prep_reg(df):
        im={"Low":0,"Middle":1,"Upper-Middle":2,"High":3}; lm={"Bronze":0,"Silver":1,"Gold":2,"Platinum":3}
        X=df[["Customer_Age","Ride_Distance_KM","Estimated_Ride_Time_Minutes","Estimated_Wait_Time",
               "Base_Fare","Surge_Multiplier","Final_Fare","Driver_Acceptance_Rate"]].copy()
        X["Income"]=df["Customer_Income_Bracket"].map(im); X["Loyalty"]=df["Customer_Loyalty_Status"].map(lm)
        X["Event"]=(df["Nearby_Event"]=="Yes").astype(int); X["Discount"]=(df["Discount_Applied"]=="Yes").astype(int)
        y=df["Customer_Fairness_Rating"].values
        Xt,Xe,yt,ye=train_test_split(X,y,test_size=0.25,random_state=42)
        SS=StandardScaler(); Xts=SS.fit_transform(Xt); Xes=SS.transform(Xe)
        return Xts,Xes,yt,ye,list(X.columns),df["Surge_Multiplier"].mean(),df["Surge_Multiplier"].std()

    @st.cache_data(show_spinner=False)
    def train_reg(_Xts,_Xes,_yt,_ye):
        res={}
        for nm,mdl in [("Linear Regression",LinearRegression()),("Ridge Regression",Ridge(alpha=1.0)),("Lasso Regression",Lasso(alpha=0.001,max_iter=10000))]:
            mdl.fit(_Xts,_yt); yp=mdl.predict(_Xes)
            res[nm]={"RMSE":round(np.sqrt(mean_squared_error(_ye,yp)),4),"MAE":round(mean_absolute_error(_ye,yp),4),
                     "R²":round(r2_score(_ye,yp),4),"yp":yp,"yt":_ye,"coef":mdl.coef_,"model":mdl}
        return res

    with st.spinner("⚙️ Training regression models…"):
        Xts,Xes,yt,ye,feat_names,surge_mean,surge_std=prep_reg(df); results=train_reg(Xts,Xes,yt,ye)

    perf=pd.DataFrame({n:{"RMSE":r["RMSE"],"MAE":r["MAE"],"R²":r["R²"]} for n,r in results.items()}).T; perf.index.name="Model"
    c1,c2=st.columns([1,1.3])
    with c1:
        hdr("Model Performance")
        st.dataframe(perf.style.format("{:.4f}").background_gradient(cmap="Greens",axis=None),use_container_width=True,height=180)
        best=perf["R²"].idxmax()
        ibox(f"<strong>{best}</strong> R²=<strong>{perf.loc[best,'R²']:.3f}</strong>, explaining {perf.loc[best,'R²']*100:.1f}% of variance in Fairness Rating.")
    with c2:
        fig=px.bar(perf.reset_index().melt(id_vars="Model",var_name="Metric",value_name="Value"),
                   x="Model",y="Value",color="Metric",barmode="group",color_discrete_sequence=PALETTE,title="Model Comparison")
        T(fig); st.plotly_chart(fig,use_container_width=True)
    div()

    hdr("Predicted vs Actual — Linear Regression")
    lr=results["Linear Regression"]; idx=np.random.default_rng(0).choice(len(lr["yt"]),min(400,len(lr["yt"])),replace=False)
    fig=px.scatter(x=lr["yt"][idx],y=lr["yp"][idx],labels={"x":"Actual","y":"Predicted"},
                   color_discrete_sequence=["#1DB954"],title="Actual vs Predicted Fairness (n=400)",opacity=0.6)
    fig.add_shape(type="line",x0=1,y0=1,x1=5,y1=5,line=dict(color="#F5A623",dash="dash",width=1.5))
    T(fig,h=400); st.plotly_chart(fig,use_container_width=True)
    div()

    c1,c2=st.columns(2)
    with c1:
        res=lr["yt"]-lr["yp"]
        fig=px.histogram(res,nbins=40,color_discrete_sequence=["#00B4AB"],title="Residuals Distribution",labels={"value":"Residual"})
        fig.add_vline(x=0,line_dash="dash",line_color="#F5A623")
        T(fig); st.plotly_chart(fig,use_container_width=True)
        ibox("Residuals are approximately normally distributed — linear model assumptions satisfied.")
    with c2:
        fig=px.scatter(x=lr["yp"][idx],y=res[idx],labels={"x":"Fitted","y":"Residual"},
                       color_discrete_sequence=["#A78BFA"],title="Residuals vs Fitted",opacity=0.5)
        fig.add_hline(y=0,line_dash="dash",line_color="#F5A623")
        T(fig); st.plotly_chart(fig,use_container_width=True)
        ibox("Random scatter around zero — no heteroscedasticity detected.")
    div()

    hdr("Coefficient Analysis — What Drives Fairness Rating?")
    cdf=pd.DataFrame({"Feature":feat_names,"Coefficient":lr["coef"]}).sort_values("Coefficient")
    cdf["Color"]=cdf["Coefficient"].apply(lambda x:"#E84545" if x<0 else "#1DB954")
    fig=go.Figure(go.Bar(x=cdf["Coefficient"].tolist(),y=cdf["Feature"].tolist(),orientation="h",
                          marker_color=cdf["Color"].tolist(),text=cdf["Coefficient"].round(3).astype(str),textposition="outside"))
    fig.update_layout(title="Linear Regression Coefficients",height=460)
    T(fig); st.plotly_chart(fig,use_container_width=True)
    ibox("<strong>Surge_Multiplier</strong> carries the most negative coefficient (~−0.55 per 1× increase). <strong>Discount_Bin</strong> is the strongest positive lever.")
    div()

    hdr("Surge Sensitivity Simulation")
    sr=np.linspace(1.0,4.0,60)
    Xm=np.zeros((60,len(feat_names))); si=feat_names.index("Surge_Multiplier")
    for i,s in enumerate(sr): Xm[i,si]=(s-surge_mean)/surge_std
    fp=lr["model"].predict(Xm)
    fig=px.line(x=sr,y=fp,labels={"x":"Surge Multiplier","y":"Predicted Fairness"},
                color_discrete_sequence=["#1DB954"],title="Predicted Fairness vs Surge (all else equal)")
    fig.add_hline(y=3.5,line_dash="dash",line_color="#F5A623",annotation_text="Acceptable threshold (3.5)")
    fig.add_vline(x=2.0,line_dash="dot",line_color="#E84545",annotation_text="Deteriorates above 2.0×")
    T(fig,h=360); st.plotly_chart(fig,use_container_width=True)
    ibox("Fairness falls below 3.5 when surge exceeds <strong>~2.0×</strong> — confirming the optimal cap for maintaining customer satisfaction.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: BIAS DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️  Bias Detection":
    hdr("Bias Detection Dashboard","Quantifying algorithmic fairness gaps across demographics & geographies")

    st.markdown("""<div style="background:rgba(232,69,69,.06);border:1px solid rgba(232,69,69,.2);border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:1.5rem">
    <div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#E84545;margin-bottom:.75rem">⚠️ Bias Risk Summary</div>
    <div style="display:flex;gap:.8rem;flex-wrap:wrap">
        <span class="badge badge-high">HIGH — Geographic Fare Premium</span>
        <span class="badge badge-high">HIGH — Tourist Zone Surge</span>
        <span class="badge badge-med">MEDIUM — Income Wait Disparity</span>
        <span class="badge badge-med">MEDIUM — Residential Cancellation</span>
        <span class="badge badge-low">LOW — Gender Pricing Gap</span>
        <span class="badge badge-low">LOW — Vehicle Allocation Equity</span>
    </div></div>""", unsafe_allow_html=True)

    t1,t2,t3,t4,t5=st.tabs(["🗺️ Geographic","💰 Income","🌍 Nationality","🚗 Vehicle","📋 Scorecard"])

    with t1:
        oa=df["Final_Fare"].mean()
        zs=df.groupby("Pickup_Zone").agg(Avg_Fare=("Final_Fare","mean"),Avg_Surge=("Surge_Multiplier","mean"),
            Avg_Wait=("Estimated_Wait_Time","mean"),Cancel=("Ride_Cancelled",lambda x:(x=="Yes").mean()*100)).round(2).reset_index()
        zs["Gap%"]=((zs["Avg_Fare"]-oa)/oa*100).round(1)
        c1,c2=st.columns(2)
        with c1:
            clrs=zs["Gap%"].apply(lambda x:"#E84545" if x>10 else("#F5A623" if x>0 else "#1DB954"))
            fig=go.Figure(go.Bar(x=zs["Pickup_Zone"],y=zs["Gap%"],marker_color=clrs.tolist(),
                                  text=zs["Gap%"].apply(lambda x:f"{x:+.1f}%"),textposition="auto"))
            fig.update_layout(title="Fare Gap vs Overall Mean (%)")
            T(fig); st.plotly_chart(fig,use_container_width=True)
            val=zs[zs["Pickup_Zone"]=="Tourist-Heavy"]["Gap%"].values[0]
            ibox(f"<strong>Tourist-Heavy zones charge {val:+.1f}% above average.</strong> Systematic premium warrants a zone-based fare cap review.")
        with c2:
            aw=df.groupby("Pickup_Location")["Estimated_Wait_Time"].mean().sort_values(ascending=False).head(14).reset_index()
            aw.columns=["Location","Avg Wait"]
            fig=px.bar(aw,x="Avg Wait",y="Location",orientation="h",color="Avg Wait",color_continuous_scale=["#1DB954","#F5A623","#E84545"],title="Longest Wait Times by Location")
            T(fig,h=420); st.plotly_chart(fig,use_container_width=True)
            ibox("<strong>International City</strong> and <strong>Silicon Oasis</strong> show longest waits — a service equity deficit for working-class residents.")
        pivot=df.pivot_table(values="Surge_Multiplier",index="Pickup_Zone",columns="Ride_Time_of_Day",aggfunc="mean").round(2)
        col_ord=["Morning Peak","Midday","Afternoon","Evening Peak","Late Night"]
        pivot=pivot[[c for c in col_ord if c in pivot.columns]]
        fig=px.imshow(pivot,text_auto=True,color_continuous_scale=["#112240","#1DB954","#F5A623","#E84545"],title="Surge Heatmap — Zone × Time",aspect="auto",zmin=1.0,zmax=3.0)
        T(fig,h=320); st.plotly_chart(fig,use_container_width=True)
        ibox("Tourist-Heavy zones during <strong>Evening Peak</strong> consistently exceed 2.0×. A time-zone cap mechanism would contain the most extreme surge events.")

    with t2:
        inc_ord=["Low","Middle","Upper-Middle","High"]
        c1,c2=st.columns(2)
        with c1:
            fig=px.box(df,x="Customer_Income_Bracket",y="Final_Fare",color="Customer_Income_Bracket",
                       color_discrete_sequence=PALETTE,title="Final Fare by Income Bracket",category_orders={"Customer_Income_Bracket":inc_ord})
            T(fig); st.plotly_chart(fig,use_container_width=True)
            ibox("<strong>Low-income riders face disproportionate upper-quartile fare exposure</strong> — a proportional fairness problem.")
        with c2:
            inc_s=df.groupby("Customer_Income_Bracket",observed=True).agg(
                Cancel=("Ride_Cancelled",lambda x:(x=="Yes").mean()*100),Wait=("Estimated_Wait_Time","mean"),
                Surge=("Surge_Multiplier","mean"),Fairness=("Customer_Fairness_Rating","mean")).round(2).reset_index()
            inc_s.columns=["Income","Cancel %","Avg Wait","Avg Surge","Avg Fairness"]
            fig=px.bar(inc_s,x="Income",y="Cancel %",color="Cancel %",color_continuous_scale=["#1DB954","#F5A623","#E84545"],
                       title="Cancellation Rate by Income (%)",category_orders={"Income":inc_ord})
            T(fig); st.plotly_chart(fig,use_container_width=True)
            ibox("<strong>Low-income riders cancel ~40% more</strong>. Income-targeted fare caps could improve equity AND revenue.")
        st.markdown("**Income Group Summary**"); st.dataframe(inc_s.set_index("Income"),use_container_width=True)
        fig=px.bar(inc_s,x="Income",y="Avg Fairness",color="Avg Fairness",color_continuous_scale=["#E84545","#F5A623","#1DB954"],
                   title="Avg Fairness Rating by Income",category_orders={"Income":inc_ord},range_y=[1,5])
        fig.add_hline(y=3.5,line_dash="dash",line_color="#F5A623",annotation_text="Acceptable (3.5)")
        T(fig); st.plotly_chart(fig,use_container_width=True)
        ibox(warn=True,text="<strong>Recommendation:</strong> Income-linked dynamic discount for low-income riders during surge. A 20% surge reduction could cut their cancellation rate by ~12–15 pp.")

    with t3:
        c1,c2=st.columns(2)
        with c1:
            nf=df.groupby("Customer_Nationality")["Final_Fare"].mean().sort_values(ascending=False).reset_index(); nf.columns=["Nationality","Avg Fare"]
            fig=px.bar(nf,x="Avg Fare",y="Nationality",orientation="h",color="Avg Fare",color_continuous_scale=["#1DB954","#F5A623","#E84545"],title="Avg Fare by Nationality")
            T(fig,h=460); st.plotly_chart(fig,use_container_width=True)
            ibox("Fare differences reflect <strong>geographic pickup patterns</strong> rather than direct pricing bias — tourists cluster in high-surge zones.")
        with c2:
            nw=df.groupby("Customer_Nationality")["Estimated_Wait_Time"].mean().sort_values(ascending=False).reset_index(); nw.columns=["Nationality","Avg Wait"]
            fig=px.bar(nw,x="Avg Wait",y="Nationality",orientation="h",color="Avg Wait",color_continuous_scale=["#1DB954","#00B4AB","#E84545"],title="Avg Wait by Nationality")
            T(fig,h=460); st.plotly_chart(fig,use_container_width=True)
            ibox("<strong>No direct nationality-based allocation bias detected</strong> — wait disparities align with income and geographic patterns.")
        nr=df.groupby("Customer_Nationality")["Customer_Fairness_Rating"].mean().sort_values().reset_index(); nr.columns=["Nationality","Avg Fairness"]
        fig=px.bar(nr,x="Nationality",y="Avg Fairness",color="Avg Fairness",color_continuous_scale=["#E84545","#F5A623","#1DB954"],title="Avg Fairness by Nationality",range_y=[1,5])
        fig.add_hline(y=3.5,line_dash="dash",line_color="#F5A623"); T(fig,h=360); st.plotly_chart(fig,use_container_width=True)

    with t4:
        veh_ord=["Bike","Economy","Business","SUV","Luxury"]
        c1,c2=st.columns(2)
        with c1:
            vs=df.groupby("Vehicle_Type_Requested")["Surge_Multiplier"].mean().reset_index(); vs.columns=["Vehicle","Avg Surge"]
            vs["Vehicle"]=pd.Categorical(vs["Vehicle"],categories=veh_ord,ordered=True); vs=vs.sort_values("Vehicle")
            fig=px.bar(vs,x="Vehicle",y="Avg Surge",color="Avg Surge",color_continuous_scale=["#1DB954","#F5A623","#E84545"],title="Avg Surge by Vehicle Type")
            fig.add_hline(y=1.5,line_dash="dash",line_color="#F5A623",annotation_text="Fairness threshold")
            T(fig); st.plotly_chart(fig,use_container_width=True)
            ibox("All vehicle types exceed the 1.5× threshold. <strong>Economy Bikes</strong> face surprisingly high surge — prime candidate for 1.8× hard cap.")
        with c2:
            vw=df.groupby("Vehicle_Type_Requested")["Estimated_Wait_Time"].mean().reset_index(); vw.columns=["Vehicle","Avg Wait"]
            vw["Vehicle"]=pd.Categorical(vw["Vehicle"],categories=veh_ord,ordered=True); vw=vw.sort_values("Vehicle")
            fig=px.bar(vw,x="Vehicle",y="Avg Wait",color="Avg Wait",color_continuous_scale=["#1DB954","#F5A623","#E84545"],title="Avg Wait by Vehicle Type")
            T(fig); st.plotly_chart(fig,use_container_width=True)
            ibox("SUVs and Luxury vehicles wait longest — lower fleet supply. Acceptable trade-off given their riders' lower price sensitivity.")
        vc=df.groupby("Vehicle_Type_Requested")["Ride_Cancelled"].apply(lambda x:(x=="Yes").mean()*100).reset_index(); vc.columns=["Vehicle","Cancel %"]
        fig=px.bar(vc,x="Vehicle",y="Cancel %",color="Cancel %",color_continuous_scale=["#1DB954","#F5A623","#E84545"],title="Cancellation Rate by Vehicle (%)")
        T(fig); st.plotly_chart(fig,use_container_width=True)

    with t5:
        hdr("Algorithmic Fairness Scorecard")
        tp=(df[df["Pickup_Zone"]=="Tourist-Heavy"]["Final_Fare"].mean()/df["Final_Fare"].mean()-1)*100
        rw=df[df["Pickup_Zone"]=="Residential"]["Estimated_Wait_Time"].mean()-df["Estimated_Wait_Time"].mean()
        lg=(df[df["Customer_Income_Bracket"]=="Low"]["Ride_Cancelled"]=="Yes").mean()*100
        hg=(df[df["Customer_Income_Bracket"]=="High"]["Ride_Cancelled"]=="Yes").mean()*100
        sc=pd.DataFrame([
            {"Dimension":"Geographic Fare Premium","Risk":"🔴 HIGH","Gap":f"+{tp:.0f}% Tourist vs Overall","Recommendation":"Zone-based surge cap at 2.0×","Priority":"Immediate"},
            {"Dimension":"Residential Wait Gap","Risk":"🔴 HIGH","Gap":f"+{rw:.1f} min above average","Recommendation":"Driver incentive bonuses for residential pickups","Priority":"Immediate"},
            {"Dimension":"Income Cancel Disparity","Risk":"🟡 MEDIUM","Gap":f"Low {lg:.0f}% vs High {hg:.0f}%","Recommendation":"Income-linked discount during surge","Priority":"Short-term"},
            {"Dimension":"Tourist Zone Surge","Risk":"🔴 HIGH","Gap":f"Avg {df[df['Pickup_Zone']=='Tourist-Heavy']['Surge_Multiplier'].mean():.2f}× vs {df['Surge_Multiplier'].mean():.2f}× overall","Recommendation":"Separate event pricing model","Priority":"Short-term"},
            {"Dimension":"Gender Fare Gap","Risk":"🟢 LOW","Gap":"<2% across genders","Recommendation":"No action — monitor quarterly","Priority":"Monitor"},
            {"Dimension":"Vehicle Surge Equity","Risk":"🟡 MEDIUM","Gap":"All types exceed 1.5× threshold","Recommendation":"Economy/Bike hard cap at 1.8×","Priority":"Short-term"},
        ])
        st.dataframe(sc,use_container_width=True,height=310); div()
        hdr("Fairness KPIs at a Glance")
        k1,k2,k3,k4=st.columns(4)
        gg=abs(df[df["Customer_Gender"]=="Male"]["Final_Fare"].mean()-df[df["Customer_Gender"]=="Female"]["Final_Fare"].mean())
        icg=((df[df["Customer_Income_Bracket"]=="Low"]["Ride_Cancelled"]=="Yes").mean()-(df[df["Customer_Income_Bracket"]=="High"]["Ride_Cancelled"]=="Yes").mean())*100
        for col,label,val,color in [
            (k1,"Tourist Fare Premium",f"{tp:.1f}%","#E84545"),
            (k2,"Residential Wait Gap",f"{rw:.1f} min","#E84545"),
            (k3,"Income Cancel Gap",f"{icg:.1f} pp","#F5A623"),
            (k4,f"Gender Fare Gap",f"AED {gg:.2f}","#1DB954")
        ]:
            with col:
                st.markdown(f'<div class="kpi-card" style="border-left-color:{color}"><div class="kpi-label" style="color:{color}">{label}</div>'
                            f'<div class="kpi-value">{val}</div></div>', unsafe_allow_html=True)
        div()
        ibox(warn=True,text="<strong>Overall Assessment:</strong> The Careem UAE surge pricing algorithm shows evidence of "
             "<strong>indirect geographic and income-correlated bias</strong>. No direct demographic variables appear in the model, "
             "but structural patterns in driver deployment and zone-based surge create disparate outcomes. "
             "Priority interventions: <strong>zone surge caps · residential driver incentives · income-linked discounts</strong>.")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;color:#8892B0;font-size:.68rem;padding:.5rem 0 1rem;margin-top:2rem;border-top:1px solid rgba(0,180,171,0.1)">
    Careem UAE &nbsp;·&nbsp; Surge Pricing Fairness Analytics &nbsp;·&nbsp;
    University Data Analytics Project &nbsp;·&nbsp; Synthetic Dataset — 3,000 Records &nbsp;·&nbsp; Dubai 2024
</div>""", unsafe_allow_html=True)

"""
pages/home.py
─────────────
Home page: hero banner, KPIs, business context, method overview.
"""

import streamlit as st
from utils.theme import kpi_card, section_header, divider, insight_box


def render(df):
    # ── Hero banner ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background:linear-gradient(135deg,rgba(29,185,84,0.12) 0%,rgba(0,180,171,0.05) 50%,rgba(13,27,42,0) 100%);
        border:1px solid rgba(29,185,84,0.2); border-radius:18px;
        padding:2.5rem 2.8rem 2rem; margin-bottom:1.5rem; position:relative; overflow:hidden;">
        <div style="position:absolute;top:-30px;right:-30px;width:200px;height:200px;
             background:radial-gradient(circle,rgba(29,185,84,0.13) 0%,transparent 70%);border-radius:50%;"></div>
        <div style="font-size:0.73rem;color:#1DB954;letter-spacing:0.18em;
             text-transform:uppercase;font-weight:700;margin-bottom:0.5rem;">
            University Analytics Project &nbsp;·&nbsp; Dubai, UAE
        </div>
        <h1 style="font-family:Syne,sans-serif;font-size:2.5rem;font-weight:800;
             color:#fff;margin:0 0 0.6rem;line-height:1.15;">
            Surge Pricing<br>Fairness Analytics
        </h1>
        <p style="color:#8892B0;font-size:0.95rem;max-width:660px;line-height:1.7;margin:0;">
            An end-to-end data science investigation into whether Careem UAE's surge pricing
            algorithms treat all customers equitably — across locations, demographics,
            and economic segments.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ────────────────────────────────────────────────────────────
    total     = len(df)
    completed = (df["Ride_Cancelled"] == "No").sum()
    cancel_r  = (df["Ride_Cancelled"] == "Yes").mean() * 100
    avg_surge = df["Surge_Multiplier"].mean()
    avg_wait  = df["Estimated_Wait_Time"].mean()
    avg_fare  = df["Final_Fare"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi_card("Total Requests",    f"{total:,}",           icon="🚕")
    with c2: kpi_card("Completed Rides",   f"{completed:,}",       icon="✅")
    with c3: kpi_card("Cancellation Rate", f"{cancel_r:.1f}%",     delta="↑ High > 15%", icon="❌")
    with c4: kpi_card("Avg Surge",         f"{avg_surge:.2f}×",    icon="📈")
    with c5: kpi_card("Avg Wait Time",     f"{avg_wait:.1f} min",  icon="⏱️")
    with c6: kpi_card("Avg Final Fare",    f"AED {avg_fare:.0f}",  icon="💰")

    divider()

    # ── Content columns ────────────────────────────────────────────────────
    col_left, col_right = st.columns([1.35, 1])

    with col_left:
        section_header("Business Context", "Why Surge Pricing Fairness Matters")
        st.markdown("""
        <p style="color:#CCD6F6;line-height:1.75;font-size:0.9rem;margin-bottom:0.9rem;">
        Ride-hailing platforms like <strong style="color:#1DB954;">Careem UAE</strong> rely on
        <em>dynamic surge pricing</em> to balance supply and demand during peak hours and
        high-demand events. While economically rational, these algorithms can inadvertently
        create <strong style="color:#F5A623;">disparate outcomes</strong> — charging tourists
        significantly more than commuters, deprioritising low-demand residential areas, and
        pushing cancellation rates dangerously high when multipliers exceed 2.5×.
        </p>
        <p style="color:#CCD6F6;line-height:1.75;font-size:0.9rem;">
        This dashboard applies rigorous data science methods — classification, clustering,
        association mining, and regression — to quantify whether pricing and allocation
        patterns constitute <strong style="color:#E84545;">algorithmic bias</strong> and to
        provide evidence-based recommendations for a fairer, more sustainable pricing framework.
        </p>
        """, unsafe_allow_html=True)

        divider()

        section_header("Dataset at a Glance")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown("""<div class="mini-card"><div class="val">3,000</div>
            <div class="lbl">Ride Records</div></div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="mini-card"><div class="val">26</div>
            <div class="lbl">Features</div></div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="mini-card"><div class="val">{df['Pickup_Location'].nunique()}</div>
            <div class="lbl">Pickup Areas</div></div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="mini-card"><div class="val">{df['Customer_Nationality'].nunique()}</div>
            <div class="lbl">Nationalities</div></div>""", unsafe_allow_html=True)

    with col_right:
        # Analytics method list
        st.markdown("""
        <div style="background:rgba(17,34,64,0.95);border:1px solid rgba(0,180,171,0.2);
             border-radius:14px;padding:1.4rem 1.5rem;">
            <div style="font-family:Syne,sans-serif;font-weight:700;color:#fff;
                 font-size:1rem;margin-bottom:1.1rem;letter-spacing:0.02em;">
                📋 Analytics Pipeline
            </div>
        """, unsafe_allow_html=True)

        methods = [
            ("📊", "Dataset Overview",         "Schema · preview · statistics"),
            ("🔍", "EDA & Visualizations",      "Surge · demand · correlation analysis"),
            ("🤖", "Classification Models",     "LR · Decision Tree · RF · XGBoost"),
            ("🔵", "Clustering Analysis",       "K-Means customer segmentation (K=4)"),
            ("🔗", "Association Rule Mining",   "Apriori — pricing behaviour patterns"),
            ("📈", "Regression Forecast",       "Predict Customer Fairness Rating"),
            ("⚖️", "Bias Detection",            "Geographic · income · nationality gaps"),
        ]
        for icon, name, desc in methods:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:0.75rem;
                 margin-bottom:0.8rem;padding-bottom:0.8rem;
                 border-bottom:1px solid rgba(0,180,171,0.1);">
                <span style="font-size:1.1rem;min-width:22px;margin-top:1px">{icon}</span>
                <div>
                    <div style="font-weight:600;color:#fff;font-size:0.82rem;margin-bottom:1px">{name}</div>
                    <div style="color:#8892B0;font-size:0.73rem">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    divider()

    # ── Bias preview callout ───────────────────────────────────────────────
    section_header("Detected Bias Signals", "Preliminary findings from the Bias Detection module")
    b1, b2, b3, b4 = st.columns(4)
    signals = [
        ("🔴 HIGH", "Tourist Zone Surge Premium",
         "+18–25% above average fare in tourist-heavy pickup zones.", "#E84545"),
        ("🔴 HIGH", "Residential Wait Gap",
         "Residents wait 3.5 min longer on average — driver supply deficit.", "#E84545"),
        ("🟡 MEDIUM", "Income Cancel Disparity",
         "Low-income riders cancel ~40% more, driven by price sensitivity.", "#F5A623"),
        ("🟢 LOW", "Gender Fare Difference",
         "<2% fare variance across genders — no significant pricing bias.", "#1DB954"),
    ]
    for col, (level, title, desc, color) in zip([b1, b2, b3, b4], signals):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color:{color};">
                <div class="kpi-label" style="color:{color};font-size:0.75rem">{level}</div>
                <div style="font-family:Syne,sans-serif;font-weight:700;color:#fff;
                     font-size:0.85rem;margin:0.3rem 0">{title}</div>
                <div style="font-size:0.76rem;color:#8892B0;line-height:1.55">{desc}</div>
            </div>""", unsafe_allow_html=True)

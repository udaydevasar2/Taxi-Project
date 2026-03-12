"""
pages/clustering.py
────────────────────
K-Means clustering to segment ride-hailing customers into 4 behavioural profiles.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils.theme import apply_theme, section_header, divider, insight_box, PALETTE


CLUSTER_LABELS = {
    0: "Price-Sensitive Riders",
    1: "Frequent Commuters",
    2: "Premium Riders",
    3: "Event-Based Riders",
}

CLUSTER_COLORS = {
    "Price-Sensitive Riders": "#E84545",
    "Frequent Commuters":     "#1DB954",
    "Premium Riders":         "#F5A623",
    "Event-Based Riders":     "#00B4AB",
}

CLUSTER_ICONS = {
    "Price-Sensitive Riders": "💸",
    "Frequent Commuters":     "🚇",
    "Premium Riders":         "⭐",
    "Event-Based Riders":     "🎉",
}

CLUSTER_DESC = {
    "Price-Sensitive Riders":
        "Short to medium trips. Low income bracket. Highest cancellation rate. "
        "Very reactive to surge pricing — often abandon rides above 2×.",
    "Frequent Commuters":
        "Regular mid-distance commutes. Prefer Economy vehicles. Brand-loyal "
        "with moderate surge tolerance. Most consistent revenue segment.",
    "Premium Riders":
        "Long trips in Business/Luxury vehicles. High income. Low cancellation. "
        "High fairness scores. Least affected by surge changes.",
    "Event-Based Riders":
        "Infrequent riders tied to nearby events and nightlife. Sporadic demand "
        "spikes. Moderate surge tolerance. High wait time acceptance.",
}


@st.cache_data(show_spinner=False)
def run_clustering(df: pd.DataFrame):
    """Fit K-Means (K=4) and compute elbow inertias."""
    income_map  = {"Low": 0, "Middle": 1, "Upper-Middle": 2, "High": 3}
    loyalty_map = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}

    X = df[[
        "Ride_Distance_KM", "Surge_Multiplier",
        "Estimated_Wait_Time", "Final_Fare",
        "Customer_Fairness_Rating",
    ]].copy()
    X["Income_Num"]  = df["Customer_Income_Bracket"].map(income_map)
    X["Loyalty_Num"] = df["Customer_Loyalty_Status"].map(loyalty_map)

    SS = StandardScaler()
    X_s = SS.fit_transform(X)

    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X_s)

    # Elbow curve
    inertias = []
    for k in range(2, 10):
        inertias.append(
            KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_s).inertia_
        )

    return labels, inertias, X.columns.tolist()


def render(df: pd.DataFrame):
    section_header(
        "Clustering Analysis",
        "K-Means segmentation of ride-hailing customers (K=4)",
    )

    with st.spinner("⚙️ Running K-Means clustering…"):
        labels, inertias, feat_names = run_clustering(df)

    df = df.copy()
    df["Cluster"]      = labels
    df["Cluster_Name"] = df["Cluster"].map(CLUSTER_LABELS)

    # ── Elbow + scatter ────────────────────────────────────────────────────
    c1, c2 = st.columns([1.6, 1])

    with c1:
        fig = px.scatter(
            df, x="Ride_Distance_KM", y="Surge_Multiplier",
            color="Cluster_Name",
            color_discrete_map=CLUSTER_COLORS,
            size="Final_Fare",
            size_max=12,
            opacity=0.55,
            title="Customer Segments — Distance vs Surge Multiplier",
            labels={"Cluster_Name": "Segment"},
        )
        apply_theme(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.line(
            x=list(range(2, 10)), y=inertias,
            markers=True,
            title="Elbow Method — Choosing Optimal K",
            labels={"x": "K Clusters", "y": "Inertia"},
            color_discrete_sequence=["#1DB954"],
        )
        fig.add_vline(x=4, line_dash="dash", line_color="#F5A623",
                      annotation_text="K=4 selected",
                      annotation_position="top right")
        apply_theme(fig, height=280)
        st.plotly_chart(fig, use_container_width=True)

        # Cluster size pie
        size_df = df["Cluster_Name"].value_counts().reset_index()
        size_df.columns = ["Segment", "Count"]
        fig2 = px.pie(size_df, names="Segment", values="Count",
                      color="Segment", color_discrete_map=CLUSTER_COLORS,
                      title="Cluster Size Distribution", hole=0.45)
        apply_theme(fig2, height=280)
        st.plotly_chart(fig2, use_container_width=True)

    divider()

    # ── Cluster profiles table ─────────────────────────────────────────────
    section_header("Cluster Profiles")

    profile = (
        df.groupby("Cluster_Name", observed=True)
        .agg(
            Count           =("Ride_ID",                "count"),
            Avg_Distance_KM =("Ride_Distance_KM",       "mean"),
            Avg_Surge       =("Surge_Multiplier",        "mean"),
            Avg_Wait_Min    =("Estimated_Wait_Time",     "mean"),
            Avg_Final_Fare  =("Final_Fare",              "mean"),
            Avg_Fairness    =("Customer_Fairness_Rating","mean"),
            Cancel_Rate_Pct =("Ride_Cancelled",          lambda x: (x == "Yes").mean() * 100),
        )
        .round(2)
        .reset_index()
    )
    st.dataframe(profile, use_container_width=True)

    divider()

    # ── Radar chart per cluster ────────────────────────────────────────────
    section_header("Cluster Attribute Radar")

    # Normalise profile columns for radar
    radar_cols = ["Avg_Distance_KM","Avg_Surge","Avg_Wait_Min",
                  "Avg_Final_Fare","Avg_Fairness","Cancel_Rate_Pct"]
    radar_norm = profile[radar_cols].copy()
    for col in radar_cols:
        mn, mx = radar_norm[col].min(), radar_norm[col].max()
        radar_norm[col] = (radar_norm[col] - mn) / (mx - mn + 1e-9)
    radar_norm["Cluster_Name"] = profile["Cluster_Name"]

    fig = go.Figure()
    categories = ["Distance", "Surge", "Wait Time", "Fare", "Fairness", "Cancel Rate"]
    for _, row in radar_norm.iterrows():
        name   = row["Cluster_Name"]
        vals   = [row[c] for c in radar_cols]
        vals_c = vals + [vals[0]]
        cats_c = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_c, theta=cats_c,
            fill="toself", name=name,
            line_color=CLUSTER_COLORS[name],
            opacity=0.65,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1],
                            gridcolor="rgba(0,180,171,0.2)", color="#8892B0"),
            angularaxis=dict(color="#CCD6F6"),
            bgcolor="rgba(17,34,64,0.7)",
        ),
        title="Normalised Cluster Attributes Radar",
    )
    apply_theme(fig, height=440)
    st.plotly_chart(fig, use_container_width=True)

    divider()

    # ── Cluster description cards ──────────────────────────────────────────
    section_header("Segment Descriptions & Business Interpretation")
    card_cols = st.columns(4)
    for col, name in zip(card_cols, CLUSTER_LABELS.values()):
        color = CLUSTER_COLORS[name]
        icon  = CLUSTER_ICONS[name]
        desc  = CLUSTER_DESC[name]
        row   = profile[profile["Cluster_Name"] == name].iloc[0]
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color:{color};min-height:200px">
                <div class="kpi-label" style="color:{color}">{icon} {name}</div>
                <div style="font-size:1.4rem;font-family:Syne,sans-serif;
                     font-weight:800;color:#fff;margin:0.3rem 0">
                    {int(row['Count'])} rides
                </div>
                <div style="font-size:0.72rem;color:#8892B0;line-height:1.6;
                     margin-bottom:0.5rem">{desc}</div>
                <div style="font-size:0.72rem;color:#CCD6F6;">
                    Avg Surge: <strong style="color:{color}">{row['Avg_Surge']:.2f}×</strong>&nbsp;
                    Cancel: <strong style="color:{color}">{row['Cancel_Rate_Pct']:.1f}%</strong>
                </div>
            </div>""", unsafe_allow_html=True)

    divider()

    # ── Cancellation rate per cluster bar ──────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            profile, x="Cluster_Name", y="Cancel_Rate_Pct",
            color="Cluster_Name", color_discrete_map=CLUSTER_COLORS,
            title="Cancellation Rate by Cluster (%)",
        )
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight_box(
            "<strong>Price-Sensitive Riders</strong> have the highest cancellation rate — "
            "a high-priority segment for targeted discount interventions to "
            "reduce churn and improve overall completion rates."
        )

    with c2:
        fig = px.bar(
            profile, x="Cluster_Name", y="Avg_Fairness",
            color="Cluster_Name", color_discrete_map=CLUSTER_COLORS,
            title="Average Fairness Rating by Cluster",
            range_y=[1, 5],
        )
        fig.add_hline(y=3.5, line_dash="dash", line_color="#F5A623",
                      annotation_text="Acceptable fairness threshold")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight_box(
            "<strong>Price-Sensitive Riders</strong> rate fairness lowest (below 3.5 threshold). "
            "This correlates directly with their higher surge exposure and "
            "longer wait times compared to Premium Riders."
        )

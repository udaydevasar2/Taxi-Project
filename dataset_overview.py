"""
pages/dataset_overview.py
──────────────────────────
Dataset preview, schema, missing values, quick charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.theme import apply_theme, section_header, divider, insight_box, PALETTE


def render(df: pd.DataFrame):
    section_header(
        "Dataset Overview",
        "3,000 synthetic ride records · 26 features · Dubai 2024",
    )

    # ── Top-level metrics ──────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows",           f"{len(df):,}")
    m2.metric("Columns",        str(len(df.columns)))
    m3.metric("Missing Values", str(int(df.isnull().sum().sum())))
    m4.metric("Cancelled",      f"{(df['Ride_Cancelled']=='Yes').mean()*100:.1f}%")
    m5.metric("Unique Riders",  f"{df['Customer_ID'].nunique():,}")

    divider()

    tab_preview, tab_schema, tab_charts, tab_stats = st.tabs([
        "📄 Data Preview",
        "🗂️ Schema",
        "📊 Quick Charts",
        "🔢 Statistics",
    ])

    # ── PREVIEW ────────────────────────────────────────────────────────────
    with tab_preview:
        n_rows = st.slider("Rows to preview", 5, 50, 15, key="ds_preview_slider")
        st.dataframe(df.head(n_rows), use_container_width=True, height=420)

    # ── SCHEMA ─────────────────────────────────────────────────────────────
    with tab_schema:
        schema_df = pd.DataFrame({
            "Column":      df.columns,
            "Dtype":       [str(df[c].dtype) for c in df.columns],
            "Unique":      [df[c].nunique() for c in df.columns],
            "Null Count":  [int(df[c].isnull().sum()) for c in df.columns],
            "Sample":      [str(df[c].iloc[0]) for c in df.columns],
            "Description": [
                "Unique ride identifier",
                "Unique customer identifier",
                "Customer age (18–66)",
                "Customer gender",
                "Customer nationality",
                "Income bracket (ordered)",
                "Loyalty tier (ordered)",
                "Pickup area name",
                "Drop-off area name",
                "Pickup zone category",
                "Time-of-day slot",
                "Day of week",
                "Whether a major event is nearby",
                "Current weather condition",
                "Trip distance in kilometres",
                "Estimated trip duration (minutes)",
                "Requested vehicle category",
                "Driver acceptance rate (0–1)",
                "Driver distance from pickup (km)",
                "Estimated wait time (minutes)",
                "Base fare before surge (AED)",
                "Surge price multiplier (1.0–4.0)",
                "Final fare after surge (AED)",
                "Whether a discount was applied",
                "Customer-perceived fairness (1–5)",
                "Target: was the ride cancelled?",
            ],
        })
        st.dataframe(schema_df, use_container_width=True, height=600)

    # ── QUICK CHARTS ───────────────────────────────────────────────────────
    with tab_charts:
        c1, c2 = st.columns(2)

        with c1:
            tod = df["Ride_Time_of_Day"].value_counts().reset_index()
            tod.columns = ["Time", "Count"]
            order = ["Morning Peak", "Midday", "Afternoon", "Evening Peak", "Late Night"]
            tod["Time"] = pd.Categorical(tod["Time"], categories=order, ordered=True)
            tod = tod.sort_values("Time")
            fig = px.bar(tod, x="Time", y="Count",
                         color="Count", color_continuous_scale=[PALETTE[1], PALETTE[0]],
                         title="Ride Requests by Time of Day")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "<strong>Evening Peak</strong> generates the highest ride volume (~28%), "
                "aligning with post-work commute patterns in Dubai. Late Night is lowest "
                "but commands the highest average fares."
            )

        with c2:
            vt = df["Vehicle_Type_Requested"].value_counts().reset_index()
            vt.columns = ["Type", "Count"]
            fig = px.pie(vt, names="Type", values="Count",
                         color_discrete_sequence=PALETTE,
                         title="Ride Distribution by Vehicle Type",
                         hole=0.42)
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "<strong>Economy</strong> dominates at ~45% — price-sensitive riders prefer "
                "affordable options. <strong>Luxury</strong> (~7%) commands the highest "
                "margins and is clustered in tourist and business zones."
            )

        c3, c4 = st.columns(2)

        with c3:
            cr = (
                df.groupby("Pickup_Zone")["Ride_Cancelled"]
                .apply(lambda x: (x == "Yes").mean() * 100)
                .reset_index()
            )
            cr.columns = ["Zone", "Cancel_Rate"]
            fig = px.bar(cr, x="Zone", y="Cancel_Rate",
                         color="Cancel_Rate",
                         color_continuous_scale=["#1DB954", "#F5A623", "#E84545"],
                         title="Cancellation Rate by Zone (%)")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "<strong>Tourist-Heavy zones</strong> show the highest cancellation rates "
                "— extreme surge acts as a demand suppressor. Residential areas suffer "
                "high abandonment from long wait times, not just price."
            )

        with c4:
            zone_surge = (
                df.groupby("Pickup_Zone")["Surge_Multiplier"]
                .mean()
                .reset_index()
            )
            zone_surge.columns = ["Zone", "Avg_Surge"]
            fig = px.bar(zone_surge, x="Zone", y="Avg_Surge",
                         color="Avg_Surge",
                         color_continuous_scale=["#1DB954", "#F5A623", "#E84545"],
                         title="Avg Surge Multiplier by Zone")
            fig.add_hline(
                y=1.5, line_dash="dash", line_color="#F5A623",
                annotation_text="Fairness threshold 1.5×",
                annotation_position="top right",
            )
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "Tourist-Heavy and Business zones consistently breach the "
                "<strong>1.5× fairness threshold</strong>, raising equity concerns "
                "particularly for non-tourist residents crossing zone boundaries."
            )

    # ── STATS ──────────────────────────────────────────────────────────────
    with tab_stats:
        st.markdown("**Numeric Summary**")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        st.dataframe(df[num_cols].describe().round(3), use_container_width=True)

        st.markdown("**Categorical Summary**")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        cat_summary = pd.DataFrame({
            "Column":    cat_cols,
            "Unique":    [df[c].nunique() for c in cat_cols],
            "Top Value": [df[c].mode()[0] for c in cat_cols],
            "Frequency": [int(df[c].value_counts().iloc[0]) for c in cat_cols],
            "Freq %":    [round(df[c].value_counts().iloc[0] / len(df) * 100, 1) for c in cat_cols],
        })
        st.dataframe(cat_summary, use_container_width=True)

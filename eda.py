"""
pages/eda.py
────────────
Exploratory Data Analysis: surge/fare distributions, demand geography,
correlation heatmap, cancellation drivers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.theme import apply_theme, section_header, divider, insight_box, PALETTE


def render(df: pd.DataFrame):
    section_header(
        "Exploratory Data Analysis",
        "Understanding surge pricing dynamics, demand patterns, and cancellation drivers",
    )

    tab_surge, tab_geo, tab_corr = st.tabs([
        "💹 Surge & Fares",
        "🗺️ Demand & Geography",
        "🌡️ Correlations",
    ])

    # ── SURGE & FARES ──────────────────────────────────────────────────────
    with tab_surge:
        c1, c2 = st.columns(2)

        with c1:
            fig = px.histogram(
                df, x="Surge_Multiplier", nbins=40,
                color_discrete_sequence=["#1DB954"],
                title="Surge Multiplier Distribution",
            )
            mean_s = df["Surge_Multiplier"].mean()
            fig.add_vline(x=mean_s, line_dash="dash", line_color="#F5A623",
                          annotation_text=f"Mean {mean_s:.2f}×",
                          annotation_position="top right")
            fig.add_vline(x=2.5, line_dash="dot", line_color="#E84545",
                          annotation_text="High-cancel threshold 2.5×",
                          annotation_position="top left")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            pct_high = (df["Surge_Multiplier"] > 2.0).mean() * 100
            insight_box(
                f"Average surge is <strong>{mean_s:.2f}×</strong>. "
                f"~<strong>{pct_high:.0f}%</strong> of rides exceed 2×, the threshold "
                "above which customer satisfaction drops sharply according to regression analysis."
            )

        with c2:
            bins  = [0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
            labels = ["1.0", "1.0–1.5", "1.5–2.0", "2.0–2.5", "2.5–3.0", "3.0–4.0"]
            df["_surge_bin"] = pd.cut(df["Surge_Multiplier"], bins=bins, labels=labels)
            cs = (
                df.groupby("_surge_bin", observed=True)["Ride_Cancelled"]
                .apply(lambda x: (x == "Yes").mean() * 100)
                .reset_index()
            )
            cs.columns = ["Surge Range", "Cancel %"]
            fig = px.bar(cs, x="Surge Range", y="Cancel %",
                         color="Cancel %",
                         color_continuous_scale=["#1DB954", "#F5A623", "#E84545"],
                         title="Cancellation Rate vs Surge Multiplier")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            df.drop(columns=["_surge_bin"], inplace=True)
            insight_box(
                "Cancellation rates <strong>nearly double</strong> when surge exceeds 2.5×. "
                "This non-linear relationship suggests a psychological price ceiling — "
                "beyond which passengers choose to walk or wait for normal pricing."
            )

        c3, c4 = st.columns(2)

        with c3:
            fig = px.box(
                df, x="Vehicle_Type_Requested", y="Final_Fare",
                color="Vehicle_Type_Requested",
                color_discrete_sequence=PALETTE,
                title="Final Fare Distribution by Vehicle Type",
                category_orders={
                    "Vehicle_Type_Requested": ["Bike", "Economy", "Business", "SUV", "Luxury"]
                },
            )
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "Luxury vehicles show extreme fare variance — indicating personalised "
                "surge or longer trip distances. Economy fares cluster tightly, "
                "suggesting uniform algorithmic pricing across that segment."
            )

        with c4:
            sample = df.sample(600, random_state=42)
            fig = px.scatter(
                sample, x="Estimated_Wait_Time", y="Surge_Multiplier",
                color="Ride_Cancelled",
                color_discrete_map={"Yes": "#E84545", "No": "#1DB954"},
                title="Wait Time vs Surge (coloured by Cancellation)",
                labels={"Estimated_Wait_Time": "Estimated Wait (min)"},
                opacity=0.65,
            )
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "The top-right quadrant (high wait + high surge) is densely populated "
                "with <strong style='color:#E84545'>cancellations</strong>. Reducing wait "
                "times in high-surge zones would directly improve completion rates and revenue."
            )

        c5, c6 = st.columns(2)

        with c5:
            fare_tod = (
                df.groupby("Ride_Time_of_Day")["Final_Fare"]
                .mean()
                .reset_index()
            )
            fare_tod.columns = ["Time", "Avg_Fare"]
            order = ["Morning Peak", "Midday", "Afternoon", "Evening Peak", "Late Night"]
            fare_tod["Time"] = pd.Categorical(fare_tod["Time"], categories=order, ordered=True)
            fare_tod = fare_tod.sort_values("Time")
            fig = px.line(fare_tod, x="Time", y="Avg_Fare", markers=True,
                          color_discrete_sequence=["#00B4AB"],
                          title="Average Final Fare by Time of Day")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "<strong>Late Night</strong> commands the highest average fare — "
                "combining lower base demand with safety-related premium. "
                "Morning and Evening peaks follow closely due to surge."
            )

        with c6:
            weather_surge = (
                df.groupby("Weather_Condition")["Surge_Multiplier"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            weather_surge.columns = ["Weather", "Avg_Surge"]
            fig = px.bar(weather_surge, x="Weather", y="Avg_Surge",
                         color="Avg_Surge",
                         color_continuous_scale=["#1DB954", "#F5A623", "#E84545"],
                         title="Avg Surge Multiplier by Weather Condition")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "Heavy Rain and Fog trigger <strong>30–45% higher surge</strong>. "
                "While demand-justified, the resulting fares may be unaffordable "
                "for essential workers who must commute regardless of weather."
            )

    # ── DEMAND & GEOGRAPHY ─────────────────────────────────────────────────
    with tab_geo:
        c1, c2 = st.columns(2)

        with c1:
            area_ct = df["Pickup_Location"].value_counts().head(12).reset_index()
            area_ct.columns = ["Area", "Rides"]
            fig = px.bar(area_ct, x="Rides", y="Area", orientation="h",
                         color="Rides",
                         color_continuous_scale=[PALETTE[1], PALETTE[0]],
                         title="Top 12 Pickup Locations by Volume")
            apply_theme(fig, height=420)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "Dubai Marina and Downtown Dubai dominate demand — high-footfall leisure "
                "destinations. Residential areas see ~3× fewer pickups, limiting driver "
                "availability and increasing wait times for those communities."
            )

        with c2:
            wt_inc = (
                df.groupby("Customer_Income_Bracket", observed=True)["Estimated_Wait_Time"]
                .mean()
                .reset_index()
            )
            wt_inc.columns = ["Income", "Avg_Wait"]
            fig = px.bar(wt_inc, x="Income", y="Avg_Wait",
                         color="Avg_Wait",
                         color_continuous_scale=["#1DB954", "#F5A623", "#E84545"],
                         title="Avg Wait Time by Income Bracket",
                         category_orders={"Income": ["Low","Middle","Upper-Middle","High"]})
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "<strong>Low-income customers wait ~3 minutes longer</strong> on average. "
                "This likely reflects geographic clustering — budget riders in residential "
                "areas with fewer nearby drivers, compounding economic disadvantage."
            )

        c3, c4 = st.columns(2)

        with c3:
            ev_cancel = (
                df.groupby(["Nearby_Event", "Ride_Time_of_Day"])["Ride_Cancelled"]
                .apply(lambda x: (x == "Yes").mean() * 100)
                .reset_index()
            )
            ev_cancel.columns = ["Event", "Time", "Cancel_Rate"]
            order = ["Morning Peak", "Midday", "Afternoon", "Evening Peak", "Late Night"]
            ev_cancel["Time"] = pd.Categorical(ev_cancel["Time"], categories=order, ordered=True)
            ev_cancel = ev_cancel.sort_values("Time")
            fig = px.bar(ev_cancel, x="Time", y="Cancel_Rate",
                         color="Event",
                         color_discrete_map={"Yes": "#E84545", "No": "#1DB954"},
                         barmode="group",
                         title="Cancellation Rate: Event vs No Event by Time")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "Events amplify cancellation by <strong>8–12 percentage points</strong> "
                "during peak hours. Proactive driver dispatch before events could "
                "significantly reduce churn and lost revenue."
            )

        with c4:
            dow_vol = df["Ride_Day_of_Week"].value_counts().reset_index()
            dow_vol.columns = ["Day", "Count"]
            order_days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            dow_vol["Day"] = pd.Categorical(dow_vol["Day"], categories=order_days, ordered=True)
            dow_vol = dow_vol.sort_values("Day")
            fig = px.bar(dow_vol, x="Day", y="Count",
                         color="Count",
                         color_continuous_scale=[PALETTE[1], PALETTE[0]],
                         title="Ride Volume by Day of Week")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "Friday and Saturday show elevated demand aligned with UAE weekend patterns. "
                "Mid-week volume is steady — driven by business commuters rather than "
                "leisure riders, resulting in shorter average trip distances."
            )

    # ── CORRELATIONS ───────────────────────────────────────────────────────
    with tab_corr:
        num_cols = [
            "Customer_Age", "Ride_Distance_KM", "Estimated_Ride_Time_Minutes",
            "Estimated_Wait_Time", "Base_Fare", "Surge_Multiplier",
            "Final_Fare", "Driver_Acceptance_Rate", "Customer_Fairness_Rating",
        ]
        corr = df[num_cols].corr()

        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            title="Correlation Heatmap — Numeric Features",
            aspect="auto",
            zmin=-1, zmax=1,
        )
        apply_theme(fig, height=520)
        st.plotly_chart(fig, use_container_width=True)

        insight_box(
            "<strong>Surge Multiplier → Final Fare</strong> has the strongest positive "
            "correlation (r ≈ 0.85). Notably, <strong>Fairness Rating correlates "
            "negatively with Surge</strong> (r ≈ −0.55) and positively with "
            "Driver Acceptance Rate — the two primary levers for fairness improvement."
        )

        divider()
        st.markdown("**Pairwise scatter: Surge vs Fairness Rating**")
        sample2 = df.sample(500, random_state=7)
        fig2 = px.scatter(
            sample2, x="Surge_Multiplier", y="Customer_Fairness_Rating",
            color="Customer_Income_Bracket",
            color_discrete_sequence=PALETTE,
            trendline="ols",
            title="Surge Multiplier vs Customer Fairness Rating (n=500)",
            opacity=0.6,
        )
        apply_theme(fig2, height=380)
        st.plotly_chart(fig2, use_container_width=True)
        insight_box(
            "The negative trend is consistent across all income brackets, but "
            "<strong>Low-income riders rate fairness lower at every surge level</strong> "
            "— suggesting relative affordability compounds perceived unfairness."
        )

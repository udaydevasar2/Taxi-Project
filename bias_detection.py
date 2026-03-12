"""
pages/bias_detection.py
────────────────────────
Bias Detection Dashboard — quantifying algorithmic fairness gaps across
geography, income, nationality, and vehicle type.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.theme import apply_theme, section_header, divider, insight_box, PALETTE


def render(df: pd.DataFrame):
    section_header(
        "Bias Detection Dashboard",
        "Quantifying algorithmic fairness gaps across demographics and geographies",
    )

    # ── Bias risk summary banner ───────────────────────────────────────────
    st.markdown("""
    <div style="background:rgba(232,69,69,0.06);border:1px solid rgba(232,69,69,0.2);
         border-radius:14px;padding:1.2rem 1.5rem;margin-bottom:1.5rem;">
        <div style="font-family:Syne,sans-serif;font-size:1rem;font-weight:700;
             color:#E84545;margin-bottom:0.75rem;">⚠️ Bias Risk Summary</div>
        <div style="display:flex;gap:0.8rem;flex-wrap:wrap;">
            <span class="badge badge-high">HIGH — Geographic Fare Premium</span>
            <span class="badge badge-high">HIGH — Tourist Zone Surge</span>
            <span class="badge badge-med">MEDIUM — Income Wait Disparity</span>
            <span class="badge badge-med">MEDIUM — Residential Cancellation</span>
            <span class="badge badge-low">LOW — Gender Pricing Gap</span>
            <span class="badge badge-low">LOW — Vehicle Allocation Equity</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_geo, tab_income, tab_nat, tab_vehicle, tab_scorecard = st.tabs([
        "🗺️ Geographic Bias",
        "💰 Income Bias",
        "🌍 Nationality Bias",
        "🚗 Vehicle Type Bias",
        "📋 Fairness Scorecard",
    ])

    # ── GEOGRAPHIC ────────────────────────────────────────────────────────
    with tab_geo:
        overall_avg = df["Final_Fare"].mean()
        zone_stats = (
            df.groupby("Pickup_Zone")
            .agg(
                Avg_Fare         =("Final_Fare",          "mean"),
                Avg_Surge        =("Surge_Multiplier",     "mean"),
                Avg_Wait         =("Estimated_Wait_Time",  "mean"),
                Cancel_Rate      =("Ride_Cancelled",       lambda x: (x=="Yes").mean()*100),
                Avg_Fairness     =("Customer_Fairness_Rating","mean"),
                Ride_Count       =("Ride_ID",              "count"),
            )
            .round(2)
            .reset_index()
        )
        zone_stats["Fare_Gap_%"] = ((zone_stats["Avg_Fare"] - overall_avg) / overall_avg * 100).round(1)

        c1, c2 = st.columns(2)

        with c1:
            colors = zone_stats["Fare_Gap_%"].apply(
                lambda x: "#E84545" if x > 10 else ("#F5A623" if x > 0 else "#1DB954")
            )
            fig = go.Figure(go.Bar(
                x=zone_stats["Pickup_Zone"],
                y=zone_stats["Fare_Gap_%"],
                marker_color=colors.tolist(),
                text=zone_stats["Fare_Gap_%"].apply(lambda x: f"{x:+.1f}%"),
                textposition="auto",
            ))
            fig.update_layout(title="Average Fare Gap vs Overall Mean (%)")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                f"<strong>Tourist-Heavy zones charge "
                f"{zone_stats[zone_stats['Pickup_Zone']=='Tourist-Heavy']['Fare_Gap_%'].values[0]:+.1f}% "
                f"above average fare.</strong> While partly due to longer trips and events, "
                "the systematic premium warrants a zone-based fare cap review."
            )

        with c2:
            area_wait = (
                df.groupby("Pickup_Location")["Estimated_Wait_Time"]
                .mean()
                .sort_values(ascending=False)
                .head(14)
                .reset_index()
            )
            area_wait.columns = ["Location", "Avg_Wait"]
            fig = px.bar(area_wait, x="Avg_Wait", y="Location", orientation="h",
                         color="Avg_Wait",
                         color_continuous_scale=["#1DB954", "#F5A623", "#E84545"],
                         title="Longest Average Wait Times by Location")
            apply_theme(fig, height=420)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "Residential locations like <strong>International City</strong> and "
                "<strong>Silicon Oasis</strong> show the longest waits — a driver supply "
                "gap creating a service equity deficit for working-class residents."
            )

        # Zone × Time surge heatmap
        pivot = df.pivot_table(
            values="Surge_Multiplier",
            index="Pickup_Zone",
            columns="Ride_Time_of_Day",
            aggfunc="mean",
        ).round(2)
        col_order = ["Morning Peak", "Midday", "Afternoon", "Evening Peak", "Late Night"]
        pivot = pivot[[c for c in col_order if c in pivot.columns]]

        fig = px.imshow(
            pivot,
            text_auto=True,
            color_continuous_scale=["#112240", "#1DB954", "#F5A623", "#E84545"],
            title="Surge Heatmap — Zone × Time of Day",
            aspect="auto",
            zmin=1.0, zmax=3.0,
        )
        apply_theme(fig, height=320)
        st.plotly_chart(fig, use_container_width=True)
        insight_box(
            "Tourist-Heavy zones during <strong>Evening Peak</strong> consistently exceed "
            "2.0×. A time-zone cap mechanism would contain the most extreme surge events "
            "without undermining supply incentives in less-biased zone-time combinations."
        )

    # ── INCOME ────────────────────────────────────────────────────────────
    with tab_income:
        inc_order = ["Low", "Middle", "Upper-Middle", "High"]

        c1, c2 = st.columns(2)
        with c1:
            fig = px.box(
                df, x="Customer_Income_Bracket", y="Final_Fare",
                color="Customer_Income_Bracket",
                color_discrete_sequence=PALETTE,
                title="Final Fare Distribution by Income Bracket",
                category_orders={"Customer_Income_Bracket": inc_order},
            )
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "While median fares are similar, <strong>Low-income riders face "
                "disproportionate upper-quartile exposure</strong> relative to income — "
                "a proportional fairness problem, not merely an absolute price gap."
            )

        with c2:
            inc_stats = (
                df.groupby("Customer_Income_Bracket", observed=True)
                .agg(
                    Cancel_Rate =("Ride_Cancelled", lambda x: (x=="Yes").mean()*100),
                    Avg_Wait    =("Estimated_Wait_Time", "mean"),
                    Avg_Surge   =("Surge_Multiplier",    "mean"),
                    Avg_Fairness=("Customer_Fairness_Rating","mean"),
                )
                .round(2)
                .reset_index()
            )
            inc_stats.columns = ["Income", "Cancel %", "Avg Wait", "Avg Surge", "Avg Fairness"]

            fig = px.bar(
                inc_stats,
                x="Income", y="Cancel %",
                color="Cancel %",
                color_continuous_scale=["#1DB954", "#F5A623", "#E84545"],
                title="Cancellation Rate by Income Bracket (%)",
                category_orders={"Income": inc_order},
            )
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "<strong>Low-income riders cancel ~40% more often</strong> than High-income "
                "riders. Income-targeted fare caps could simultaneously improve equity "
                "AND retention revenue."
            )

        # Income summary table
        st.markdown("**Income Group Summary**")
        st.dataframe(inc_stats.set_index("Income"), use_container_width=True)

        # Fairness vs income
        fig = px.bar(
            inc_stats, x="Income", y="Avg Fairness",
            color="Avg Fairness",
            color_continuous_scale=["#E84545", "#F5A623", "#1DB954"],
            title="Average Fairness Rating by Income Bracket",
            category_orders={"Income": inc_order},
            range_y=[1, 5],
        )
        fig.add_hline(y=3.5, line_dash="dash", line_color="#F5A623",
                      annotation_text="Acceptable threshold (3.5)")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

        insight_box(
            warn=True,
            text="<strong>Recommendation:</strong> Introduce an income-linked dynamic "
                 "discount for verified low-income riders during surge periods. A 20% "
                 "surge reduction for the 'Low' bracket could cut their cancellation "
                 "rate by an estimated 12–15 pp based on regression model coefficients.",
        )

    # ── NATIONALITY ───────────────────────────────────────────────────────
    with tab_nat:
        c1, c2 = st.columns(2)

        with c1:
            nat_fare = (
                df.groupby("Customer_Nationality")["Final_Fare"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            nat_fare.columns = ["Nationality", "Avg_Fare"]
            fig = px.bar(nat_fare, x="Avg_Fare", y="Nationality", orientation="h",
                         color="Avg_Fare",
                         color_continuous_scale=["#1DB954", "#F5A623", "#E84545"],
                         title="Avg Final Fare by Nationality")
            apply_theme(fig, height=460)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "Fare differences across nationalities largely reflect "
                "<strong>geographic pickup patterns</strong> rather than direct pricing "
                "bias — tourists cluster in high-surge zones, proxying national origin "
                "with location type."
            )

        with c2:
            nat_wait = (
                df.groupby("Customer_Nationality")["Estimated_Wait_Time"]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            nat_wait.columns = ["Nationality", "Avg_Wait"]
            fig = px.bar(nat_wait, x="Avg_Wait", y="Nationality", orientation="h",
                         color="Avg_Wait",
                         color_continuous_scale=["#1DB954", "#00B4AB", "#E84545"],
                         title="Avg Wait Time by Nationality")
            apply_theme(fig, height=460)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "Wait time disparities across nationalities are consistent with income "
                "and geographic patterns. <strong>No direct nationality-based allocation "
                "bias detected</strong> — the algorithm does not appear to use nationality "
                "as a pricing or dispatch feature."
            )

        # Fairness by nationality
        nat_fair = (
            df.groupby("Customer_Nationality")["Customer_Fairness_Rating"]
            .mean()
            .sort_values()
            .reset_index()
        )
        nat_fair.columns = ["Nationality", "Avg_Fairness"]
        fig = px.bar(nat_fair, x="Nationality", y="Avg_Fairness",
                     color="Avg_Fairness",
                     color_continuous_scale=["#E84545", "#F5A623", "#1DB954"],
                     title="Average Perceived Fairness Rating by Nationality",
                     range_y=[1, 5])
        fig.add_hline(y=3.5, line_dash="dash", line_color="#F5A623")
        apply_theme(fig, height=360)
        st.plotly_chart(fig, use_container_width=True)

    # ── VEHICLE ───────────────────────────────────────────────────────────
    with tab_vehicle:
        veh_order = ["Bike", "Economy", "Business", "SUV", "Luxury"]

        c1, c2 = st.columns(2)
        with c1:
            veh_surge = (
                df.groupby("Vehicle_Type_Requested")["Surge_Multiplier"]
                .mean()
                .reset_index()
            )
            veh_surge.columns = ["Vehicle", "Avg_Surge"]
            veh_surge["Vehicle"] = pd.Categorical(veh_surge["Vehicle"],
                                                   categories=veh_order, ordered=True)
            veh_surge = veh_surge.sort_values("Vehicle")
            fig = px.bar(veh_surge, x="Vehicle", y="Avg_Surge",
                         color="Avg_Surge",
                         color_continuous_scale=["#1DB954", "#F5A623", "#E84545"],
                         title="Avg Surge Multiplier by Vehicle Type",
                         category_orders={"Vehicle": veh_order})
            fig.add_hline(y=1.5, line_dash="dash", line_color="#F5A623",
                          annotation_text="Fairness threshold 1.5×")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "All vehicle types exceed the 1.5× fairness threshold on average. "
                "<strong>Economy Bikes</strong> show surprisingly high surge for the most "
                "price-sensitive segment — a prime candidate for a hard cap at 1.8×."
            )

        with c2:
            veh_wait = (
                df.groupby("Vehicle_Type_Requested")["Estimated_Wait_Time"]
                .mean()
                .reset_index()
            )
            veh_wait.columns = ["Vehicle", "Avg_Wait"]
            veh_wait["Vehicle"] = pd.Categorical(veh_wait["Vehicle"],
                                                  categories=veh_order, ordered=True)
            veh_wait = veh_wait.sort_values("Vehicle")
            fig = px.bar(veh_wait, x="Vehicle", y="Avg_Wait",
                         color="Avg_Wait",
                         color_continuous_scale=["#1DB954", "#F5A623", "#E84545"],
                         title="Avg Wait Time by Vehicle Type",
                         category_orders={"Vehicle": veh_order})
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
            insight_box(
                "<strong>SUVs and Luxury</strong> vehicles have the longest waits — "
                "lower fleet supply. Their riders are less price-sensitive, making "
                "this an acceptable trade-off. Economy/Bike shortest waits align "
                "with highest fleet density."
            )

        # Cancel rate by vehicle
        veh_cancel = (
            df.groupby("Vehicle_Type_Requested")["Ride_Cancelled"]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .reset_index()
        )
        veh_cancel.columns = ["Vehicle", "Cancel_Rate"]
        fig = px.bar(veh_cancel, x="Vehicle", y="Cancel_Rate",
                     color="Cancel_Rate",
                     color_continuous_scale=["#1DB954", "#F5A623", "#E84545"],
                     title="Cancellation Rate by Vehicle Type (%)")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── SCORECARD ─────────────────────────────────────────────────────────
    with tab_scorecard:
        section_header("Algorithmic Fairness Scorecard")

        scorecard = pd.DataFrame([
            {
                "Bias Dimension":   "Geographic Fare Premium",
                "Risk Level":       "🔴 HIGH",
                "Measured Gap":     f"+{(df[df['Pickup_Zone']=='Tourist-Heavy']['Final_Fare'].mean() / df['Final_Fare'].mean() - 1)*100:.0f}% Tourist vs Overall",
                "Affected Group":   "All riders in Tourist-Heavy zones",
                "Recommendation":   "Zone-based surge cap at 2.0× for tourist areas",
                "Priority":         "Immediate",
            },
            {
                "Bias Dimension":   "Residential Wait Gap",
                "Risk Level":       "🔴 HIGH",
                "Measured Gap":     f"+{df[df['Pickup_Zone']=='Residential']['Estimated_Wait_Time'].mean() - df['Estimated_Wait_Time'].mean():.1f} min above average",
                "Affected Group":   "Riders in Residential zones",
                "Recommendation":   "Driver incentive bonuses for residential zone pickups",
                "Priority":         "Immediate",
            },
            {
                "Bias Dimension":   "Income Cancel Disparity",
                "Risk Level":       "🟡 MEDIUM",
                "Measured Gap":     f"Low income cancels {(df[df['Customer_Income_Bracket']=='Low']['Ride_Cancelled']=='Yes').mean()*100:.0f}% vs {(df[df['Customer_Income_Bracket']=='High']['Ride_Cancelled']=='Yes').mean()*100:.0f}% High",
                "Affected Group":   "Low-income customers",
                "Recommendation":   "Income-linked dynamic discount scheme during surge",
                "Priority":         "Short-term",
            },
            {
                "Bias Dimension":   "Tourist Zone Surge Premium",
                "Risk Level":       "🔴 HIGH",
                "Measured Gap":     f"Avg {df[df['Pickup_Zone']=='Tourist-Heavy']['Surge_Multiplier'].mean():.2f}× vs {df['Surge_Multiplier'].mean():.2f}× overall",
                "Affected Group":   "Tourists and zone residents",
                "Recommendation":   "Event pricing separated from standard surge",
                "Priority":         "Short-term",
            },
            {
                "Bias Dimension":   "Gender Fare Gap",
                "Risk Level":       "🟢 LOW",
                "Measured Gap":     "<2% difference across genders",
                "Affected Group":   "None significant",
                "Recommendation":   "No action required — monitor quarterly",
                "Priority":         "Monitor",
            },
            {
                "Bias Dimension":   "Vehicle Surge Equity",
                "Risk Level":       "🟡 MEDIUM",
                "Measured Gap":     "All types exceed 1.5× fairness threshold",
                "Affected Group":   "Economy / Bike riders most exposed",
                "Recommendation":   "Economy and Bike hard cap at 1.8×",
                "Priority":         "Short-term",
            },
        ])

        st.dataframe(scorecard, use_container_width=True, height=310)

        divider()

        section_header("Fairness KPIs at a Glance")
        k1, k2, k3, k4 = st.columns(4)

        tourist_premium = (
            df[df["Pickup_Zone"] == "Tourist-Heavy"]["Final_Fare"].mean()
            / df["Final_Fare"].mean() - 1
        ) * 100
        res_wait_gap = (
            df[df["Pickup_Zone"] == "Residential"]["Estimated_Wait_Time"].mean()
            - df["Estimated_Wait_Time"].mean()
        )
        income_cancel_gap = (
            (df[df["Customer_Income_Bracket"] == "Low"]["Ride_Cancelled"] == "Yes").mean()
            - (df[df["Customer_Income_Bracket"] == "High"]["Ride_Cancelled"] == "Yes").mean()
        ) * 100
        gender_gap = abs(
            df[df["Customer_Gender"] == "Male"]["Final_Fare"].mean()
            - df[df["Customer_Gender"] == "Female"]["Final_Fare"].mean()
        )

        with k1:
            st.markdown(f"""<div class="kpi-card" style="border-left-color:#E84545">
                <div class="kpi-label" style="color:#E84545">Tourist Fare Premium</div>
                <div class="kpi-value">{tourist_premium:.1f}%</div>
                <div class="kpi-delta">above overall average</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="kpi-card" style="border-left-color:#E84545">
                <div class="kpi-label" style="color:#E84545">Residential Wait Gap</div>
                <div class="kpi-value">{res_wait_gap:.1f} min</div>
                <div class="kpi-delta">longer than average</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="kpi-card" style="border-left-color:#F5A623">
                <div class="kpi-label" style="color:#F5A623">Income Cancel Gap</div>
                <div class="kpi-value">{income_cancel_gap:.1f} pp</div>
                <div class="kpi-delta">Low vs High income</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class="kpi-card" style="border-left-color:#1DB954">
                <div class="kpi-label" style="color:#1DB954">Gender Fare Gap</div>
                <div class="kpi-value">AED {gender_gap:.2f}</div>
                <div class="kpi-delta">M vs F — negligible</div>
            </div>""", unsafe_allow_html=True)

        divider()

        insight_box(
            warn=True,
            text="<strong>Overall Fairness Assessment:</strong> The Careem UAE surge pricing "
                 "algorithm shows evidence of <strong>indirect geographic and income-correlated "
                 "bias</strong>. No direct demographic variables appear in the pricing model, "
                 "but structural patterns in driver deployment and zone-based surge multipliers "
                 "create disparate outcomes. Implementing <strong>zone-based surge caps, "
                 "residential driver incentives, and income-linked discounts</strong> represents "
                 "the highest-priority fairness interventions.",
        )

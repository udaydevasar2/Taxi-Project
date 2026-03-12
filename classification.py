"""
pages/classification.py
────────────────────────
Classification models to predict Ride_Cancelled.
Models: Logistic Regression, Decision Tree, Random Forest, XGBoost (if available).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

from utils.theme import apply_theme, section_header, divider, insight_box, PALETTE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ── Feature engineering ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def prepare_classification_data(df: pd.DataFrame):
    """Encode features and split into train/test sets."""
    feature_cols = [
        "Customer_Age", "Ride_Distance_KM", "Estimated_Ride_Time_Minutes",
        "Estimated_Wait_Time", "Base_Fare", "Surge_Multiplier",
        "Final_Fare", "Driver_Acceptance_Rate", "Driver_Distance_to_Pickup",
        "Customer_Fairness_Rating",
    ]
    cat_cols = [
        "Vehicle_Type_Requested", "Customer_Income_Bracket",
        "Customer_Loyalty_Status", "Ride_Time_of_Day",
        "Pickup_Zone", "Nearby_Event", "Weather_Condition",
    ]

    X = df[feature_cols].copy()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(df[col].astype(str))

    y = (df["Ride_Cancelled"] == "Yes").astype(int)
    all_features = feature_cols + cat_cols

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    return X_tr, X_te, y_tr, y_te, all_features


@st.cache_data(show_spinner=False)
def train_models(_X_tr, _X_te, _y_tr, _y_te):
    """Train all classifiers and return performance metrics."""
    SS = StandardScaler()
    X_tr_s = SS.fit_transform(_X_tr)
    X_te_s  = SS.transform(_X_te)

    classifiers = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=42,
                                                    class_weight="balanced"), True),
        "Decision Tree":       (DecisionTreeClassifier(max_depth=6, random_state=42,
                                                       class_weight="balanced"), False),
        "Random Forest":       (RandomForestClassifier(n_estimators=120, max_depth=8,
                                                        n_jobs=-1, random_state=42,
                                                        class_weight="balanced"), False),
    }
    if HAS_XGB:
        n_neg = int((_y_tr == 0).sum())
        n_pos = int((_y_tr == 1).sum())
        classifiers["XGBoost"] = (
            XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1,
                          scale_pos_weight=n_neg / max(n_pos, 1),
                          eval_metric="logloss",
                          random_state=42, n_jobs=-1),
            False,
        )

    results = {}
    for name, (model, scaled) in classifiers.items():
        Xtr = X_tr_s if scaled else _X_tr
        Xte = X_te_s if scaled else _X_te
        model.fit(Xtr, _y_tr)
        yp = model.predict(Xte)
        results[name] = {
            "Accuracy":  round(accuracy_score(_y_te, yp)  * 100, 2),
            "Precision": round(precision_score(_y_te, yp) * 100, 2),
            "Recall":    round(recall_score(_y_te, yp)    * 100, 2),
            "F1":        round(f1_score(_y_te, yp)        * 100, 2),
            "CM":        confusion_matrix(_y_te, yp),
            "model":     model,
            "yp":        yp,
            "scaled":    scaled,
        }
    return results


def render(df: pd.DataFrame):
    section_header(
        "Classification Models",
        "Predicting Ride_Cancelled using machine learning",
    )

    with st.spinner("⚙️ Training classifiers — please wait…"):
        X_tr, X_te, y_tr, y_te, feature_names = prepare_classification_data(df)
        results = train_models(X_tr, X_te, y_tr, y_te)

    # ── Metrics table & comparison bar ────────────────────────────────────
    metrics_df = pd.DataFrame({
        name: {k: v for k, v in r.items() if k in ["Accuracy","Precision","Recall","F1"]}
        for name, r in results.items()
    }).T
    metrics_df.index.name = "Model"

    c1, c2 = st.columns([1.3, 1])

    with c1:
        section_header("Metric Comparison")
        melt = metrics_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.bar(melt, x="Model", y="Score", color="Metric",
                     barmode="group",
                     color_discrete_sequence=PALETTE,
                     title="Accuracy / Precision / Recall / F1 (%)",
                     range_y=[50, 100])
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        section_header("Results Table")
        st.dataframe(
            metrics_df.style.format("{:.2f}").background_gradient(cmap="Greens", axis=None),
            use_container_width=True,
            height=220,
        )
        best_model = metrics_df["F1"].idxmax()
        best_f1    = metrics_df.loc[best_model, "F1"]
        insight_box(
            f"🏆 <strong>{best_model}</strong> achieves the highest F1 score "
            f"(<strong>{best_f1:.1f}%</strong>). It is the recommended model for "
            "production deployment to predict and prevent ride cancellations."
        )

    divider()

    # ── Confusion matrices ─────────────────────────────────────────────────
    section_header("Confusion Matrices")
    cm_cols = st.columns(len(results))
    for col, (name, res) in zip(cm_cols, results.items()):
        with col:
            cm_arr = res["CM"]
            labels = ["Completed", "Cancelled"]
            fig = ff.create_annotated_heatmap(
                z=cm_arr, x=labels, y=labels,
                colorscale=[[0, "#112240"], [1, "#1DB954"]],
                showscale=False,
                font_colors=["white", "white"],
            )
            fig.update_layout(
                title=dict(text=name, font=dict(size=13)),
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=260,
                margin=dict(l=5, r=5, t=40, b=5),
            )
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    divider()

    # ── Feature importances (Random Forest) ───────────────────────────────
    section_header("Feature Importances — Random Forest")
    rf_model = results["Random Forest"]["model"]
    fi = (
        pd.DataFrame({"Feature": feature_names, "Importance": rf_model.feature_importances_})
        .sort_values("Importance", ascending=True)
        .tail(15)
    )
    fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                 color="Importance",
                 color_continuous_scale=[PALETTE[1], PALETTE[0]],
                 title="Top Feature Importances (Random Forest)")
    apply_theme(fig, height=460)
    st.plotly_chart(fig, use_container_width=True)
    insight_box(
        "<strong>Surge_Multiplier</strong> and <strong>Final_Fare</strong> are the top "
        "predictors of cancellation. <strong>Estimated_Wait_Time</strong> ranks third, "
        "validating driver supply as a key retention lever alongside pricing strategy."
    )

    divider()

    # ── F1 radar chart ─────────────────────────────────────────────────────
    section_header("Model Performance Radar")
    categories = ["Accuracy", "Precision", "Recall", "F1"]
    fig = go.Figure()
    for i, (name, res) in enumerate(results.items()):
        vals = [res[m] for m in categories]
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill="toself", name=name,
            line_color=PALETTE[i],
            opacity=0.6,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[50, 100],
                            gridcolor="rgba(0,180,171,0.2)", color="#8892B0"),
            angularaxis=dict(color="#CCD6F6"),
            bgcolor="rgba(17,34,64,0.7)",
        ),
        title="Classification Metrics Radar",
        showlegend=True,
    )
    apply_theme(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)

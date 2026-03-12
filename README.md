# 🚖 Careem UAE — Surge Pricing Fairness Analytics Dashboard

> **University Data Analytics Project**  
> An end-to-end data science investigation into algorithmic fairness in ride-hailing surge pricing.

---

## 🔗 Live App
Deploy to [Streamlit Cloud](https://streamlit.io/cloud) by pushing this repo and connecting it.

---

## 📋 Overview

This Streamlit dashboard analyzes whether **Careem UAE's** surge pricing strategy treats all customers fairly across locations, income groups, and demographics. It uses a 3,000-record synthetic dataset generated with realistic Dubai ride-hailing demand patterns and intentional bias signals.

---

## 🗂️ Dashboard Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview, KPIs, business context |
| 📊 Dataset Overview | Schema, preview, quick charts |
| 🔍 EDA & Visualizations | Surge distributions, demand maps, correlations |
| 🤖 Classification Models | Logistic Regression, Decision Tree, Random Forest, XGBoost |
| 🔵 Clustering Analysis | K-Means customer segmentation (4 clusters) |
| 🔗 Association Rule Mining | Apriori patterns — surge, cancellation, events |
| 📈 Regression Forecast | Predict Customer Fairness Rating |
| ⚖️ Bias Detection | Geographic, income, nationality, vehicle type bias |

---

## 🚀 Deployment

### Local
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud
1. Push `app.py` and `requirements.txt` to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → select `app.py` → Deploy

---

## 📦 Dataset

The dataset is **generated automatically** at runtime — no external files needed.

- **3,000** synthetic ride requests
- **26 columns** covering rider demographics, trip details, surge pricing, and outcomes
- **Bias patterns embedded**: tourist-zone premiums, residential wait gaps, income-correlated cancellations
- **Target variable**: `Ride_Cancelled` (Yes / No)

---

## 🛠️ Tech Stack

- **Streamlit** — dashboard framework
- **Pandas / NumPy** — data manipulation
- **Scikit-learn** — ML models (classification, clustering, regression)
- **XGBoost** — gradient boosting classifier
- **Mlxtend** — association rule mining (Apriori)
- **Plotly** — interactive visualizations

---

## 🎨 Design

Dark theme inspired by professional BI dashboards:
- Careem brand colors (green `#1DB954`, teal `#00B4AB`)
- Syne (display) + DM Sans (body) typography
- KPI cards, insight callouts, bias scorecard

---

*Synthetic data only. No real customer or business data used.*

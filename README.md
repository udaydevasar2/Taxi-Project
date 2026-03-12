# 🚖 Careem UAE — Surge Pricing Fairness Analytics

> **University Data Analytics Project**  
> End-to-end investigation into algorithmic fairness in Dubai ride-hailing surge pricing.

---

## 🗂️ Repository Structure

```
careem-surge-analytics/
├── app.py                        ← Main entry point
├── requirements.txt              ← Python dependencies
├── README.md
│
├── data/
│   ├── __init__.py
│   └── generate_data.py          ← Synthetic dataset generator (3,000 records)
│
├── utils/
│   ├── __init__.py
│   └── theme.py                  ← Shared CSS, Plotly theme, UI helpers
│
└── pages/
    ├── __init__.py
    ├── home.py                   ← Hero banner, KPIs, business context
    ├── dataset_overview.py       ← Preview, schema, quick charts
    ├── eda.py                    ← Surge, fare, demand, correlation charts
    ├── classification.py         ← LR, DT, RF, XGBoost — predict Ride_Cancelled
    ├── clustering.py             ← K-Means (K=4) customer segmentation
    ├── association_rules.py      ← Apriori pattern mining
    ├── regression.py             ← Predict Customer Fairness Rating
    └── bias_detection.py         ← Geographic, income, nationality, vehicle bias
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/careem-surge-analytics.git
cd careem-surge-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repository to GitHub (public or private)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → Select your repo → Set main file to `app.py`
4. Click **Deploy**

The dataset is **generated automatically at runtime** — no CSV files needed.

---

## 📊 Dashboard Pages

| Page | What it shows |
|------|--------------|
| 🏠 **Home** | Hero banner, 6 KPIs, business context, bias preview |
| 📊 **Dataset Overview** | Schema, data preview, quick charts, statistics |
| 🔍 **EDA & Visualizations** | Surge/fare/demand/weather/correlation charts |
| 🤖 **Classification** | LR · DT · RF · XGBoost — confusion matrices, feature importance |
| 🔵 **Clustering** | K-Means segments: Price-Sensitive, Commuter, Premium, Event-Based |
| 🔗 **Association Rules** | Apriori patterns — live-mined or fallback table |
| 📈 **Regression** | Linear · Ridge · Lasso — fairness rating prediction + coefficients |
| ⚖️ **Bias Detection** | Zone · income · nationality · vehicle — scorecard + KPIs |

---

## 🗄️ Dataset

Synthetic 3,000-record ride-hailing dataset for Dubai 2024.

**Key bias patterns embedded:**
- Tourist-Heavy zones: +15–25% fare premium via higher surge
- Residential zones: +3.5 min wait due to lower driver acceptance rate
- Low-income riders: 40% higher cancellation rate from price sensitivity
- Extreme surge (>2.5×): cancellation probability nearly doubles

**Target variable:** `Ride_Cancelled` (Yes / No)

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| Streamlit | Dashboard framework |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | Classification · Clustering · Regression |
| XGBoost | Gradient boosting classifier |
| Mlxtend | Apriori association rule mining |
| Plotly | All interactive visualisations |

---

*All data is synthetic. No real Careem customer or business data was used.*

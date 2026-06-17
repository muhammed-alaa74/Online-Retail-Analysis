import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
import os
import sys

# ─── Path Setup ───────────────────────────────────────────────────────────────
# Works whether running from root or from app/ directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(ROOT, "data", "final_customer_data.csv")
MODEL_PATH = os.path.join(ROOT, "models", "supervised_model.pkl")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Executive Sales Intelligence",
    page_icon="🛍️",
    layout="wide"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f1f5f9; }
    [data-testid="stMetricValue"] {
        color: #1e3a8a !important;
        font-size: 1.8rem !important;
        font-weight: bold !important;
    }
    [data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-top: 4px solid #1e3a8a;
    }
    h1, h2, h3 { color: #1e3a8a; font-family: 'Segoe UI', sans-serif; }
    .stButton > button {
        background-color: #1e3a8a;
        color: white;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        width: 100%;
    }
    .prediction-box {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #f0f4ff, #e8edff);
        border-radius: 16px;
        border: 2px solid #1e3a8a;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data & Model Loading ──────────────────────────────────────────────────────
@st.cache_resource
def load_data_and_model():
    if not os.path.exists(DATA_PATH):
        st.error(
            "⚠️ Dataset not found!\n\n"
            "Please run `notebooks/01_EDA.ipynb` first to generate `data/final_customer_data.csv`."
        )
        st.stop()

    if not os.path.exists(MODEL_PATH):
        st.error(
            "⚠️ Model not found!\n\n"
            "Please run `notebooks/02_Supervised_Model.ipynb` to train and save the model."
        )
        st.stop()

    cols = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'TotalPrice', 'StockCode', 'Country', 'Hour']
    df = pd.read_csv(DATA_PATH, usecols=cols, low_memory=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Month_Name'] = df['InvoiceDate'].dt.month_name()
    df['Month_Year'] = df['InvoiceDate'].dt.to_period('M').astype(str)

    ref_date = df['InvoiceDate'].max()
    df_cust = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (ref_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum'),
        Variety=('StockCode', 'nunique')
    ).reset_index()

    model = joblib.load(MODEL_PATH)
    return df, df_cust, model


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛍️ Navigation")
    page = st.radio("Select View:", ["📊 Executive Summary", "🤖 Customer Value Predictor"])
    st.markdown("---")
    st.caption("Executive Sales Intelligence System\nPowered by Machine Learning")


df_raw, df_cust, model = load_data_and_model()


# ─── PAGE 1: Executive Summary ────────────────────────────────────────────────
if page == "📊 Executive Summary":
    st.title("📊 Executive Sales Dashboard")
    st.markdown("Real-time insights from retail transactions.")
    st.markdown("---")

    total_rev     = df_raw['TotalPrice'].sum()
    peak_month    = df_raw.groupby('Month_Name')['TotalPrice'].sum().idxmax()
    avg_basket    = df_raw.groupby('InvoiceNo')['TotalPrice'].sum().mean()
    total_customers = df_cust['CustomerID'].nunique()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("💰 Total Revenue",    f"${total_rev:,.0f}")
    m2.metric("👥 Total Customers",  f"{total_customers:,}")
    m3.metric("🏆 Peak Month",       peak_month)
    m4.metric("🛒 Avg. Basket Value",f"${avg_basket:.2f}")

    st.markdown("---")

    c1, c2 = st.columns(2)
    with c1:
        top_countries = (
            df_raw.groupby('Country')['TotalPrice']
            .sum().nlargest(10).reset_index()
        )
        fig = px.bar(top_countries, x='TotalPrice', y='Country', orientation='h',
                     title="🌍 Top 10 Countries by Revenue",
                     color='TotalPrice', color_continuous_scale='Blues',
                     labels={'TotalPrice': 'Revenue ($)', 'Country': ''})
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        hour_stats = df_raw.groupby('Hour')['InvoiceNo'].nunique().reset_index()
        fig = px.line(hour_stats, x='Hour', y='InvoiceNo',
                      title="⏰ Peak Transaction Hours", markers=True,
                      labels={'InvoiceNo': 'Transactions', 'Hour': 'Hour of Day'})
        fig.update_traces(line_color='#1e3a8a', marker_color='#1e3a8a')
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Monthly Revenue Trends")
    monthly = (
        df_raw.groupby('Month_Year')['TotalPrice']
        .sum().reset_index().sort_values('Month_Year')
    )
    fig = px.area(monthly, x='Month_Year', y='TotalPrice',
                  title="Revenue Growth Over Time",
                  color_discrete_sequence=['#1e3a8a'],
                  labels={'TotalPrice': 'Revenue ($)', 'Month_Year': 'Month'})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


# ─── PAGE 2: CLV Predictor ────────────────────────────────────────────────────
elif page == "🤖 Customer Value Predictor":
    st.title("🤖 Customer Lifetime Value Predictor")
    st.markdown("Enter customer behavior metrics to estimate their **predicted future spend**.")
    st.markdown("---")

    st.subheader("📝 Customer Metrics Input")
    p1, p2, p3 = st.columns(3)
    with p1: rec  = st.number_input("📅 Recency (Days)",   min_value=0,   max_value=500, value=30)
    with p2: freq = st.number_input("🔁 Frequency (Orders)",min_value=1,  max_value=500, value=5)
    with p3: var  = st.number_input("📦 Variety (Items)",  min_value=1,   max_value=500, value=10)

    m_val = st.number_input("💵 Current Monetary Value ($)", min_value=0.0, max_value=100_000.0, value=500.0, step=50.0)

    if st.button("🚀 Generate CLV Prediction"):
        try:
            prediction = model.predict(np.array([[rec, freq, var, m_val]]))[0]
        except ValueError:
            prediction = model.predict(np.array([[rec, freq, var, m_val, 0]]))[0]

        st.markdown(f"""
        <div class="prediction-box">
            <h3 style="color:#475569; margin-bottom:5px;">Predicted Future Monetary Value</h3>
            <h1 style="font-size:64px; color:#1e3a8a; margin:10px 0;">${prediction:,.2f}</h1>
            <p style="color:#64748b;">Based on RFV analysis and historical purchase patterns</p>
        </div>
        """, unsafe_allow_html=True)

        if prediction > m_val:
            st.success(f"📈 Expected to **increase** spend by ${prediction - m_val:,.2f}")
        elif prediction < m_val * 0.8:
            st.warning("📉 May **decrease** activity — consider a retention campaign.")
        else:
            st.info("➡️ Expected to maintain a **stable** spending level.")

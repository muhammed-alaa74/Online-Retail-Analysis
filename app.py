import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Executive Sales Intelligence", layout="wide")

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
    .stButton>button { 
        background-color: #1e3a8a; 
        color: white; 
        border-radius: 8px; 
        height: 3em;
        font-weight: bold;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_data():
    cols = ['CustomerID', 'InvoiceDate', 'InvoiceNo', 'TotalPrice', 'StockCode', 'Country', 'Hour']
    df = pd.read_csv('final_customer_data.csv', usecols=cols, low_memory=True)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Month_Name'] = df['InvoiceDate'].dt.month_name()
    df['Month_Year'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    
    ref_date = df['InvoiceDate'].max()
    df_cust = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (ref_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum',
        'StockCode': 'nunique'
    }).reset_index()
    df_cust.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Variety']
    
    model = joblib.load('supervised_model.pkl')
    return df, df_cust, model

try:
    df_raw, df_cust, model = load_data()

    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Select View:", ["Executive Summary", "Customer Value Predictor"])
        st.markdown("---")
        st.write("Professional Sales Analytics System")

    if page == "Executive Summary":
        st.title("Executive Sales Dashboard")
        
        m1, m2, m3, m4 = st.columns(4)
        total_rev = df_raw['TotalPrice'].sum()
        peak_month = df_raw.groupby('Month_Name')['TotalPrice'].sum().idxmax()
        avg_basket = df_raw.groupby('InvoiceNo')['TotalPrice'].sum().mean()
        
        m1.metric("Total Revenue", f"${total_rev:,.0f}")
        m2.metric("Total Customers", f"{df_cust['CustomerID'].nunique():,}")
        m3.metric("Peak Sales Month", peak_month)
        m4.metric("Avg. Basket Value", f"${avg_basket:.2f}")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            top_c = df_raw.groupby('Country')['TotalPrice'].sum().nlargest(10).reset_index()
            fig_country = px.bar(top_c, x='TotalPrice', y='Country', orientation='h', 
                                 title="Top 10 Countries by Revenue",
                                 color='TotalPrice', color_continuous_scale='Blues')
            st.plotly_chart(fig_country, use_container_width=True)
        
        with c2:
            hour_s = df_raw.groupby('Hour')['InvoiceNo'].nunique().reset_index()
            fig_hour = px.line(hour_s, x='Hour', y='InvoiceNo', title="Peak Transaction Hours",
                               markers=True)
            fig_hour.update_traces(line_color='#1e3a8a')
            st.plotly_chart(fig_hour, use_container_width=True)

        st.subheader("Monthly Revenue Trends")
        monthly_sales = df_raw.groupby('Month_Year')['TotalPrice'].sum().reset_index()
        fig_trend = px.area(monthly_sales, x='Month_Year', y='TotalPrice', 
                            title="Revenue Growth Over Time",
                            color_discrete_sequence=['#1e3a8a'])
        st.plotly_chart(fig_trend, use_container_width=True)

    elif page == "Customer Value Predictor":
        st.title("Customer Lifetime Value Predictor")
        st.write("Input customer behavior metrics to estimate future monetary value.")
        
        with st.container():
            st.markdown('<div style="background-color:white; padding:30px; border-radius:15px; border:1px solid #e2e8f0;">', unsafe_allow_html=True)
            p1, p2, p3 = st.columns(3)
            with p1: rec = st.number_input("Recency (Days)", 0, 500, 30)
            with p2: freq = st.number_input("Frequency (Orders)", 1, 500, 5)
            with p3: var = st.number_input("Variety (Items)", 1, 500, 10)
            
            m_val = st.number_input("Current Monetary Value ($)", 0.0, 100000.0, 500.0)
            
            if st.button("Generate Prediction"):
                try:
                    prediction = model.predict([[rec, freq, var, m_val]])
                except:
                    prediction = model.predict([[rec, freq, var, m_val, 0]])
                
                st.markdown(f"""
                    <div style='text-align:center; padding:20px;'>
                        <h2 style='color:#1e3a8a;'>Predicted Future Value</h2>
                        <h1 style='font-size:60px;'>${prediction[0]:,.2f}</h1>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"System Error: {e}")
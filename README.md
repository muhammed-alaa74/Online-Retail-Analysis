# 🛍️ Executive Sales Intelligence & Customer CLV Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-Visualization-3F4F75?logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
</p>

A comprehensive Data Science project that transforms raw transactional data into actionable business intelligence. The system integrates **Exploratory Data Analysis (EDA)**, **Customer Segmentation (RFM)**, and **Supervised Machine Learning** to predict Customer Lifetime Value (CLV), deployed as an interactive Streamlit dashboard.

---

## 🚀 Live Demo

> **[🔗 Launch App on Streamlit Cloud](https://your-streamlit-link-here)**

---

## 📌 Project Overview

This project analyzes a UK-based online retail dataset (~500K transactions) and builds a complete data science pipeline from raw data to a deployed predictive model.

### Business Questions Answered:
- What are the peak revenue months and high-value customer segments?
- Which countries generate the most revenue?
- What are the busiest transaction hours during the day?
- What is the predicted **future monetary value** of a customer given their behavior?

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📊 Executive Dashboard | Real-time KPIs: Total Revenue, Total Customers, Avg. Basket Value |
| 🗓️ Peak Sales Analysis | Identifies highest-performing months with visual trends |
| 🌍 Geo Insights | Top 10 countries by revenue with interactive bar chart |
| ⏰ Hourly Patterns | Transaction density across hours of the day |
| 📈 Revenue Trends | Monthly area chart showing revenue growth over time |
| 🤖 AI CLV Predictor | Predicts future customer spend using RFV (Recency, Frequency, Variety) |

---

## 🧠 ML Pipeline

```
Raw Data (UCI Online Retail)
        │
        ▼
   Data Cleaning & Feature Engineering
   (Remove nulls, returns, outliers → TotalPrice, Hour)
        │
        ▼
   RFM Analysis
   (Recency | Frequency | Monetary | Variety per Customer)
        │
        ├──► Unsupervised: KMeans Segmentation (Customer Clusters)
        │
        └──► Supervised: Regression Model → Predict Future Monetary Value
                    (Input: Recency, Frequency, Variety, Current Monetary)
                    (Output: Predicted Future Spend)
```

**Model Performance (Supervised Regression):**
- Algorithm: Gradient Boosting / Random Forest Regressor
- Target: Future Monetary Value
- Features: Recency, Frequency, Variety, Monetary

---

## 🗂️ Project Structure

```
Online-Retail-Analysis/
│
├── notebooks/
│   ├── 01_EDA.ipynb                        # Data exploration, cleaning & feature engineering
│   ├── 02_Supervised_Model.ipynb           # CLV regression model training & evaluation
│   └── 03_Unsupervised_Segmentation.ipynb  # KMeans customer segmentation
│
├── app/
│   └── app.py                              # Streamlit dashboard application
│
├── models/
│   └── supervised_model.pkl                # Trained & serialized ML model
│
├── data/
│   └── README.md                           # Dataset info & download instructions
│
├── assets/                                 # Screenshots & demo images
├── requirements.txt                        # Python dependencies
├── .gitignore
└── README.md
```

> **Note:** Raw data files (`*.csv`, `*.xlsx`) are excluded from version control.  
> Run `notebooks/01_EDA.ipynb` to generate `data/final_customer_data.csv`, or see [`data/README.md`](data/README.md) for download instructions.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Dashboard | Streamlit |
| Data | Pandas, NumPy |
| Visualization | Plotly Express |
| ML | Scikit-Learn |
| Serialization | Joblib |

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/muhammed-alaa74/Online-Retail-Analysis.git
cd Online-Retail-Analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the processed dataset (run notebook first)
# Open Notebooks/EDA.ipynb and run all cells → saves final_customer_data.csv

# 4. Launch the dashboard
streamlit run app.py
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [UCI ML Repository - Online Retail](https://archive.ics.uci.edu/ml/datasets/online+retail) |
| Period | 01/12/2010 – 09/12/2011 |
| Records | ~541,909 transactions |
| Countries | 38 |
| Features | InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country |

---

## 👤 Author

**Muhammed Alaa**
- 🎓 CS Student @ Zagazig University, Faculty of Computers & Information
- 💼 AI Co-Lead @ ZagEng Student Community
- 🔗 [GitHub](https://github.com/muhammed-alaa74)

---

## 📄 License

This project is licensed under the MIT License.

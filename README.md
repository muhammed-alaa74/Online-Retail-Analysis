# Executive Sales Intelligence & Customer CLV Predictor

A comprehensive Data Science project that integrates **Exploratory Data Analysis (EDA)**, **Customer Segmentation**, and **Supervised Machine Learning** to predict future customer spending. This project is deployed as an interactive dashboard using Streamlit.

---

## Live Demo
[Insert your Streamlit Cloud Link Here]

---

## Project Overview
This project aims to transform raw transactional data into actionable business insights. It provides an executive-level view of sales performance and utilizes AI to predict the potential value of customers based on their historical behavior.

### Key Features:
* **Executive Dashboard:** Real-time KPIs including Total Revenue, Total Customers, and Average Basket Value.
* **Peak Sales Analysis:** Identification of the highest-performing sales months.
* **Sales Insights:** Geographical and time-based (hourly) distribution of transactions.
* **Revenue Trends:** Visual representation of monthly revenue growth.
* **AI Value Predictor:** A Supervised Learning model that predicts the **Future Monetary Value** of a customer using Recency, Frequency, and Variety metrics.

---

## Tech Stack
* **Python:** Main programming language.
* **Streamlit:** For building the interactive web dashboard.
* **Pandas & NumPy:** For data manipulation and aggregation.
* **Plotly:** For high-end, interactive visualizations.
* **Scikit-Learn:** For training the supervised regression model.
* **Joblib:** For model serialization and deployment.

---

## Project Structure
* `app.py`: The main Streamlit application file.
* `EDA_Customer_Analysis.ipynb`: Notebook containing data exploration and cleaning.
* `Model_Training_Supervised.ipynb`: Notebook for model training and evaluation.
* `supervised_model.pkl`: The trained AI model.
* `final_customer_data.csv`: Processed dataset used for the dashboard.
* `requirements.txt`: List of Python dependencies.

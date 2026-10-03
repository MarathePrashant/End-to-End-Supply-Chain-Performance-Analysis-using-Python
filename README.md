# 🚚 Supply Chain Operations & Late Delivery Risk Analytics | Python

An end-to-end Data Analytics and Predictive Risk Modeling project built using Python to analyze global supply chain operations, fulfillment bottlenecks, shipping SLA performance, geographic delay trends, and order profitability through interactive data visualisations and machine learning models.

The project transforms raw transactional supply chain data into actionable business insights using data preparation, exploratory data analysis (EDA), data visualisations, performance benchmarking, and predictive modeling using Random Forest.

---

## 🔎 Project Overview

This project analyzes **172,765 supply chain transactions** across multiple order regions, customer segments, shipping classes, product categories, and delivery statuses.

The objective is to understand:
* **Fulfillment Performance:** Uncovering the overall late delivery rate across global orders.
* **Shipping SLA Effectiveness:** Evaluating fulfillment efficiency across different shipping modes (First Class, Second Class, Standard Class, Same Day).
* **Geographic Trends:** Identifying international regions and markets facing severe delivery friction.
* **Profit Exposure:** Quantifying the total profit associated with delayed orders to assess financial risk.
* **Predictive Analytics:** Building classification models to flag high-risk delayed orders before warehouse dispatch.

The final output is a structured **Data Analytics & Predictive Risk Pipeline** in Python that generates clear visual charts and actionable recommendations to help the company reduce delays and protect profit margins.

---

## 🎯 Business Objectives

The analysis was designed to answer key operational questions:
* What percentage of total orders fail to meet scheduled delivery timelines?
* How does delivery performance vary across different shipping classes?
* Which order regions suffer from the highest late delivery rates?
* How much revenue and profit are tied to delayed shipments?
* Can machine learning accurately predict order delay risks early in the fulfillment process?
* What operational strategies can the company implement to improve on-time fulfillment and lower costs?

---

## 📁 Dataset

The project utilizes the `DataCoSupplyChainDataset.csv` containing **172,765 cleaned order records** across global supply chain operations.

### Key Data Fields

| Field | Description |
| :--- | :--- |
| **order date (DateOrders)** | Timestamp when the customer placed the order |
| **shipping date (DateOrders)** | Timestamp when the order was dispatched |
| **Days for shipment (scheduled)** | Target promised delivery duration (SLA) |
| **Shipping Mode** | Delivery class (First Class, Second Class, Standard Class, Same Day) |
| **Order Region** | Destination region of the customer |
| **Order City / Country** | Geographic location detail |
| **Sales** | Gross sales revenue per order |
| **Order Item Profit Ratio** | Profitability metric associated with the order item |
| **Delivery Status** | Fulfillment outcome (Late delivery, Advance shipping, Shipping on time) |

### Dataset Coverage

* **172,765** transaction records analyzed (excluding canceled shipments)
* **94,523** late delivery instances identified
* **$3,806,420.63** in total generated profit analyzed
* **23** global order regions evaluated
* **4** core shipping classes analyzed

---

## 🛠️ Tools & Technologies

### Python Stack

* **Pandas & NumPy:** Data cleaning, date conversion, feature engineering, and metrics aggregation.
* **Matplotlib & Seaborn:** Custom bar charts, performance comparisons, and delay distributions.
* **Scikit-Learn:** Data preprocessing, train-test splitting, Logistic Regression, Random Forest classification, cross-validation, and evaluation metrics.
* **Imbalanced-Learn:** Synthetic Minority Over-sampling Technique (SMOTE).
* **Jupyter Notebook / Python Scripting:** Exploratory Data Analysis and model execution environment

### Analytics & Data Science Skills

* **Data Preparation & Cleaning:** Handling missing values, dropping non-informative columns, datetime processing.
* **Exploratory Data Analysis (EDA):** Grouping, pivot aggregation, and bottleneck identification.
* **Logistics & Operations Analytics:** SLA tracking, shipping class failure analysis, regional transit tracking.
* **Financial Risk Analysis:** Profit exposure tracking across delayed shipments.
* **Predictive Analytics & Machine Learning:** Classification modeling, handling target leakage, cross-validation, ROC-AUC benchmarking.
* **Data Visualisation:** Visual storytelling using clean, annotated charts.
* **Business Intelligence & Strategy:** Translating model insights into actionable recommendations for company growth.

---

## 🔄 Project Workflow

```text
Raw Supply Chain Data (172k+ Records)
        ↓
Data Cleaning & Missing Value Handling
        ↓
Feature Engineering & Target Leakage Removal
        ↓
Exploratory Data Analysis (EDA)
        ↓
SLA, Region & Profit Exposure Metrics
        ↓
Data Visualisation & Chart Generation
        ↓
Machine Learning (Logistic Regression vs. Random Forest)
        ↓
Model Evaluation & Benchmarking (Accuracy, F1-Score, ROC-AUC)
        ↓
Strategic Business Recommendations & Financial Impact

# 🚚 Supply Chain Performance Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square\&logo=python)](#)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=flat-square\&logo=pandas)](#)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4c72b0?style=flat-square)](#)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)](#)

## 📌 Executive Summary & Business Problem

In global supply chain management, operational delays, carrier lead-time volatility, and fulfillment bottlenecks can increase inventory carrying costs and affect SLA compliance.

This project analyzes end-to-end logistics transaction data to identify the key drivers of shipping delays, measure lead-time variability across shipping modes, and evaluate fulfillment reliability across geographic routes.

## 🛠️ Data Pipeline & Technical Approach

* **Data Cleaning & Transformation:** Cleaned raw logistics records using Pandas, standardized shipping-related fields, handled missing transit attributes, and removed invalid data anomalies.
* **Feature Engineering:** Created operational KPIs such as `Lead Time (Days)`, `Shipping Delay Variance`, and `Route Fulfillment Reliability Rate (%)`.
* **Exploratory Data Analysis (EDA):** Used Matplotlib and Seaborn to analyze lead-time distributions, correlations, shipping delays, and route-level performance patterns.
* **Business Analysis:** Translated analytical findings into actionable recommendations focused on carrier performance, inventory planning, warehouse operations, and fulfillment reliability.

## 📂 Project Structure

```text
├── data/               # Raw and processed transactional supply chain datasets
├── notebooks/          # Jupyter Notebook containing data cleaning and EDA
├── visuals/            # Exported analysis charts and visualizations
└── README.md           # Project documentation, methodology, and insights
```

## 📊 Key Business Insights

* **Shipping Mode Bottleneck:** Standard shipping accounted for **over 60% of recorded transit delays**, indicating a significant opportunity to improve fulfillment and staging processes.
* **Lead-Time Volatility:** High-volume product categories experienced a **22% longer fulfillment window**, highlighting the impact of peak-period warehouse handling.
* **Regional Route Inefficiencies:** Secondary transit lanes connecting Tier-2 distribution hubs showed approximately **1.4× higher delivery-time variability** than primary routes.
* **Delay-to-Return Relationship:** Shipments delayed by more than three business days were associated with an **18% increase in order cancellation and return rates**, indicating a measurable relationship between delivery performance and customer outcomes.

## 💡 Strategic Business Recommendations

* **Carrier SLA Restructuring:** Introduce performance-based carrier agreements for non-priority routes, with defined service-level thresholds and escalation mechanisms for significant delays.
* **Dynamic Safety Stock:** Maintain targeted buffer inventory for high-variance SKUs at regional distribution hubs to reduce the impact of unpredictable fulfillment times.
* **Automated Staging Alerts:** Implement automated monitoring alerts when warehouse staging exceeds predefined thresholds, enabling operations teams to intervene before delays affect final delivery.

## 🚀 How to Explore This Project

1. **Review the EDA Notebook:** Explore `/notebooks` to understand the data-cleaning process, feature engineering, KPI calculations, and analytical workflow.
2. **Review the Visualizations:** Open `/visuals` to explore lead-time distributions, delay analysis, correlation charts, and route-level performance insights.

## 👤 Author

**Prashant Marathe**

* **LinkedIn:** https://www.linkedin.com/in/prashantmarathe17
* **Portfolio:** https://prashant-marathe.framer.website/
* **Email:** [p04747391@gmail.com](mailto:p04747391@gmail.com)
* **Location:** Pune, Maharashtra, India

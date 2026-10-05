# 🚚 End-to-End Supply Chain Performance Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](#)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?style=flat-square&logo=pandas)](#)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4c72b0?style=flat-square)](#)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)](#)

## 📌 Executive Summary & Business Problem

In global supply chain management, operational delays, carrier lead-time volatility, and fulfillment bottlenecks drive up inventory carrying costs and reduce client SLA compliance. 

This project analyzes end-to-end logistics transactional logs to diagnose root causes of shipping delays, quantify lead-time variance across carrier modes, and determine fulfillment reliability across geographic routes.

## 🛠️ Data Pipeline & Technical Architecture

- **Data Wrangling & Normalization:** Cleaned raw logistics records with Pandas, standardized shipping timestamps, handled missing transit attributes, and removed invalid data anomalies.
- **Feature Engineering:** Calculated core operational KPIs including `Lead Time (Days)`, `Shipping Delay Variance`, and `Route Fulfillment Reliability Rate (%)`.
- **Exploratory Data Analysis (EDA):** Leveraged Matplotlib and Seaborn to map correlation matrices, lead-time distribution curves, and route-level bottleneck charts.

```text
├── data/               # Raw and processed transactional supply chain datasets
├── notebooks/          # Modularized Jupyter Notebook containing step-by-step EDA
├── visuals/            # Exported lead-time distributions and route charts
└── README.md           # Business case study and insights 

## 📊 Key Business Insights

- **Shipping Mode Bottleneck:** Standard shipping was responsible for **over 60% of all recorded transit delays**, primarily stemming from fulfillment hub staging rather than carrier movement.
- **Lead-Time Volatility:** High-volume product categories experienced a **22% longer fulfillment window** due to peak warehouse handling bottlenecks.
- **Regional Route Inefficiencies:** Secondary transit lanes connecting Tier-2 distribution hubs demonstrated a **1.4x higher standard deviation** in delivery timelines compared to primary arterial corridors.
- **Delay-to-Return Correlation:** Shipments delayed by more than 3 business days showed an **18% increase in order cancellation and return rates**.

## 💡 Strategic Business Recommendations

- **Carrier SLA Restructuring:** Transition non-priority transit routes to performance-based contracts that enforce fee penalties on delays exceeding 48 hours.
- **Dynamic Safety Stock:** Maintain targeted buffer inventory at regional hubs for high-variance SKUs to absorb staging and lead-time volatility.
- **Automated Staging Alerts:** Deploy automated tracking alerts when warehouse staging time exceeds 24 hours to enable proactive logistics intervention.

## 🚀 How to Explore This Project

1. **Review EDA Notebook:** Check `/notebooks` to review the end-to-step Python data cleaning, feature calculations, and statistical plots.
2. **Review Visuals:** Open `/visuals` for high-resolution distribution graphs and correlation matrices.

## 👤 Author

**Prashant Marathe**
- LinkedIn: [linkedin.com/in/prashantmarathe17](https://www.linkedin.com/in/prashantmarathe17)[cite: 1]
- Portfolio Website: [prashant-marathe.framer.website](https://prashant-marathe.framer.website/)[cite: 1]
- Email: [p04747391@gmail.com](mailto:p04747391@gmail.com)[cite: 1]
- Location: Pune, Maharashtra, India[cite: 1]

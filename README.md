# 🚚 End-to-End Supply Chain Performance Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=flat-square&logo=pandas)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4c72b0?style=flat-square)](https://seaborn.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)](#)

## 📌 Executive Summary & Business Context
In global logistics operations, shipping delays, carrier variance, and inventory holding costs directly erode gross margins and client SLA compliance. 

This project analyzes end-to-end supply chain transactional data to diagnose delivery bottlenecks, quantify shipping lead times, and evaluate order fulfillment reliability across product categories and carrier routes.

---

## 🛠️️ Data Pipeline & Technical Workflow

* **Data Extraction & Hygiene:** Cleaned raw transactional datasets using Pandas; standardized shipping timestamps, resolved missing order attributes, and eliminated redundant records.
* **Exploratory Data Analysis (EDA):** Evaluated shipment route distribution, lead time variance, customer delivery windows, and defect/cancellation rates across segments.
* **Feature Engineering:** Derived operational metrics including `Lead Time (Days)`, `Delivery Delay Variance`, and `Route Fulfillment Efficiency`.
* **Statistical Visualization:** Leveraged Matplotlib and Seaborn to map correlation matrices, lead-time distribution curves, and route-level bottleneck charts.

```text
├── data/               # Raw and processed supply chain datasets
├── notebooks/          # Clean, modularized Jupyter Notebook with annotated EDA
├── visuals/            # Exported distribution plots and correlation charts
└── README.md           # Business impact documentation

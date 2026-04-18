# 📉 Customer Churn Prediction & Business Intelligence Dashboard

![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/churn-dashboard)

An end-to-end machine learning system that predicts customer churn and converts model outputs into actionable business strategy — built for SaaS and subscription startups.

---

## Problem Statement

Startups spend $200–$400 to acquire a customer, then lose them silently. This project flags at-risk customers **before** they leave, explains *why* they're likely to churn using SHAP, and quantifies the **dollar value** of targeted retention outreach.

---

## Architecture

```
Raw CSV → Preprocessing + Feature Engineering → XGBoost Model
                                                      ↓
                                              SHAP Explainability
                                                      ↓
                                         Business Impact Calculator
                                                      ↓
                                       Streamlit Interactive Dashboard
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Accuracy | ~82% |
| Recall (Churn) | ~79% |
| ROC-AUC | ~0.86 |
| Precision @ Top 20% | ~65% |
| Net Revenue Saved (est.) | $42,000 / month per 1000 customers |

> **Business insight:** Flagging top-decile at-risk users and running a $30/customer retention campaign yields an estimated **14:1 ROI** compared to re-acquiring churned customers at $300 CAC.

---

## Features

- **EDA Dashboard** — Churn distribution, service breakdowns, tenure vs charge scatter
- **Model Performance** — Confusion matrix, ROC curve, probability calibration
- **SHAP Explainability** — Global feature importance + per-customer "why did it predict this?"
- **Business Impact Tab** — Interactive ROI simulator with threshold optimization
- **At-Risk Export** — Download CSV of high-risk customers, ready for CRM upload

---

## Feature Engineering Highlights

Beyond raw columns, the model benefits from:

- `ChargesPerTenureMonth` — Revenue per month normalized by customer age (loyalty-adjusted value)
- `ChargeInflation` — How much their bill has grown relative to initial pricing
- `NumServices` — Total add-on services subscribed (strong retention predictor)

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Data Wrangling | Pandas, NumPy |
| ML Model | XGBoost (with `scale_pos_weight` for imbalance) |
| Explainability | SHAP (TreeExplainer) |
| Visualization | Plotly, Streamlit |
| Deployment | Hugging Face Spaces |

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/churn-prediction.git
cd churn-prediction

pip install -r requirements.txt

# Download dataset from Kaggle:
# https://www.kaggle.com/blastchar/telco-customer-churn
# Place as: data/telco_churn.csv

streamlit run app.py
```

---

## Dataset

[Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) — 7,043 customers, 21 features including contract type, internet service, monthly charges, and tenure.

---

## Project Structure

```
churn_project/
├── app.py                  # Streamlit dashboard (main)
├── src/
│   ├── preprocess.py       # Data cleaning + feature engineering
│   ├── model.py            # XGBoost training + SHAP values
│   └── insights.py         # Business impact calculator
├── data/
│   └── telco_churn.csv     # (not committed — download from Kaggle)
├── requirements.txt
└── README.md
```

---

## Business Impact Framing

This project doesn't just predict churn — it answers:
- Which customers are about to leave? *(model)*
- Why are they leaving? *(SHAP)*
- What's the cost of losing them? *(ARPU × LTV)*
- Is it worth reaching out? *(retention ROI calculator)*
- Who specifically should we call first? *(ranked export)*

---

*Built as part of a data analyst portfolio targeting B2C/SaaS startup roles.*

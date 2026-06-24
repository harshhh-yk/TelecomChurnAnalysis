# Telco Customer Churn — Analysis & Prediction

## Overview
This project analyzes customer churn for a telecom company and builds a model
that scores every customer's likelihood of leaving. The goal isn't just
prediction accuracy — it's turning that prediction into something a
retention team can actually act on.

## Dataset
- 7,043 customers, 21 attributes (demographics, account info, services subscribed)
- Target: `Churn` (Yes/No)
- Source: IBM/Kaggle Telco Customer Churn dataset

## What this project does

**1. Data Cleaning**
- Standardized column names, dropped the non-predictive `customerID`
- Fixed `TotalCharges` (stored as text due to blank values for new customers)
  by converting to numeric and imputing missing values as `tenure × monthly charges`
- Removed exact duplicate rows

**2. Exploratory Data Analysis**
- Churn distribution, demographic cuts (gender, senior citizen)
- Churn rate by contract type, internet service, and payment method
- Tenure, monthly charges, and total charges distributions split by churn
- Correlation heatmap across numeric features

**3. Modeling**
- Built a `ColumnTransformer` pipeline (StandardScaler for numeric, OneHotEncoder for categorical)
- Trained and compared three models: **Logistic Regression**, **Random Forest**, **XGBoost**
- Evaluated with Accuracy, ROC-AUC, Precision-Recall (AP), and 5-fold stratified cross-validation

**4. Interpretability**
- Logistic regression coefficients to identify the strongest churn drivers
- SHAP values (TreeExplainer on XGBoost) as an independent cross-check on feature importance

**5. Risk Segmentation**
- Converted churn probabilities into Low / Medium / High risk tiers
- Scored the full customer base to size the actionable retention workload

## Key Findings
- Month-to-month customers churn at **~15x** the rate of two-year contract customers
- Fiber-optic customers churn at **more than double** the rate of DSL customers
- Logistic Regression outperformed Random Forest and XGBoost on this dataset (ROC-AUC: 0.840 vs 0.819 vs 0.834) — simpler model, stronger generalization
- **~21%** of customers can be flagged High Risk today for proactive outreach

## Tech Stack
`Python` · `pandas` · `scikit-learn` · `XGBoost` · `SHAP` · `seaborn` / `matplotlib`

## Repo Contents
- `telco_churn_analysis.py` — full, commented analysis & modeling pipeline
- `Churn_Recommendations.pptx` — business-facing summary deck
- `telco_churn_cleaned.csv` — cleaned dataset (output of the cleaning step)

## How to Run
```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
python telco_churn_analysis.py
```

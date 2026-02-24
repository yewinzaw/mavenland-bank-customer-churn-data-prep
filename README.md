> “As software becomes a commodity, so do common practices — but craftsmanship and software quality do not follow.”  
> — Jonathan YE
# 🏦 Mavenland Bank Customer Churn
## Data Cleaning & Feature Engineering Portfolio Project
End-to-end data preparation workflow for customer churn analysis using Python, Pandas, and data visualizations.

---

## 📌 Overview

This project cleans and prepares a messy banking dataset for machine learning modeling.The implementation in `CleanUpChurn.py` 
focuses on data integrity, structured EDA, and practical feature engineering.

---
## 📂 Dataset

**File:** `Bank_Churn_Messy.xlsx`

**Sheets:**
- `Account_Info`
- `Customer_Info`

The dataset intentionally includes:
- Duplicate records  
- Currency-formatted numeric fields  
- Inconsistent country labels  
- Missing values  
- Placeholder values (e.g., -999999)  
- Redundant columns  

---

## 🎯 Objectives & Workflow

### ✅ Objective 1 — Import & QA 

- Load and validate both Excel sheets  
- Remove duplicate `CustomerId` records
- Clean currency fields (`Balance`, `EstimatedSalary`)
- Convert categorical fields to numeric format
- Drop redundant columns
- Left join on `CustomerId`

---

### ✅ Objective 2 — Data Cleaning

- Fix data types
- Impute missing numeric values (median)
- Fill missing categorical values 
- Standardize country names (e.g., `FRA`, `French` → `France`)
- Replace invalid salary placeholders (-999999) with median 

---

### ✅ Objective 3 — Exploratory Data Analysis (EDA)

- Target distribution: churned (`Exited = 1`) vs retained (`Exited = 0`)  
- Churn rate by Geography and Gender  
- Box plots of numeric features by churn status  
- Histograms by churn status

---

### ✅ Objective 4 — Modeling Preparation

- Removed `CustomerId` and `Surname`  
- One-hot encoded `Geography`  
- Created ratio feature: `balance_v_income = Balance / EstimatedSalary`  
- Filtered extreme outliers before visualization  
---

##  🏁  Conclusion

From inconsistencies to clarity. 
This project demonstrates a hands-on approach to transforming raw, fragmented data into a dependable foundation for machine learning.
---
👤 **Author:** Jonathan YE  
📬 **Contact:** [yewinzaw@gmail.com](mailto:yewinzaw@gmail.com)  
🔗 **Reference**: mavenanalytics.io/guided_projects




# Hospital Readmission Prediction

Predicting 30-day hospital readmissions for diabetic patients using machine learning on 70,416 clinical records from 130 US hospitals.

---

## Problem Statement

The US healthcare system spends over $26 billion yearly on hospital readmissions within 30 days. Some diabetic patients face higher readmission risk due to comorbidities and complex medication regimens; it is crucial to identify high-risk individuals before discharging is done. This project builds a machine learning pipeline that flags high-risk patients so clinical teams can prioritize post-discharge interventions.

---
 
## Dashboard Preview

### Executive Summary
![Executive Summary](page1_executive_summary.png)

### Patient Risk Explorer
![Patient Risk Explorer](page2_patient_risk_explorer.png)

### Model Insights
![Model Insights](page3_model_insights.png)

---

## Key Findings

- **6.9%** of unique patients were readmitted within 30 days, indicating a severe class imbalance addressed using `class_weight='balanced'` 
- **5,927 patients** flagged as high risk out of 70,416 analyzed.
- **Patients aged 70-80** represent the largest high-risk group (2,024 patients), with those over 60 accounting for **77% of all high-risk readmissions**
- **Length of stay** was the strongest predictor of readmission (importance: 0.123), followed by prior inpatient visits (0.106) and total clinical interactions (0.097)
- **Discharge disposition** ranked 4th in importance, which reveals that where a patient goes after discharge significantly influences readmission risk
- The Random Forest model achieved **ROC-AUC of 0.682**, correctly flagging **53% of true readmissions** across the full dataset

---

## Project Structure

```
hospital_readmission_project
├── data
│   ├── diabetic_data_clean.csv         # Cleaned dataset after python preprocessing
│   ├── diabetic_data_features.csv      # Engineered features ready for modelling
│   ├── predictions.csv                 # Model predictions with risk scores and categories
│   └── feature_importances.csv         # Top 15 features by importance score
├── notebooks
│   ├── 01_data_exploration.ipynb       # Data cleaning, EDA and feature engineering
│   └── 02_modelling.ipynb              # Model training, evaluation and export
├── dashboard
│   └── Hospital_readmission_Dashboard.pbix  # Power BI dashboard (3 pages)
├── visuals
│   ├── roc_curves.png
│   ├── feature_importance.png
│   ├── precision_recall_tradeoff.png
│   ├── page1_executive_summary.png
│   ├── page2_patient_risk_explorer.png
│   └── page3_model_insights.png
└── README.md
```

---

## Dataset

**Source:** [Diabetes 130-US Hospitals (1999–2008)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) — UCI Machine Learning Repository

| Detail | Value |
|---|---|
| Original rows | 101,766 |
| Unique patients (after deduplication) | 70,416 |
| Features | 50 raw → 46 engineered |
| Target | Readmitted within 30 days (binary) |
| Class balance | 93.1% not readmitted / 6.9% readmitted |

### Data Quality Issues Addressed
- Missing values encoded as `?` — replaced with `NaN` before any analysis
- Multiple encounters per patient removed — kept first encounter per patient to prevent data leakage
- Columns with >90% missing values dropped (`weight`, `max_glu_serum`, `medical_specialty`, `payer_code)
- `A1Cresult`  missing values converted to a binary feature (`A1C_tested`) - absence of the test is itself a clinical signal

---

## Methodology

### 1. Data Cleaning (Python)
- Replaced `?` placeholders with `NaN`
- Removed duplicate patient encounters
- Dropped high-missingness columns
- Created binary `A1C_tested` feature from missing `A1Cresult`

### 2. Feature Engineering
- Converted age brackets to ordinal numbers (`age_num`)
- Created `total_interactions` = lab procedures + procedures + medications
- Created `prior_hospital_use` = outpatient + emergency + inpatient visits
- Created `long_stay` binary flag for stays ≥ 7 days
- Encoded 23 medication columns as binary (prescribed / not prescribed)
- Created `total_meds_prescribed` count feature
- Encoded categorical columns using `LabelEncoder`

### 3. Model Training
Two models trained and compared:

| Model | ROC-AUC | Recall | Precision |
|---|---|---|---|
| Logistic Regression | 0.675 | 39% | 14% |
| Random Forest | **0.682** | **39%** | **15%** |

**Random Forest selected** as the final model based on higher ROC-AUC and better handling of feature interactions.

### 4. Handling Class Imbalance
- Used `class_weight='balanced'` on both models instead of SMOTE
- Applied `stratify=y` during train/test split to preserve class ratios
- Optimal decision threshold found using precision-recall curve - threshold set to **0.13** (lower than default 0.5 due to 7% minority class)

### 5. Evaluation Metric
**ROC-AUC used as primary metric** due to class imblance. Accuracy is misleading on imbalanced datasets as a model predicting "no readmission" for every patient would score 93% accuracy but identify zero high-risk patients.

---

## Model Performance

```
Random Forest — optimal threshold: 0.13
ROC-AUC: 0.682

              precision    recall    f1-score
Not readmitted    0.95      0.83      0.89
Readmitted        0.15      0.39      0.21

Confusion Matrix:
  TP (correctly flagged):        2,555
  FP (false alarms):            10,271
  FN (missed readmissions):      2,279
  TN (correctly low risk):      55,311
```

**Note on high FP rate:** The threshold of 0.13 was deliberately set low to maximize recall. In a clinical setting, missing a high-risk patient (FN) is far more costly than a false alarm (FP). A hospital case manager reviewing 10,271 flagged patients to find 2,555 true readmissions is a clinically acceptable trade-off.

---

## Risk Category Breakdown

| Category | Patients | Threshold |
|---|---|---|
| Low risk | 16,815 | Probability < 0.30 |
| Medium risk | 47,674 | Probability 0.30–0.60 |
| High risk | 5,927 | Probability > 0.60 |

---

## Business Impact

This model is designed to support hospital decision-making by identifying patients at high risk of 30-day readmission before discharge.

### 1. Early Intervention Targeting
The model flags **5,927 high-risk patients**, enabling care teams to prioritize the following:
- Post-discharge follow-ups
- Medication reconciliation
- Patient education
- Home care support

### 2. Cost Reduction Potential
Hospital readmissions are estimated to cost over $26 billion annually in the U.S. Even a modest reduction in readmissions can lead to significant savings.

If targeted interventions reduce readmissions among flagged patients by just:
- **10% → ~255 avoided readmissions**
- **20% → ~511 avoided readmissions**

This translates into **millions of dollars in potential cost savings**, based on the average cost per readmission.

### 3. Operational Efficiency
Instead of applying interventions to all patients, hospitals can:
- Focus on **~8.4% high-risk patients**
- Allocate limited resources more effectively
- Reduce clinician workload while improving outcomes

### 4. Clinical Decision Support
The model provides interpretable drivers of risk (e.g., length of stay, prior hospital use), helping clinicians:
- Understand *why* a patient is high-risk
- Make informed discharge decisions

### 5. Scalable Integration
The model can be integrated into:
- Electronic Health Record (EHR) systems
- Discharge planning workflows
- Hospital dashboards (as demonstrated in Power BI)

This enables **real-time risk scoring at scale**.

## Power BI Dashboard

The dashboard has 3 pages built in Power BI Desktop:

**Page 1 - Executive Summary**
KPI cards, readmission rate by age group, patient risk distribution donut, readmission rate by length of stay, and readmission rate by prior hospital visits.

**Page 2 - Patient Risk Explorer**
Interactive slicers for risk category, age group, length of stay, and prior visits. Scatter plot of individual patient risk scores, bar chart by age group, and high-risk patient detail table.

**Page 3 - Model Insights**
Feature importance chart, prediction breakdown donut, confusion matrix (TP/FP/FN/TN), model performance notes, and improvement roadmap.

---

## Tools and Technologies

| Tool | Purpose |
|---|---|
| Python 3.11 | Data processing and modelling |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | Model training and evaluation |
| Matplotlib / Seaborn | Exploratory visualisations |
| Power BI Desktop | Interactive dashboard |
| python | Initial data exploration |
| Jupyter Notebook | Analysis environment |
| GitHub | Version control and portfolio |

---

## What Would Improve This Model

- **Engineer ICD-9 diagnosis codes** into 9 clinical categories (Circulatory, Respiratory, Diabetes, etc.) -currently dropped due to 700+ unique values
- **Add Charlson Comorbidity Index** -a standard clinical score that weights comorbidities for readmission risk
- **Try XGBoost or LightGBM**; gradient boosting models typically outperform Random Forest on tabular medical data.
- **Use SHAP values** for patient-level explainability -it tells clinicians exactly why each patient received their risk score.
- **More recent data** -current dataset covers 1999–2008; clinical practices have evolved significantly
- **Add social determinants of health**, such as insurance type, socioeconomic factors, and geography, which are important for readmission risk.

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Cephasadamskumah-ds2025/hospital-readmission-project.git
cd hospital-readmission-project
```

### 2. Install dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. Download the dataset
Download `diabetic_data.csv` from [Kaggle](https://www.kaggle.com/datasets/brandao/diabetes) or [UCI](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) and place it in the `data/` folder.
 

### 4. Run the notebooks in order
```bash
jupyter notebook
```
Open and run `01_data_exploration.ipynb` first, then `02_modelling.ipynb`.

### 5. Open the dashboard
Open `dashboard/Hospital_readmission_Dashboard.pbix` in Power BI Desktop. If prompted to refresh the data source, point it to your local `data/predictions.csv` file.

---

## Author

**CEPHAS ADAMS KUMAH**
Data Science Graduate | Healthcare Analytics

[LinkedIn](https://linkedin.com/in/Cephas-Adams-Kumah) · [GitHub](https://github.com/Cephasadamskumah-ds2025)

---

## License

This project uses the [Diabetes 130-US Hospitals dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).

Citation:
> Clore, J., Cios, K., DeShazo, J., & Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008. UCI Machine Learning Repository.

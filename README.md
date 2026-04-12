# Telco Customer Churn Prediction

> **End-to-end ML system** that identifies customers at risk of churning, quantifies business impact and serves predictions through Streamlit application with drift monitoring and audit logging.

[![Python 3.11+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

## Business Problem

A telecom company loses an estimated **Rs. 21.5 lakh per retention cycle** due to customer churn. 
Traditional approaches flag customers too late or waste retention budget on customers who would have stayed anyway. 
This system predicts churn probability **before** the customer leaves, enabling targeted, cost-efficient intervention.

### Business Impact (Production Model)

| Metric | Value |
|---|---|
| Dataset | 7,032 Telco customers |
| Actual churn rate | 26.5% -> 1,869 churners |
| Churners caught by model (Recall 77.3%) | ~1,434 customers |
| Average Customer Lifetime Value | Rs. 1,500 |
| Revenue protected per cycle | **Rs. 21.5 lakh** |
| Retention campaign cost (7K × Rs. 30) | Rs. 2.1 lakh |
| **Net ROI per cycle** | **Rs. 19.4 lakh** |

---


## Model Performance
 
| Metric | Score | Benchmark |
|---|---|---|
| **AUC-ROC** | **0.8403** | > 0.80 required |
| **Recall** | **0.7727** | > 0.70 required |
| F1 Score | 0.6208 | — |
| Precision | 0.5199 | — |
| Accuracy | 0.7743 | — |
| CV AUC-ROC (5-fold) | 0.8480 | — |
 
**SMOTE impact:** Recall improved from 0.546 -> 0.773 (+41.6%) by oversampling the minority class (26.5% churn rate) during training.
 
### Top Churn Drivers (Cramér's V)
 
| Feature | Association | Key Insight |
|---|---|---|
| `contract` | V = 0.410 | Month-to-month: 42.7% churn vs Two year: 2.8% |
| `tenure_group` | V = 0.350 | 0-12 months: 47.7% churn vs 61-72 months: 6.6% |
| `onlinesecurity` | V = 0.347 | No security: 41.8% churn vs Yes: 14.6% |
| `techsupport` | V = 0.336 | No support: 41.6% churn vs Yes: 15.2% |
| `internetservice` | V = 0.331 | Fiber optic: 41.9% churn (highest) |
| `paymentmethod` | V = 0.316 | Electronic check: 45.3% churn (highest) |

---

## Project Structure

```
Telco-Customer-Churn-Prediction/
├── main.py
├── pages/
│   ├── 00_Home.py
│   ├── 01_EDA_Report.py     # Exploratory data analysis + visualizations
│   ├── 02_Live_Predictor.py # CSV upload -> batch predictions
│   ├── 03_Monitoring.py     # Drift detection dashboard
├── src/
│   ├── __init__.py   
│   ├── config.py
│   ├── database.py
│   ├── extract_training_stats.py  
│   ├── logger.py
│   ├── model.py
│   ├── monitoring.py
│   ├── preprocessing.py
├── notebooks/
│   ├── churn_eda_ml.ipynb
├── docs/
│   ├── DECISIONS.md
│   └── MODEL_CARD.md
├── models/
│   ├── adaboost_churn_model.pkl
│   ├── preprocessor.pkl
│   └── top20_features.pkl
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # training_data
│   └── audit.db
├── reports/
│   ├── assets  # EDA Report slides
│   ├── figures # Charts
│   └── Telco Customer Churn EDA Report.pdf
├── tests/
│   ├── conftest.py
│   ├── test_model.py
│   ├── test_preprocessing.py
├── .github/workflows/
│   └── test.yml
├── requirements.txt
├── requirements-test.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/abhishek-tiwari-nitrr/Telco-Customer-Churn-Prediction.git
cd Telco-Customer-Churn-Prediction
pip install -r requirements.txt

# 2. Run the app
streamlit run main.py

# 3. Run the test suite
pip install -r requirements-test.txt
pytest test/ -v
```

---

## Key Design Decisions
 
See [`docs/DECISIONS.md`](docs/DECISIONS.md) for the full decision log. Summary:
 
- **AdaBoost over XGBoost** - highest CV AUC-ROC (0.8403 vs 0.8238), simpler hyperparameter space, faster inference
- **SMOTE over class_weight** - +41.6% recall improvement justified the pipeline complexity
- **SQLite over PostgreSQL** - audit trail requirement met without operational overhead for this scale
- **OneHotEncoder** - production preprocessor uses OHE (`sparse_output=False`); top 20 SHAP-selected features are then used for inference
 
---
 
## Tech Stack
 
| Layer | Technology |
|---|---|
| ML | scikit-learn, imbalanced-learn (SMOTE), joblib |
| Explainability | SHAP (KernelExplainer - feature selection during training) |
| App | Streamlit |
| Data | Pandas, NumPy |
| Testing | pytest |
| Logging | Python logging (RotatingFileHandler) |
| Storage | SQLite (audit trail) |
| CI/CD | GitHub Actions |
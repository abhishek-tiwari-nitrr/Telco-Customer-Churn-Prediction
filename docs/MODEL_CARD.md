# Model Card: Telecom Customer Churn Prediction

## Model Details

- **Model Name**: Telecom_Churn_Predictor
- **Model Type**: Binary Classification
- **Algorithm**: AdaBoost Classifier (inside an ImbPipeline with SMOTE)
- **Version**: 1.0
- **Date Created**: April 2026
- **Developed By**: Abhishek Tiwari
- **Framework**: scikit-learn + imbalanced-learn + pandas
- **Task**: Predict whether a telecom customer will churn (`Churn = Yes / 1`) or stay (`Churn = No / 0`)
- **Input**: Tabular customer features (19 columns after dropping customerID)
- **Output**: Churn probability (0–1) and binary prediction

---

## Factors

- **Relevant Factors**: Contract type, tenure, MonthlyCharges, TotalCharges, InternetService,
  OnlineSecurity, TechSupport, PaymentMethod, StreamingTV, StreamingMovies, MultipleLines,
  PaperlessBilling, Dependents, OnlineBackup, DeviceProtection.
- **Data Imbalance**: ~26.5% churn rate (minority class) - handled via SMOTE inside training pipeline.

---

## Training Data

- **Dataset**: Telco Customer Churn (Kaggle) - 7,043 records, 21 columns
- **Source**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn/data
- **Target**: `Churn` (Yes = 1 / No = 0)
- **Class Distribution**: 73.4% Retained | 26.6% Churned

### Key Features Used

Top 20 selected by **Mean |SHAP| value** using `shap.KernelExplainer` on 200 sampled test instances. Feature names use the OHE ColumnTransformer
prefix convention (`num__` for numeric, `cat__` for categorical).

| Feature | Type | SHAP Tier |
|---|---|---|
| `num__monthlycharges` | Numerical | High |
| `num__totalcharges` | Numerical | High |
| `num__tenure` | Numerical | High |
| `cat__internetservice_Fiber optic` | Categorical | High |
| `cat__contract_Month-to-month` | Categorical | High |
| `cat__contract_Two year` | Categorical | High |
| `cat__streamingmovies_Yes` | Categorical | Moderate |
| `cat__streamingtv_Yes` | Categorical | Moderate |
| `cat__multiplelines_Yes` | Categorical | Moderate |
| `cat__paymentmethod_Electronic check` | Categorical | Moderate |
| `cat__paperlessbilling_No` | Categorical | Moderate |
| `cat__onlinesecurity_No` | Categorical | Moderate |
| `cat__dependents_Yes` | Categorical | Moderate |
| `cat__techsupport_No` | Categorical | Moderate |
| `cat__streamingtv_No internet service` | Categorical | Low |
| `cat__techsupport_No internet service` | Categorical | Low |
| `cat__streamingmovies_No internet service` | Categorical | Low |
| `cat__onlinesecurity_No internet service` | Categorical | Low |
| `cat__deviceprotection_No internet service` | Categorical | Low |
| `cat__onlinebackup_No internet service` | Categorical | Low |

### Preprocessing Pipeline

- `customerID` dropped before any processing
- `TotalCharges` - `pd.to_numeric(errors='coerce')`, 11 missing values imputed with median
- **Numerical features** (`seniorcitizen`, `tenure`, `monthlycharges`, `totalcharges`):
  median imputation -> StandardScaler
- **Categorical features** (remaining 15 columns):
  mode imputation -> `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
- **Class imbalance**: `SMOTE(random_state=42)` applied inside `ImbPipeline` after preprocessing,
  on training data only - test set is untouched real data
- **Train/Test split**: 80/20 stratified by target (`random_state=42`)

---

## Model Artifact Details

| File | Contents | Notes |
|---|---|---|
| `adaboost_churn_model.pkl` | Fitted `ImbPipeline([("smote", SMOTE), ("model", AdaBoostClassifier)])` | SMOTE step is inert at inference; only AdaBoost runs |
| `preprocessor.pkl` | Fitted `ColumnTransformer` with StandardScaler + OneHotEncoder | Must be applied before the model; produces ~45 OHE columns |
| `top20_features.pkl` | Python list of 20 OHE feature name strings | Used to slice the encoded DataFrame before `predict_proba` |

---

## Evaluation Data

- Held-out test set: 20% of data (≈1,409 records), stratified by churn
- SMOTE applied only to training folds - no synthetic data leaks into evaluation
- Evaluation focused on the minority (churn) class due to the 50:1 cost asymmetry

---

## Quantitative Analysis

### Model Selection Journey

| Stage | AUC-ROC | Test Recall | Test F1 |
|---|---|---|---|
| Base AdaBoost (no balancing, no tuning) | 0.8433 | 0.5455 | 0.5939 |
| + SMOTE | 0.8394 | 0.7674 | 0.6199 |
| + Top 20 SHAP features + GridSearchCV | 0.8403 | 0.7727 | 0.6208 |

### Final Model Performance (held-out test set)

| Metric | Value | Notes |
|---|---|---|
| Accuracy | 0.7743 | Weak signal - dummy majority classifier baseline is 73.5% |
| Precision (Churn) | 0.5199 | ~1 in 2 flagged customers are false positives |
| **Recall (Churn)** | **0.7727** | **Primary metric - catches 77.3% of churners** |
| F1-Score (Churn) | 0.6208 | |
| AUC-ROC | 0.8403 | |
| **CV AUC-ROC (5-fold)** | **0.8480** | `GridSearchCV(scoring="roc_auc")` |


### Best Hyperparameters

Found via `GridSearchCV` over `{n_estimators: [50,100,200,300,500], learning_rate: [0.01,0.05,0.1,0.5,1.0]}`,
scored by `roc_auc`, 5-fold CV on the training set with SMOTE inside each fold.

| Parameter | Value |
|---|---|
| n_estimators | 200 |
| learning_rate | 1.0 |

### Baseline Comparison

| Model | AUC-ROC |
|---|---|
| Dummy (majority class always) | 0.5000 |
| AdaBoost Final | 0.8403 |
| Improvement | +0.3403 |
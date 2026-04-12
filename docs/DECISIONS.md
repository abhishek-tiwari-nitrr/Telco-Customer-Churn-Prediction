# Architecture Decision Record - Telco Churn Predictor

This document records every significant technical decision made during this project: the alternatives considered, the data that drove the choice, and the trade-offs accepted.

---

## ADR-001: Algorithm Selection - AdaBoost over XGBoost

### Context
Binary classifier on tabular data: 7,043 samples, 19 features after preprocessing, mixed numeric and categorical types, 26.5% class imbalance.

### Decision
Use **AdaBoostClassifier** with decision stumps (`max_depth=1`) as the production model.

### Alternatives Considered

| Algorithm | CV AUC-ROC | CV Recall | Training Time | Notes |
|---|---|---|---|---|
| **AdaBoost (chosen)** | **0.8403** | **0.7727** | ~2s | Simple, fast, interpretable importances |
| XGBoost | 0.8238 | 0.7912 | ~8s | Higher tuning cost, marginal AUC gain |
| Random Forest | 0.8101 | 0.7543 | ~5s | Lower recall on imbalanced data |
| Logistic Regression | 0.7823 | 0.7210 | <1s | Good baseline, underfits non-linear patterns |
| LightGBM | 0.8312 | 0.8203 | ~4s | Comparable to AdaBoost, more complex to tune |


### Rationale
- AdaBoost achieved the highest test AUC-ROC (0.8403) among all candidates on this dataset.
- Simpler hyperparameter space (`n_estimators`, `learning_rate`) - less tuning surface, more reproducible results.
- Decision stumps are inherently interpretable: each stump splits on one feature at one threshold.
- Faster inference than complex ensembles - relevant for batch prediction latency.

### Trade-offs Accepted
- AdaBoost is more sensitive to outliers than gradient boosting (outliers receive increasing weight across rounds).
  - Mitigated by: StandardScaler on numeric features and bounds validation in the validation pipeline.
- AdaBoost cannot natively handle missing values - XGBoost can.
  - Mitigated by: `SimpleImputer` in `preprocessor_base_5` before AdaBoost sees any data.

---

## ADR-002: Class Imbalance Strategy - SMOTE over class_weight

### Context
Target distribution: 73.5% No Churn / 26.5% Churn. Without intervention, the model optimises for accuracy by predicting the majority class, missing the churners that matter most.

### Decision
Use **SMOTE** (Synthetic Minority Oversampling Technique) inside an `ImbPipeline`, applied after preprocessing and before the classifier.

### Alternatives Considered

| Strategy | Recall Before | Recall After | Notes |
|---|---|---|---|
| No balancing | 0.546 | — | Misses 45% of churners |
| **SMOTE (chosen)** | 0.546 | **0.773** | +41.6% improvement |
| class_weight='balanced' | 0.546 | 0.701 | +28.4% improvement |

### Rationale
- SMOTE produced the largest recall lift (+41.6% vs +28.4% for class_weight).
- Recall is the primary metric: a missed churner costs ~Rs. 1,500 CLV; a false alarm costs ~Rs. 30 retention offer. The miss:false-alarm cost ratio is ~50:1.
- SMOTE interpolates between nearest neighbours, generating plausible synthetic profiles rather than simple duplication.
- Placed inside `ImbPipeline` so SMOTE only sees training folds during cross-validation - no data leakage.

### Trade-offs Accepted
- Synthetic samples did not actually exist - interpolation may produce unrealistic feature combinations.
  - Mitigated by: `k_neighbors=5` (default) keeps synthetic points close to real minority members.
- Training is slower due to the larger synthetic training set.
  - Accepted: training runs offline; SMOTE is inert at inference time.

---

## ADR-003: Encoding Strategy - OneHotEncoder (production) with OrdinalEncoder explored

### Context
The dataset has 15 categorical features. The notebook explored four preprocessing combinations before selecting the final production configuration.

### Decision
Use **OneHotEncoder(`handle_unknown='ignore'`, `sparse_output=False`)** for all categorical features in the production preprocessor (`preprocessor_base_5`).

### Preprocessing Configurations Explored

| Config | Numeric | Categorical | Used for |
|---|---|---|---|
| `preprocessor_base_1` | MedianImputer | OrdinalEncoder | Broad model sweep |
| `preprocessor_base_2` | MedianImputer | OneHotEncoder | Broad model sweep |
| `preprocessor_base_3` | MedianImputer + StandardScaler | OrdinalEncoder | Top-5 model sweep |
| `preprocessor_base_4` | MedianImputer + StandardScaler | OneHotEncoder | Top-5 model sweep |
| **`preprocessor_base_5` (chosen)** | MedianImputer + StandardScaler | **OneHotEncoder (sparse_output=False)** | SMOTE tuning + saved to `preprocessor.pkl` |

### Rationale
- `preprocessor_base_5` with OHE was selected because it produced the best combined recall and AUC-ROC across the top-5 model comparison.
- `sparse_output=False` is required for compatibility with SMOTE, which cannot operate on sparse matrices.
- `handle_unknown='ignore'` encodes unseen categories as an all-zero vector, which is safe for production inference without crashing.
- OHE produces ~45 binary columns from 15 categorical features. SHAP-based top-20 selection then reduces this to a manageable, informative subset before the final GridSearchCV.

### Trade-offs Accepted
- OHE expands the feature space (~45 columns vs 15 with OrdinalEncoder) - mitigated by the top-20 SHAP filter.
- Unseen categories at inference are silently zeroed (`handle_unknown='ignore'`) with no warning to the caller.
  - Mitigated by: the validation pipeline in `src/preprocessing.py` warns on unexpected category values before encoding.


---

## ADR-004: Storage - SQLite over PostgreSQL for Audit Trail

### Context
Every prediction batch must be logged for audit, compliance, and retrospective analysis. Four tables required: `prediction_batches`, `prediction_records`, `drift_alerts`, `validation_errors`.

### Decision
Use **SQLite** via Python's built-in `sqlite3` module.

### Rationale
- Zero operational overhead - SQLite is a file, not a server process.
- Sufficient access pattern: sequential writes during prediction, occasional reads for dashboard and audit queries.
- ACID-compliant - writes are transactional, no partial records.
- Single-file database ships with the application - no setup required for deployment.
- At 7,043 customers per batch, SQLite handles millions of prediction records before performance becomes relevant.

### Trade-offs Accepted
- No concurrent write support from multiple processes.
  - Accepted: Streamlit is single-process; concurrent writes are not a concern at this scale.
- `check_same_thread=False` is set in `get_connection()` to suppress the SQLite thread warning.
  - Accepted for Streamlit's single-process model; would need revisiting under a multi-worker deployment.

---

## ADR-005: Validation Strategy - Warnings not Errors for Data Quality Issues

### Context
The validation pipeline must handle real-world data quality issues without blocking the operations team from getting predictions.

### Decision
Data quality issues produce **warnings**, not errors. Only structurally invalid data produces errors.

### Error Conditions (batch rejected)
- Missing required columns - model cannot run without its expected input features.
- Zero valid rows remaining after cleaning - nothing to predict.

### Warning Conditions (batch continues with caution flag)
- Missing values detected (will be imputed by the preprocessor pipeline).
- Non-numeric values in numeric columns (coerced to NaN, then imputed).
- Unexpected category values (encoded as all-zero vector by `handle_unknown='ignore'`).
- `Churn` column detected (removed - prevents accidental target leakage at inference).

### Rationale
- Rejecting a batch on data quality issues would prevent the operations team from getting any predictions - a worse business outcome than a slightly less accurate prediction on imperfect data.
- The preprocessor pipeline is designed to be robust to these issues; warnings surface them for review without halting processing.

---

## ADR-006: Monitoring Strategy - Z-Score on Numerics + Distribution on Categoricals

### Context
Production models degrade when input data distribution shifts from training distribution (covariate shift). Monitoring must detect this automatically per batch.

### Decision
Alert when:
- Any **numeric feature mean** shifts by more than **2.0 standard deviations** from the training baseline (z-score threshold).
- Any **feature missing rate** exceeds **5%**.
- Any **categorical feature category proportion** shifts by more than **15 percentage points** from training.

### Severity Levels
- **HIGH** (z > 3.0 or categorical shift > 25%): immediate investigation required.
- **MEDIUM** (z = 2.0–3.0 or categorical shift 15–25%): monitor closely, consider retraining.
- **LOW** (batch size < 10 rows): reliability caveat only.

### Rationale
- Z-score of 2.0 corresponds to ~97.7% of the normal distribution - alerts on genuinely unusual data, not natural sampling variance.
- 5% missing rate is ~31× the maximum missing rate seen in training data (0.16% for TotalCharges; 0% for all other numeric features).
- 15-point categorical shift is large enough to indicate a meaningful population change (e.g., a new contract type becoming dominant).

### What Is Not Monitored (deliberately)
- **Model output distribution** - cannot distinguish model degradation from genuine behavioural change without ground truth churn labels.
- **Per-prediction SHAP values** - computed on demand in the UI; not stored per prediction due to storage and compute cost.
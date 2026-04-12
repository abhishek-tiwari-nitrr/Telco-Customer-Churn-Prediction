from pathlib import Path

# base path
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
ASSETS_DIR = BASE_DIR / "reports/assets"

# model artifacts
MODEL_PATH = MODEL_DIR / "adaboost_churn_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
FEATURES_PATH = MODEL_DIR / "top20_features.pkl"

# data path
TRAINING_DATA_PATH = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
AUDIT_DB_PATH = DATA_DIR / "audit.db"

# logging
LOG_FILE = LOG_DIR / "app.log"
LOG_LEVEL = "INFO"
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per log file
LOG_BACKUP_COUNT = 3  # Keep 3 rotated files

# model meta data
MODEL_NAME = "AdaBoost Churn Predictor"
MODEL_VERSION = "1.0.0"
MODEL_AUTHOR = "Abhishek Tiwari"
MODEL_AUC = 0.8403
MODEL_RECALL = 0.7727
MODEL_F1 = 0.6208
MODEL_PRECISION = 0.5199
MODEL_ACCURACY = 0.7743

# app ui
APP_TITLE = "Telco Customer Churn Prediction System"
APP_ICON = "🛰️"
GREEN = "#15803D"
ORANGE = "#C2410C"
RED = "#B91C1C"
BLOOD_RED = "#DC2626"
LIGHT_GREEN = "#F0FDF4"
LIGHT_YELLOW = "#FFFBEB"
LIGHT_RED = "#FEF2F2"
GREY = "#64748B"

# input validation
REQUIRED_COLUMNS = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "tenure",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
    "monthlycharges",
    "totalcharges",
]
NUMERIC_COLUMNS = ["seniorcitizen", "tenure", "monthlycharges", "totalcharges"]
CATEGORICAL_COLUMNS = [c for c in REQUIRED_COLUMNS if c not in NUMERIC_COLUMNS]
VALID_CATEGORIES = {
    "gender": ["Male", "Female"],
    "partner": ["Yes", "No"],
    "dependents": ["Yes", "No"],
    "phoneservice": ["Yes", "No"],
    "multiplelines": ["Yes", "No", "No phone service"],
    "internetservice": ["DSL", "Fiber optic", "No"],
    "onlinesecurity": ["Yes", "No", "No internet service"],
    "onlinebackup": ["Yes", "No", "No internet service"],
    "deviceprotection": ["Yes", "No", "No internet service"],
    "techsupport": ["Yes", "No", "No internet service"],
    "streamingtv": ["Yes", "No", "No internet service"],
    "streamingmovies": ["Yes", "No", "No internet service"],
    "contract": ["Month-to-month", "One year", "Two year"],
    "paperlessbilling": ["Yes", "No"],
    "paymentmethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# drift detection thresholds
DRIFT_NUMERIC_Z_THRESHOLD = 2.0  # flag if mean shifts by >2 std devs
DRIFT_MISSING_THRESHOLD = 0.05  # flag if missing rate >5%

# risk tier thresholds
RISK_HIGH_THRESHOLD = 0.7
RISK_MEDIUM_THRESHOLD = 0.4
RISK_LABELS = {"high": "High Risk", "medium": "Medium Risk", "low": "Low Risk"}

# Training Stats - run extract_training_stats to get this
TRAINING_STATS = {
    "seniorcitizen": {
        "mean": 0.1621,
        "std": 0.3686,
        "min": 0.0,
        "max": 1.0,
        "missing_rate": 0.0,
    },
    "tenure": {
        "mean": 32.3711,
        "std": 24.5595,
        "min": 0.0,
        "max": 72.0,
        "missing_rate": 0.0,
    },
    "monthlycharges": {
        "mean": 64.7617,
        "std": 30.09,
        "min": 18.25,
        "max": 118.75,
        "missing_rate": 0.0,
    },
    "totalcharges": {
        "mean": 2283.3004,
        "std": 2266.7714,
        "min": 18.8,
        "max": 8684.8,
        "missing_rate": 0.0016,
    },
    "churn_rate": 0.2654,
    "contract_dist": {"Month-to-month": 0.5502, "Two year": 0.2407, "One year": 0.2091},
    "internetservice_dist": {"Fiber optic": 0.4396, "DSL": 0.3437, "No": 0.2167},
    "paymentmethod_dist": {
        "Electronic check": 0.3358,
        "Mailed check": 0.2289,
        "Bank transfer (automatic)": 0.2192,
        "Credit card (automatic)": 0.2161,
    },
    "partner_dist": {"No": 0.517, "Yes": 0.483},
    "dependents_dist": {"No": 0.7004, "Yes": 0.2996},
    "phoneservice_dist": {"Yes": 0.9032, "No": 0.0968},
}

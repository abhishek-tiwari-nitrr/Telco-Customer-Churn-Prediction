import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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


def _make_customer(**overrides) -> dict:
    base = {
        "gender": "Male",
        "seniorcitizen": 0,
        "partner": "Yes",
        "dependents": "No",
        "tenure": 12,
        "phoneservice": "Yes",
        "multiplelines": "No",
        "internetservice": "Fiber optic",
        "onlinesecurity": "No",
        "onlinebackup": "No",
        "deviceprotection": "No",
        "techsupport": "No",
        "streamingtv": "No",
        "streamingmovies": "No",
        "contract": "Month-to-month",
        "paperlessbilling": "Yes",
        "paymentmethod": "Electronic check",
        "monthlycharges": 70.35,
        "totalcharges": 844.20,
    }
    base.update(overrides)
    return base


@pytest.fixture
def valid_customer() -> dict:
    return _make_customer()


@pytest.fixture
def valid_df() -> pd.DataFrame:
    """10-row DataFrame, all valid, varied values."""
    rows = [
        _make_customer(
            tenure=1,
            monthlycharges=29.85,
            totalcharges=29.85,
            contract="Month-to-month",
            internetservice="DSL",
            paymentmethod="Electronic check",
        ),
        _make_customer(
            tenure=6,
            monthlycharges=56.95,
            totalcharges=341.70,
            contract="Month-to-month",
            internetservice="DSL",
            paymentmethod="Mailed check",
        ),
        _make_customer(
            tenure=12,
            monthlycharges=70.70,
            totalcharges=848.40,
            contract="One year",
            internetservice="Fiber optic",
            paymentmethod="Bank transfer (automatic)",
        ),
        _make_customer(
            tenure=24,
            monthlycharges=89.10,
            totalcharges=2138.40,
            contract="One year",
            internetservice="DSL",
            paymentmethod="Mailed check",
        ),
        _make_customer(
            tenure=36,
            monthlycharges=20.00,
            totalcharges=720.00,
            contract="Two year",
            internetservice="No",
            paymentmethod="Credit card (automatic)",
            seniorcitizen=1,
        ),
        _make_customer(
            tenure=48,
            monthlycharges=104.80,
            totalcharges=5030.40,
            contract="Two year",
            internetservice="Fiber optic",
            paymentmethod="Credit card (automatic)",
        ),
        _make_customer(
            tenure=60,
            monthlycharges=90.45,
            totalcharges=5427.00,
            contract="Two year",
            internetservice="Fiber optic",
            paymentmethod="Bank transfer (automatic)",
            onlinesecurity="Yes",
        ),
        _make_customer(
            tenure=72,
            monthlycharges=115.00,
            totalcharges=8280.00,
            contract="Month-to-month",
            internetservice="Fiber optic",
            paymentmethod="Electronic check",
            techsupport="Yes",
        ),
        _make_customer(
            tenure=3,
            monthlycharges=45.20,
            totalcharges=135.60,
            contract="Month-to-month",
            internetservice="DSL",
            paymentmethod="Mailed check",
            gender="Female",
        ),
        _make_customer(
            tenure=18,
            monthlycharges=79.65,
            totalcharges=1433.70,
            contract="One year",
            internetservice="No",
            paymentmethod="Bank transfer (automatic)",
        ),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def training_like_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    rows = []
    for _ in range(n):
        tenure = int(np.clip(np.random.exponential(30), 0, 72))
        monthly = round(float(np.random.uniform(18.0, 120.0)), 2)
        total = round(monthly * max(tenure, 1) * np.random.uniform(0.8, 1.1), 2)
        contract = np.random.choice(
            ["Month-to-month", "One year", "Two year"], p=[0.55, 0.24, 0.21]
        )
        internet = np.random.choice(["Fiber optic", "DSL", "No"], p=[0.44, 0.34, 0.22])
        payment = np.random.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            p=[0.34, 0.23, 0.22, 0.21],
        )
        rows.append(
            _make_customer(
                tenure=tenure,
                monthlycharges=monthly,
                totalcharges=total,
                contract=contract,
                internetservice=internet,
                paymentmethod=payment,
                gender=np.random.choice(["Male", "Female"]),
                partner=np.random.choice(["Yes", "No"]),
                seniorcitizen=int(np.random.choice([0, 1], p=[0.84, 0.16])),
            )
        )
    return pd.DataFrame(rows)


@pytest.fixture
def mock_preprocessor(training_like_df):
    num_cols = NUMERIC_COLUMNS
    cat_cols = CATEGORICAL_COLUMNS

    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )
    preprocessor.fit(training_like_df)
    return preprocessor

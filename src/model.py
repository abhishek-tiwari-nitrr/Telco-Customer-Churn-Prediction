import pandas as pd
import numpy as np
import joblib
from dataclasses import dataclass
import time
from src.config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    FEATURES_PATH,
    MODEL_VERSION,
    RISK_HIGH_THRESHOLD,
    RISK_MEDIUM_THRESHOLD,
    RISK_LABELS,
)
from src.logger import get_logger

logger = get_logger(__name__)


# result structure
@dataclass
class PredictionOutput:
    result_df: pd.DataFrame
    processing_ms: int
    model_version: str
    total: int
    predicted_churn: int
    churn_rate: float
    high_risk: int
    medium_risk: int
    low_risk: int


def _get_risk_tier(prob: float) -> str:
    if prob >= RISK_HIGH_THRESHOLD:
        return RISK_LABELS["high"]
    elif prob >= RISK_MEDIUM_THRESHOLD:
        return RISK_LABELS["medium"]
    return RISK_LABELS["low"]


# Singleton
class ChurnModel:
    _instance = None  # to store single instance of the class - class variable

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)  # create a fresh instance of Singleton
            cls._instance._loaded = False  # initial state
        return cls._instance

    def load(self):
        if self._loaded:
            return
        try:
            logger.info(f"Loading model artifacts from {MODEL_PATH.parent}")
            self.model = joblib.load(MODEL_PATH)
            self.preprocessor = joblib.load(PREPROCESSOR_PATH)
            self.top_features = joblib.load(FEATURES_PATH)
            self._loaded = True  # update state
            logger.info(
                f"Model loaded successfully | version={MODEL_VERSION} | features={len(self.top_features)}"
            )
        except FileNotFoundError as e:
            logger.critical(f"Model artifact not found: {e}")
            raise
        except Exception as e:
            logger.critical(f"Failed to load model: {e}")
            raise

    def predict(
        self, df: pd.DataFrame, customer_ids: str | None = None
    ) -> PredictionOutput:
        if not self._loaded:
            self.load()
        logger.info(f"Starting inference | rows={len(df)}")
        start_time = time.time()
        try:
            encoded = self.preprocessor.transform(df)
            encoded_df = pd.DataFrame(
                encoded, columns=self.preprocessor.get_feature_names_out()
            )
            probability = self.model.predict_proba(encoded_df[self.top_features])[:, 1]
            predict = self.model.predict(encoded_df[self.top_features])
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

        processing_ms = int((time.time() - start_time) * 1000)

        # result df
        result_df = df.copy().reset_index(drop=True)
        if customer_ids is not None:
            result_df.insert(0, "customer_id", customer_ids.reset_index(drop=True))
        result_df.insert(
            (
                result_df.columns.get_loc("customer_id") + 1
                if "customer_id" in result_df.columns
                else 0
            ),
            "churn_prediction",
            ["Yes" if p == 1 else "No" for p in predict],
        )
        result_df.insert(
            result_df.columns.get_loc("churn_prediction") + 1,
            "churn_probability",
            np.round(probability, 4),
        )
        result_df.insert(
            result_df.columns.get_loc("churn_probability") + 1,
            "risk_tier",
            [_get_risk_tier(p) for p in probability],
        )

        # summary stats
        total = len(result_df)
        predicted_churn = int((result_df["churn_prediction"] == "Yes").sum())
        churn_rate = round(predicted_churn / total, 4)
        high_risk = int(result_df["risk_tier"].str.contains("High").sum())
        medium_risk = int(result_df["risk_tier"].str.contains("Medium").sum())
        low_risk = int(result_df["risk_tier"].str.contains("Low").sum())

        logger.info(
            f"Inference complete | ms={processing_ms} | total={total} | churn={predicted_churn} ({churn_rate:.1%}) | high={high_risk} | medium={medium_risk} | low={low_risk}"
        )

        return PredictionOutput(
            result_df=result_df,
            processing_ms=processing_ms,
            model_version=MODEL_VERSION,
            total=total,
            predicted_churn=predicted_churn,
            churn_rate=churn_rate,
            high_risk=high_risk,
            medium_risk=medium_risk,
            low_risk=low_risk,
        )


# Module-level singleton
churn_model = ChurnModel()

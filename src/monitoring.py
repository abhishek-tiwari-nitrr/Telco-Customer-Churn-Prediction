import pandas as pd
from dataclasses import dataclass
from typing import List, Dict
from src.config import (
    TRAINING_STATS,
    NUMERIC_COLUMNS,
    DRIFT_NUMERIC_Z_THRESHOLD,
    DRIFT_MISSING_THRESHOLD,
)
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DriftAlert:
    feature: str
    alert_type: str
    severity: str
    message: str
    training_value: float
    upload_value: float
    deviation: float


@dataclass
class DriftReport:
    has_drift: bool
    alerts: List[DriftAlert]
    summary: str


def detect_drift(upload_df: pd.DataFrame, batch_id: str = "") -> DriftReport:
    alerts = []
    df = upload_df.copy()
    df.columns = df.columns.str.lower().str.strip()
    df["totalcharges"] = pd.to_numeric(
        df.get("totalcharges", pd.Series(dtype=float)), errors="coerce"
    )

    logger.info(f"Running drift detection | batch_id={batch_id} | rows={len(df)}")

    for col in NUMERIC_COLUMNS:
        train_stats = TRAINING_STATS[col]
        upload_mean = float(df[col].mean())
        train_mean = train_stats["mean"]
        train_std = train_stats["std"]

        if train_std == 0:
            continue

        z_score = abs(upload_mean - train_mean) / train_std

        if z_score > DRIFT_NUMERIC_Z_THRESHOLD:
            severity = "HIGH" if z_score > 3 else "MEDIUM"
            alerts.append(
                DriftAlert(
                    feature=col,
                    alert_type="mean_shift",
                    training_value=train_mean,
                    upload_value=upload_mean,
                    deviation=round(z_score, 3),
                    severity=severity,
                    message=f"'{col}' mean shifted by {z_score:.2f} std devs  (training: {train_mean:.2f}, upload: {upload_mean:.2f})",
                )
            )
            logger.warning(f"Drift | {col} | z={z_score:.2f} | severity={severity}")

        upload_missing = float(df[col].isna().mean())
        if upload_missing > DRIFT_MISSING_THRESHOLD:
            alerts.append(
                DriftAlert(
                    feature=col,
                    alert_type="missing_rate",
                    training_value=train_stats["missing_rate"],
                    upload_value=upload_missing,
                    deviation=round(upload_missing - train_stats["missing_rate"], 3),
                    severity="HIGH",
                    message=f"'{col}' has {upload_missing:.1%} missing values (threshold: {DRIFT_MISSING_THRESHOLD:.0%})",
                )
            )

    cat_check_cols = ["contract", "internetservice", "paymentmethod"]
    for col in cat_check_cols:
        key = f"{col}_dist"

        train_dist = TRAINING_STATS[key]
        upload_dist = df[col].value_counts(normalize=True).to_dict()

        for category, train_pct in train_dist.items():
            upload_pct = upload_dist.get(category, 0.0)
            deviation = abs(upload_pct - train_pct)

            if deviation > 0.15:  # >15% shift in any category
                severity = "HIGH" if deviation > 0.25 else "MEDIUM"
                alerts.append(
                    DriftAlert(
                        feature=f"{col}_{category}",
                        alert_type="category_distribution",
                        training_value=round(train_pct, 4),
                        upload_value=round(upload_pct, 4),
                        deviation=round(deviation, 4),
                        severity=severity,
                        message=f"'{col}={category}' proportion shifted "
                        f"from {train_pct:.1%} to {upload_pct:.1%}",
                    )
                )

    if len(df) < 10:
        alerts.append(
            DriftAlert(
                feature="dataset_size",
                alert_type="small_batch",
                training_value=7032,
                upload_value=len(df),
                deviation=len(df),
                severity="LOW",
                message=f"Small batch detected ({len(df)} rows) — predictions may be less reliable",
            )
        )

    has_drift = any(a.severity in ["HIGH", "MEDIUM"] for a in alerts)

    report = DriftReport(
        has_drift=has_drift,
        alerts=alerts,
        summary={
            "total_alerts": len(alerts),
            "high": sum(1 for a in alerts if a.severity == "HIGH"),
            "medium": sum(1 for a in alerts if a.severity == "MEDIUM"),
            "low": sum(1 for a in alerts if a.severity == "LOW"),
            "batch_id": batch_id,
        },
    )

    logger.info(
        f"Drift detection complete | alerts={len(alerts)} | "
        f"high={report.summary['high']} | medium={report.summary['medium']}"
    )
    return report

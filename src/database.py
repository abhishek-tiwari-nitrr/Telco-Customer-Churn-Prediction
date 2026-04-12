import sqlite3
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import List, Dict, Any

from src.config import AUDIT_DB_PATH
from src.logger import get_logger

logger = get_logger(__name__)

# Tables -
# 1. prediction_batches - Batch level summary
# 2. prediction_records - Customer level predictions
# 3. drift_alerts - Data drift monitoring
# 4. validation_errors - Data quality issues

SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_batches (
    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT,
    uploaded_at TEXT,
    total_customers INTEGER,
    predicted_churn INTEGER,
    churn_rate REAL,
    high_risk INTEGER,
    medium_risk INTEGER,
    low_risk INTEGER,
    model_version TEXT,
    processing_ms INTEGER,
    status TEXT,
    error_message TEXT
    );

CREATE TABLE IF NOT EXISTS prediction_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    customer_id TEXT,
    predicted_at TEXT,
    churn_prediction TEXT,
    churn_probability REAL,
    risk_tier TEXT,
    tenure REAL,
    monthly_charges REAL, 
    total_charges REAL,
    contract TEXT,
    internet_service TEXT,
    payment_method TEXT,
    FOREIGN KEY (batch_id) REFERENCES prediction_batches(batch_id)
    );

CREATE TABLE IF NOT EXISTS drift_alerts(
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT,
    alerted_at TEXT,
    feature TEXT,
    alert_type TEXT,
    training_value REAL,
    upload_value REAL,
    deviation REAL,
    severity TEXT,
    FOREIGN KEY (batch_id) REFERENCES prediction_batches(batch_id)
    );

CREATE TABLE IF NOT EXISTS validation_errors(
    error_id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    logged_at TEXT NOT NULL,
    error_type TEXT NOT NULL,
    error_detail TEXT NOT NULL,
    row_count INTEGER,
    FOREIGN KEY (batch_id) REFERENCES prediction_batches(batch_id)
    );
"""


@contextmanager
def get_connection():
    AUDIT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(AUDIT_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


def init_db():
    try:
        with get_connection() as conn:
            conn.executescript(SCHEMA)
        logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        logger.exception(f"Failed to initialize database: {e}")
        raise


# Write
def log_batch(
    filename: str,
    total_customers: int,
    predicted_churn: int,
    churn_rate: float,
    high_risk: int,
    medium_risk: int,
    low_risk: int,
    model_version: str,
    processing_ms: int,
    status: str = "success",
    error_message: str | None = None,
) -> str:
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO prediction_batches (filename, uploaded_at, total_customers, predicted_churn, churn_rate, high_risk, medium_risk, low_risk, model_version, processing_ms, status, error_message) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                filename,
                datetime.now(timezone.utc).isoformat(),
                total_customers,
                predicted_churn,
                round(churn_rate, 4),
                high_risk,
                medium_risk,
                low_risk,
                model_version,
                processing_ms,
                status,
                error_message,
            ),
        )
        batch_id = cursor.lastrowid
    logger.info(
        f"Batch logged | batch_id={batch_id} | file={filename} | customers={total_customers}"
    )
    return batch_id


def log_predictions(batch_id: int, records: List[Dict[str, Any]]):
    rows = [
        (
            batch_id,
            r.get("customer_id"),
            datetime.now(timezone.utc).isoformat(),
            r["churn_prediction"],
            round(r["churn_probability"], 4),
            r["risk_tier"],
            r.get("tenure"),
            r.get("monthlycharges"),
            r.get("totalcharges"),
            r.get("contract"),
            r.get("internetservice"),
            r.get("paymentmethod"),
        )
        for r in records
    ]
    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO prediction_records (batch_id, customer_id, predicted_at, churn_prediction, churn_probability, risk_tier, tenure, monthly_charges,  total_charges, contract, internet_service, payment_method) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
    logger.info(f"Logged {len(rows)} customer predictions for batch_id={batch_id}")


def log_drift_alert(
    batch_id: int,
    feature: str,
    alert_type: str,
    training_value: float,
    upload_value: float,
    deviation: float,
    severity: str,
):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO drift_alerts (batch_id, alerted_at, feature, alert_type, training_value, upload_value, deviation, severity) VALUES (?,?,?,?,?,?,?,?)""",
            (
                batch_id,
                datetime.now(timezone.utc).isoformat(),
                feature,
                alert_type,
                round(training_value, 4),
                round(upload_value, 4),
                round(deviation, 4),
                severity,
            ),
        )
    logger.warning(
        f"Drift alert | feature={feature} | type={alert_type} | severity={severity} | deviation={deviation:.4f}"
    )


def log_validation_error(
    batch_id: int, error_type: str, error_detail: str, row_count: int = 0
):
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO validation_errors (batch_id, logged_at, error_type, error_detail, row_count) VALUES (?,?,?,?,?)""",
            (
                batch_id,
                datetime.now(timezone.utc).isoformat(),
                error_type,
                error_detail,
                row_count,
            ),
        )
    logger.warning(f"Validation error | type={error_type} | detail={error_detail}")


# Read
def get_batch_history(limit: int = 50) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM prediction_batches ORDER BY uploaded_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_batch_records(batch_id: int) -> List[Dict]:
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT * FROM prediction_records WHERE batch_id = ? ORDER BY churn_probability DESC""",
            (batch_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_drift_alerts(batch_id: int | None = None, limit: int = 10) -> List[Dict]:
    with get_connection() as conn:
        if batch_id:
            rows = conn.execute(
                """SELECT * FROM drift_alerts WHERE batch_id = ? ORDER BY alerted_at DESC""",
                (batch_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM drift_alerts ORDER BY alerted_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def get_summary_stats() -> Dict:
    with get_connection() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS total_batches, SUM(total_customers) AS total_customers, SUM(predicted_churn) AS total_churned, AVG(churn_rate) AS avg_churn_rate, SUM(high_risk) AS total_high_risk FROM prediction_batches WHERE status = 'success' """
        ).fetchone()
    return dict(row) if row else {}

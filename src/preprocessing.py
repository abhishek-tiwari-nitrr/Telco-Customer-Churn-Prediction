import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from src.config import REQUIRED_COLUMNS, NUMERIC_COLUMNS, VALID_CATEGORIES

from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    original_rows: int = 0
    cleaned_rows: int = 0
    customer_ids: Optional[pd.Series] = None


def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationResult]:
    result = ValidationResult(is_valid=True, original_rows=len(df))
    df = df.copy()
    logger.info(f"Starting validation | rows={len(df)} | cols={len(df.columns)}")
    df.columns = df.columns.str.lower().str.strip()

    if "customerid" in df.columns:
        result.customer_ids = df["customerid"].astype(str)
        df = df.drop(columns=["customerid"])
        logger.debug("customerid column extracted")

    if "churn" in df.columns:
        df = df.drop(columns=["churn"])
        result.warnings.append(
            "'Churn' column found and removed - it is the target variable"
        )

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        result.errors.append(f"Missing required columns: {missing_cols}")
        result.is_valid = False
        logger.error(f"Missing columns: {missing_cols}")
        return df, result

    for col in NUMERIC_COLUMNS:
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after = df[col].isna().sum()
        if after > before:
            result.warnings.append(
                f"'{col}': {after - before} non numeric values coerced to NaN"
            )

    for col, valid_vals in VALID_CATEGORIES.items():
        if col not in df.columns:
            continue
        actual_vals = df[col].dropna().unique().tolist()
        unexpected = [v for v in actual_vals if v not in valid_vals]
        if unexpected:
            result.warnings.append(
                f"'{col}': unexpected values {unexpected} - will be handled by encoder"
            )

    total_missing = df.isnull().sum().sum()
    if total_missing > 0:
        missing_summary = df.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0].to_dict()
        result.warnings.append(
            f"Missing values detected: {missing_summary} - will be imputed by pipeline"
        )
        logger.warning(f"Missing values: {missing_summary}")

    result.cleaned_rows = len(df)
    result.rows_dropped = result.original_rows - result.cleaned_rows

    if len(df) == 0:
        result.errors.append("No valid rows remaining after cleaning")
        result.is_valid = False

    logger.info(
        f"Validation complete | valid={result.is_valid} | rows={result.cleaned_rows} | warnings={len(result.warnings)} | errors={len(result.errors)}"
    )
    return df, result

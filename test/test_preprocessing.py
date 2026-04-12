import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

from conftest import (
    REQUIRED_COLUMNS,
    NUMERIC_COLUMNS,
    NUMERIC_BOUNDS,
    VALID_CATEGORIES,
)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    rows_dropped: int = 0
    original_rows: int = 0
    cleaned_rows: int = 0
    customer_ids: Optional[pd.Series] = None


def validate_and_clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, ValidationResult]:
    result = ValidationResult(is_valid=True, original_rows=len(df))
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    if "customerid" in df.columns:
        result.customer_ids = df["customerid"].astype(str)
        df = df.drop(columns=["customerid"])

    if "churn" in df.columns:
        df = df.drop(columns=["churn"])
        result.warnings.append("'churn' column found and removed")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        result.errors.append(f"Missing required columns: {missing_cols}")
        result.is_valid = False
        return df, result

    for col in NUMERIC_COLUMNS:
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after = df[col].isna().sum()
        if after > before:
            result.warnings.append(
                f"'{col}': {after - before} non-numeric values coerced to NaN"
            )

    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in df.columns:
            continue
        oob = df[col].dropna()
        oob = oob[(oob < lo) | (oob > hi)]
        if len(oob):
            result.warnings.append(f"'{col}': {len(oob)} values outside [{lo}, {hi}]")

    for col, valid_vals in VALID_CATEGORIES.items():
        if col not in df.columns:
            continue
        unexpected = [v for v in df[col].dropna().unique() if v not in valid_vals]
        if unexpected:
            result.warnings.append(f"'{col}': unexpected values {unexpected}")

    total_missing = df.isnull().sum().sum()
    if total_missing:
        result.warnings.append("Missing values detected — will be imputed")

    result.cleaned_rows = len(df)
    result.rows_dropped = result.original_rows - result.cleaned_rows

    if len(df) == 0:
        result.errors.append("No valid rows remaining after cleaning")
        result.is_valid = False

    return df, result


def test_valid_df_passes(valid_df):
    _, result = validate_and_clean(valid_df)
    assert result.is_valid is True
    assert result.errors == []


def test_missing_required_column_fails(valid_df):
    df = valid_df.drop(columns=["contract"])
    _, result = validate_and_clean(df)
    assert result.is_valid is False
    assert any("contract" in e for e in result.errors)


def test_multiple_missing_columns_all_reported(valid_df):
    df = valid_df.drop(columns=["contract", "tenure", "monthlycharges"])
    _, result = validate_and_clean(df)
    assert result.is_valid is False
    error = result.errors[0]
    assert "contract" in error
    assert "tenure" in error
    assert "monthlycharges" in error


def test_column_names_lowercased(valid_df):
    df = valid_df.copy()
    df.columns = [c.upper() for c in df.columns]
    _, result = validate_and_clean(df)
    assert result.is_valid is True


def test_column_names_stripped_of_whitespace(valid_df):
    df = valid_df.copy()
    df.columns = [f"  {c}  " for c in df.columns]
    _, result = validate_and_clean(df)
    assert result.is_valid is True


def test_extra_columns_allowed(valid_df):
    df = valid_df.copy()
    df["extra_column_xyz"] = "irrelevant"
    _, result = validate_and_clean(df)
    assert result.is_valid is True


def test_empty_dataframe_fails():
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    _, result = validate_and_clean(df)
    assert result.is_valid is False
    assert len(result.errors) > 0


def test_single_row_df_passes(valid_customer):
    df = pd.DataFrame([valid_customer])
    _, result = validate_and_clean(df)
    assert result.is_valid is True
    assert result.cleaned_rows == 1


def test_no_customerid_still_valid(valid_df):
    assert "customerid" not in valid_df.columns
    _, result = validate_and_clean(valid_df)
    assert result.is_valid is True
    assert result.customer_ids is None


def test_string_numbers_coerced(valid_df):
    df = valid_df.copy()
    df["tenure"] = df["tenure"].astype(str)
    df["monthlycharges"] = df["monthlycharges"].astype(str)
    df["totalcharges"] = df["totalcharges"].astype(str)
    cleaned, result = validate_and_clean(df)
    assert result.is_valid is True
    assert pd.api.types.is_numeric_dtype(cleaned["tenure"])
    assert pd.api.types.is_numeric_dtype(cleaned["monthlycharges"])


def test_all_numeric_columns_remain_numeric(valid_df):
    cleaned, _ = validate_and_clean(valid_df)
    for col in NUMERIC_COLUMNS:
        assert pd.api.types.is_numeric_dtype(
            cleaned[col]
        ), f"'{col}' should be numeric after cleaning"


def test_negative_tenure_triggers_warning(valid_df):
    df = valid_df.copy()
    df.loc[0, "tenure"] = -5
    _, result = validate_and_clean(df)
    assert result.is_valid is True
    assert any("tenure" in w for w in result.warnings)


def test_tenure_over_max_triggers_warning(valid_df):
    df = valid_df.copy()
    df.loc[0, "tenure"] = 999
    _, result = validate_and_clean(df)
    assert any("tenure" in w for w in result.warnings)


def test_negative_monthly_charges_triggers_warning(valid_df):
    df = valid_df.copy()
    df.loc[0, "monthlycharges"] = -10.0
    _, result = validate_and_clean(df)
    assert any("monthlycharges" in w for w in result.warnings)


def test_valid_bounds_produce_no_bounds_warning(valid_df):
    _, result = validate_and_clean(valid_df)
    assert not any("outside" in w for w in result.warnings)


def test_all_missing_numeric_column_flagged(valid_df):
    df = valid_df.copy()
    df["totalcharges"] = np.nan
    _, result = validate_and_clean(df)
    assert any("totalcharges" in w or "missing" in w.lower() for w in result.warnings)


def test_valid_categories_produce_no_category_warnings(valid_df):
    _, result = validate_and_clean(valid_df)
    assert not any("unexpected" in w.lower() for w in result.warnings)


def test_case_mismatch_detected(valid_df):
    df = valid_df.copy()
    df.loc[0, "partner"] = "YES"
    _, result = validate_and_clean(df)
    assert any("partner" in w for w in result.warnings)


def test_original_rows_counted_correctly(valid_df):
    _, result = validate_and_clean(valid_df)
    assert result.original_rows == len(valid_df)


def test_cleaned_rows_matches_original_for_valid_data(valid_df):
    _, result = validate_and_clean(valid_df)
    assert result.cleaned_rows == result.original_rows


def test_rows_dropped_is_zero_for_valid_data(valid_df):
    _, result = validate_and_clean(valid_df)
    assert result.rows_dropped == 0


def test_is_valid_true_for_clean_data(valid_df):
    _, result = validate_and_clean(valid_df)
    assert result.is_valid is True


def test_is_valid_false_for_missing_columns(valid_df):
    df = valid_df.drop(columns=["tenure"])
    _, result = validate_and_clean(df)
    assert result.is_valid is False


def test_errors_list_empty_for_valid_data(valid_df):
    _, result = validate_and_clean(valid_df)
    assert result.errors == []

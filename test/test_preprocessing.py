import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import validate_and_clean

from conftest import REQUIRED_COLUMNS, NUMERIC_COLUMNS


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

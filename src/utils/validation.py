"""Data validation and schema checks akin to Great Expectations."""
from __future__ import annotations

import pandas as pd
from typing import Dict, List, Tuple


def check_schema(df: pd.DataFrame, expected: Dict[str, str]) -> List[str]:
    errors = []
    for col, dtype in expected.items():
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
            continue
        if dtype == "datetime":
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                errors.append(f"Column {col} not datetime")
        elif dtype == "float":
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Column {col} not numeric")
    return errors


def check_missing(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [col for col in columns if df[col].isna().any()]


def check_ranges(df: pd.DataFrame, ranges: Dict[str, Tuple[float, float]]) -> List[str]:
    errors = []
    for col, (low, high) in ranges.items():
        if col in df.columns:
            if (df[col] < low).any() or (df[col] > high).any():
                errors.append(f"{col} outside range {low}-{high}")
    return errors

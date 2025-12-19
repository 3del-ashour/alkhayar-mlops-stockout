import pandas as pd
from src.utils import validation


def test_check_schema_detects_missing():
    df = pd.DataFrame({"A": [1]})
    errors = validation.check_schema(df, {"A": "float", "B": "float"})
    assert "Missing column: B" in errors


def test_check_ranges():
    df = pd.DataFrame({"x": [0, 5]})
    errors = validation.check_ranges(df, {"x": (1, 4)})
    assert errors

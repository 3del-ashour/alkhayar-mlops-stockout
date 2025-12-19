"""Data loading utilities with robust date parsing and chunked ingestion."""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Iterator, List, Optional


DATE_COLS = ["Date", "LastUpdatedAt", "OpeningDate"]


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def read_csv_chunks(path: Path, chunksize: int = 50000, usecols: Optional[List[str]] = None) -> Iterator[pd.DataFrame]:
    """Yield dataframe chunks with parsed dates."""
    for chunk in pd.read_csv(path, chunksize=chunksize, usecols=usecols):
        yield _parse_dates(chunk)


def read_csv_full(path: Path, usecols: Optional[List[str]] = None) -> pd.DataFrame:
    """Read full CSV with parsed dates, falling back to chunk concat if large."""
    try:
        df = pd.read_csv(path, usecols=usecols)
        return _parse_dates(df)
    except MemoryError:
        chunks = list(read_csv_chunks(path, usecols=usecols))
        return pd.concat(chunks, ignore_index=True)

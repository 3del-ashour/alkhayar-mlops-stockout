"""Monitoring utilities for drift and continued model evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy
from typing import Dict, Tuple


def population_stability_index(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected_counts, bin_edges = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    expected_pct = expected_counts / (expected_counts.sum() + 1e-8)
    actual_pct = actual_counts / (actual_counts.sum() + 1e-8)
    return float(np.sum((actual_pct - expected_pct) * np.log((actual_pct + 1e-8) / (expected_pct + 1e-8))))


def categorical_drift(expected: pd.Series, actual: pd.Series) -> float:
    expected_freq = expected.value_counts(normalize=True)
    actual_freq = actual.value_counts(normalize=True)
    aligned = expected_freq.align(actual_freq, fill_value=1e-8)
    return float(entropy(aligned[0], aligned[1]))


def rolling_window_metrics(df: pd.DataFrame, target_col: str, pred_col: str, window: int = 7) -> pd.DataFrame:
    df_sorted = df.sort_values("Date")
    df_sorted["rolling_f1"] = (
        2
        * (df_sorted[target_col] * df_sorted[pred_col]).rolling(window).sum()
        / (
            (df_sorted[target_col] + df_sorted[pred_col]).rolling(window).sum().replace(0, np.nan)
        )
    )
    return df_sorted


def threshold_check(psi: float, kl: float, psi_limit: float = 0.2, kl_limit: float = 0.5) -> Dict[str, bool]:
    return {"psi_ok": psi < psi_limit, "kl_ok": kl < kl_limit}

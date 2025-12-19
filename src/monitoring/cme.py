"""Continued Model Evaluation and drift detection."""
from __future__ import annotations

import mlflow
import pandas as pd
from scipy.stats import entropy
from typing import Dict

from src import config
from src.features.build_features import build_feature_matrix, create_label
from src.utils import mlflow_utils, monitoring


def run_cme(df_sales: pd.DataFrame, df_stock: pd.DataFrame, reference_preds: pd.Series, horizon_days: int = config.DEFAULT_HORIZON_DAYS) -> Dict:
    mlflow_utils.setup_mlflow()
    labeled = create_label(df_sales, df_stock, horizon_days)
    X_sparse, X_numeric, y, _ = build_feature_matrix(labeled)
    _ = (X_sparse, X_numeric, y)  # placeholders for potential future use
    psi = monitoring.population_stability_index(reference_preds, labeled["label_stockout"])
    kl = entropy(reference_preds + 1e-8, labeled["label_stockout"] + 1e-8)
    drift_ok = monitoring.threshold_check(psi, kl)

    with mlflow.start_run(run_name="cme"):
        mlflow.log_metric("psi", psi)
        mlflow.log_metric("kl", kl)
        for k, v in drift_ok.items():
            mlflow.log_metric(k, int(v))

    fallback_triggered = not drift_ok["psi_ok"] or not drift_ok["kl_ok"]
    rollback_version = None
    baseline_rule = None
    if fallback_triggered:
        rollback_version = mlflow_utils.rollback_production()
        if rollback_version is None:
            baseline_rule = labeled.apply(
                lambda row: int(row["CurrentQuantity"] < row["SafetyStockLevel"]), axis=1
            ).tolist()
    return {
        "psi": psi,
        "kl": kl,
        "drift_ok": drift_ok,
        "fallback": fallback_triggered,
        "rollback_version": rollback_version,
        "baseline_rule": baseline_rule,
    }

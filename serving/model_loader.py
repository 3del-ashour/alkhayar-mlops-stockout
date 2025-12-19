"""Utility to load production model and preprocess requests."""
from __future__ import annotations

import pandas as pd
import mlflow
from scipy.sparse import csr_matrix, hstack

from src.features.build_features import build_feature_matrix
from src.utils import mlflow_utils
from src import config


def load_model():
    mlflow_utils.setup_mlflow()
    client = mlflow.MlflowClient()
    versions = client.get_latest_versions(config.MLFLOW_MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError("No Production model available")
    model_uri = versions[0].source
    model = mlflow.lightgbm.load_model(model_uri)
    return model, versions[0].version


def prepare_features(payload: dict):
    df = pd.DataFrame([payload])
    df["LastUpdatedAt"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
    df["future_sales"] = df.get("future_sales", 0)
    df["projected_stock"] = df["CurrentQuantity"] - df["ReservedQuantity"] - df["future_sales"]
    df["label_stockout"] = 0
    X_sparse, X_numeric, _, _ = build_feature_matrix(df)
    return hstack([csr_matrix(X_numeric), X_sparse]).tocsr()

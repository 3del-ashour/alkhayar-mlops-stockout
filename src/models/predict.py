"""Batch prediction utilities."""
from __future__ import annotations

import mlflow
import numpy as np
from scipy.sparse import csr_matrix, hstack

from src import config
from src.features.build_features import build_feature_matrix
from src.utils import mlflow_utils


def load_production_model():
    mlflow_utils.setup_mlflow()
    client = mlflow.MlflowClient()
    prods = client.get_latest_versions(config.MLFLOW_MODEL_NAME, stages=["Production"])
    if not prods:
        raise RuntimeError("No production model found")
    model_uri = prods[0].source
    return mlflow.lightgbm.load_model(model_uri)


def predict(df_features):
    model = load_production_model()
    X_sparse, X_numeric, _, _ = build_feature_matrix(df_features)
    X_combined = hstack([csr_matrix(X_numeric), X_sparse]).tocsr()
    probs = model.predict(X_combined)
    return probs

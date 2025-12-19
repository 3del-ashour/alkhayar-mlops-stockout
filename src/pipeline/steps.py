"""Prefect tasks for the MLOps pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import mlflow
import pandas as pd
from prefect import task
from scipy.sparse import csr_matrix, hstack

from src import config
from src.features.build_features import build_feature_matrix, create_label
from src.models.train import train_model
from src.models.evaluate import evaluate_predictions
from src.utils import io, validation, mlflow_utils


@task
def ingest_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    sales = io.read_csv_full(data_dir / "sales_transactions.csv")
    stock = io.read_csv_full(data_dir / "stock_current.csv")
    movement = io.read_csv_full(data_dir / "stock_movement.csv")
    return {"sales": sales, "stock": stock, "movement": movement}


@task
def validate_data(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    for name, df in data.items():
        key = name
        if name == "sales":
            key = "sales_transactions"
        elif name == "movement":
            key = "stock_movement"
        schema_errors = validation.check_schema(df, config.SCHEMA.get(key, {}))
        if schema_errors:
            raise ValueError(f"Schema errors in {name}: {schema_errors}")
    return data


@task
def engineer_features(data: Dict[str, pd.DataFrame], horizon_days: int):
    labeled = create_label(data["sales"], data["stock"], horizon_days, data.get("movement"))
    X_sparse, X_numeric, y, feature_names = build_feature_matrix(labeled)
    return {"X_sparse": X_sparse, "X_numeric": X_numeric, "y": y, "feature_names": feature_names, "df": labeled}


@task
def train(features: Dict[str, object]):
    return train_model(features["X_sparse"], features["X_numeric"], features["y"])


@task
def evaluate_run(features: Dict[str, object], metrics: Dict[str, float]):
    X_combined = hstack([csr_matrix(features["X_numeric"]), features["X_sparse"]]).tocsr()
    model_uri = f"runs:/{metrics['run_id']}/model"
    model = mlflow.lightgbm.load_model(model_uri)
    preds = model.predict(X_combined)
    eval_metrics = evaluate_predictions(features["y"], preds)
    mlflow_utils.log_params_and_metrics({}, {f"eval_{k}": v for k, v in eval_metrics.items()})
    return eval_metrics


@task
def promote_if_good(metrics: Dict[str, float]):
    if metrics.get("val_f1", 0) >= config.ACCEPTANCE_THRESHOLD:
        mlflow_utils.promote_to_production(str(int(metrics.get("model_version"))))
        return "promoted"
    return "staging"

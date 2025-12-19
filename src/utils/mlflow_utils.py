"""MLflow helper utilities for tracking and registry operations."""
from __future__ import annotations

import mlflow
from mlflow import MlflowClient
from typing import Dict

from src import config


def setup_mlflow() -> MlflowClient:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    return MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)


def log_params_and_metrics(params: Dict, metrics: Dict):
    for k, v in params.items():
        mlflow.log_param(k, v)
    for k, v in metrics.items():
        mlflow.log_metric(k, v)


def register_and_transition(model_uri: str, stage: str = "Staging") -> str:
    client = setup_mlflow()
    mv = mlflow.register_model(model_uri, config.MLFLOW_MODEL_NAME)
    client.transition_model_version_stage(
        name=config.MLFLOW_MODEL_NAME, version=mv.version, stage=stage
    )
    return mv.version


def promote_to_production(version: str):
    client = setup_mlflow()
    client.transition_model_version_stage(
        name=config.MLFLOW_MODEL_NAME, version=version, stage="Production", archive_existing_versions=True
    )


def rollback_production():
    client = setup_mlflow()
    versions = client.get_latest_versions(config.MLFLOW_MODEL_NAME, stages=["Production", "Staging"])
    if len(versions) < 2:
        return None
    versions_sorted = sorted(versions, key=lambda v: v.version, reverse=True)
    previous = versions_sorted[1]
    client.transition_model_version_stage(
        name=config.MLFLOW_MODEL_NAME, version=previous.version, stage="Production", archive_existing_versions=True
    )
    return previous.version

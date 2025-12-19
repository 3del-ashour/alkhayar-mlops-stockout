"""Prefect flow wiring the MLOps pipeline."""
from __future__ import annotations

from pathlib import Path
from prefect import flow

from src import config
from src.pipeline.steps import ingest_data, validate_data, engineer_features, train, evaluate_run, promote_if_good


@flow(name="stockout-pipeline")
def run_pipeline(data_dir: Path = config.SAMPLE_DATA_DIR, horizon_days: int = config.DEFAULT_HORIZON_DAYS):
    datasets = ingest_data(data_dir)
    datasets = validate_data(datasets)
    features = engineer_features(datasets, horizon_days)
    metrics = train(features)
    eval_metrics = evaluate_run(features, metrics)
    status = promote_if_good(metrics)
    return {"train_metrics": metrics, "eval_metrics": eval_metrics, "status": status}


if __name__ == "__main__":
    run_pipeline()

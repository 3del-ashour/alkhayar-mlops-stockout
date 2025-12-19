from pathlib import Path

from src.pipeline.orchestrate import run_pipeline
from src import config


def test_pipeline_runs_on_sample(tmp_path):
    result = run_pipeline.fn(data_dir=config.SAMPLE_DATA_DIR, horizon_days=3)
    assert "train_metrics" in result
    assert result["train_metrics"].get("val_f1") is not None

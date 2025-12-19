"""Training script for LightGBM classifier with imbalance handling and checkpoints."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import lightgbm as lgb
import mlflow
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split

from src import config
from src.utils import mlflow_utils


def handle_imbalance(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos_ratio = y.mean()
    if pos_ratio < config.IMBALANCE_THRESHOLD:
        sampler = RandomOverSampler(random_state=config.SEED)
        X_res, y_res = sampler.fit_resample(X, y)
    elif (1 - pos_ratio) < config.IMBALANCE_THRESHOLD:
        sampler = RandomUnderSampler(random_state=config.SEED)
        X_res, y_res = sampler.fit_resample(X, y)
    else:
        X_res, y_res = X, y
    return X_res, y_res


def train_model(X_sparse: csr_matrix, X_numeric: np.ndarray, y: np.ndarray) -> Dict:
    mlflow_utils.setup_mlflow()
    mlflow.lightgbm.autolog()

    X_combined = hstack([csr_matrix(X_numeric), X_sparse]).tocsr()
    stratify = y if min(np.bincount(y)) >= 2 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_combined, y, test_size=0.2, random_state=config.SEED, stratify=stratify
    )
    X_train_bal, y_train_bal = handle_imbalance(X_train, y_train)

    lgb_train = lgb.Dataset(X_train_bal, label=y_train_bal)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "seed": config.SEED,
        "deterministic": True,
        "verbose": -1,
    }

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    callbacks = [
        lgb.record_evaluation({}),
        lgb.early_stopping(30, verbose=False),
        lgb.log_evaluation(period=10),
        lgb.reset_parameter(learning_rate=lambda current_iter: 0.05 * (0.99 ** current_iter)),
    ]

    # Custom checkpointing
    def callback(env):
        iteration = env.iteration
        if iteration % 50 == 0 and iteration > 0:
            checkpoint_path = Path(config.CHECKPOINT_DIR) / f"model_iter_{iteration}.txt"
            env.model.save_model(checkpoint_path)

    callbacks.append(callback)

    with mlflow.start_run() as run:
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_val],
            num_boost_round=500,
            callbacks=callbacks,
        )

        val_pred = model.predict(X_val)
        val_label = (val_pred >= 0.5).astype(int)

        metrics = {
            "val_auc": float(roc_auc_score(y_val, val_pred)),
            "val_precision": float(precision_recall_fscore_support(y_val, val_label, average="binary")[0]),
            "val_recall": float(precision_recall_fscore_support(y_val, val_label, average="binary")[1]),
            "val_f1": float(f1_score(y_val, val_label)),
        }
        mlflow_utils.log_params_and_metrics(params, metrics)

        model_path = Path(config.MODEL_DIR)
        model_path.mkdir(parents=True, exist_ok=True)
        saved_model = model_path / "model.txt"
        model.save_model(saved_model)
        mlflow.log_artifact(saved_model)

        mlflow.lightgbm.log_model(model, artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        version = mlflow_utils.register_and_transition(model_uri, stage="Staging")
        metrics["model_version"] = version
        metrics["run_id"] = run.info.run_id
        return metrics

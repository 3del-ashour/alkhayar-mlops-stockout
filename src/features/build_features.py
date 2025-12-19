"""Feature engineering pipeline for stockout prediction."""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Tuple

from src import config
from src.utils.hashing import hash_categorical, hash_feature_cross, stack_sparse


def _aggregate_movement(df_movement: pd.DataFrame) -> pd.DataFrame:
    if df_movement.empty:
        return pd.DataFrame(columns=["BranchID", "ItemCode", "net_movement"])
    df_movement["Date"] = pd.to_datetime(df_movement["Date"], errors="coerce", utc=True)
    outflow = (
        df_movement.groupby(["FromBranchID", "ItemCode"]).QuantityMoved.sum().reset_index().rename(
            columns={"FromBranchID": "BranchID", "QuantityMoved": "outflow"}
        )
    )
    inflow = (
        df_movement.groupby(["ToBranchID", "ItemCode"]).QuantityMoved.sum().reset_index().rename(
            columns={"ToBranchID": "BranchID", "QuantityMoved": "inflow"}
        )
    )
    movement = outflow.merge(inflow, on=["BranchID", "ItemCode"], how="outer").fillna(0)
    movement["net_movement"] = movement["inflow"] - movement["outflow"]
    return movement[["BranchID", "ItemCode", "net_movement"]]


def create_label(df_sales: pd.DataFrame, df_stock: pd.DataFrame, horizon_days: int, df_movement: pd.DataFrame | None = None) -> pd.DataFrame:
    future_sales = (
        df_sales.groupby(["BranchID", "ItemCode"])
        .rolling(f"{horizon_days}D", on="Date")
        .QuantitySold.sum()
        .reset_index()
        .rename(columns={"QuantitySold": "future_sales"})
    )
    merged = df_stock.merge(future_sales, on=["BranchID", "ItemCode"], how="left")
    merged["future_sales"].fillna(0, inplace=True)
    merged["net_movement"] = 0
    if df_movement is not None:
        movement = _aggregate_movement(df_movement)
        merged = merged.merge(movement, on=["BranchID", "ItemCode"], how="left")
        merged["net_movement"].fillna(0, inplace=True)
    merged["projected_stock"] = merged["CurrentQuantity"] - merged["ReservedQuantity"] - merged["future_sales"] + merged["net_movement"]
    merged["label_stockout"] = (merged["projected_stock"] < merged["SafetyStockLevel"]).astype(int)
    # placeholders to satisfy hashing requirements
    merged["FromBranchID"] = merged["BranchID"]
    merged["ToBranchID"] = merged["BranchID"]
    return merged


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    numeric_cols = ["CurrentQuantity", "ReservedQuantity", "SafetyStockLevel", "future_sales", "projected_stock", "net_movement"]
    X_numeric = df[numeric_cols].fillna(0).to_numpy()

    hashed_item = hash_categorical(df["ItemCode"], config.HASH_SPACE)
    hashed_branch = hash_categorical(df["BranchID"], config.HASH_SPACE)
    hashed_from_branch = hash_categorical(df.get("FromBranchID", df["BranchID"]), config.HASH_SPACE)
    hashed_to_branch = hash_categorical(df.get("ToBranchID", df["BranchID"]), config.HASH_SPACE)
    hashed_cross = hash_feature_cross(df["ItemCode"], df["BranchID"], config.CROSS_HASH_SPACE)
    month = df["LastUpdatedAt"].dt.month.fillna(0).astype(int)
    hashed_item_month = hash_feature_cross(df["ItemCode"], month.astype(str), config.CROSS_HASH_SPACE)

    X_sparse = stack_sparse([hashed_item, hashed_branch, hashed_from_branch, hashed_to_branch, hashed_cross, hashed_item_month])
    y = df["label_stockout"].to_numpy()
    feature_names = numeric_cols + ["hashed_features"]

    return X_sparse, X_numeric, y, feature_names

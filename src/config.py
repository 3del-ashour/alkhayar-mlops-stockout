"""Global configuration for the stockout MLOps project."""
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
MODEL_DIR = ARTIFACTS_DIR / "models"
SEED = 42
DEFAULT_HORIZON_DAYS = 7
HASH_SPACE = 2 ** 12
CROSS_HASH_SPACE = 2 ** 10

MLFLOW_TRACKING_URI = f"file:{PROJECT_ROOT / 'mlruns'}"
MLFLOW_EXPERIMENT = "stockout_prediction"
MLFLOW_MODEL_NAME = "stockout_classifier"

ACCEPTANCE_THRESHOLD = 0.7  # minimum F1 for promotion
IMBALANCE_THRESHOLD = 0.2  # minority proportion threshold

SCHEMA: Dict[str, Dict[str, str]] = {
    "stock_current": {
        "BranchID": "string",
        "BranchName": "string",
        "ItemCode": "string",
        "ItemName": "string",
        "CurrentQuantity": "float",
        "ReservedQuantity": "float",
        "SafetyStockLevel": "float",
        "LastUpdatedAt": "datetime",
    },
    "sales_transactions": {
        "Date": "datetime",
        "BranchID": "string",
        "BranchName": "string",
        "InvoiceNumber": "string",
        "ItemCode": "string",
        "ItemName": "string",
        "QuantitySold": "float",
    },
    "stock_movement": {
        "MovementID": "string",
        "Date": "datetime",
        "FromBranchID": "string",
        "FromBranchName": "string",
        "ToBranchID": "string",
        "ToBranchName": "string",
        "ItemCode": "string",
        "ItemName": "string",
        "QuantityMoved": "float",
    },
}

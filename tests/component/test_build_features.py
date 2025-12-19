import pandas as pd
from src.features.build_features import create_label, build_feature_matrix


def test_label_creation():
    sales = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "BranchID": ["B1", "B1"],
            "BranchName": ["Central", "Central"],
            "InvoiceNumber": ["1", "2"],
            "ItemCode": ["ITM1", "ITM1"],
            "ItemName": ["A", "A"],
            "QuantitySold": [10, 20],
        }
    )
    stock = pd.DataFrame(
        {
            "BranchID": ["B1"],
            "BranchName": ["Central"],
            "ItemCode": ["ITM1"],
            "ItemName": ["A"],
            "CurrentQuantity": [50],
            "ReservedQuantity": [5],
            "SafetyStockLevel": [30],
            "LastUpdatedAt": pd.to_datetime(["2024-01-02"], utc=True),
        }
    )
    labeled = create_label(sales, stock, horizon_days=7)
    assert "label_stockout" in labeled.columns


def test_build_feature_matrix_shapes():
    df = pd.DataFrame(
        {
            "BranchID": ["B1", "B2"],
            "ItemCode": ["ITM1", "ITM2"],
            "CurrentQuantity": [50, 60],
            "ReservedQuantity": [5, 10],
            "SafetyStockLevel": [30, 20],
            "future_sales": [10, 5],
            "projected_stock": [35, 45],
            "net_movement": [0, 0],
            "LastUpdatedAt": pd.to_datetime(["2024-01-02", "2024-01-02"], utc=True),
            "label_stockout": [0, 1],
        }
    )
    X_sparse, X_numeric, y, _ = build_feature_matrix(df)
    assert X_sparse.shape[0] == df.shape[0]
    assert len(y) == df.shape[0]

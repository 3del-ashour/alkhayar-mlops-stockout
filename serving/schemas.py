from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional


class PredictionRequest(BaseModel):
    BranchID: str
    ItemCode: str
    Date: datetime
    CurrentQuantity: float
    ReservedQuantity: float
    SafetyStockLevel: float
    future_sales: Optional[float] = Field(default=0)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

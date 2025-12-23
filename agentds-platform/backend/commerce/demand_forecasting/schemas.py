from pydantic import BaseModel
from typing import List, Optional, Any

class TrainingResponse(BaseModel):
    success: bool
    message: str
    metrics: Optional[dict] = None

class ForecastInput(BaseModel):
    sku_id: int
    week: int
    price: float
    promo_flag: int

class BatchForecastInput(BaseModel):
    inputs: List[ForecastInput]

class ForecastPrediction(BaseModel):
    sku_id: int
    week: int
    predicted_units_sold: float

class PredictionResponse(BaseModel):
    success: bool
    predictions: List[ForecastPrediction]

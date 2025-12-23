from pydantic import BaseModel
from typing import List, Optional

class RecTrainingResponse(BaseModel):
    success: bool
    message: str
    metrics: Optional[dict] = None

class RecInput(BaseModel):
    customer_id: int
    product_id: int

class BatchRecInput(BaseModel):
    inputs: List[RecInput]

class RecPrediction(BaseModel):
    customer_id: int
    product_id: int
    predicted_score: float

class RecPredictionResponse(BaseModel):
    success: bool
    predictions: List[RecPrediction]

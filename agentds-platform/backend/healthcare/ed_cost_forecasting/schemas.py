from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional, Any

class BatchRecords(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")

class PredictionOutput(BaseModel):
    predictions: List[float]
    model_version: str = "v1"

class TrainOutput(BaseModel):
    success: bool
    metrics: Dict[str, float]
    input_shape: tuple
    artifact_path: str

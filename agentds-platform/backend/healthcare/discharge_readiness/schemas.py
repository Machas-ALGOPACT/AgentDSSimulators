from pydantic import BaseModel, Field
from typing import List, Dict, Union, Optional, Any

class BatchRecords(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="List of feature dictionaries")

class PredictionOutput(BaseModel):
    predictions: List[Union[float, int, str]]
    probabilities: Optional[List[List[float]]] = None
    model_version: str = "v1"

class TrainOutput(BaseModel):
    success: bool
    metrics: Dict[str, float]
    input_shape: tuple
    artifact_path: str

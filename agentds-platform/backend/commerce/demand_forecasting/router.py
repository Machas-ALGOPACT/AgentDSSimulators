from fastapi import APIRouter, HTTPException
from backend.common.schemas.base import BaseResponse
from backend.commerce.demand_forecasting.schemas import BatchForecastInput, PredictionResponse, TrainingResponse
from backend.commerce.demand_forecasting.service import service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/demand-forecasting",
    tags=["Commerce - Demand Forecasting"],
    responses={404: {"description": "Not found"}},
)

@router.get("/health", response_model=BaseResponse)
async def health_check():
    return BaseResponse(
        success=True, 
        message="Demand Forecasting service is healthy",
        metadata={"domain": "commerce", "service": "demand_forecasting"}
    )

@router.post("/train", response_model=TrainingResponse)
async def train():
    result = service.train()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return TrainingResponse(
        success=True,
        message="Training completed successfully",
        metrics=result.get("metrics")
    )

@router.post("/predict", response_model=PredictionResponse)
async def predict(payload: BatchForecastInput):
    try:
        preds = service.predict(payload)
        return PredictionResponse(
            success=True,
            predictions=preds
        )
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model not trained. Please call /train first.")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

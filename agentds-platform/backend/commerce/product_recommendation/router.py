from fastapi import APIRouter, HTTPException
from backend.common.schemas.base import BaseResponse
from backend.commerce.product_recommendation.schemas import BatchRecInput, RecPredictionResponse, RecTrainingResponse
from backend.commerce.product_recommendation.service import service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/product-recommendation",
    tags=["Commerce - Product Recommendation"],
    responses={404: {"description": "Not found"}},
)

@router.get("/health", response_model=BaseResponse)
async def health_check():
    return BaseResponse(
        success=True, 
        message="Product Recommendation service is healthy",
        metadata={"domain": "commerce", "service": "product_recommendation"}
    )

@router.post("/train", response_model=RecTrainingResponse)
async def train():
    result = service.train()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return RecTrainingResponse(
        success=True,
        message="Training completed successfully",
        metrics=result.get("metrics")
    )

@router.post("/predict", response_model=RecPredictionResponse)
async def predict(payload: BatchRecInput):
    try:
        preds = service.predict(payload)
        return RecPredictionResponse(
            success=True,
            predictions=preds
        )
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model not trained. Please call /train first.")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

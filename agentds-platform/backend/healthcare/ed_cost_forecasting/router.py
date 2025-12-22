from fastapi import APIRouter, HTTPException
from backend.common.schemas.base import BaseResponse
from backend.healthcare.ed_cost_forecasting.schemas import BatchRecords, PredictionOutput
from backend.healthcare.ed_cost_forecasting.service import service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ed-cost-forecasting",
    tags=["Healthcare - ED Cost Forecasting"],
    responses={404: {"description": "Not found"}},
)

@router.get("/health", response_model=BaseResponse)
async def health_check():
    return BaseResponse(
        success=True, 
        message="ED Cost Forecasting service is healthy",
        metadata={"domain": "healthcare", "service": "ed_cost_forecasting"}
    )

@router.post("/train", response_model=BaseResponse)
async def train():
    try:
        result = service.train()
        if not result["success"]:
             raise HTTPException(status_code=500, detail=result.get("error"))
        return BaseResponse(
            success=True,
            message="Training completed successfully",
            data=result,
            metadata={"metrics": result.get("metrics")}
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", response_model=BaseResponse)
async def predict(payload: BatchRecords):
    try:
        preds = service.predict(payload.records)
        return BaseResponse(
            success=True,
            message="Prediction successful",
            data=PredictionOutput(predictions=preds)
        )
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model not trained. Please call /train first.")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

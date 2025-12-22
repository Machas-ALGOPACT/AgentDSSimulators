from fastapi import APIRouter, HTTPException
from backend.common.schemas.base import BaseResponse
from backend.healthcare.readmission_prediction.schemas import BatchRecords, TrainOutput, PredictionOutput
from backend.healthcare.readmission_prediction.service import service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/readmission-prediction",
    tags=["Healthcare - Readmission Prediction"],
    responses={404: {"description": "Not found"}},
)

@router.get("/health", response_model=BaseResponse)
async def health_check():
    return BaseResponse(
        success=True, 
        message="Readmission Prediction service is healthy",
        metadata={"domain": "healthcare", "service": "readmission_prediction"}
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
        preds, proba = service.predict(payload.records)
        return BaseResponse(
            success=True,
            message="Prediction successful",
            data=PredictionOutput(predictions=preds, probabilities=proba)
        )
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model not trained. Please call /train first.")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, HTTPException
from backend.common.schemas.base import BaseResponse
from backend.commerce.coupon_redemption.schemas import BatchCouponInput, CouponPredictionResponse, CouponTrainingResponse
from backend.commerce.coupon_redemption.service import service
import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/coupon-redemption",
    tags=["Commerce - Coupon Redemption"],
    responses={404: {"description": "Not found"}},
)

@router.get("/health", response_model=BaseResponse)
async def health_check():
    return BaseResponse(
        success=True, 
        message="Coupon Redemption service is healthy",
        metadata={"domain": "commerce", "service": "coupon_redemption"}
    )

@router.post("/train", response_model=CouponTrainingResponse)
async def train():
    result = service.train()
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error"))
    return CouponTrainingResponse(
        success=True,
        message="Training completed successfully",
        metrics=result.get("metrics")
    )

@router.post("/predict", response_model=CouponPredictionResponse)
async def predict(payload: BatchCouponInput):
    try:
        preds = service.predict(payload)
        return CouponPredictionResponse(
            success=True,
            predictions=preds
        )
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Model not trained. Please call /train first.")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

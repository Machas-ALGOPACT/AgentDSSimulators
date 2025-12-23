from fastapi import APIRouter
from backend.commerce.demand_forecasting.router import router as demand_router
from backend.commerce.product_recommendation.router import router as rec_router
from backend.commerce.coupon_redemption.router import router as coupon_router

router = APIRouter(
    prefix="/commerce",
    responses={404: {"description": "Commerce endpoint not found"}},
)

# Register Problem Statements
router.include_router(demand_router)
router.include_router(rec_router)
router.include_router(coupon_router)

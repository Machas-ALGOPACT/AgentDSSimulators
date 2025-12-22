from fastapi import APIRouter
from backend.insurance.fraud_detection.router import router as fraud_router
from backend.insurance.claims_complexity.router import router as claims_router
from backend.insurance.risk_pricing.router import router as risk_router

# -----------------------------------------------------------------------------
# DOMAIN ROUTER AGGREGATION
# -----------------------------------------------------------------------------
# Domain: Insurance
# Root Path: /api/v1/insurance
#
# RESPONSIBILITY:
# This router aggregates all 'Insurance' related Problem Statements (PS).
# It acts as a namespace wrapper.
#
# INSTRUCTIONS FOR NEW PS:
# 1. Import your new PS router above.
# 2. Add `router.include_router(your_new_router)` below.
# -----------------------------------------------------------------------------

router = APIRouter(
    prefix="/insurance",
    responses={404: {"description": "Insurance endpoint not found"}},
)

# Register Problem Statements
router.include_router(fraud_router)
router.include_router(claims_router)
router.include_router(risk_router)

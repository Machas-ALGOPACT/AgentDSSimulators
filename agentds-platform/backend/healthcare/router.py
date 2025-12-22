from fastapi import APIRouter
from backend.healthcare.readmission_prediction.router import router as readmission_router
from backend.healthcare.ed_cost_forecasting.router import router as ed_cost_router
from backend.healthcare.discharge_readiness.router import router as discharge_router

# -----------------------------------------------------------------------------
# DOMAIN ROUTER AGGREGATION
# -----------------------------------------------------------------------------
# Domain: Healthcare
# Root Path: /api/v1/healthcare
#
# RESPONSIBILITY:
# This router aggregates all 'Healthcare' related Problem Statements (PS).
# -----------------------------------------------------------------------------

router = APIRouter(
    prefix="/healthcare",
    responses={404: {"description": "Healthcare endpoint not found"}},
)

# Register Problem Statements
router.include_router(readmission_router)
router.include_router(ed_cost_router)
router.include_router(discharge_router)

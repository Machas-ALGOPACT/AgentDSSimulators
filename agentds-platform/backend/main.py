from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.common.logging.logger import setup_logger

# Import Domain Routers
# NOTE: Add new domain imports here as they are developed.
from backend.insurance.router import router as insurance_router
from backend.healthcare.router import router as healthcare_router
# from backend.manufacturing.router import router as manufacturing_router
from backend.commerce.router import router as commerce_router
# from backend.food_production.router import router as food_router
# from backend.retail_banking.router import router as banking_router

# -----------------------------------------------------------------------------
# APP CONFIGURATION
# -----------------------------------------------------------------------------

logger = setup_logger(__name__)

app = FastAPI(
    title="AgentDS Platform API",
    description="Unified Analytics Platform for Insurance, Healthcare, Manufacturing, and more.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# GLOBAL ROUTES
# -----------------------------------------------------------------------------

@app.get("/health")
async def root_health():
    """global health check for the entire platform."""
    return {"status": "active", "platform": "AgentDS"}

# -----------------------------------------------------------------------------
# DOMAIN ROUTER REGISTRATION
# -----------------------------------------------------------------------------
# DEVELOPER INSTRUCTION:
# Plug in your domain routers here. 
# Ensure the 'prefix' is set correctly in the domain router file itself.

# /api/v1/insurance/...
app.include_router(insurance_router, prefix="/api/v1")

# /api/v1/healthcare/...
app.include_router(healthcare_router, prefix="/api/v1")

# Future Domains (Uncomment when routers are created)
# app.include_router(manufacturing_router, prefix="/api/v1")
app.include_router(commerce_router, prefix="/api/v1")
# app.include_router(food_router, prefix="/api/v1")
# app.include_router(banking_router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting AgentDS Backend...")
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

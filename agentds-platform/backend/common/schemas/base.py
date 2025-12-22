from pydantic import BaseModel
from typing import Optional, Any

class BaseResponse(BaseModel):
    """
    Standard response schema for all API endpoints.
    
    All ML predictions should try to follow this format or extend it
    to ensure the frontend can parse responses consistently.
    """
    success: bool
    message: str
    data: Optional[Any] = None
    metadata: Optional[dict] = None

class HealthCheckResponse(BaseResponse):
    """
    Response model for health checks.
    """
    pass

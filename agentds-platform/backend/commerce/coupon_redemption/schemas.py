from pydantic import BaseModel
from typing import List, Optional

class CouponTrainingResponse(BaseModel):
    success: bool
    message: str
    metrics: Optional[dict] = None

class CouponInput(BaseModel):
    customer_id: int
    offer_id: int
    sku_id: int
    category: str
    discount_pct: float
    price_tier: str
    hist_spend: float
    email_open_rate: float
    avg_basket_value: float
    
class BatchCouponInput(BaseModel):
    inputs: List[CouponInput]

class CouponPrediction(BaseModel):
    customer_id: int
    offer_id: int
    redemption_probability: float
    will_redeem: bool

class CouponPredictionResponse(BaseModel):
    success: bool
    predictions: List[CouponPrediction]

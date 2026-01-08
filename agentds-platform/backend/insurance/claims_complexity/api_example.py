"""
API Example for Claims Complexity Model Integration

This shows how to integrate the model with a web API (FastAPI, Flask, etc).
Use this as reference when building your UI backend.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import pandas as pd
import os

app = FastAPI(title="Claims Complexity Predictor")

# Load models once at startup
model = None
pipeline = None
label_encoder = None

@app.on_event("startup")
async def load_models():
    global model, pipeline, label_encoder
    try:
        model = joblib.load("models/ensemble_model.joblib")
        pipeline = joblib.load("models/preprocessing_pipeline.joblib")
        label_encoder = joblib.load("models/label_encoder.joblib")
    except Exception as e:
        print(f"Error loading models: {e}")


# Request schema
class ClaimInput(BaseModel):
    # Required fields
    claim_id: str = Field(..., description="Unique claim ID")
    policy_id: str = Field(..., description="Unique policy ID")
    claim_date: str = Field(..., description="Date/time of claim (YYYY-MM-DD HH:MM)")
    claim_type: str = Field(..., description="Type of claim")
    reported_damage: float = Field(..., ge=0, description="Damage amount in dollars")
    num_parties: int = Field(..., ge=1, description="Number of parties involved")
    description: str = Field(..., min_length=50, description="Claim description")
    
    # Optional fields
    holder_age: Optional[float] = Field(None, ge=16, le=120)
    vehicle_type: Optional[str] = None
    annual_mileage: Optional[float] = Field(None, ge=0)
    location_urban: Optional[int] = Field(None, description="1=Urban, 0=Rural")
    credit_score: Optional[float] = Field(None, ge=0, le=1)
    policy_start: Optional[str] = None
    policy_end: Optional[str] = None


# Response schema
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    claim_id: str


@app.post("/predict", response_model=PredictionResponse)
async def predict_claim_complexity(claim: ClaimInput):
    """
    Predict claim complexity level
    
    Returns:
        PredictionResponse with prediction, confidence, and probabilities
    """
    try:
        # Convert input to DataFrame
        data = {
            "ClaimID": claim.claim_id,
            "PolicyID": claim.policy_id,
            "ClaimDate": claim.claim_date,
            "ClaimType": claim.claim_type,
            "ReportedDamage": claim.reported_damage,
            "NumParties": claim.num_parties,
            "Description": claim.description,
            "HolderAge": claim.holder_age,
            "VehicleType": claim.vehicle_type,
            "AnnualMileage": claim.annual_mileage,
            "LocationUrban": claim.location_urban,
            "CreditScore": claim.credit_score,
            "PolicyStart": claim.policy_start,
            "PolicyEnd": claim.policy_end,
        }
        
        df = pd.DataFrame([data])
        
        # Transform features
        X = pipeline.transform(df)
        
        # Predict
        pred_numeric = model.predict(X)[0]
        pred_label = label_encoder.inverse_transform([pred_numeric])[0]
        
        # Get probabilities
        probs = model.predict_proba(X)[0]
        prob_dict = {
            label: float(prob) 
            for label, prob in zip(label_encoder.classes_, probs)
        }
        
        return PredictionResponse(
            prediction=pred_label,
            confidence=float(probs[pred_numeric]),
            probabilities=prob_dict,
            claim_id=claim.claim_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Usage example:
# uvicorn api_example:app --reload
# POST http://localhost:8000/predict

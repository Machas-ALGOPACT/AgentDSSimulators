from fastapi import APIRouter
from backend.common.schemas.base import BaseResponse
from src.utils.config import load_config
from src.utils.persistence import load_object
import os
import pandas as pd

# -----------------------------------------------------------------------------
# ROUTER CONFIGURATION
# -----------------------------------------------------------------------------
# This router handles the 'Claims Complexity' problem statement.
# Root Path: /api/v1/insurance/claims-complexity
#
# DEVELOPER INSTRUCTIONS:
# 1. Define your Request/Response schemas locally or in common/schemas.
# 2. Implement your ML inference logic in the /predict endpoint.
# 3. Do NOT change the router prefix or tags without team approval.
# -----------------------------------------------------------------------------

router = APIRouter(
    prefix="/claims-complexity",
    tags=["Insurance - Claims Complexity"],
    responses={404: {"description": "Not found"}},
)

@router.get("/health", response_model=BaseResponse)
async def health_check():
    """
    Health check for Claims Complexity service.
    """
    return BaseResponse(
        success=True, 
        message="Claims Complexity service is healthy",
        metadata={"domain": "insurance", "service": "claims_complexity"}
    )

@router.post("/predict", response_model=BaseResponse)
async def predict(payload: dict):
    """
    Main inference endpoint for Claims Complexity.

    Payload: single claim as a dict with keys (e.g., ClaimID, PolicyID, ReportedDamage, NumParties, Description)
    Returns: predicted label and probabilities (if available)
    """
    config = load_config()
    model_path = os.path.join(config['paths']['models'], f"{config.get('active_model','ensemble')}_model.joblib")

    # Load model
    try:
        model = load_object(model_path)
    except Exception as e:
        return BaseResponse(success=False, message=f"Model not found: {e}")

    # Prepare single-row dataframe
    try:
        df = pd.DataFrame([payload])
    except Exception as e:
        return BaseResponse(success=False, message=f"Invalid payload: {e}")

    # First try to load a persisted preprocessing pipeline (preferred)
    pipeline_path = os.path.join(config['paths']['models'], 'preprocessing_pipeline.joblib')
    X = None
    try:
        pipeline = load_object(pipeline_path)
        X = pipeline.transform(df)
    except Exception:
        # Fallback: ad-hoc text + scaler steps
        from src.features.text_features import TextFeatureEngineer
        tfe = TextFeatureEngineer(config)
        df = tfe.extract_basic_text_features(df, 'Description')

        vec_path = os.path.join(config['paths']['models'], 'tfidf_vectorizer.joblib')
        try:
            vectorizer = load_object(vec_path)
            df = tfe.transform_tfidf(df, 'Description', vectorizer=vectorizer)
        except Exception:
            # No TF-IDF available, continue with other features
            vectorizer = None

        # Scaling numeric features if scaler exists
        scaler_path = os.path.join(config['paths']['models'], 'scaler.joblib')
        try:
            scaler_obj = load_object(scaler_path)
            num_cols = [c for c in config.get('features', {}).get('numerical', []) if c in df.columns]
            if num_cols:
                df[num_cols] = scaler_obj.transform(df[num_cols])
        except Exception:
            scaler_obj = None

        # Construct feature vector: TF-IDF features + numerical features
        feature_cols = [c for c in df.columns if c.startswith('tfidf_')] + [c for c in config.get('features', {}).get('numerical', []) if c in df.columns]
        if not feature_cols:
            return BaseResponse(success=False, message="No features available to predict", data={})

        X = df[feature_cols]

    # Load label encoder for human-readable output
    le_path = os.path.join(config['paths']['models'], 'label_encoder.joblib')
    try:
        le = load_object(le_path)
    except Exception as e:
        return BaseResponse(success=False, message=f"Label encoder not found: {e}")

    # Ensure the feature order is deterministic
    try:
        y_pred_encoded = model.predict(X)[0]
        # Inverse transform to get human-readable label
        y_pred = le.inverse_transform([y_pred_encoded])[0]
    except Exception as e:
        return BaseResponse(success=False, message=f"Prediction failed: {e}")

    prob = None
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            # Map probabilities to class names
            class_proba = dict(zip([str(c) for c in le.classes_], proba.tolist()))
            prob = class_proba
    except Exception:
        prob = None

    return BaseResponse(success=True, message="Prediction successful", data={"prediction": str(y_pred), "probabilities": prob})

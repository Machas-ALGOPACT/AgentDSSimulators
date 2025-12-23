import os
import joblib
import pandas as pd
from backend.commerce.product_recommendation.model.features import feature_engineering_recommendation
from backend.commerce.shared.constants import MODEL_FILENAME

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, MODEL_FILENAME)

_MODEL_CACHE = None

def load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model not found. Please train first.")
        _MODEL_CACHE = joblib.load(MODEL_PATH)
    return _MODEL_CACHE

def predict_recommendation(input_df: pd.DataFrame):
    model = load_model()
    df_feats = feature_engineering_recommendation(input_df)
    
    # Ensure columns match training (simplified)
    # We assume schema matches exactly what was trained
    preds = model.predict(df_feats)
    return preds

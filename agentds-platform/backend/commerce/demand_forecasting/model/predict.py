import os
import joblib
import pandas as pd
import logging
from backend.commerce.demand_forecasting.model.features import feature_engineering_forecasting
from backend.commerce.shared.constants import MODEL_FILENAME

logger = logging.getLogger(__name__)

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

def predict_demand(input_df: pd.DataFrame):
    model = load_model()
    
    # Feature Engineering
    df_feats = feature_engineering_forecasting(input_df)
    
    # Align Columns (reorder to match training if needed, but XGB handles names usually)
    # Ideally should ensure column order matches X_train.columns
    # For now, simplistic.
    
    preds = model.predict(df_feats)
    return preds

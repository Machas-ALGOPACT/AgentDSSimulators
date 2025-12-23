import os
import joblib
import pandas as pd
from backend.commerce.coupon_redemption.model.features import feature_engineering_coupon
from backend.commerce.shared.constants import MODEL_FILENAME
from backend.commerce.shared.preprocessing import encode_categorical

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

def predict_coupon(input_df: pd.DataFrame):
    model = load_model()
    df_feats = feature_engineering_coupon(input_df)
    
    # Encode Categoricals (Simplistic - mismatch risk!)
    cat_cols = ["category", "price_tier"]
    df_feats = encode_categorical(df_feats, cat_cols)
    
    # Output prob and class
    # Enforce column order to match training
    expected_cols = ['offer_id', 'customer_id', 'sku_id', 'category', 'discount_pct', 'price_tier', 'hist_spend', 'email_open_rate', 'avg_basket_value']
    df_feats = df_feats[expected_cols]
    
    probs = model.predict_proba(df_feats)[:, 1]
    preds = model.predict(df_feats)
    
    return preds, probs

import os
import joblib
import xgboost as xgb
import pandas as pd
from backend.commerce.shared.dataset_loader import load_commerce_dataset
from backend.commerce.shared.constants import (
    TASK_COUPON_REDEMPTION, COL_REDEEMED, MODEL_FILENAME, METRICS_FILENAME
)
from backend.commerce.shared.preprocessing import split_data, fill_missing_values, encode_categorical
from backend.commerce.shared.evaluation import evaluate_classification
from backend.commerce.coupon_redemption.model.features import feature_engineering_coupon

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(ARTIFACTS_DIR, MODEL_FILENAME)
METRICS_PATH = os.path.join(ARTIFACTS_DIR, METRICS_FILENAME)

def train_coupon_model():
    # 1. Load
    df = load_commerce_dataset(TASK_COUPON_REDEMPTION)
    
    # 2. Preprocess
    df = fill_missing_values(df)
    df = feature_engineering_coupon(df)
    
    # Encode Categoricals (Category, Price Tier)
    # We must encode them to numeric for XGBoost
    cat_cols = ["category", "price_tier"]
    df = encode_categorical(df, cat_cols)
    
    # 3. Split
    if COL_REDEEMED not in df.columns:
        raise ValueError(f"Target {COL_REDEEMED} not found")
        
    X_train, X_test, y_train, y_test = split_data(df, target_col=COL_REDEEMED)
    
    # 4. Train
    # Handle Class Imbalance via scale_pos_weight
    ratio = float(len(y_train[y_train == 0])) / len(y_train[y_train == 1]) if len(y_train[y_train == 1]) > 0 else 1.0
    
    model = xgb.XGBClassifier(
        n_estimators=100, 
        objective='binary:logistic', 
        scale_pos_weight=ratio,
        eval_metric='logloss',
        use_label_encoder=False,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_classification(y_test, preds, probs)
    
    # 6. Save
    joblib.dump(model, MODEL_PATH)
    import json
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f)
        
    return metrics

import os
import joblib
import xgboost as xgb
import pandas as pd
from backend.commerce.shared.dataset_loader import load_commerce_dataset
from backend.commerce.shared.constants import (
    TASK_DEMAND_FORECASTING, COL_WEEKLY_SALES, MODEL_FILENAME, METRICS_FILENAME
)
from backend.commerce.shared.preprocessing import split_data, fill_missing_values
from backend.commerce.shared.evaluation import evaluate_regression
from backend.commerce.demand_forecasting.model.features import feature_engineering_forecasting

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(ARTIFACTS_DIR, MODEL_FILENAME)
METRICS_PATH = os.path.join(ARTIFACTS_DIR, METRICS_FILENAME)

def train_demand_model():
    # 1. Load Data
    df = load_commerce_dataset(TASK_DEMAND_FORECASTING)
    
    # 2. Preprocess
    df = fill_missing_values(df, strategy="mean")
    df = feature_engineering_forecasting(df)
    
    # 3. Split
    # Target is now 'units_sold' (mapped to COL_WEEKLY_SALES const)
    X_train, X_test, y_train, y_test = split_data(df, target_col=COL_WEEKLY_SALES)
    
    # 4. Train
    model = xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror', n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    preds = model.predict(X_test)
    metrics = evaluate_regression(y_test, preds)
    
    # 6. Save
    joblib.dump(model, MODEL_PATH)
    import json
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f)
        
    return metrics

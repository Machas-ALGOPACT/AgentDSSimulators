import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from backend.healthcare.shared.dataset_loader import loader
from backend.healthcare.shared.preprocessing import create_preprocessing_pipeline, clean_data
from backend.healthcare.shared.evaluation import evaluate_classification
from backend.common.utils.paths import get_artifacts_path
from backend.common.utils.io import save_artifact, save_json
import logging

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = get_artifacts_path("healthcare", "discharge_readiness")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
PIPELINE_PATH = ARTIFACTS_DIR / "pipeline.pkl"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

TARGET_COL = "ready_for_discharge" 

def train_model():
    logger.info("Starting training for Discharge Readiness...")
    
    try:
        df = loader.get_discharge_data()
    except Exception as e:
         return {"success": False, "error": str(e)}
         
    if df.empty:
         return {"success": False, "error": "No data found for Discharge Readiness."}

    df = clean_data(df, target_col=TARGET_COL)
    
    if TARGET_COL not in df.columns:
        possible = [c for c in df.columns if 'ready' in c.lower() or 'status' in c.lower()]
        if possible:
            target = possible[0]
        else:
            return {"success": False, "error": f"Target column '{TARGET_COL}' not found."}
    else:
        target = TARGET_COL

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = create_preprocessing_pipeline(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('clf', model)])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = evaluate_classification(y_test, y_pred)
    logger.info(f"Training metrics: {metrics}")

    save_artifact(pipeline, MODEL_PATH)
    
    metadata = {
        "features": list(X.columns),
        "target": target,
        "metrics": metrics
    }
    save_json(metadata, METADATA_PATH)

    return {
        "success": True, 
        "metrics": metrics, 
        "input_shape": X.shape, 
        "artifact_path": str(MODEL_PATH)
    }

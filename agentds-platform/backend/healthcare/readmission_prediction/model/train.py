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

ARTIFACTS_DIR = get_artifacts_path("healthcare", "readmission_prediction")
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
PIPELINE_PATH = ARTIFACTS_DIR / "pipeline.pkl"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"

TARGET_COL = "readmitted" # Validated from similar datasets, might need adjustment if actual column differs.

def train_model():
    logger.info("Starting training for Readmission Prediction...")
    
    # 1. Load Data
    try:
        df = loader.get_readmission_data()
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return {"success": False, "error": str(e)}

    # 2. Cleanup & Split
    df = clean_data(df, target_col=TARGET_COL)
    
    if TARGET_COL not in df.columns:
        # Attempt minimal inferrence if standard name fails
        # Heuristic: verify if 'target', 'label' exists
        possible = [c for c in df.columns if c.lower() in ['target', 'label', 'class', 'readmission']]
        if possible:
            target = possible[0]
        else:
            return {"success": False, "error": f"Target column '{TARGET_COL}' not found."}
    else:
        target = TARGET_COL

    X = df.drop(columns=[target])
    y = df[target]

    # Handle class imbalance checks or mapping if y is object
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Preprocessing
    preprocessor = create_preprocessing_pipeline(X_train)
    
    # 4. Model
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])

    # 5. fit
    clf.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = clf.predict(X_test)
    metrics = evaluate_classification(y_test, y_pred)
    logger.info(f"Training metrics: {metrics}")

    # 7. Save
    save_artifact(clf, MODEL_PATH)
    
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

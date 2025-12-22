import pandas as pd
from backend.common.utils.io import load_artifact, load_json
from backend.healthcare.readmission_prediction.model.train import MODEL_PATH, METADATA_PATH

feature_metadata = None
model_pipeline = None

def load_resources():
    global model_pipeline, feature_metadata
    if model_pipeline is None:
        model_pipeline = load_artifact(MODEL_PATH)
    if feature_metadata is None:
        feature_metadata = load_json(METADATA_PATH)

def make_prediction(data: list):
    load_resources()
    
    # ensure dataframe matches expected schema
    # The pipeline handles missing columns if handled in preprocessor, 
    # but best to ensure alignment?
    # Our simple preprocessor assumes exact columns often, 
    # but scikit-learn transformers are robust if configured right.
    
    df = pd.DataFrame(data)
    
    # Align columns could be added here if needed
    
    preds = model_pipeline.predict(df)
    proba = model_pipeline.predict_proba(df)
    
    return preds.tolist(), proba.tolist()

import pandas as pd
from backend.common.utils.io import load_artifact, load_json
from backend.healthcare.ed_cost_forecasting.model.train import MODEL_PATH, METADATA_PATH

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
    df = pd.DataFrame(data)
    preds = model_pipeline.predict(df)
    return preds.tolist()

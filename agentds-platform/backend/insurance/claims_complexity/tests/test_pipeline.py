import os
import pandas as pd
import numpy as np
from src.preprocessing.pipeline import PreprocessingPipeline
from src.utils.persistence import save_object, load_object


def test_pipeline_fit_transform_save_load(tmp_path):
    # Sample df
    df = pd.DataFrame({
        'Description': ['rear end collision', 'minor scratch', 'side impact injury'],
        'ReportedDamage': [100.0, 50.0, 200.0],
        'NumParties': [1, 1, 2],
        'HolderAge': [34, 45, 29],
        'AnnualMileage': [12000, 8000, 15000],
        'CreditScore': [700, 650, 720]
    })

    pipeline = PreprocessingPipeline(config=None)
    pipeline.fit(df)
    X = pipeline.transform(df)

    assert isinstance(X, pd.DataFrame)
    assert X.shape[0] == df.shape[0]
    assert len(pipeline.feature_columns) == X.shape[1]

    # Save and load pipeline
    path = os.path.join(tmp_path, 'pipeline.joblib')
    pipeline.save(path)

    loaded = PreprocessingPipeline.load(path)
    X2 = loaded.transform(df)
    # Transformed outputs should have same shape and column names
    assert list(X.columns) == list(X2.columns)
    assert X2.shape == X.shape

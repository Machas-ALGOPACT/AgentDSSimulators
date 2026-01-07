import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.utils.persistence import save_object
import asyncio
from importlib.machinery import SourceFileLoader

# Helper to load router module by file path
def load_router_module():
    router_path = os.path.join(os.path.dirname(__file__), '..', 'router.py')
    router_path = os.path.abspath(router_path)
    return SourceFileLoader('claims_router', router_path).load_module()


def test_end_to_end_predict_temp_model(tmp_path):
    # Setup a temporary models directory inside the claims_complexity module
    project_root = os.path.join(os.path.dirname(__file__), '..')
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Create TF-IDF vectorizer and fit on sample texts
    sample_texts = ["rear end collision", "side impact injury", "minor scratch"]
    vec = TfidfVectorizer(max_features=3)
    vec.fit(sample_texts)
    feature_names = vec.get_feature_names_out()

    # Create dummy training set for model with tfidf features + numerical features
    num_cols = ['ReportedDamage', 'NumParties', 'HolderAge', 'AnnualMileage', 'CreditScore']

    tfidf_dummy = pd.DataFrame(np.random.rand(20, len(feature_names)), columns=[f"tfidf_{n}" for n in feature_names])
    num_dummy = pd.DataFrame(np.random.rand(20, len(num_cols)), columns=num_cols)

    X_train = pd.concat([tfidf_dummy, num_dummy], axis=1)
    y_train = np.random.choice(['Simple', 'Moderate', 'Complex'], size=20)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)

    # Save artifacts to expected model paths
    model_path = os.path.join(models_dir, 'ensemble_model.joblib')
    save_object(clf, model_path)

    vec_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    save_object(vec, vec_path)

    scaler = StandardScaler().fit(num_dummy)
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    save_object(scaler, scaler_path)

    # Now call router.predict with a realistic payload
    router = load_router_module()

    payload = {
        'ClaimID': 'C123',
        'PolicyID': 'P123',
        'ReportedDamage': 150.0,
        'NumParties': 1,
        'HolderAge': 40,
        'AnnualMileage': 10000,
        'CreditScore': 700,
        'Description': 'minor scratch and dent from low speed collision'
    }

    # Async call
    response = asyncio.run(router.predict(payload))

    # Validate response schema
    assert response.success is True
    assert 'prediction' in response.data

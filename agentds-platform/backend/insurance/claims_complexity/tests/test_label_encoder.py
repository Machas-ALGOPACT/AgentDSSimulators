import os
import tempfile
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from src.utils.persistence import save_object, load_object


def test_label_encoder_save_load():
    """Test LabelEncoder persistence"""
    le = LabelEncoder()
    classes = ['Simple', 'Moderate', 'Complex']
    y = np.array(['Simple', 'Moderate', 'Complex', 'Simple', 'Moderate'])
    y_encoded = le.fit_transform(y)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'label_encoder.joblib')
        save_object(le, path)
        
        loaded_le = load_object(path)
        # Check classes match (they're sorted alphabetically by sklearn)
        assert sorted(list(loaded_le.classes_)) == sorted(classes)
        
        # Check inverse transform works
        y_decoded = loaded_le.inverse_transform(y_encoded)
        assert list(y_decoded) == list(y)


def test_label_encoder_returns_readable_labels():
    """Test that label encoder maps back to human-readable labels"""
    le = LabelEncoder()
    classes = np.array(['Simple', 'Moderate', 'Complex'])
    le.fit(classes)
    
    # Encode and decode
    encoded = le.transform(['Simple', 'Complex', 'Moderate'])
    decoded = le.inverse_transform(encoded)
    
    assert list(decoded) == ['Simple', 'Complex', 'Moderate']
    assert decoded[0] == 'Simple'  # Human-readable

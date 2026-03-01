import os
import tempfile
from src.utils.persistence import save_object, load_object
from sklearn.preprocessing import StandardScaler


def test_save_load_scaler_roundtrip():
    scaler = StandardScaler()
    # Fit on a tiny sample
    X = [[1.0, 2.0], [2.0, 3.0]]
    scaler.fit(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'scaler.joblib')
        save_object(scaler, path)
        assert os.path.exists(path)

        loaded = load_object(path)
        # Check that loaded scaler retains mean attributes
        assert hasattr(loaded, 'mean_')
        assert loaded.mean_.shape == scaler.mean_.shape

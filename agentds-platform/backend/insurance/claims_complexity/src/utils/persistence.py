import joblib
import os
from typing import Any


def save_object(obj: Any, path: str) -> str:
    """Save an object to `path` using joblib. Creates directories if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_object(path: str) -> Any:
    """Load an object from `path` using joblib."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Object not found: {path}")
    return joblib.load(path)

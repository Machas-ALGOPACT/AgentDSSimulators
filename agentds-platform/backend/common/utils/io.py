import joblib
import json
import os
from pathlib import Path
from typing import Any, Dict

def save_artifact(obj: Any, path: Path):
    """Saves a python object using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_artifact(path: Path) -> Any:
    """Loads a python object using joblib."""
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found at {path}")
    return joblib.load(path)

def save_json(data: Dict, path: Path):
    """Saves a dictionary to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_json(path: Path) -> Dict:
    """Loads a dictionary from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found at {path}")
    with open(path, "r") as f:
        return json.load(f)

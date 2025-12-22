import os
from pathlib import Path

def get_project_root() -> Path:
    """Returns the project root directory."""
    # current file is in backend/common/utils/paths.py
    # root is 3 levels up
    return Path(__file__).parent.parent.parent.parent

def get_artifacts_path(domain: str, ps: str) -> Path:
    """Returns the artifacts path for a specific problem statement."""
    return get_project_root() / "backend" / domain / ps / "model" / "artifacts"

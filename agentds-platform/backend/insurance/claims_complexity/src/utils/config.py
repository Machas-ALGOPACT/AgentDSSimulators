import yaml
import os

def load_config(config_path="config/config.yaml"):
    """
    Load configuration from a YAML file.
    """
    if not os.path.exists(config_path):
        # Fallback if called from src/ or other subdirs
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        config_path = os.path.join(root_dir, "config", "config.yaml")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_full_path(relative_path, base_dir=None):
    """
    Get full absolute path from a relative path.
    """
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(base_dir, relative_path)

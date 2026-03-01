from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Optional
from src.utils.logger import setup_logger
from src.utils.persistence import save_object, load_object
import os

logger = setup_logger(__name__)

class ScalerWrapper:
    def __init__(self, config: dict = None, method: str = 'standard'):
        self.config = config or {}
        self.method = method
        self.scaler = None

    def fit(self, df):
        """Fit the scaler on a dataframe or array-like."""
        if self.method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = StandardScaler()
        self.scaler.fit(df)
        logger.info(f"Fitted {self.method} scaler on data with shape {getattr(df, 'shape', None)}")
        return self

    def transform(self, df):
        """Transform data using the fitted scaler."""
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit() before transform().")
        transformed = self.scaler.transform(df)
        return transformed

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_object(self.scaler, path)
        logger.info(f"Scaler saved to {path}")
        return path

    def load(self, path: str):
        self.scaler = load_object(path)
        logger.info(f"Scaler loaded from {path}")
        return self

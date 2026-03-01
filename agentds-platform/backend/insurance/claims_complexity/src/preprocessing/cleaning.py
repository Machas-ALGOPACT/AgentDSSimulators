import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataCleaner:
    def __init__(self, config):
        self.config = config
        
    def handle_missing_values(self, df):
        """
        Handle missing values in the dataframe.
        - Numeric: Median imputation
        - Categorical: Mode imputation or 'Unknown'
        """
        df = df.copy()
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Imputed missing values in numeric column {col} with median: {median_val}")
                
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])
                    logger.info(f"Imputed missing values in categorical column {col} with mode: {mode_val[0]}")
                else:
                    df[col] = df[col].fillna('Unknown')
                    logger.info(f"Imputed missing values in categorical column {col} with 'Unknown'")
                    
        return df

    def standardize_text(self, df, text_cols):
        """
        Standardize text fields (lowercase, strip whitespace).
        """
        df = df.copy()
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().str.strip()
                logger.info(f"Standardized text column: {col}")
        return df

    def handle_outliers(self, df, numeric_cols, method='clip'):
        """
        Handle outliers using IQR or clipping.
        """
        df = df.copy()
        if method == 'clip':
            for col in numeric_cols:
                if col in df.columns:
                    q1 = df[col].quantile(0.01)
                    q99 = df[col].quantile(0.99)
                    df[col] = df[col].clip(lower=q1, upper=q99)
                    logger.info(f"Clipped outliers for {col} at 1st and 99th percentiles")
        return df

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits dataframe into X_train, X_test, y_train, y_test.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def drop_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops rows with missing values.
    """
    initial_len = len(df)
    df_clean = df.dropna()
    dropped = initial_len - len(df_clean)
    if dropped > 0:
        logger.info(f"Dropped {dropped} rows with missing values.")
    return df_clean

def fill_missing_values(df: pd.DataFrame, strategy: str = "mean", fill_value = None) -> pd.DataFrame:
    """
    Fills missing values.
    """
    if strategy == "constant":
        return df.fillna(fill_value)
    # Simple numeric fill for now
    numeric_cols = df.select_dtypes(include=['number']).columns
    if strategy == "mean":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == "median":
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

def encode_categorical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Label encodes specified categorical columns.
    """
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in columns:
        if col in df_encoded.columns:
            # Convert to string to ensure consistent type before encoding
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded

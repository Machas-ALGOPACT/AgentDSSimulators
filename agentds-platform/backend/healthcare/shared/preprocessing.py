from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np

def create_preprocessing_pipeline(X: pd.DataFrame):
    """
    Creates a robust preprocessing pipeline by inferring column types.
    """
    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop columns not specified (e.g., weird metadata)
    )

    return preprocessor

def clean_data(df: pd.DataFrame, target_col: str = None):
    """
    Basic cleanup. 
    1. Drop duplicates.
    2. Drop IDs if looks like an ID (high cardinality categorical matching row count - strict heuristic).
    """
    df = df.drop_duplicates()
    
    # Heuristic: Drop columns that are obviously IDs (all unique, object/int type)
    # Be careful not to drop the target.
    drop_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        if df[col].nunique() == len(df):
            # Likely an ID
            drop_cols.append(col)
            
    if drop_cols:
        df = df.drop(columns=drop_cols)
        
    return df

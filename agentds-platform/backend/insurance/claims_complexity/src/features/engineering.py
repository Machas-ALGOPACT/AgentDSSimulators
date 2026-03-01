import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        
    def create_temporal_features(self, df, date_col):
        """
        Create temporal features from a date column.
        """
        if date_col not in df.columns:
            logger.warning(f"Date column {date_col} not found. Skipping temporal features.")
            return df
            
        df = df.copy()
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            
            df[f'{date_col}_Month'] = df[date_col].dt.month
            df[f'{date_col}_DayOfWeek'] = df[date_col].dt.dayofweek
            df[f'{date_col}_Hour'] = df[date_col].dt.hour
            df[f'{date_col}_IsWeekend'] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
            
            logger.info(f"Created temporal features from {date_col}")
        except Exception as e:
            logger.error(f"Error creating temporal features: {e}")
            
        return df

    def create_interaction_features(self, df):
        """
        Create interaction features between numeric columns.
        """
        df = df.copy()
        
        # Example: Damage per Party
        if 'ReportedDamage' in df.columns and 'NumParties' in df.columns:
            # Avoid division by zero
            df['DamagePerParty'] = df['ReportedDamage'] / df['NumParties'].replace(0, 1)
            logger.info("Created feature: DamagePerParty")
            
        # Example: Age * Credit (Risk proxy)
        if 'HolderAge' in df.columns and 'CreditScore' in df.columns:
            df['Age_Credit_Interaction'] = df['HolderAge'] * df['CreditScore']
            logger.info("Created feature: Age_Credit_Interaction")
            
        return df

    def encode_categorical_features(self, df, cat_cols=None):
        """
        One-hot encode categorical features.
        """
        df = df.copy()
        if cat_cols is None:
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            # Exclude target and IDs if they are in cat_cols
            exclude = [self.config['data']['target_col'], self.config['data']['id_col'], self.config['data']['join_col']]
            cat_cols = [c for c in cat_cols if c not in exclude]
            
        if not cat_cols:
            return df
            
        logger.info(f"One-hot encoding columns: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        return df

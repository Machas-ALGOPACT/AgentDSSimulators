import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.logger import setup_logger
from src.utils.persistence import save_object, load_object
from typing import List, Optional

logger = setup_logger(__name__)

class PreprocessingPipeline:
    """
    Preprocessing pipeline that encapsulates TF-IDF vectorization, scaling, and feature alignment.
    
    This pipeline:
    1. Extracts TF-IDF features from text column (if present)
    2. Scales numeric columns
    3. Passes through all other numeric columns as-is
    4. Ensures consistent feature alignment for transform
    """
    def __init__(self, config: dict = None, text_col: str = 'Description', numeric_cols: Optional[List[str]] = None):
        self.config = config or {}
        self.text_col = text_col
        self.numeric_cols = numeric_cols or self.config.get('features', {}).get('numerical', [])
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.scaler = None
        self.feature_columns: List[str] = []
        # Only exclude target/ID columns and raw text - KEEP all engineered features
        self.exclude_cols = [self.config.get('data', {}).get('target_col', 'ClaimComplexityLabel'),
                             self.config.get('data', {}).get('id_col', 'ClaimID'),
                             self.config.get('data', {}).get('join_col', 'PolicyID'),
                             'Description', 'ClaimType', 'VehicleType', 'PolicyStart', 'PolicyEnd', 'ClaimDate']

    def fit(self, df: pd.DataFrame):
        """
        Fit all transformers on the data.
        
        This method:
        1. Looks for pre-existing TF-IDF features (columns starting with 'tfidf_') 
           OR extracts TF-IDF from text column if it exists
        2. Scales configured numeric columns
        3. Collects all other numeric columns (one-hot encoded, temporal, aggregates, interactions)
        4. Combines them into a unified feature matrix
        """
        df = df.copy()
        
        # 1. Check for pre-existing TF-IDF features OR extract from text
        tfidf_cols_list = [c for c in df.columns if c.startswith('tfidf_')]
        
        if tfidf_cols_list:
            # TF-IDF features already exist (created earlier in pipeline)
            tfidf_df = df[tfidf_cols_list]
            # Sanitize column names for LightGBM compatibility
            tfidf_df.columns = [c.replace(' ', '_').replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(':', '').replace(',', '').replace('"', '') for c in tfidf_df.columns]
            logger.info(f"Found {len(tfidf_cols_list)} pre-existing TF-IDF features in dataframe")
            self.vectorizer = None  # No need to fit vectorizer if features already exist
        elif self.text_col in df.columns:
            # Extract TF-IDF from text column
            texts = df[self.text_col].astype(str).fillna('')
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.get('features', {}).get('tfidf', {}).get('max_features', 500),
                ngram_range=tuple(self.config.get('features', {}).get('tfidf', {}).get('ngram_range', (1, 1))),
                stop_words='english'
            )
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            tfidf_cols_list = [f"tfidf_{name}".replace(' ', '_').replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace(':', '').replace(',', '').replace('"', '') for name in self.vectorizer.get_feature_names_out()]
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols_list, index=df.index)
            logger.info(f"Fitted TF-IDF vectorizer with {len(tfidf_cols_list)} features")
        else:
            # No TF-IDF features and no text column
            tfidf_cols_list = []
            tfidf_df = pd.DataFrame(index=df.index)
            self.vectorizer = None

        # 2. Scale explicitly configured numeric features
        scaled_cols_list = []
        scaled_df = pd.DataFrame(index=df.index)
        num_cols_to_scale = [c for c in self.numeric_cols if c in df.columns]
        if num_cols_to_scale:
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.scaler.fit(df[num_cols_to_scale])
            scaled = self.scaler.transform(df[num_cols_to_scale])
            scaled_df = pd.DataFrame(scaled, columns=num_cols_to_scale, index=df.index)
            scaled_cols_list = num_cols_to_scale
            logger.info(f"Fitted scaler on {len(num_cols_to_scale)} numeric columns")

        # 3. CRITICAL: Collect ALL other numeric columns
        #    (one-hot encoded, aggregates, temporal, interactions, etc.)
        from pandas.api.types import is_numeric_dtype
        other_numeric_cols = [c for c in df.columns 
                             if is_numeric_dtype(df[c])
                             and c not in self.exclude_cols 
                             and c not in tfidf_cols_list
                             and c not in scaled_cols_list]
        
        if other_numeric_cols:
            other_numeric_df = df[other_numeric_cols]
            logger.info(f"Including {len(other_numeric_cols)} additional numeric features (one-hot, aggregates, temporal, interactions)")
        else:
            other_numeric_df = pd.DataFrame(index=df.index)

        # 4. Combine all features in order: TF-IDF + scaled numeric + other numeric
        features = pd.concat([tfidf_df, scaled_df, other_numeric_df], axis=1)
        self.feature_columns = features.columns.tolist()

        logger.info(f"Preprocessing pipeline fitted. Total feature columns: {len(self.feature_columns)}")
        logger.info(f"  - TF-IDF: {len(tfidf_cols_list)} features")
        logger.info(f"  - Scaled numeric: {len(scaled_cols_list)} features")
        logger.info(f"  - Other numeric: {len(other_numeric_cols)} features")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted transformers and return aligned feature matrix.
        
        This method:
        1. Looks for pre-existing tfidf_* columns OR applies vectorizer if available
        2. Scales numeric columns if scaler was fitted
        3. Collects all other numeric columns
        4. Returns features aligned to the columns learned during fit
        """
        df = df.copy()
        out = pd.DataFrame(index=df.index)

        # 1. Handle TF-IDF columns (either pre-existing or generated)
        tfidf_cols_in_df = [c for c in df.columns if c.startswith('tfidf_')]
        if tfidf_cols_in_df:
            # Use pre-existing TF-IDF columns
            out = pd.concat([out, df[tfidf_cols_in_df]], axis=1)
        elif self.vectorizer is not None and self.text_col in df.columns:
            # Apply fitted vectorizer to text column
            texts = df[self.text_col].astype(str).fillna('')
            tfidf_matrix = self.vectorizer.transform(texts)
            tfidf_cols = [f"tfidf_{name}" for name in self.vectorizer.get_feature_names_out()]
            tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols, index=df.index)
            out = pd.concat([out, tfidf_df], axis=1)

        # 2. Numeric scaling transform
        num_cols_to_scale = [c for c in self.numeric_cols if c in df.columns]
        if self.scaler is not None and num_cols_to_scale:
            scaled = self.scaler.transform(df[num_cols_to_scale])
            scaled_df = pd.DataFrame(scaled, columns=num_cols_to_scale, index=df.index)
            out = pd.concat([out, scaled_df], axis=1)

        # 3. Add all other numeric columns
        from pandas.api.types import is_numeric_dtype
        other_numeric_cols = [c for c in df.columns 
                             if is_numeric_dtype(df[c])
                             and c not in self.exclude_cols 
                             and c not in out.columns]
        if other_numeric_cols:
            out = pd.concat([out, df[other_numeric_cols]], axis=1)

        # 4. Ensure all expected columns present (fill missing with 0)
        for c in self.feature_columns:
            if c not in out.columns:
                out[c] = 0

        # 5. Reorder to match fitted feature columns and drop any extras
        out = out[self.feature_columns]
        return out

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_object(self, path)
        logger.info(f"Saved preprocessing pipeline to {path}")
        return path

    @staticmethod
    def load(path: str):
        obj = load_object(path)
        logger.info(f"Loaded preprocessing pipeline from {path}")
        return obj

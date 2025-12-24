import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils.logger import setup_logger
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

logger = setup_logger(__name__)

class TextFeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.tfidf_params = config['features']['tfidf']
        self.vectorizer = TfidfVectorizer(
            max_features=self.tfidf_params['max_features'],
            ngram_range=tuple(self.tfidf_params['ngram_range']),
            stop_words='english'
        )
        self.text_col = 'Description' # Should be configurable, but defaulting to known col
        
    def extract_basic_text_features(self, df, text_col):
        """
        Extract length, word count, etc.
        """
        if text_col not in df.columns:
            return df
            
        df = df.copy()
        df[f'{text_col}_Length'] = df[text_col].astype(str).str.len()
        df[f'{text_col}_WordCount'] = df[text_col].astype(str).apply(lambda x: len(x.split()))
        
        logger.info(f"Created basic text features for {text_col}")
        return df

    def fit_transform_tfidf(self, df, text_col):
        """
        Fit TF-IDF and transform the text column. Returns df with new columns.
        """
        if text_col not in df.columns:
            logger.warning(f"Text column {text_col} not found for TF-IDF.")
            return df, None
            
        logger.info(f"Fitting TF-IDF on {text_col}...")
        text_data = df[text_col].astype(str).fillna('')
        tfidf_matrix = self.vectorizer.fit_transform(text_data)
        
        feature_names = [f"tfidf_{name}" for name in self.vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=df.index)
        
        logger.info(f"Created {len(feature_names)} TF-IDF features")
        
        # Concat original df with tfidf features
        # We generally drop the original text column before modeling, but here we just add features
        df_new = pd.concat([df, tfidf_df], axis=1)
        
        return df_new, self.vectorizer
    
    def transform_tfidf(self, df, text_col, vectorizer=None):
        """
        Transform using existing vectorizer.
        """
        if vectorizer is None:
            vectorizer = self.vectorizer
            
        if text_col not in df.columns:
            return df
            
        text_data = df[text_col].astype(str).fillna('')
        tfidf_matrix = vectorizer.transform(text_data)
        
        feature_names = [f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names, index=df.index)
        
        df_new = pd.concat([df, tfidf_df], axis=1)
        return df_new

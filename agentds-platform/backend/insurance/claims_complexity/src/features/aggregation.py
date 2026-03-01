import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class AggregateFeatureEngineer:
    def __init__(self, config):
        self.config = config
        
    def create_policy_aggregates(self, claims_df):
        """
        Create aggregate features based on PolicyID.
        """
        if 'PolicyID' not in claims_df.columns:
            return claims_df
            
        logger.info("Creating aggregate features by PolicyID...")
        
        # Count claims per policy
        claim_counts = claims_df.groupby('PolicyID').size().reset_index(name='Policy_ClaimCount')
        
        # Average damage per policy (if multiple claims exist)
        avg_damage = claims_df.groupby('PolicyID')['ReportedDamage'].mean().reset_index(name='Policy_AvgDamage')
        
        # Merge back
        df_agg = pd.merge(claim_counts, avg_damage, on='PolicyID')
        
        logger.info(f"Created aggregate features for {len(df_agg)} policies")
        return df_agg

    def merge_aggregates(self, df, agg_df, on='PolicyID'):
        """
        Merge aggregate features into the main dataframe.
        """
        if agg_df is None or agg_df.empty:
            return df
            
        logger.info(f"Merging aggregate features into main dataframe...")
        df_merged = pd.merge(df, agg_df, on=on, how='left')
        return df_merged

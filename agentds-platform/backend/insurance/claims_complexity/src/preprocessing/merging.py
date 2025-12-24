import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataMerger:
    def __init__(self, config):
        self.config = config
        self.join_col = config['data']['join_col']
        
    def merge_claims_policies(self, claims_df, policies_df):
        """
        Left join claims to policies.
        """
        if policies_df is None:
            logger.warning("Policies dataframe is None. Returning claims dataframe as is.")
            return claims_df
            
        logger.info(f"Merging claims ({len(claims_df)} rows) with policies ({len(policies_df)} rows) on {self.join_col}")
        
        merged_df = pd.merge(
            claims_df, 
            policies_df, 
            on=self.join_col, 
            how='left',
            validate='many_to_one' # Assuming one policy can have multiple claims
        )
        
        logger.info(f"Merged dataframe has {len(merged_df)} rows and {len(merged_df.columns)} columns")
        
        # Check for claims without policy info
        missing_policies = merged_df[policies_df.columns[1]].isnull().sum() if len(policies_df.columns) > 1 else 0
        if missing_policies > 0:
            logger.warning(f"{missing_policies} claims did not find matching policy information")
            merged_df['HasPolicyInfo'] = merged_df[policies_df.columns[1]].notnull().astype(int)
            
        return merged_df

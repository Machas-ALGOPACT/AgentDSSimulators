import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataValidator:
    def __init__(self, config):
        self.config = config
        self.target_col = config['data']['target_col']
        self.id_col = config['data']['id_col']
        
    def validate_schema(self, df, required_cols, name="dataframe"):
        """
        Check if required columns exist in the dataframe.
        """
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in {name}: {missing_cols}")
            return False, missing_cols
        logger.info(f"Schema validation passed for {name}")
        return True, []

    def check_missing_values(self, df, name="dataframe"):
        """
        Identify and report missing values.
        """
        missing_counts = df.isnull().sum()
        cols_with_missing = missing_counts[missing_counts > 0]
        if not cols_with_missing.empty:
            logger.warning(f"Missing values in {name}:\n{cols_with_missing}")
        else:
            logger.info(f"No missing values found in {name}")
        return cols_with_missing

    def check_duplicates(self, df, id_col, name="dataframe"):
        """
        Detect duplicate IDs.
        """
        duplicates = df[id_col].duplicated().sum()
        if duplicates > 0:
            logger.error(f"Found {duplicates} duplicate {id_col} in {name}")
            return False
        logger.info(f"No duplicate {id_col} found in {name}")
        return True

    def validate_ranges(self, df, name="dataframe"):
        """
        Validate value ranges (e.g., ReportedDamage >= 0).
        """
        valid = True
        if 'ReportedDamage' in df.columns:
            negative_damage = (df['ReportedDamage'] < 0).sum()
            if negative_damage > 0:
                logger.error(f"Found {negative_damage} rows with negative ReportedDamage in {name}")
                valid = False
        
        if 'NumParties' in df.columns:
            invalid_parties = (df['NumParties'] < 1).sum()
            if invalid_parties > 0:
                logger.warning(f"Found {invalid_parties} rows with NumParties < 1 in {name}")
                # Not necessarily an error, but worth logging
                
        return valid

    def check_for_leakage(self, train_df, test_df, name="split"):
        """
        Check for data leakage (e.g., test labels in training, or test IDs in training).
        """
        if test_df is not None and self.target_col in test_df.columns:
            logger.error(f"CRITICAL: Target column {self.target_col} found in test set!")
            return False
            
        if test_df is not None:
            common_ids = set(train_df[self.id_col]).intersection(set(test_df[self.id_col]))
            if common_ids:
                logger.warning(f"Overlap detected: {len(common_ids)} {self.id_col} appear in both train and test")
                
        return True

    def run_all_checks(self, train_claims, train_policies=None, test_claims=None, test_policies=None):
        """
        Run all validation checks on the provided datasets.
        """
        logger.info("Starting comprehensive data validation...")
        
        # Validate train claims
        required_train_claims = [self.id_col, self.config['data']['join_col'], self.target_col]
        s1, _ = self.validate_schema(train_claims, required_train_claims, "train_claims")
        d1 = self.check_duplicates(train_claims, self.id_col, "train_claims")
        r1 = self.validate_ranges(train_claims, "train_claims")
        
        # Validate train policies if provided
        s2, d2, r2 = True, True, True
        if train_policies is not None:
            required_policies = [self.config['data']['join_col']]
            s2, _ = self.validate_schema(train_policies, required_policies, "train_policies")
            d2 = self.check_duplicates(train_policies, self.config['data']['join_col'], "train_policies")
            r2 = self.validate_ranges(train_policies, "train_policies")
            
        # Leakage check
        l1 = self.check_for_leakage(train_claims, test_claims)
        
        logger.info("Validation complete.")
        return all([s1, d1, r1, s2, d2, r2, l1])

if __name__ == "__main__":
    from src.data.loader import DataLoader
    from src.utils.config import load_config
    config = load_config()
    loader = DataLoader(config)
    data = loader.load_all_data()
    validator = DataValidator(config)
    validator.run_all_checks(data['train_claims'], data['train_policies'])

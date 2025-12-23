import pandas as pd

def feature_engineering_coupon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features for coupon redemption.
    """
    df = df.copy()
    
    # Example placeholder: Convert 'discount_pct' or similar if present
    # We maintain IDs for features but might drop them for pure model training if high cardinality
    # For this task, we'll keep it simple.
    
    return df

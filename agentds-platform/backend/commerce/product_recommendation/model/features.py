import pandas as pd
from backend.commerce.shared.constants import COL_CUSTOMER_ID, COL_PRODUCT_ID

def feature_engineering_recommendation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features for recommendation. For a basic Collaborative Filtering or Regression approach,
    we ensure IDs are present. 
    """
    df = df.copy()
    
    # In a real system, we'd add user/item metadata features here.
    # For now, we rely on ID encodings or pass through for the model to handle.
    
    return df

import pandas as pd
# No specific derived features needed for now as 'week' is already int
# and we are missing 'date'.

def feature_engineering_forecasting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pass-through or basic casting.
    """
    df = df.copy()
    return df

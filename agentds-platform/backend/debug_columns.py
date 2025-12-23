from backend.commerce.shared.dataset_loader import load_commerce_dataset
from backend.commerce.shared.constants import TASK_DEMAND_FORECASTING, TASK_COUPON_REDEMPTION

try:
    print("--- Demand Forecasting ---")
    df = load_commerce_dataset(TASK_DEMAND_FORECASTING)
    print(df.columns.tolist())
    print(df.head(1))
except Exception as e:
    print(e)
    
try:
    print("\n--- Coupon Redemption ---")
    df = load_commerce_dataset(TASK_COUPON_REDEMPTION)
    print(df.columns.tolist())
    print(df.head(1))
except Exception as e:
    print(e)

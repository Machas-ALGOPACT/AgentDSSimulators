
# Problem Statements
TASK_DEMAND_FORECASTING = "demand_forecasting"
TASK_PRODUCT_RECOMMENDATION = "product_recommendation"
TASK_COUPON_REDEMPTION = "coupon_redemption"

# Dataset Keys
DATASET_SALES_HISTORY = "sales_history"
DATASET_PURCHASES = "purchases"
DATASET_COUPON_OFFERS = "coupon_offers"

# File Mappings
DATASET_FILES = {
    TASK_DEMAND_FORECASTING: "Commerce/sales_history_train.csv",
    TASK_PRODUCT_RECOMMENDATION: "Commerce/purchases_train.csv",
    TASK_COUPON_REDEMPTION: "Commerce/coupon_offers_train.csv"
}

# Column Names - Demand Forecasting
COL_SKU_ID = "sku_id"
COL_WEEK = "week"
COL_WEEKLY_SALES = "units_sold" # Updated from 'weekly_sales'
COL_PRICE = "price"
COL_PROMO = "promo_flag"

# Column Names - Product Recommendation
COL_CUSTOMER_ID = "customer_id"
COL_PRODUCT_ID = "sku_id" # Updated from 'product_id'
COL_RATING = "rating"
COL_PURCHASE_DATE = "purchase_date"

# Column Names - Coupon Redemption
COL_OFFER_ID = "offer_id"
COL_REDEEMED = "target_redeem" # Updated from 'redeemed'

# Model Artifact Paths (Relative to domain dirs)
MODEL_FILENAME = "model.joblib"
METRICS_FILENAME = "metrics.json"

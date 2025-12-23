# Commerce Domain

This domain handles analytics and ML predictions for retail/commerce scenarios.

## Problem Statements

### 1. Demand Forecasting
- **Goal**: Predict weekly sales for store departments.
- **Type**: Regression (XGBoost)
- **Routes**: `/api/v1/commerce/demand-forecasting`

### 2. Product Recommendation
- **Goal**: Personalize product recommendations / predict ranking score.
- **Type**: Ranking/Regression (RandomForest)
- **Routes**: `/api/v1/commerce/product-recommendation`

### 3. Coupon Redemption
- **Goal**: Predict likelihood of coupon redemption.
- **Type**: Binary Classification (XGBoost)
- **Routes**: `/api/v1/commerce/coupon-redemption`

## Development
See individual folders for specific implementation details.

### Shared Infrastructure
- `shared/dataset_loader.py`: Handles loading specific CSVs from Hugging Face.
- `shared/preprocessing.py`: Shared cleaning/encoding logic.
- `shared/evaluation.py`: Shared metrics calculation.

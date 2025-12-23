# Demand Forecasting (Commerce)

## Problem Statement
Predict weekly sales for store departments based on historical data.

## Dataset
- **File**: `sales_history_train.csv` (loaded via shared dataset loader)
- **Features**: `is_holiday`, `temperature`, `fuel_price`, `cpi`, `unemployment`, `year`, `month`, `week`.
- **Target**: `weekly_sales`.

## Metrics
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)

## API Endpoints

### 1. Health Check
`GET /api/v1/commerce/demand-forecasting/health`

### 2. Train Model
`POST /api/v1/commerce/demand-forecasting/train`
- Triggers training on CPU using XGBoost.
- Returns metrics.

### 3. Predict
`POST /api/v1/commerce/demand-forecasting/predict`
- **Body**:
  ```json
  {
    "inputs": [
      {
        "store_id": 1,
        "sku_id": 1,
        "date": "2023-11-24",
        "is_holiday": true,
        "temperature": 45.0,
        "fuel_price": 3.5,
        "cpi": 211.0,
        "unemployment": 8.0
      }
    ]
  }
  ```

## Implementation Details
- Uses `XGBRegressor` for high performance on regression tasks.
- Feature engineering extracts temporal features from `date`.

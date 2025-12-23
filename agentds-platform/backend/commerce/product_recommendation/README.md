# Product Recommendation (Commerce)

## Problem Statement
Personalized product recommendation (ranking based on predicted score).

## Dataset
- **File**: `purchases_train.csv`
- **Features**: `customer_id`, `product_id`.
- **Target**: `rating` (inferred or explicit).

## Metrics
- **RMSE** (predicting score)

## API Endpoints

### 1. Health Check
`GET /api/v1/commerce/product-recommendation/health`

### 2. Train Model
`POST /api/v1/commerce/product-recommendation/train`

### 3. Predict
`POST /api/v1/commerce/product-recommendation/predict`
- **Body**:
  ```json
  {
    "inputs": [
      {
        "customer_id": 123,
        "product_id": 456
      }
    ]
  }
  ```

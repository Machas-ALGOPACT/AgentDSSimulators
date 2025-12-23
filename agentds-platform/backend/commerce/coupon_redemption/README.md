# Coupon Redemption (Commerce)

## Problem Statement
Predict whether a user will redeem a coupon (Binary Classification).

## Dataset
- **File**: `coupon_offers_train.csv`
- **Features**: `offer_id`, `customer_id`.
- **Target**: `redeemed` (or `target_redeem`).

## Metrics
- **Macro-F1** (Imbalanced data)
- **AUC**

## API Endpoints

### 1. Health Check
`GET /api/v1/commerce/coupon-redemption/health`

### 2. Train Model
`POST /api/v1/commerce/coupon-redemption/train`
- Handles class imbalance using `scale_pos_weight`.

### 3. Predict
`POST /api/v1/commerce/coupon-redemption/predict`
- **Body**:
  ```json
  {
    "inputs": [
      {
        "customer_id": 123,
        "offer_id": 999
      }
    ]
  }
  ```

# ED Cost Forecasting

## Problem Statement
Forecast the cost associated with Emergency Department visits.

## Data
Source: `lainmn/AgentDS-Healthcare` (Keyword: `cost` or `ed`)

## Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

## Usage

### Train
```bash
curl -X POST "http://localhost:8000/api/v1/healthcare/ed-cost-forecasting/train"
```

### Predict
```bash
curl -X POST "http://localhost:8000/api/v1/healthcare/ed-cost-forecasting/predict" \
-H "Content-Type: application/json" \
-d '{
  "records": [
    {"procedure_code": "A123", "duration_minutes": 45},
    {"procedure_code": "B999", "duration_minutes": 120}
  ]
}'
```

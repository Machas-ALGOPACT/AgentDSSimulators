# Readmission Prediction Service

## Problem Statement
Predict whether a patient will be readmitted to the hospital within 30 days of discharge.

## Data
Source: `lainmn/AgentDS-Healthcare` (Keyword: `readmission`)

## Metrics
- Macro-F1 Score
- Accuracy

## Usage

### Train
```bash
curl -X POST "http://localhost:8000/api/v1/healthcare/readmission-prediction/train"
```

### Predict
```bash
curl -X POST "http://localhost:8000/api/v1/healthcare/readmission-prediction/predict" \
-H "Content-Type: application/json" \
-d '{
  "records": [
    {"age": 65, "diabetes": 1, "visits": 5},
    {"age": 30, "diabetes": 0, "visits": 1}
  ]
}'
```

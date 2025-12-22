# Discharge Readiness

## Problem Statement
Determine if a patient is clinically and operationally ready for discharge.

## Data
Source: `lainmn/AgentDS-Healthcare` (Keyword: `discharge`)

## Metrics
- Macro-F1

## Usage

### Train
```bash
curl -X POST "http://localhost:8000/api/v1/healthcare/discharge-readiness/train"
```

### Predict
```bash
curl -X POST "http://localhost:8000/api/v1/healthcare/discharge-readiness/predict" \
-H "Content-Type: application/json" \
-d '{
  "records": [
    {"vital_stability": 0.9, "checklist_score": 10},
    {"vital_stability": 0.2, "checklist_score": 2}
  ]
}'
```

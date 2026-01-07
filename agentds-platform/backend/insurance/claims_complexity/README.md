# Claims Complexity - Quick Guide ✅

This folder contains the code for the Claims Complexity prediction pipeline.

## Saved artifacts (models/)
- `ensemble_model.joblib` (or `<model>_model.joblib`) — trained model artifact
- `tfidf_vectorizer.joblib` — fitted TF-IDF vectorizer used for text features
- `scaler.joblib` — fitted numeric scaler (StandardScaler)
- `preprocessing_pipeline.joblib` — fitted `PreprocessingPipeline` (TF-IDF + scaler + feature metadata)
- `label_encoder.joblib` — fitted `LabelEncoder` for mapping between encoded labels (0,1,2) and human-readable labels (Simple, Moderate, Complex)

These artifacts are saved to the path configured in `config/config.yaml` under `paths.models`.

## How to run the full pipeline
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python main.py` — this will train the configured model, save model and preprocessing artifacts, and (if test data exists) generate a submission in `outputs/`.

## Inference (API)
- Router: `/claims-complexity/predict` (POST)
- Payload: JSON with claim fields (e.g., `ClaimID`, `PolicyID`, `ReportedDamage`, `NumParties`, `Description`, `HolderAge`, `AnnualMileage`, `CreditScore`)
- The endpoint will attempt to load `preprocessing_pipeline.joblib` and `ensemble_model.joblib` from the `models/` folder and return a structured response with `prediction` and optional `probabilities`.

Example (curl):

```
curl -X POST http://localhost:8000/api/v1/insurance/claims-complexity/predict \
  -H 'Content-Type: application/json' \
  -d '{"ClaimID": "C123", "PolicyID": "P1", "ReportedDamage": 120.0, "NumParties": 1, "Description": "minor scratch" }'
```

## Tests
- Run `pytest` in this package to exercise persistence, pipeline and an end-to-end basic inference test.

## Notes & Next Steps
- The `PreprocessingPipeline` currently includes TF-IDF on `Description` and StandardScaler on numeric features listed in `config/config.yaml`.
- Next enhancements: add support for categorical one-hot alignment, save LabelEncoder mapping, and improve/stabilize inference schema validation.

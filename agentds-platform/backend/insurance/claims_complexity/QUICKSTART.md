# Claims Complexity Prediction — Complete Guide

## The Problem (in plain English)

Your company insures cars. When someone files a claim (e.g., "I got in a car accident"), an insurance agent has to decide: **Is this claim simple, moderate, or complex?**

- **Simple**: Quick to process, clear liability, straightforward payout (e.g., minor scratches, no injury).
- **Moderate**: Some complications—multiple parties involved, injury questions, disputed fault.
- **Complex**: Very complicated—major damage, multiple injuries, unclear liability, fraud risk.

**Why does this matter?** Because:
- Simple claims can be auto-approved in 1 day → saves money.
- Moderate claims need a human reviewer → takes 5-7 days.
- Complex claims need a specialist + investigation → takes weeks.

**The Goal**: Build a machine learning system that **automatically reads claim details** (damage amount, number of people involved, accident description, driver age, credit score, etc.) and **predicts the complexity level**. This lets you route claims to the right team immediately instead of manually reviewing every single one.

---

## How Your Code Solves This

You have a **machine learning pipeline** (a step-by-step process) that:

1. **Reads raw data** (CSV files with claim and policy information)
2. **Cleans it** (fills in missing values, fixes errors)
3. **Extracts features** (turns raw data into numbers the ML model can understand)
4. **Trains a model** (teaches it to recognize patterns between claim details and complexity)
5. **Evaluates it** (checks if it's accurate)
6. **Saves it** (stores the trained model so you can use it later without retraining)
7. **Makes predictions** (uses the saved model to predict complexity for new claims)

---

## What We Just Built (The Improvements)

You already had steps 1–5 done. But **steps 6 & 7 had problems**:

### ❌ The Original Problem:
When your code trained the model, it was like baking a cake and throwing away the recipe. The next day:
- The **TF-IDF vectorizer** (used to convert text like "rear-end collision" into numbers) was lost.
- The **scaler** (used to normalize numbers like damage amounts) was lost.
- The **label mapping** (Simple=0, Moderate=1, Complex=2 in the model, but you need to convert back to the original names) was lost.

So if someone tried to use the model later for predictions, it would either crash or give the wrong answer.

### ✅ What We Fixed:

**1. Added persistence (saving transformers)**
   - Created `src/utils/persistence.py` — a utility to save and load any Python object using `joblib`.
   - Now when you train, the code saves:
     - `tfidf_vectorizer.joblib` — the text transformer
     - `scaler.joblib` — the number normalizer
     - `label_encoder.joblib` — the label name mapper (Simple ↔ 0)
     - `preprocessing_pipeline.joblib` — all of the above bundled together

**2. Added scaling**
   - Created `src/preprocessing/scaling.py` — standardizes numeric features (e.g., turning damage amounts into a standard scale).
   - This is saved and loaded so predictions use the exact same scaling as training.

**3. Created a preprocessing pipeline**
   - Created `src/preprocessing/pipeline.py` — bundles TF-IDF + scaling + feature alignment into one reusable object.
   - Train once, save it, then use the exact same transformations for new data.

**4. Implemented API predictions**
   - Updated `router.py` (`/predict` endpoint) to load the saved model + pipeline and make predictions.
   - Returns **human-readable labels** ("Simple", not "0").

**5. Added tests**
   - `tests/test_persistence.py`, `tests/test_pipeline.py`, `tests/test_label_encoder.py` — verify everything saves/loads correctly.

---

## How to Run It

### **Step 1: Install dependencies**
```bash
cd c:\Users\bohar\DRIVE F\My Masters Project\AgentDSSimulators\agentds-platform\backend\insurance\claims_complexity
pip install -r requirements.txt
```

### **Step 2: Run the training pipeline**
```bash
python main.py
```

**What happens:**
- Loads train data from `data/raw/` (train_claims.csv, train_policies_subset.csv)
- Cleans and engineers features
- Trains the model (default: ensemble of Random Forest + XGBoost + LightGBM)
- **Saves everything** to `models/`:
  - `ensemble_model.joblib`
  - `tfidf_vectorizer.joblib`
  - `scaler.joblib`
  - `label_encoder.joblib`
  - `preprocessing_pipeline.joblib`
- If test data exists, generates submission in `outputs/submission_<timestamp>.csv`

**Expected output (in console):**
```
Starting Auto Insurance Claims Complexity Prediction Pipeline
Loading data from data/raw/train_claims.csv
Loaded 1000 rows and 9 columns from train_claims.csv
...
Training Ensemble Model (VotingClassifier: RF, XGB, LGBM)...
Model training complete.
Ensemble Evaluation - Macro-F1: 0.6234
...
Saved preprocessing pipeline to models/preprocessing_pipeline.joblib
Ensemble model saved to models/ensemble_model.joblib
Pipeline execution finished successfully.
```

### **Step 3: Make predictions via API**
```bash
# Start the FastAPI server (this runs from the backend root)
cd c:\Users\bohar\DRIVE F\My Masters Project\AgentDSSimulators\agentds-platform\backend
uvicorn main:app --reload
```

Then call the `/predict` endpoint:
```bash
curl -X POST http://localhost:8000/api/v1/insurance/claims-complexity/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ClaimID": "C12345",
    "PolicyID": "P5678",
    "ReportedDamage": 500.0,
    "NumParties": 2,
    "HolderAge": 35,
    "AnnualMileage": 12000,
    "CreditScore": 720,
    "Description": "multi-car collision on highway"
  }'
```

**Expected response:**
```json
{
  "success": true,
  "message": "Prediction successful",
  "data": {
    "prediction": "Complex",
    "probabilities": {
      "Simple": 0.05,
      "Moderate": 0.20,
      "Complex": 0.75
    }
  }
}
```

---

## Saved Artifacts

After training, your `models/` folder contains:

| File | Purpose |
|------|---------|
| `ensemble_model.joblib` | The trained ML model (Random Forest + XGBoost + LightGBM voting) |
| `preprocessing_pipeline.joblib` | Bundles TF-IDF vectorizer, scaler, and feature logic |
| `tfidf_vectorizer.joblib` | Converts text descriptions into numeric features |
| `scaler.joblib` | Normalizes numeric features (damage, age, mileage, etc.) |
| `label_encoder.joblib` | Maps between encoded labels (0,1,2) and human-readable names (Simple, Moderate, Complex) |

These are loaded automatically by:
- `main.py` when processing test data for submission
- `router.py` `/predict` endpoint for real-time predictions

---

## How to Know It's Working

### **1. After running `main.py`:**
- Check `models/` folder — should have 5 files
- Check console output — should see `Macro-F1: 0.5X` (ideally > 0.50)
- Check `outputs/` folder — should have a `submission_<timestamp>.csv`

### **2. After running tests:**
```bash
pytest tests/
```

Should see:
```
tests/test_persistence.py::test_save_load_scaler_roundtrip PASSED
tests/test_pipeline.py::test_pipeline_fit_transform_save_load PASSED
tests/test_label_encoder.py::test_label_encoder_save_load PASSED
...
```

### **3. After calling `/predict`:**
- Response `success: true`
- `prediction` is one of: "Simple", "Moderate", "Complex" (human-readable)
- `probabilities` sums to ~1.0 and shows confidence for each class

---

## The Big Picture (Why This Matters)

**Before our changes:**
```
Train model → Save only model → Next day: Can't use it (transformers missing) → Crashes or wrong predictions
```

**After our changes:**
```
Train model → Save model + ALL transformers + metadata → Next day: Load everything → Use exact same pipeline → Correct predictions every time
```

This is called **reproducibility** — the backbone of production ML systems. It ensures:
- ✅ Your predictions are consistent day after day
- ✅ New team members can use the model without errors
- ✅ Auditors can verify "how did you get that prediction?"
- ✅ The API always works correctly

---

## File Structure

```
insurance/claims_complexity/
├── main.py                           # Entry point for training
├── router.py                         # FastAPI endpoint (/predict)
├── config/
│   └── config.yaml                   # Configuration (model type, paths, hyperparams)
├── data/
│   ├── raw/                          # Original CSV files
│   ├── processed/                    # Cleaned data (if needed)
│   └── features/                     # Engineered features (if needed)
├── models/                           # Saved artifacts after training
│   ├── ensemble_model.joblib
│   ├── preprocessing_pipeline.joblib
│   ├── tfidf_vectorizer.joblib
│   ├── scaler.joblib
│   └── label_encoder.joblib
├── outputs/
│   ├── logs/                         # Execution logs
│   └── submission_*.csv              # Generated predictions for test set
├── src/
│   ├── data/
│   │   ├── loader.py                 # Load CSVs
│   │   └── validator.py              # Validate data integrity
│   ├── preprocessing/
│   │   ├── cleaning.py               # Handle missing values, outliers
│   │   ├── merging.py                # Merge claims + policies
│   │   ├── scaling.py                # Normalize numeric features
│   │   └── pipeline.py               # Bundle all preprocessing steps
│   ├── features/
│   │   ├── engineering.py            # Create temporal, interaction features
│   │   ├── text_features.py          # TF-IDF for descriptions
│   │   └── aggregation.py            # Aggregate statistics per policy
│   ├── models/
│   │   ├── baseline.py               # Random Forest
│   │   ├── advanced.py               # XGBoost, LightGBM
│   │   ├── ensemble.py               # Voting ensemble
│   │   └── tuning.py                 # Hyperparameter tuning with Optuna
│   ├── evaluation/
│   │   ├── analysis.py               # Error analysis, feature importance
│   │   └── submission.py             # Generate submission CSV
│   └── utils/
│       ├── config.py                 # Load YAML config
│       ├── logger.py                 # Setup logging
│       └── persistence.py            # Save/load objects
├── tests/
│   ├── test_persistence.py
│   ├── test_pipeline.py
│   ├── test_label_encoder.py
│   └── test_inference.py
├── requirements.txt                  # Python dependencies
└── README.md                         # Overview
```

---

## Next Steps (Future Improvements)

While the core pipeline is solid, here are potential enhancements:

1. **Feature Selection** — Implement correlation-based filtering or RFE to reduce feature count
2. **Cross-Validation** — Add K-Fold CV for more robust evaluation metrics
3. **Experiment Tracking** — Integrate MLflow or Weights & Biases to track model versions
4. **Hyperparameter Optimization** — Full grid/random search for all model types
5. **Unit Tests** — Expand test coverage for all critical modules
6. **Documentation** — Add docstrings and type hints throughout

---

## Questions?

- **Model not converging?** Check `config.yaml` for model hyperparameters and adjust.
- **API endpoint not responding?** Ensure FastAPI is running and port 8000 is available.
- **Predictions seem wrong?** Verify test data has same columns as training data; check logs for preprocessing errors.
- **Out of memory?** Reduce TF-IDF max_features or use stratified sampling in config.

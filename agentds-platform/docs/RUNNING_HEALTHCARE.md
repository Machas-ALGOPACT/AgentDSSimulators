# Running the Healthcare Backend with Local Data

## 1. Local Data Setup

You have chosen to use local CSV files. Please follow these steps to place them correctly:

1.  Navigate to the **backend** folder in this repository.
2.  If it doesn't exist, create a folder named `data`.
3.  Inside `data`, create format folder named `healthcare`.
4.  Copy your downloaded CSV/JSON files into `backend/data/healthcare/`.

**Expected file structure:**
```
agentds-platform/
└── backend/
    └── data/
        └── healthcare/
            ├── admissions_train.csv
            ├── admissions_test.csv
            ├── ed_cost_train.csv
            ├── ed_cost_test.csv
            ├── stays_train.csv
            ├── patients.csv
            └── ... (other files)
```

## 2. Installation & Running

1.  **Install Requirements**:
    ```bash
    pip install -r backend/requirements.txt
    ```

2.  **Start the Server**:
    From the root `agentds-platform` directory:
    ```bash
    # CMD / PowerShell
    uvicorn backend.main:app --reload
    ```
    The API will run at `http://localhost:8000`.

## 3. Training & Prediction (Using Local Data)

The system is configured to look in `backend/data/healthcare` first. If files are found, it uses them; otherwise, it attempts to download from Hugging Face.

### Problem Statement 1: Readmission Prediction
*   **Data Used**: `admissions_train.csv` (+ `patients.csv` if available)
*   **Target**: `readmit_30d` (Ensure this column exists in `admissions_train.csv`)

**Train:**
```bash
curl -X POST "http://localhost:8000/api/v1/healthcare/readmission-prediction/train"
```

**Predict:**
```bash
curl -X POST "http://localhost:8000/api/v1/healthcare/readmission-prediction/predict" \
-H "Content-Type: application/json" \
-d '{
  "records": [
    {"age": 65, "gender": "M", "admission_type": "EMERGENCY"} 
  ]
}'
```

### Problem Statement 2: ED Cost Forecasting
*   **Data Used**: `ed_cost_train.csv`
*   **Target**: `ed_cost_next3y_usd`

**Train:**
```bash
curl -X POST "http://localhost:8000/api/v1/healthcare/ed-cost-forecasting/train"
```

### Problem Statement 3: Discharge Readiness
*   **Data Used**: `stays_train.csv`
*   **Target**: `ready_for_discharge` (You might need to verify the column name in your CSV)

**Train:**
```bash
curl -X POST "http://localhost:8000/api/v1/healthcare/discharge-readiness/train"
```

## Troubleshooting
*   **"FileNotFoundError"**: Check that the `.csv` files are exactly in `backend/data/healthcare/` and match the names listed above.
*   **"Target column not found"**: Open your CSV and check the exact header name for the label. If it differs (e.g., `Readmitted` vs `readmit_30d`), you may need to update the `TARGET_COL` variable in `backend/healthcare/<ps>/model/train.py`.

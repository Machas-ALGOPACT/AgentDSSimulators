# 📋 UNIVERSAL CODING STANDARDS & GUIDELINES

**For all problem statements and projects in this repository**

Version: 1.0 | Last Updated: January 8, 2026

---

## 1. PROJECT STRUCTURE TEMPLATE

Every problem statement must follow this directory structure:

```
[project_name]/
├── config/
│   └── config.yaml                    # Centralized configuration (NO hardcoding)
├── data/
│   ├── raw/                           # Original input files (never modify)
│   ├── processed/                     # Cleaned/transformed data
│   └── features/                      # Feature matrices
├── models/                            # Trained artifacts (joblib serialization)
├── outputs/                           # Predictions, submissions, results
├── notebooks/                         # EDA & exploratory analysis (ipynb)
├── src/                               # Production code (CRITICAL STRUCTURE)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                  # Data loading from files
│   │   ├── validator.py               # Data validation logic
│   │   └── schema.py                  # Data schema definitions
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaning.py                # Data cleaning/standardization
│   │   ├── merging.py                 # Data joining/merging
│   │   ├── pipeline.py                # Preprocessing orchestration
│   │   └── transformers.py            # Fit/transform classes
│   ├── features/
│   │   ├── __init__.py
│   │   ├── [feature_type_1].py        # e.g., text_features.py
│   │   ├── [feature_type_2].py        # e.g., temporal_features.py
│   │   ├── [feature_type_3].py        # e.g., interaction_features.py
│   │   └── engineering.py             # Feature orchestration
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline.py                # Simple baseline models
│   │   ├── advanced.py                # Complex models (ensemble, boosting)
│   │   └── training.py                # Training orchestration
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Metric calculations
│   │   ├── analysis.py                # Error analysis, diagnostics
│   │   └── submission.py              # Output/submission generation
│   └── utils/
│       ├── __init__.py
│       ├── config.py                  # YAML config loader
│       ├── logger.py                  # Logging setup
│       ├── persistence.py             # Model save/load (joblib)
│       └── helpers.py                 # Common utilities
├── tests/
│   ├── __init__.py
│   ├── test_data_loading.py           # Data I/O tests
│   ├── test_preprocessing.py          # Pipeline tests
│   ├── test_features.py               # Feature engineering tests
│   ├── test_models.py                 # Model training tests
│   └── test_inference.py              # End-to-end prediction tests
├── main.py                            # Full orchestration script
├── router.py                          # API endpoint (if applicable)
├── requirements.txt                   # Python dependencies (pinned versions)
├── .gitignore                         # Exclude: models/, outputs/, __pycache__/
├── README.md                          # Quick overview (1-2 pages)
├── QUICKSTART.md                      # How to run in 5 minutes
└── [PROJECT_NAME]_SUMMARY.md          # Comprehensive reference
```

**RULES**:
- Use this structure for ALL projects
- Never deviate without explicit justification
- One logical component per file (max ~300 lines)
- All imports use relative paths within src/

---

## 2. CONFIGURATION MANAGEMENT

### Rule: NO HARDCODING

Every configurable value must be in `config/config.yaml`.

#### Mandatory config.yaml structure:

```yaml
# Data paths
data:
  raw_data_dir: data/raw
  processed_data_dir: data/processed
  features_dir: data/features
  # List all input files explicitly
  input_files:
    - file1.csv
    - file2.csv

# Feature engineering (document ALL features you plan to create)
features:
  # Columns requiring scaling
  scale_columns: [col1, col2, col3]
  
  # Feature engineering flags
  text_features: true/false
  temporal_features: true/false
  interaction_features: true/false
  aggregate_features: true/false
  
  # Feature dimensions
  expected_features_count: [number]

# Model hyperparameters
models:
  model_name_1:
    type: algorithm_type
    hyperparameters:
      param1: value1
      param2: value2
  model_name_2:
    type: algorithm_type
    hyperparameters: {}

# Paths for output
paths:
  models_dir: models
  outputs_dir: outputs
  logs_dir: outputs/logs

# Data split & validation
validation:
  train_test_split: 0.8  # or cross-validation params
  stratified: true/false
  random_state: 42

# Logging
logging:
  level: INFO  # DEBUG, INFO, WARNING, ERROR
  file: outputs/logs/pipeline.log
```

#### Implementation Pattern:

```python
# src/utils/config.py
import yaml

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Usage in any module:
from src.utils.config import load_config
config = load_config()
raw_data_dir = config['data']['raw_data_dir']
```

**RULES**:
- All paths must be relative to project root
- All hyperparameters must be in config
- Never use `config['key']['key']['key']` without verification (use .get() with defaults)
- Document expected values in config comments

---

## 3. CODE ORGANIZATION & MODULARITY

### Principle: SINGLE RESPONSIBILITY

Each module/class must have ONE clear responsibility.

#### Module Responsibilities:

| Module | Responsibility | Output |
|--------|-----------------|--------|
| `data/loader.py` | Load files from disk → DataFrame | Raw data (unmodified) |
| `data/validator.py` | Check schema, duplicates, ranges | Validation report, warnings |
| `preprocessing/cleaning.py` | Standardize names, impute missing | Clean DataFrame |
| `preprocessing/merging.py` | Join tables, handle mismatches | Merged DataFrame |
| `preprocessing/pipeline.py` | Orchestrate all transforms | Fitted transformer object |
| `features/*.py` | Engineer specific feature types | Feature matrix/DataFrame |
| `models/baseline.py` | Train simple models | Baseline model object |
| `models/advanced.py` | Train complex models | Trained model object |
| `evaluation/metrics.py` | Calculate performance metrics | Metric dictionary |
| `evaluation/analysis.py` | Analyze errors, produce diagnostics | Analysis report |

### Class & Function Patterns:

#### Pattern 1: Transformer Class (for reusable transforms)
```python
class MyTransformer:
    """Transform data in fit/transform pattern."""
    
    def fit(self, X):
        """Learn from training data."""
        # Store learned parameters
        self.learned_param_ = ...
        return self
    
    def transform(self, X):
        """Apply learned transformation."""
        # Use self.learned_param_
        return X_transformed
    
    def fit_transform(self, X):
        """Convenience method."""
        return self.fit(X).transform(X)
    
    def save(self, path):
        """Persist to disk."""
        import joblib
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path):
        """Load from disk."""
        import joblib
        return joblib.load(path)
```

#### Pattern 2: Service Class (for orchestration)
```python
class DataPipeline:
    """Orchestrate data processing steps."""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__)
    
    def run(self, X):
        """Execute full pipeline."""
        self.logger.info(f"Input shape: {X.shape}")
        
        X = self.clean(X)
        X = self.engineer_features(X)
        X = self.scale(X)
        
        self.logger.info(f"Output shape: {X.shape}")
        return X
    
    def clean(self, X):
        """Step 1."""
        # Implementation
        return X
    
    def engineer_features(self, X):
        """Step 2."""
        # Implementation
        return X
```

#### Pattern 3: Utility Functions (no state)
```python
def calculate_metric(y_true, y_pred):
    """Pure function - no side effects."""
    metric_value = ...
    return metric_value

def get_numeric_columns(df):
    """Pure function - returns list."""
    return [col for col in df.columns if is_numeric(df[col])]
```

**RULES**:
- Transformers use `fit/transform` pattern
- Services orchestrate multiple transformers
- Utilities are pure functions (no state)
- Use `self.logger` in classes for logging
- Use `*_` suffix for learned parameters (e.g., `learned_param_`)

---

## 4. DATA PIPELINE PRINCIPLES

### Mandatory Pipeline Steps

Every project's `main.py` must follow this sequence:

```python
# 1. Setup
config = load_config()
logger = setup_logger(__name__)
logger.info("=" * 50)
logger.info("STARTING PIPELINE")
logger.info("=" * 50)

# 2. Load
logger.info("Step 1: Loading data...")
loader = DataLoader(config)
data = loader.load_all()
logger.info(f"  Loaded {len(data)} records")

# 3. Validate
logger.info("Step 2: Validating data...")
validator = DataValidator(config)
validator.run_all_checks(data)
logger.info("  Validation passed")

# 4. Clean
logger.info("Step 3: Cleaning data...")
cleaner = DataCleaner(config)
data = cleaner.run(data)
logger.info(f"  Cleaned: {len(data)} records")

# 5. Transform (if applicable)
logger.info("Step 4: Transforming data...")
# Merge tables, derive columns, etc.
logger.info(f"  Transformed: {data.shape}")

# 6. Engineer Features
logger.info("Step 5: Engineering features...")
features = FeatureEngineer(config)
X = features.run(data)
logger.info(f"  Features shape: {X.shape}")

# 7. Preprocess
logger.info("Step 6: Preprocessing...")
preprocessor = PreprocessingPipeline(config)
X = preprocessor.fit_transform(X)  # Only on training data!
logger.info(f"  Final shape: {X.shape}")

# 8. Split
logger.info("Step 7: Splitting train/test...")
X_train, X_test, y_train, y_test = split_data(X, y, config)
logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")

# 9. Train
logger.info("Step 8: Training model...")
model = train_model(X_train, y_train, config)
logger.info("  Model trained")

# 10. Evaluate
logger.info("Step 9: Evaluating model...")
metrics = evaluate_model(model, X_test, y_test)
logger.info(f"  Metrics: {metrics}")

# 11. Predict
logger.info("Step 10: Generating predictions...")
predictions = model.predict(X_test)
logger.info(f"  Predictions: {predictions.shape}")

# 12. Analyze
logger.info("Step 11: Analyzing results...")
analyzer = ErrorAnalyzer()
analysis = analyzer.run(y_test, predictions)
logger.info(f"  Errors: {analysis}")

# 13. Save & Output
logger.info("Step 12: Saving artifacts...")
save_model(model, config)
save_predictions(predictions, config)
logger.info("=" * 50)
logger.info("PIPELINE COMPLETE")
logger.info("=" * 50)
```

**RULES**:
- Number every step (helps with debugging)
- Log at every step entry/exit
- Log data shapes/counts at transformations
- Never modify raw data (keep original)
- Only fit on training data
- Test set must be held completely separate

---

## 5. LOGGING STANDARDS

### Mandatory Logging Setup

```python
# src/utils/logger.py
import logging
import sys

def setup_logger(name, log_file=None, level=logging.INFO):
    """Create a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    
    # File handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    ch.setFormatter(formatter)
    if log_file:
        fh.setFormatter(formatter)
    
    logger.addHandler(ch)
    return logger
```

### Logging Requirements

**MUST LOG**:
- Entry/exit of each major step
- Data counts/shapes at transformations
- Missing value counts and imputation details
- Model hyperparameters before training
- Performance metrics after evaluation
- Errors and warnings with context

**PATTERN**:
```python
logger.info(f"Loading data from {file_path}")
logger.info(f"  Records loaded: {len(df)}")
logger.info(f"  Columns: {df.shape[1]}")

logger.warning(f"Column '{col}' has {missing_count} missing values")
logger.info(f"  Imputing with: {impute_method}")

logger.debug(f"Feature engineering complete. Output shape: {X.shape}")

logger.error(f"Validation failed: {error_msg}")
```

**RULES**:
- Use `logger.info()` for major steps
- Use `logger.warning()` for issues that don't stop execution
- Use `logger.error()` for failures
- Use `logger.debug()` for detailed diagnostics
- Every module uses: `logger = setup_logger(__name__)`

---

## 6. FEATURE ENGINEERING STANDARDS

### Document ALL Features

Before running feature engineering, create this in your README:

```markdown
## Feature Engineering Plan

### Input Features: X (from data preprocessing)
- Column1: description
- Column2: description

### Output Features: X_engineered
1. **Text Features** (method: TF-IDF)
   - Count: 500 features
   - Columns: tfidf_word1, tfidf_word2, ...

2. **Temporal Features** (extracted from dates)
   - ClaimDate_Month
   - ClaimDate_Hour
   - ClaimDate_DayOfWeek
   - PolicyAge_Days
   - Count: 4 features

3. **Interaction Features** (derived relationships)
   - DamagePerParty = ReportedDamage / NumParties
   - Count: 1 feature

4. **Aggregate Features** (group-level statistics)
   - Policy_ClaimCount (per policy)
   - Policy_AvgDamage (per policy)
   - Count: 2 features

5. **One-Hot Encoded** (categorical)
   - VehicleType: 5 categories → 5 columns
   - ClaimType: 3 categories → 3 columns
   - Count: 8 features

### Total Feature Count: 520 features
```

### Feature Engineering Rules

**MUST FOLLOW**:
1. Create separate files for each feature type
2. Document expected feature count BEFORE execution
3. Log feature creation at each step
4. Verify final count matches expectation
5. Never drop features silently
6. Preserve feature names (no special characters that confuse models)
7. Document feature importance after training

**PATTERN**:
```python
# src/features/text_features.py
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_text_features(df, config):
    """Extract TF-IDF features from text column."""
    vectorizer = TfidfVectorizer(
        max_features=config['features']['tfidf_max_features'],
        min_df=2,
        max_df=0.95
    )
    tfidf_matrix = vectorizer.fit_transform(df['text_column'])
    
    # Sanitize feature names for model compatibility
    feature_names = [
        f"tfidf_{name.replace(' ', '_').replace('-', '_')}"
        for name in vectorizer.get_feature_names_out()
    ]
    
    logger.info(f"Extracted {tfidf_matrix.shape[1]} TF-IDF features")
    
    return tfidf_matrix, feature_names
```

---

## 7. PREPROCESSING PIPELINE STANDARDS

### CRITICAL: Fit ONLY on Training Data

```python
# src/preprocessing/pipeline.py
class PreprocessingPipeline:
    """Bundle all transformations for reproducibility."""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__)
        self.scaler = None
        self.tfidf_vectorizer = None
        # Store all learned parameters
    
    def fit(self, X_train):
        """Learn transformations from training data ONLY."""
        self.logger.info(f"Fitting pipeline on training data: {X_train.shape}")
        
        # Learn scaling from training data
        self.scaler = StandardScaler()
        self.scaler.fit(X_train[self.config['features']['scale_columns']])
        
        self.logger.info(f"Pipeline fitted")
        return self
    
    def transform(self, X):
        """Apply learned transformations."""
        if self.scaler is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        X_transformed = X.copy()
        
        # Apply learned scaling
        scale_cols = self.config['features']['scale_columns']
        X_transformed[scale_cols] = self.scaler.transform(X[scale_cols])
        
        return X_transformed
    
    def fit_transform(self, X_train):
        """Fit and transform training data."""
        return self.fit(X_train).transform(X_train)
    
    def save(self, path):
        """Persist pipeline to disk."""
        import joblib
        joblib.dump(self, path)
        self.logger.info(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path):
        """Load pipeline from disk."""
        import joblib
        pipeline = joblib.load(path)
        return pipeline
```

**USAGE IN main.py**:
```python
# FIT ONLY ON TRAINING DATA
logger.info("Fitting preprocessing pipeline...")
pipeline = PreprocessingPipeline(config)
X_train = pipeline.fit_transform(X_train)  # FIT & TRANSFORM
logger.info(f"Training data shape: {X_train.shape}")

# TRANSFORM TEST DATA ONLY (don't fit)
X_test = pipeline.transform(X_test)  # TRANSFORM ONLY
logger.info(f"Test data shape: {X_test.shape}")

# SAVE FOR PRODUCTION
pipeline.save('models/preprocessing_pipeline.joblib')
```

**RULES**:
- `fit()` learns parameters from training data
- `transform()` applies learned parameters
- `fit_transform()` combines both (training data only)
- Never fit on test data
- Always save fitted pipeline
- Load and use same pipeline for inference

---

## 8. MODEL DEVELOPMENT STANDARDS

### Baseline Model (Required)

Before implementing complex models, create a simple baseline:

```python
# src/models/baseline.py
def train_baseline(X_train, y_train, config):
    """Simple baseline for comparison."""
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression(random_state=config['validation']['random_state'])
    model.fit(X_train, y_train)
    
    logger.info("Baseline model trained")
    return model
```

### Ensemble Models (Recommended)

Use multiple diverse base learners:

```python
# src/models/advanced.py
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import xgboost as xgb

def train_ensemble(X_train, y_train, config):
    """Train ensemble of diverse models."""
    
    # Base learner 1: Random Forest
    rf = RandomForestClassifier(
        n_estimators=config['models']['random_forest']['hyperparameters']['n_estimators'],
        max_depth=config['models']['random_forest']['hyperparameters']['max_depth'],
        random_state=config['validation']['random_state']
    )
    
    # Base learner 2: XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=config['models']['xgboost']['hyperparameters']['n_estimators'],
        max_depth=config['models']['xgboost']['hyperparameters']['max_depth'],
        learning_rate=config['models']['xgboost']['hyperparameters']['learning_rate'],
        random_state=config['validation']['random_state']
    )
    
    # Combine with soft voting
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_model)],
        voting='soft'  # Probability averaging
    )
    
    ensemble.fit(X_train, y_train)
    logger.info("Ensemble model trained")
    
    return ensemble
```

**RULES**:
- Always train a baseline first
- Use 2+ diverse base learners in ensembles
- Use soft voting (probability averaging)
- Document hyperparameters in config.yaml
- Save all trained models via joblib

---

## 9. EVALUATION STANDARDS

### Required Metrics

**For Classification**:
```python
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

def evaluate_classification(y_true, y_pred):
    """Calculate all classification metrics."""
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
    }
    
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    
    return metrics
```

**For Regression**:
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_regression(y_true, y_pred):
    """Calculate all regression metrics."""
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
    }
    
    logger.info(f"MAE: {metrics['mae']:.4f}")
    logger.info(f"RMSE: {metrics['rmse']:.4f}")
    logger.info(f"R²: {metrics['r2']:.4f}")
    
    return metrics
```

### Error Analysis (Required)

```python
# src/evaluation/analysis.py
class ErrorAnalyzer:
    """Analyze model errors and produce diagnostics."""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(__name__)
    
    def run(self, y_true, y_pred):
        """Produce error analysis report."""
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.logger.info(f"Confusion Matrix:\n{cm}")
        
        # Per-class metrics
        report = classification_report(y_true, y_pred)
        self.logger.info(f"Classification Report:\n{report}")
        
        # Error samples
        errors = y_true != y_pred
        error_indices = np.where(errors)[0]
        self.logger.info(f"Total errors: {len(error_indices)} / {len(y_true)}")
        
        return {
            'confusion_matrix': cm,
            'error_indices': error_indices,
            'error_count': len(error_indices),
        }
```

**RULES**:
- Calculate metrics on validation/test set ONLY
- Log all metrics with 4 decimal precision
- Produce confusion matrix visualization
- Identify and log misclassified samples
- Compare against baseline performance

---

## 10. TESTING STANDARDS

### Mandatory Test Coverage

Create `tests/` with these files:

```python
# tests/test_data_loading.py
import pytest
from src.data.loader import DataLoader

def test_loader_loads_file():
    """DataLoader must load CSV files."""
    loader = DataLoader({'data': {'raw_data_dir': 'data/raw'}})
    df = loader.load_csv('test_data.csv')
    assert len(df) > 0

def test_loader_returns_dataframe():
    """DataLoader.load_csv() must return DataFrame."""
    loader = DataLoader({'data': {'raw_data_dir': 'data/raw'}})
    df = loader.load_csv('test_data.csv')
    assert isinstance(df, pd.DataFrame)
```

```python
# tests/test_preprocessing.py
def test_pipeline_fit_transform():
    """Pipeline fit/transform must work."""
    X_train = np.random.rand(100, 10)
    pipeline = PreprocessingPipeline({})
    X_transformed = pipeline.fit_transform(X_train)
    assert X_transformed.shape[0] == X_train.shape[0]

def test_pipeline_serialization():
    """Pipeline must save and load."""
    X_train = np.random.rand(100, 10)
    pipeline = PreprocessingPipeline({})
    pipeline.fit(X_train)
    
    pipeline.save('/tmp/test_pipeline.joblib')
    loaded_pipeline = PreprocessingPipeline.load('/tmp/test_pipeline.joblib')
    
    X1 = pipeline.transform(X_train)
    X2 = loaded_pipeline.transform(X_train)
    
    assert np.allclose(X1, X2)
```

```python
# tests/test_features.py
def test_feature_count():
    """Feature engineering must produce expected count."""
    X = np.random.rand(100, 20)
    features = FeatureEngineer(config)
    X_engineered = features.run(X)
    
    assert X_engineered.shape[1] == 520  # Expected count from plan
```

```python
# tests/test_models.py
def test_model_training():
    """Model must train without errors."""
    X = np.random.rand(100, 50)
    y = np.random.randint(0, 3, 100)
    
    model = train_model(X, y, config)
    assert model is not None

def test_model_reproducibility():
    """Same random_state must produce identical predictions."""
    X = np.random.rand(100, 50)
    y = np.random.randint(0, 3, 100)
    
    model1 = train_model(X, y, config)
    model2 = train_model(X, y, config)
    
    pred1 = model1.predict(X)
    pred2 = model2.predict(X)
    
    assert np.array_equal(pred1, pred2)
```

**RULES**:
- Run tests before every commit: `pytest tests/`
- Maintain >80% passing rate
- Test fit/transform pattern
- Test serialization round-trips
- Test reproducibility (with random_state)
- Use temporary files for I/O tests

---

## 11. MODEL PERSISTENCE & PRODUCTION

### Serialization Standards

```python
# src/utils/persistence.py
import joblib

def save_model(model, path):
    """Save model to disk."""
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")

def load_model(path):
    """Load model from disk."""
    model = joblib.load(path)
    logger.info(f"Model loaded from {path}")
    return model

def save_all_artifacts(model, pipeline, encoder, config):
    """Save all models and transformers."""
    models_dir = config['paths']['models_dir']
    
    save_model(model, f"{models_dir}/model.joblib")
    save_model(pipeline, f"{models_dir}/preprocessing_pipeline.joblib")
    save_model(encoder, f"{models_dir}/label_encoder.joblib")
    
    logger.info("All artifacts saved")
```

**MUST PERSIST**:
- Trained model (*.joblib)
- Preprocessing pipeline (*.joblib)
- Label encoder (*.joblib)
- Feature names (saved in pipeline)
- Vectorizers (TF-IDF, Count, etc.)

**USAGE IN main.py**:
```python
# After training
save_all_artifacts(model, pipeline, encoder, config)
logger.info("Production artifacts saved to models/")
```

---

## 12. API ENDPOINT STANDARDS

If creating FastAPI/Flask endpoint:

```python
# router.py or api.py
from fastapi import APIRouter
from pydantic import BaseModel
from src.utils.persistence import load_model, load_pipeline

router = APIRouter()

# Define request schema
class PredictionRequest(BaseModel):
    field1: str
    field2: float
    field3: int
    # Document all required fields

# Define response schema
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict

# Load models at startup (not per request)
model = load_model('models/model.joblib')
pipeline = load_pipeline('models/preprocessing_pipeline.joblib')

@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Generate prediction from input."""
    
    try:
        # Convert input to DataFrame
        X = pd.DataFrame([request.dict()])
        
        # Apply preprocessing
        X = pipeline.transform(X)
        
        # Generate prediction
        pred = model.predict(X)
        prob = model.predict_proba(X)
        
        return PredictionResponse(
            prediction=pred[0],
            confidence=float(prob[0].max()),
            probabilities={
                "class_0": float(prob[0][0]),
                "class_1": float(prob[0][1]),
            }
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```

**RULES**:
- Use Pydantic models for request/response validation
- Load models at startup (singleton pattern)
- Handle errors gracefully with logging
- Return structured JSON responses
- Document all fields in docstrings

---

## 13. GIT WORKFLOW STANDARDS

### Commit Message Format

```
[type]: [description] - [brief reason]

Types:
- feat: New feature implementation
- fix: Bug fix or correction
- docs: Documentation changes
- refactor: Code restructuring (no logic change)
- test: Test additions or fixes
- chore: Maintenance, dependency updates

Examples:
feat: Implement preprocessing pipeline - handle dtype detection correctly
fix: Correct StandardScaler column detection using pd.api.types
docs: Add SUMMARY.md - comprehensive project reference for team
test: Add preprocessing pipeline serialization tests
refactor: Separate feature engineering into modular files
```

### Git Rules

**MUST FOLLOW**:
- Commit before major changes: `git status`
- Use meaningful commit messages (not "updates" or "fixes")
- One logical change per commit
- Never commit without passing tests: `pytest tests/`
- Never commit with uncommitted changes
- Always push to remote after local commits

**BEFORE COMMITTING**:
```bash
# Check status
git status

# Run all tests
pytest tests/

# Verify no hardcoding (grep for absolute paths)
grep -r "C:/" .

# Stage changes
git add [files]

# Commit with message
git commit -m "feat: Description - reasoning"

# Push to remote
git push origin [branch]
```

---

## 14. DOCUMENTATION STANDARDS

### Required Documentation Files

| File | Purpose | Owner | Audience |
|------|---------|-------|----------|
| **README.md** | Project overview | Data scientist | Everyone |
| **QUICKSTART.md** | Run in 5 minutes | Developer | Developers/agents |
| **[PROJECT]_SUMMARY.md** | Comprehensive reference | Lead data scientist | Team members |
| **config/config.yaml** | All parameters | Developer | Code |
| **main.py** | Numbered steps | Developer | Code |
| **Inline comments** | Complex logic | Developer | Code reviewers |

### README.md Template

```markdown
# Project Name

## Problem Statement
[1-2 sentence description of what the model solves]

## Data
- **Source**: [where data comes from]
- **Records**: [number of training/test records]
- **Features**: [number of input features]
- **Target**: [what are we predicting]

## Approach
[Brief description of data pipeline and model]

## Results
- **Metric 1**: X.XX
- **Metric 2**: X.XX
- **Baseline comparison**: [improvement over baseline]

## Files
- `main.py`: Full pipeline orchestration
- `config/config.yaml`: All parameters
- `src/data/`: Data loading & validation
- `src/preprocessing/`: Data cleaning & transformation
- `src/features/`: Feature engineering
- `src/models/`: Model training
- `src/evaluation/`: Metric calculation & analysis

## How to Run
```bash
python main.py
```

## Configuration
See `config/config.yaml` for all parameters.
```

### QUICKSTART.md Template

```markdown
# Quick Start

## Setup
```bash
pip install -r requirements.txt
```

## Run Full Pipeline
```bash
python main.py
```

## Expected Output
- `models/model.joblib`: Trained model
- `outputs/submission.csv`: Predictions
- `outputs/logs/pipeline.log`: Execution log

## Verify Results
Check `outputs/logs/pipeline.log` for metrics.
```

### [PROJECT]_SUMMARY.md Template

```markdown
# Project Summary

## Architecture
[Describe overall design]

## Data Pipeline
[Describe 12+ steps in main.py]

## Components
[List key classes and responsibilities]

## Feature Engineering
[Document all features created]

## Model
[Describe training approach]

## Results
[Metrics and comparison]

## Patterns Applied
[List design patterns used]
```

---

## 15. QUALITY GATES (MANDATORY CHECKS)

Before committing, verify ALL of these:

```
✅ Code
  ☐ No hardcoded paths (use config.yaml)
  ☐ No hardcoded parameters (use config.yaml)
  ☐ All modules use relative imports
  ☐ Every class/function has docstring
  ☐ Max 300 lines per file
  ☐ Consistent naming convention (snake_case)

✅ Functionality
  ☐ main.py runs without errors
  ☐ All data logged at each step
  ☐ Feature count verified vs. expected
  ☐ No silent record drops
  ☐ Preprocessing pipeline serialized
  ☐ Model saved to models/
  ☐ Predictions saved to outputs/

✅ Testing
  ☐ All tests pass: pytest tests/
  ☐ >80% test pass rate
  ☐ Serialization round-trip verified
  ☐ Reproducibility verified (random_state=42)

✅ Documentation
  ☐ README.md written
  ☐ QUICKSTART.md written
  ☐ [PROJECT]_SUMMARY.md written
  ☐ Inline comments for complex logic
  ☐ Docstrings for all classes/functions

✅ Logging
  ☐ Entry/exit logged for each step
  ☐ Data counts logged
  ☐ Errors logged with context
  ☐ Performance metrics logged

✅ Git
  ☐ git status shows clean working tree
  ☐ No uncommitted files
  ☐ Commit message follows format
  ☐ Commit explains WHY (not just WHAT)

✅ Configuration
  ☐ config.yaml contains all parameters
  ☐ No magic numbers in code
  ☐ All paths use config values
  ☐ Expected feature count documented
```

---

## 16. QUICK REFERENCE CHECKLIST

**For each new project, use this checklist**:

- [ ] Create directory structure from template
- [ ] Create config.yaml with all parameters
- [ ] Create src/ subdirectories (data, preprocessing, features, models, evaluation, utils)
- [ ] Implement src/utils/ (config.py, logger.py, persistence.py)
- [ ] Create main.py with 13-step skeleton
- [ ] Create tests/ with 4 test files
- [ ] Implement data loading (src/data/)
- [ ] Implement data cleaning (src/preprocessing/)
- [ ] Implement feature engineering (src/features/ - separate files)
- [ ] Implement preprocessing pipeline (src/preprocessing/pipeline.py)
- [ ] Implement models (src/models/)
- [ ] Implement evaluation (src/evaluation/)
- [ ] Run main.py and verify metrics
- [ ] Run tests and achieve >80% pass rate
- [ ] Create README.md, QUICKSTART.md, [PROJECT]_SUMMARY.md
- [ ] Create router.py (if API needed)
- [ ] Verify all quality gates pass
- [ ] Commit with meaningful message
- [ ] Push to GitHub

---

## 17. COMMON PITFALLS TO AVOID

| Pitfall | Impact | Solution |
|---------|--------|----------|
| Hardcoded paths | Breaks on different systems | Use config.yaml |
| Fitting on test data | Overly optimistic metrics | Only fit transformers on train |
| Silent data loss | Wrong counts, missing records | Log counts at each step |
| No logging | Can't debug failures | Log entry/exit every step |
| No serialization | Can't reproduce in production | Save all transformers |
| Wrong dtype detection | Feature collapse | Use `pd.api.types.is_numeric_dtype()` |
| Uncommitted changes | Lost work | Commit before major changes |
| No tests | Breaks silently | Run `pytest tests/` before commit |
| Hardcoded features | Not generalizable | Document in config |
| Single learner | Poor generalization | Use ensembles |

---

## 18. SUPPORT & ESCALATION

If you encounter issues:

1. **Check the logs**: `cat outputs/logs/pipeline.log`
2. **Re-run with logging**: Check main.py for step-by-step output
3. **Run tests**: `pytest tests/ -v`
4. **Verify config**: Ensure config.yaml has all required fields
5. **Check git status**: `git status` and `git diff`
6. **Review this document**: Most answers are here

---

## VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 8, 2026 | Initial release |

---

**Last Updated**: January 8, 2026  
**Applies To**: ALL projects in this repository  
**For**: Any coding agent, developer, or team member

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.utils.config import load_config
from src.utils.logger import setup_logger, get_default_log_path
from src.data.loader import DataLoader
from src.data.validator import DataValidator
from src.preprocessing.merging import DataMerger
from src.preprocessing.cleaning import DataCleaner
from src.models.baseline import BaselineModel

def main():
    # 1. Load Configuration
    config = load_config()
    
    # 2. Setup Logging
    log_path = get_default_log_path(config)
    logger = setup_logger("MainPipeline", log_file=log_path)
    logger.info("Starting Auto Insurance Claims Complexity Prediction Pipeline")
    
    # 3. Load Data
    loader = DataLoader(config)
    data = loader.load_all_data()
    
    train_claims = data['train_claims']
    train_policies = data['train_policies']
    
    if train_claims is None:
        logger.error("Train claims data not found. Exiting.")
        return

    # 4. Validate Data
    validator = DataValidator(config)
    if not validator.run_all_checks(train_claims, train_policies):
        logger.warning("Data validation failed or found issues. Proceeding with caution.")
    
    # 5. Merge Data
    merger = DataMerger(config)
    df = merger.merge_claims_policies(train_claims, train_policies)
    
    # 6. Clean Data
    cleaner = DataCleaner(config)
    df = cleaner.handle_missing_values(df)
    
    # 7. Feature Engineering
    logger.info("Starting Feature Engineering...")
    
    # 7.1 Tabular Features
    from src.features.engineering import FeatureEngineer
    fe = FeatureEngineer(config)
    df = fe.create_temporal_features(df, 'ClaimDate')
    df = fe.create_interaction_features(df)
    
    # 7.2 Text Features
    from src.features.text_features import TextFeatureEngineer
    tfe = TextFeatureEngineer(config)
    df = tfe.extract_basic_text_features(df, 'Description')
    df, vectorizer = tfe.fit_transform_tfidf(df, 'Description')
    
    # 7.3 Aggregate Features
    from src.features.aggregation import AggregateFeatureEngineer
    afe = AggregateFeatureEngineer(config)
    # Calculate aggregates from the original training claims (before split if possible, or just on current df)
    # Ideally should be calculated on historical data, but here using current df
    agg_df = afe.create_policy_aggregates(train_claims) # Use original claims data for aggregation source
    df = afe.merge_aggregates(df, agg_df)
    
    # 7.4 Encode Categorical
    df = fe.encode_categorical_features(df)
    
    # 8. Feature Selection for Modeling
    target_col = config['data']['target_col']
    
    # Drop non-numeric and target columns for X
    drop_cols = [target_col, config['data']['id_col'], config['data']['join_col'], 'ClaimDate', 'Description', 'ClaimType', 'VehicleType', 'PolicyStart', 'PolicyEnd']
    # Also drop original categorical columns that were encoded (if any remained) or just select numeric
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=[np.number]) # Final safety check
    
    logger.info(f"Final feature set shape: {X.shape}")
    logger.info(f"Features: {X.columns.tolist()[:10]} ...")
    
    y = df[target_col]
    
    # Encode Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    logger.info(f"Target classes: {le.classes_}")
    
    # 8. Train/Test Split
    try:
        if len(np.unique(y_encoded)) > 1:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=0.2, random_state=config['random_seed'], stratify=y_encoded
            )
        else:
             # Fallback for single class
             X_train, X_val, y_train, y_val = X, X, y_encoded, y_encoded
    except ValueError:
        logger.warning("Stratified split failed (likely too few samples per class). Falling back to random split.")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, test_size=0.2, random_state=config['random_seed']
        )
    
    # ... (Feature selection block remains) ...
    
    # 9. Model Selection & Training
    model_type = config.get('active_model', 'baseline') # 'baseline', 'xgboost', 'lightgbm'
    run_tuning = config.get('run_tuning', False)
    
    if run_tuning and model_type in ['xgboost', 'lightgbm']:
        from src.models.tuning import HyperparameterTuner
        tuner = HyperparameterTuner(config, X_train, y_train)
        best_params = tuner.tune(model_type, n_trials=10)
        # Update config with best params (in memory)
        config['model']['advanced'][model_type].update(best_params)
    
    if model_type == 'baseline':
        from src.models.baseline import BaselineModel
        model = BaselineModel(config)
    elif model_type in ['xgboost', 'lightgbm']:
        from src.models.advanced import AdvancedModel
        model = AdvancedModel(config, model_type=model_type)
    elif model_type == 'ensemble':
        from src.models.ensemble import EnsembleModel
        model = EnsembleModel(config)
    else:
        logger.error(f"Unknown model type: {model_type}")
        return

    model.train(X_train, y_train, X_val, y_val)
    
    # 10. Evaluation
    results = model.evaluate(X_val, y_val)
    
    # 11. Error Analysis
    from src.evaluation.analysis import ErrorAnalyzer
    analyzer = ErrorAnalyzer(config)
    try:
        if hasattr(model, 'model'): # Unwrap for some custom classes if needed, or DuckTyping
            # Our wrappers usually expose predict, but getting feature importance might require the underlying model
            pass
        
        analyzer.analyze_errors(model, X_val, y_val, le.classes_)
        # Feature importance might not work for Ensemble or generic wrappers without extra logic
        if model_type not in ['ensemble']: 
             # Try to get underlying model for feature importance
             underlying = model.model if hasattr(model, 'model') else model
             analyzer.plot_feature_importance(underlying, X.columns)
    except Exception as e:
        logger.warning(f"Error analysis (optional) failed: {e}")

    # Save Model
    model.save(f"{model_type}_model.joblib")
    
    # 12. Inference on Test Set (if available)
    test_claims = data['test_claims']
    test_policies = data['test_policies']
    
    if test_claims is not None:
        logger.info("Processing Test Set for Submission...")
        
        # Merge
        df_test = merger.merge_claims_policies(test_claims, test_policies)
        # Clean
        df_test = cleaner.handle_missing_values(df_test)
        
        # Feature Engineering (Must reuse the same objects if they hold state, but here they are mostly stateless approx)
        # Ideally, we should fit transformers on train and transform test. 
        # Using the same `fe`, `tfe` instances etc.
        # Tabular
        df_test = fe.create_temporal_features(df_test, 'ClaimDate')
        df_test = fe.create_interaction_features(df_test)
        
        # Text - Use the SAME vectorizer fitted on train
        df_test = tfe.extract_basic_text_features(df_test, 'Description')
        df_test = tfe.transform_tfidf(df_test, 'Description', vectorizer=vectorizer)
        
        # Aggregates - Map from training aggregates or re-calculate if appropriate?
        # Usually we map from training knowledge. 
        # Here `agg_df` was computed on `train_claims`. We merge it into test.
        df_test = afe.merge_aggregates(df_test, agg_df)
        
        # Encode
        df_test = fe.encode_categorical_features(df_test)
        
        # Align Features (Ensure test has same columns as train X)
        missing_cols = set(X.columns) - set(df_test.columns)
        for c in missing_cols:
            df_test[c] = 0
            
        # Select features
        X_test_final = df_test[X.columns]
        
        # Generate Submission
        from src.evaluation.submission import generate_submission
        test_ids = test_claims[config['data']['id_col']]
        generate_submission(model, X_test_final, test_ids, le.classes_, output_dir=config['paths']['outputs'])
        
    logger.info("Pipeline execution finished successfully.")

if __name__ == "__main__":
    main()

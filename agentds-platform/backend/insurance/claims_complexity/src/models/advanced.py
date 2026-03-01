from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report
import os
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class AdvancedModel:
    def __init__(self, config, model_type='xgboost'):
        self.config = config
        self.model_type = model_type
        self.model_params = config['model']['advanced'].get(model_type, {})
        
        if model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                max_depth=self.model_params.get('max_depth', 6),
                random_state=config['random_seed'],
                n_jobs=-1
            )
        elif model_type == 'lightgbm':
            self.model = LGBMClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                num_leaves=self.model_params.get('num_leaves', 31),
                random_state=config['random_seed'],
                n_jobs=-1,
                verbose=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
    def train(self, X_train, y_train, X_val=None, y_val=None):
        logger.info(f"Training {self.model_type} with params: {self.model_params}")
        
        # XGBoost and LightGBM support early stopping if eval_set is provided
        fit_params = {}
        if X_val is not None and y_val is not None:
             # Scikit-learn API changes for early stopping in recent versions
             # For simplicity, we'll just fit normally for now, or add eval_set if needed
             # self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)
             pass
        
        self.model.fit(X_train, y_train)
        logger.info("Model training complete.")
        
    def predict(self, X):
        return self.model.predict(X)
        
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        logger.info(f"{self.model_type} Evaluation - Macro-F1: {macro_f1:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        return {
            'macro_f1': macro_f1,
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
        
    def save(self, name=None):
        if name is None:
            name = f"{self.model_type}_model.joblib"
        path = os.path.join(self.config['paths']['models'], name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
        return path

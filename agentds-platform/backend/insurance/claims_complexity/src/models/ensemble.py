from sklearn.ensemble import VotingClassifier
from src.models.baseline import BaselineModel
from src.models.advanced import AdvancedModel
from src.utils.logger import setup_logger
import os
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib

logger = setup_logger(__name__)

class EnsembleModel:
    def __init__(self, config):
        self.config = config
        self.random_seed = config['random_seed']
        
        # Initialize base models
        self.rf = BaselineModel(config).model
        self.xgb = AdvancedModel(config, 'xgboost').model
        self.lgbm = AdvancedModel(config, 'lightgbm').model
        
        self.model = VotingClassifier(
            estimators=[
                ('rf', self.rf),
                ('xgb', self.xgb),
                ('lgbm', self.lgbm)
            ],
            voting='soft',
            n_jobs=-1
        )
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        logger.info("Training Ensemble Model (VotingClassifier: RF, XGB, LGBM)...")
        self.model.fit(X_train, y_train)
        logger.info("Ensemble training complete.")
        
    def predict(self, X):
        return self.model.predict(X)
        
    def evaluate(self, X_test, y_test):
        from sklearn.metrics import f1_score, classification_report
        y_pred = self.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        logger.info(f"Ensemble Evaluation - Macro-F1: {macro_f1:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        return {
            'macro_f1': macro_f1,
            'report': classification_report(y_test, y_pred, output_dict=True)
        }

    def save(self, name="ensemble_model.joblib"):
        path = os.path.join(self.config['paths']['models'], name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Ensemble model saved to {path}")
        return path

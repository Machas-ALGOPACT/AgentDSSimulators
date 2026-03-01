from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
try:
    import joblib
except ImportError:
    from sklearn.externals import joblib
import os
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class BaselineModel:
    def __init__(self, config):
        self.config = config
        self.model_params = config['model']['baseline']
        self.model = RandomForestClassifier(
            n_estimators=self.model_params['n_estimators'],
            max_depth=self.model_params['max_depth'],
            class_weight=self.model_params['class_weight'],
            random_state=config['random_seed']
        )
        
    def train(self, X_train, y_train):
        """
        Train the Random Forest baseline model.
        """
        logger.info(f"Training Baseline Random Forest with params: {self.model_params}")
        self.model.fit(X_train, y_train)
        logger.info("Model training complete.")
        
    def predict(self, X):
        """
        Generate predictions.
        """
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using Macro-F1 and other metrics.
        """
        y_pred = self.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        logger.info(f"Evaluation Results - Macro-F1: {macro_f1:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred))
        
        return {
            'macro_f1': macro_f1,
            'report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
    def save(self, name="baseline_rf.joblib"):
        """
        Save the model artifact.
        """
        path = os.path.join(self.config['paths']['models'], name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
        return path

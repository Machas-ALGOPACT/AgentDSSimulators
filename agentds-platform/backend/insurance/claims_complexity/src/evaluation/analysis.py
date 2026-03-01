import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ErrorAnalyzer:
    def __init__(self, config):
        self.config = config
        
    def analyze_errors(self, model, X_test, y_test, le_classes):
        """
        Analyze misclassifications.
        """
        logger.info("Starting Error Analysis...")
        y_pred = model.predict(X_test)
        
        # Create a dataframe of errors
        results = X_test.copy()
        results['Actual'] = y_test
        results['Predicted'] = y_pred
        results['Actual_Label'] = [le_classes[i] for i in y_test]
        results['Predicted_Label'] = [le_classes[i] for i in y_pred]
        
        errors = results[results['Actual'] != results['Predicted']]
        logger.info(f"Total Errors: {len(errors)} out of {len(y_test)} samples")
        
        if not errors.empty:
            logger.info("Error Sample:\n" + str(errors.head()))
            
            # Confusion Matrix Analysis
            confusion = pd.crosstab(results['Actual_Label'], results['Predicted_Label'])
            logger.info("Confusion Matrix:\n" + str(confusion))
            
        return errors

    def plot_feature_importance(self, model, feature_names, top_n=20):
        """
        Plot feature importance/coefficients.
        """
        logger.info("Generating feature importance report...")
        importance = None
        
        # Handle different model types
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
            
        if importance is not None:
            fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            fi_df = fi_df.sort_values(by='Importance', ascending=False).head(top_n)
            
            logger.info("Top Feature Importances:\n" + str(fi_df))
            return fi_df
        else:
            logger.warning("Model does not provide feature importances.")
            return None

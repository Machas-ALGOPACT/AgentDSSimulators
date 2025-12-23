import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, roc_auc_score, accuracy_score
from typing import Dict

def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    """
    Calculates MAE and RMSE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mae),
        "rmse": float(rmse)
    }

def evaluate_classification(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    """
    Calculates Classification metrics.
    """
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    metrics = {
        "macro_f1": float(f1),
        "accuracy": float(acc)
    }
    
    if y_prob is not None:
        try:
            # Handle binary vs multi-class AUC implicitly if possible, or assume binary for now as per coupon task
            if len(np.unique(y_true)) == 2:
                auc = roc_auc_score(y_true, y_prob)
                metrics["auc"] = float(auc)
        except Exception:
            pass
            
    return metrics

from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_classification(y_true, y_pred, y_proba=None):
    metrics = {
        "macro_f1": f1_score(y_true, y_pred, average='macro'),
        "weighted_f1": f1_score(y_true, y_pred, average='weighted'),
    }
    return metrics

def evaluate_regression(y_true, y_pred):
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred)
    }
    return metrics

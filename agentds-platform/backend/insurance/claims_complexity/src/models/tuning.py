import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class HyperparameterTuner:
    def __init__(self, config, X, y):
        self.config = config
        self.X = X
        self.y = y
        self.random_seed = config['random_seed']
        
    def objective_xgboost(self, trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': self.random_seed,
            'n_jobs': -1
        }
        
        clf = XGBClassifier(**param)
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_seed)
        scores = cross_val_score(clf, self.X, self.y, cv=cv, scoring='f1_macro')
        return scores.mean()

    def objective_lightgbm(self, trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'random_state': self.random_seed,
            'n_jobs': -1,
            'verbose': -1
        }
        
        clf = LGBMClassifier(**param)
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_seed)
        scores = cross_val_score(clf, self.X, self.y, cv=cv, scoring='f1_macro')
        return scores.mean()

    def tune(self, model_type='xgboost', n_trials=20):
        if len(self.y) < 20 or len(set(self.y)) < 2:
            logger.warning("Insufficient data for tuning. Returning default parameters.")
            return {}

        logger.info(f"Starting tuning for {model_type} with {n_trials} trials...")
        
        study = optuna.create_study(direction='maximize')
        
        try:
            if model_type == 'xgboost':
                study.optimize(self.objective_xgboost, n_trials=n_trials)
            elif model_type == 'lightgbm':
                study.optimize(self.objective_lightgbm, n_trials=n_trials)
            else:
                raise ValueError(f"Unknown model type for tuning: {model_type}")
        except Exception as e:
            logger.error(f"Tuning failed: {e}")
            return {}
            
        logger.info(f"Best trial: {study.best_trial.value}")
        logger.info(f"Best params: {study.best_trial.params}")
        
        return study.best_trial.params

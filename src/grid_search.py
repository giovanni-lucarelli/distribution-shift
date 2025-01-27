from sklearn.model_selection import GridSearchCV
#import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import product
import xgboost as xgb

best_params = {
    "RandomForestClassifier" : {
        "n_estimators": 125,
        "criterion": "gini",
        "max_depth": 6,
        #"min_samples_leaf": 1, [= default]
        "min_samples_split": 5,
        "random_state": 0
    },
    "GradientBoostingClassifier" : {
        "n_estimators": 500,
        "learning_rate": 0.005,
        "max_depth": 3,
        "subsample": 0.4,
        "random_state": 0
    },
    "XGBoost" : {
        #'n_estimators': 100, [default]
        'learning_rate': 0.1,
        'max_depth': 6,
        #'min_child_weight': 2,
        'subsample': 0.7,
        #'colsample_bytree': 1.0,
        #'reg_alpha': 1,
        #'reg_lambda': 1,
        'gamma': 5
    },
}

def grid_search_cv(estimator, param_grid, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1):

    # Initialize GridSearchCV
    grid_searcher = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose)
    # Perform the grid search
    grid_searcher.fit(X_train, y_train)
    # Get the best parameters and the best model
    best_model = grid_searcher.best_estimator_

    if verbose:
        print(f"Best parameters found: {grid_searcher.best_params_}")
    
    return best_model

def _evaluate_params(args):
    """Optimized parameter evaluation with early stopping (GPU enabled)."""
    params, X, y, nfold, early_stopping = args
    local_params = params.copy()
    # Set objective, eval metric, and GPU support
    local_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',   # Use GPU hist for faster training
        'device': 'cuda'        # Use GPU acceleration
    })
    # Default n_estimators to 100 if not present
    num_boost_round = local_params.pop('n_estimators', 100)
    dtrain = xgb.DMatrix(X, label=y)
    cv_results = xgb.cv(
        params=local_params,
        dtrain=dtrain,
        nfold=nfold,
        metrics="auc",
        seed=0,
        verbose_eval=False,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping
    )
    # Get best AUC and corresponding round
    best_auc = cv_results['test-auc-mean'].max()
    best_round = cv_results['test-auc-mean'].idxmax() + 1
    return best_auc, best_round, params

def grid_search_cv_xgb(param_grid, X, y, nfold=5, early_stopping=10, n_jobs=-1, verbose=True):
    """Grid search for XGBoost with GPU acceleration."""
    # Create all hyperparameter combinations
    param_combinations = [
        dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())
    ]
    
    if verbose:
        print(f"Starting grid search with {len(param_combinations)} combinations, "
              f"training {len(param_combinations) * nfold} models")
    
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    n_jobs = min(n_jobs, len(param_combinations))
    if verbose:
        print(f"Running grid search with {n_jobs} cores")
    
    args = [(params, X, y, nfold, early_stopping) for params in param_combinations]
    
    # Parallel processing
    with Pool(processes=n_jobs) as pool:
        results = pool.map(_evaluate_params, args)
    
    # Find best parameters based on AUC
    best_auc, best_round, best_params = max(results, key=lambda x: x[0])
    
    model_params = best_params.copy()
    model_params['n_estimators'] = best_round
    model_params['tree_method'] = 'hist'  # Ensure GPU is used in final model
    model_params['device'] = 'cuda'
    
    if verbose:
        print(f"Best Parameters: {best_params}")
        
    best_model = xgb.XGBClassifier(**model_params)
    best_model.fit(X, y)
    
    return best_model


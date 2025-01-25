from sklearn.model_selection import GridSearchCV
#import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import product
import xgboost as xgb
import time

best_params = {
    "LogisticGAM" : {
        #"max_iter": 1000,
        #"n_splines": 10,
        #"lam": 0.6
    },
    "DecisionTreeClassifier" : {
        "criterion": "gini",
        "max_depth": 5,#10,  #8, 10
        "min_samples_leaf": 16,#12, #10, 12
        "splitter": "best"
    },
    "GradientBoostingClassifier" : {
        "learning_rate": 0.025,
        "max_depth": 3,
        "n_estimators": 125, #150, 125
        "subsample": 0.4    #0.5, 0.4
    },
    "RandomForestClassifier" : {
        "n_estimators": 125,#50, #50, 50
        "criterion": "gini",
        "max_depth": 5,  #7,7
        "min_samples_leaf": 1,#2,  #2, 2
        "min_samples_split": 5,#10, #10, 10
        "max_samples": 0.8,
        "random_state": 0
    },
    "XGBoost" : {
        'n_estimators': 50,
        'learning_rate': 0.1,
        'max_depth': 6,
        'min_child_weight': 2,
        'subsample': 0.7,
        'colsample_bytree': 1.0,
        'reg_alpha': 1,
        'reg_lambda': 1,
        'gamma': 5
    },
    "LogisticRegression" : {
        "penalty": "l2",
        "C": 0.1,
        "solver": "liblinear"
    }
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

def _fit_and_score(params, model, X_train, y_train):
    model = model.__class__(**params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    score = roc_auc_score(y_train, predictions)
    return score, params, model

def _check_params(params):
    if params['penalty'] is None and params['solver'] == 'liblinear':
        return False
    if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
        return False
    if params['penalty'] == 'l2' and params['solver'] not in ['liblinear', 'lbfgs', 'saga', 'newton-cg']:
        return False
    return True

def overfit_models(X_train, y_train, model, param_grid, verbose=1, n_jobs=-1):
    grid = list(ParameterGrid(param_grid))
    best_score = -1
    best_params = None
    best_model = None
    
    print(model.__class__.__name__)
    if model.__class__.__name__ == 'LogisticRegression':
        grid = [params for params in grid if _check_params(params)]
    
    if verbose:
        print(f"Searching for best parameters for {model.__class__.__name__}, training {len(grid)} models")

    # Create a partial function with fixed arguments
    worker = partial(_fit_and_score, model=model, X_train=X_train, y_train=y_train)
    
    # Use all available CPU cores
    if n_jobs == -1:
        n_jobs = cpu_count()
    n_jobs = min(n_jobs, len(grid))
    print(f"Running grid search with {n_jobs} cores")
    
    # Run the grid search in parallel
    with Pool(processes=n_jobs) as pool:
        results = pool.map(worker, grid)
    
    # Find the best result
    for score, params, model in results:
        if score > best_score:
            best_score = score
            best_params = params
            best_model = model
            
    if verbose:
        print(f"Best parameters found: {best_params}")
    
    return best_model, best_params, best_score

def _evaluate_params(args):
    """Optimized parameter evaluation + early stopping"""
    params, X, y, nfold, early_stopping = args
    local_params = params.copy()
    # set objective and eval metric
    local_params.update({'objective': 'binary:logistic', 'eval_metric': 'auc'})
    # extract n_estimators and remove from local_params
    # if not present, default to 100
    num_boost_round = local_params.pop('n_estimators', 100)
    # Create DMatrix
    dtrain = xgb.DMatrix(X, label=y)
    # Execute CV
    cv_results = xgb.cv(
        params=local_params,
        dtrain=dtrain,
        nfold=nfold,
        metrics="auc",
        seed=0,
        verbose_eval=False,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping  # <--- EARLY STOPPING
    )
    # best auc e best round
    best_auc = cv_results['test-auc-mean'].max()
    best_round = cv_results['test-auc-mean'].idxmax() + 1  # +1 -> index starts from 0
    return best_auc, best_round, params

def grid_search_cv_xgb(param_grid, X, y, nfold=5, early_stopping=10, n_jobs=-1, verbose=True):
    start_time = time.time()
    
    # Crea tutte le combinazioni di iperparametri
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
    
    # Prepara gli argomenti per l'evaluazione in parallelo
    # Passiamo anche nfold ed early_stopping
    args = [(params, X, y, nfold, early_stopping) for params in param_combinations]
    
    # Run parallel evaluation
    with Pool(processes=n_jobs) as pool:
        results = pool.map(_evaluate_params, args)
    
    # results contiene tuple: (best_auc, best_round, original_params)
    # Cerchiamo la tupla con best_auc massima
    best_auc, best_round, best_params = max(results, key=lambda x: x[0])
    
    # Ricostruisci il modello XGBClassifier con i best params
    # Imposta n_estimators = best_round (ottenuto dall’Early Stopping in CV)
    # Così il modello finale avrà il numero di alberi “ottimale”
    model_params = best_params.copy()
    model_params['n_estimators'] = best_round  # Imposto la best iteration
    
    if verbose:
        print(f"Best Parameters: {best_params}")
        
    best_model = xgb.XGBClassifier(**model_params)
    best_model.fit(X, y)
    
    return best_model

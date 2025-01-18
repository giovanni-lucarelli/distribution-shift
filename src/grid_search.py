from sklearn.model_selection import GridSearchCV
#import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score

best_params = {
    "LogisticGAM" : {
        #"max_iter": 1000,
        #"n_splines": 10,
        #"lam": 0.6
    },
    "DecisionTreeClassifier" : {
        "criterion": "entropy",
        "max_depth": 9,#10,  #8, 10
        "min_samples_leaf": 10,#12, #10, 12
        "splitter": "random"
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
        #"max_depth": 5,
        #"n_estimators": 100
    },
}

# #? overfitting params
# best_params = {
#     "LogisticGAM" : {
#         "max_iter": 1000,
#         "n_splines": 10,
#         "lam": 0.6
#     },
#     "DecisionTreeClassifier" : {
#         'criterion': 'gini',
#         'max_depth': 14,
#         'min_samples_leaf': 10,
#         'splitter': 'best'
#     },
#     "GradientBoostingClassifier" : {
#         'learning_rate': 0.025,
#         'max_depth': 8,
#         'max_features': None,
#         'n_estimators': 165,
#         'subsample': 0.7
#     },
#     "RandomForestClassifier" : {
#         'criterion': 'gini', 
#         'max_depth': 12, 
#         'min_samples_leaf': 2, 
#         'min_samples_split': 10,
#         'n_estimators': 175,
#         'random_state': 0
#     },
#     "XGBoost" : {
#         'learning_rate': 0.03,
#         'max_depth': 0,
#         'n_estimators': 200,
#         'subsample': 0.7
#     },
# }

#? vanilla params
# best_params = {
#     "LogisticGAM" : {},
#     "DecisionTreeClassifier" : {},
#     "GradientBoostingClassifier" : {},
#     "RandomForestClassifier" : {},
#     "XGBoost" : {},
# }


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

def overfit_models(X_train, y_train, model, param_grid):
    
    grid = ParameterGrid(param_grid)
    best_score = -1
    best_params = None
    best_model = None

    for params in grid:
        
        model.set_params(**params)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_train)
        score = roc_auc_score(y_train, predictions)
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_model = model
    
    return best_model, best_params, best_score



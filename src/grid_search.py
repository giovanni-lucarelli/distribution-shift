from sklearn.model_selection import GridSearchCV
import pandas as pd

best_params = {
    "LogisticGAM" : {
        "max_iter": 1000,
        "n_splines": 10,
        "lam": 0.6
    },
    "DecisionTreeClassifier" : {
        "criterion": "entropy",
        "max_depth": 8,
        "min_samples_leaf": 5,
        "splitter": "random"
    },
    "GradientBoostingClassifier" : {
        "learning_rate": 0.1,
        "max_depth": 3,
        "n_estimators": 100
    },
    "RandomForestClassifier" : {
        "max_depth": 10, 
        "min_samples_leaf": 4, 
        "min_samples_split": 5, 
        "n_estimators": 150},
    "XGBoost" : {
        "max_depth": 5,
        "n_estimators": 100
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
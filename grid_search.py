from sklearn.model_selection import GridSearchCV

def grid_search_cv(estimator, X_train, y_train, param_grid, cv=5, scoring='roc_auc'):

    # Initialize GridSearchCV
    grid_searcher = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring=scoring)
    # Perform the grid search
    grid_searcher.fit(X_train, y_train)
    # Get the best parameters and the best model
    best_params = grid_searcher.best_params_
    best_model = grid_searcher.best_estimator_

    print(f"Best parameters found: {best_params}")
    print(f"Best model: {best_model}")
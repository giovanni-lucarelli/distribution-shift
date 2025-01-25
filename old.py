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
#     "LogisticRegression" : {
#         'C': 10,
#         'max_iter': 500,
#         'penalty': 'l1',
#         'solver': 'liblinear'
#     }
# }

# #? vanilla params
# best_params = {
#     "LogisticGAM" : {},
#     "DecisionTreeClassifier" : {},
#     "GradientBoostingClassifier" : {},
#     "RandomForestClassifier" : {},
#     "XGBoost" : {},
#     "LogisticRegression" : {},
# }

#? decision tree classifier
# Define the parameter grid
dtc_grid = {
    'criterion'         : ['gini', 'entropy', 'log_loss'],
    'max_depth'         : [3, 4, 5, 6],#, 7, 8], #9, 10, 11, 12],
    'min_samples_leaf'  : [2, 4, 8, 16],
    'splitter'          : ['best', 'random']
}

if GRID_SEARCH:
    dtc_model = grid_search_cv(DecisionTreeClassifier(), dtc_grid, X_train, y_train, n_jobs=-1)
else:
    dtc_model = DecisionTreeClassifier(**best_params["DecisionTreeClassifier"])
    dtc_model.fit(X_train, y_train)

if OVERFIT:

    ## Logistic GAM

    lgam_params = {
        "terms"     : s(0, lam  = 0.01, n_splines = 30) + s(1, lam  = 0.01, n_splines = 30) + s(2, lam  = 0.01, n_splines = 30) + te(0, 1, lam  = 0.01) + te(0, 2, lam  = 0.01) + te(1, 2, lam  = 0.01),
        "max_iter"  : 10000
    }

    X_train_np = X_train.values  # Convert to NumPy array
    y_train_np = y_train.values  # Convert to NumPy array

    lgam_model_of = LogisticGAM(**lgam_params).fit(X_train_np, y_train_np)

    ## Random Forest

    dtc_grid = {
        "max_depth"         : [5, 10, 25, 50, 100, 150, 200],
        "min_samples_leaf"  : [4, 8, 16, 32, 64, 128],
        "criterion"         : ['gini', 'entropy', 'log_loss'],
        "splitter"          : ['random', 'best']
    }

    dtc_model_of, dtc_params_of, dtc_score_of = overfit_models(X_train, y_train, DecisionTreeClassifier(), dtc_grid)
    #{'criterion': 'log_loss', 'max_depth': 200, 'min_samples_leaf': 4, 'splitter': 'best'}

    ## Random Forest

    rfc_grid = {
        'criterion'         : ['gini', 'entropy'], #'log_loss'],
        'n_estimators'      : [75, 100, 175],
        'max_depth'         : [16, 32, 64],
        'min_samples_split' : [2, 3, 4],
        'min_samples_leaf'  : [1, 2, 4],
        'random_state'      : [0]
    }

    rfc_model_of, rfc_params_of, rfc_score_of = overfit_models(X_train, y_train, RandomForestClassifier(), rfc_grid)
    #{'criterion': 'gini', 'max_depth': 32, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 75, 'random_state': 0}

    ## Gradient Boosting

    gbc_grid = {
        'n_estimators'  : [100, 175, 200],
        'learning_rate' : [0.05, 0.1],
        'max_depth'     : [16, 32, 64],
        'subsample'     : [1.0],
        'max_features'  : [None]#, 'sqrt', 'log2']
    }

    gbc_model_of, gbc_params_of, gbc_score_of = overfit_models(X_train, y_train, GradientBoostingClassifier(), gbc_grid)
    #{'learning_rate': 0.05, 'max_depth': 16, 'max_features': None, 'n_estimators': 175, 'subsample': 1.0}

    ## XGBoost

    xgb_grid = {
        'learning_rate'     : [0.05, 0.075, 0.1],
        'max_depth'         : [0, 16, 32], 
        #'subsample'         : [1.0],#[0.5, 0.7],
        #'colsample_bytree'  : [0.5, 0.7],
        'n_estimators'      : [250, 500, 700]
    }

    xgb_model_of, xgb_params_of, xgb_score_of = overfit_models(X_train, y_train, xgb.XGBClassifier(), xgb_grid)
    #{'learning_rate': 0.05, 'max_depth': 32, 'n_estimators': 700, 'subsample': 1.0}
    
    # Define the models to evaluate
    models = {
        "DecisionTreeClassifier"        : dtc_model_of,
        "RandomForestClassifier"        : rfc_model_of, 
        "GradientBoostingClassifier"    : gbc_model_of,
        "XGBoost"                       : xgb_model_of,
        "LogisticGAM"                   : lgam_model_of
    }

    # Assuming df_dict is a dictionary with keys from 0.1 to 1.0
    test_datasets = [(key, df.drop(['Y','Z'], axis=1), df['Y']) for key, df in df_dict.items() if 0.0 <= key <= 1.0]
    #test_datasets = [("Train", df_train.drop(['Y', 'Z'], axis=1), df_train['Y'])]
    evaluator = ModelEvaluator(models, test_datasets)
    evaluator.evaluate_models(show_metrics=False) # = True)
    evaluator.plot_roc_curves()
    evaluator.plot_roc_curves_per_dataset()
    evaluator.plot_auc()
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, classification_report

# TODO: search for the best hyperparameters for the models

def evaluate_models_on_shifts(folder: str = "data", target = 'Y'): 
    """
    1) Train a DecisionTree and a GradientBoosting model on 'train.csv'.
    2) Evaluate each model on all 'mix_*' CSVs in the folder.
    3) Print metrics (Accuracy, F1, AUC) and plot ROC curves.

    Parameters
    ----------
    folder : str, optional
        Folder containing 'train.csv' and 'mix_*.csv'.
    """
    # Load original => train data
    df_orig = pd.read_csv(os.path.join(folder, "train.csv"))
    X_train = df_orig.drop(target, axis=1)
    y_train = df_orig[target]
    
    # Fit Decision Tree
    dtc = DecisionTreeClassifier(max_depth=4, min_samples_leaf=13)
    dtc.fit(X_train, y_train)
    
    # Fit GradientBoosting
    gbc = GradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_features='log2',
        min_samples_leaf=13,
        n_estimators=100,
        subsample=0.7
    )
    gbc.fit(X_train, y_train)
    
    # Evaluate on shifted sets
    test_files = [f for f in os.listdir(folder) if f.startswith("mix_")]
    
    plt.figure(figsize=(10,5))
    color_map = {"DecisionTree": "blue", "GradientBoosting": "red"}
    
    for model, name in zip([dtc, gbc], ["DecisionTree", "GradientBoosting"]):
        color = color_map[name]
        for test_file in sorted(test_files):
            df_test = pd.read_csv(os.path.join(folder, test_file))
            X_test = df_test[X_train.columns]
            y_test = df_test[target]
            
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            f1_ = f1_score(y_test, y_pred)
            auc_ = roc_auc_score(y_test, y_pred_proba)
            
            print(f"=== {name} on {test_file} ===")
            print(f"Accuracy: {acc:.3f}, F1: {f1_:.3f}, AUC: {auc_:.3f}")
            print(classification_report(y_test, y_pred))
            print("---------------------------------------------------")
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            label_str = f"{name}-{test_file} (AUC={auc_:.3f})"
            plt.plot(fpr, tpr, label=label_str, color=color, alpha=0.3)
    
    
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves on Shifted Test Sets")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

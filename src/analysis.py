import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, classification_report
from src.models import adversarial_training

# TODO: search for the best hyperparameters for the models

def evaluate_models_on_shifts(
    models,
    folder: str = "data",
    target: str = 'Y',
    fig_size=(15, 5),
    color_map=None
) -> None:
     
    """
    1) Train models on 'train.csv'.
    2) Evaluate each model on all 'mix_*' CSVs in the folder.
    3) Print metrics (Accuracy, F1, AUC) and plot ROC curves.

    Parameters
    ----------
    models : dict
        Dictionary of models to train and evaluate.
    folder : str, optional
        Folder containing 'train.csv' and 'mix_*.csv'.
    target : str, optional
        Target column name.
    fig_size : tuple, optional
        Figure size for ROC plot.
    color_map : dict, optional
        Color mapping for models.
    """
    # Load original => train data
    df_orig = pd.read_csv(os.path.join(folder, "train.csv"))
    X_train = df_orig.drop(target, axis=1)
    # y_train = df_orig[target]
    
    plt.figure(figsize=fig_size)
    if not color_map:
        color_map = {
            "DecisionTreeClassifier": "blue", 
            "GradientBoostingClassifier": "red"}  # default map
    
    for name, model in models.items():
        # model.fit(X_train, y_train)
        color = color_map.get(name, "green")    # fallback color
        # Evaluate on shifted sets
        test_files = [f for f in os.listdir(folder) if f.startswith("mix_")]
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

def compare_adversarial_training(
    folder: str = "data",
    target='Y',
    base_model_class=None,
    base_model_params=None,
    adv_params=None
) -> None:
    """
    Demonstrates the effect of adversarial training:
      1) Train a normal GradientBoosting on train.csv.
      2) Train an adversarially-augmented GradientBoosting on the same dataset.
      3) Compare performance on each shifted dataset.
    """
    # 1) Load original => train data
    df_orig = pd.read_csv(os.path.join(folder, "train.csv"))
    X_train = df_orig.drop(target, axis=1)
    y_train = df_orig[target]
    
    # Use fallback to GBC if none provided
    if base_model_class is None:
        from sklearn.ensemble import GradientBoostingClassifier
        base_model_class = GradientBoostingClassifier
    if base_model_params is None:
        base_model_params = {
            'learning_rate': 0.05,
            'max_depth': 4,
            'max_features': 'log2',
            'min_samples_leaf': 13,
            'n_estimators': 100,
            'subsample': 0.7
        }
    
    # Normal model
    normal_model = base_model_class(**base_model_params)
    normal_model.fit(X_train, y_train)
    
    if adv_params is None:
        adv_params = {'epsilon': 0.2}
    
    # Adversarial training
    adv_model, _, _ = adversarial_training(
        X_train, y_train,
        base_model_class=base_model_class,
        base_model_params=base_model_params,
        **adv_params
    )
    
    # 2) Evaluate on shifted sets
    test_files = [f for f in os.listdir(folder) if f.startswith("mix_")]
    
    print("\n=== Evaluate Normal vs. Adversarially Trained Model ===\n")
    for test_file in sorted(test_files):
        df_test = pd.read_csv(os.path.join(folder, test_file))
        X_test = df_test[X_train.columns]
        y_test = df_test[target]
        
        # Normal
        y_pred_n = normal_model.predict(X_test)
        y_proba_n = normal_model.predict_proba(X_test)[:, 1]
        acc_n = accuracy_score(y_test, y_pred_n)
        f1_n = f1_score(y_test, y_pred_n)
        auc_n = roc_auc_score(y_test, y_proba_n)
        
        # Adversarial
        y_pred_a = adv_model.predict(X_test)
        y_proba_a = adv_model.predict_proba(X_test)[:, 1]
        acc_a = accuracy_score(y_test, y_pred_a)
        f1_a = f1_score(y_test, y_pred_a)
        auc_a = roc_auc_score(y_test, y_proba_a)
        
        print(f"Shifted file: {test_file}")
        print(f"  Normal Model   => Acc: {acc_n:.3f}, F1: {f1_n:.3f}, AUC: {auc_n:.3f}")
        print(f"  AdvTrain Model => Acc: {acc_a:.3f}, F1: {f1_a:.3f}, AUC: {auc_a:.3f}")
        print("---------------------------------------------------")

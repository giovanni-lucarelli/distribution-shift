import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, classification_report
from src.robust_training import adversarial
from src.utils import pred_proba_1d, get_color_gradient
from src.robust_training.adversarial import AdversarialTrainer

# TODO: search for the best hyperparameters for the models

def evaluate_models_on_shifts(
    models,
    #? old
    folder: str = "data",
    #? new
    df_dict: dict = None,
    original_file: float = 0.0,
    #?
    target: str = 'Y',
    fig_size=(8, 8),
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
    
    if not color_map:
        color_map = {
            "DecisionTreeClassifier": "blue", 
            "GradientBoostingClassifier": "red",
            "LogisticGAM" : "orange"
        }
    
    for name, model in models.items():
        plt.figure(figsize=fig_size)
        color = color_map.get(name, "green")    # fallback color
        
        #? new
        #shifted_dict = df_dict.copy()
        #
        ## Pop the original dataset from the copied dictionary
        #df_orig = shifted_dict.pop(original_file, None)
        #
        #if df_orig is None:
        #    raise ValueError(f"Original dataset with key {original_file} not found in df_dict")
        #
        #shifted_dict = dict(sorted(shifted_dict.items()))
        #
        #for mix, shifted_df in shifted_dict.items():
        #    X_test = shifted_df.drop(columns=[target])
        #    y_test = shifted_df[target]
        #    
        #    y_pred = model.predict(X_test)
        #    y_pred_proba = pred_proba_1d(model, X_test)
        #    
        #    acc = accuracy_score(y_test, y_pred)
        #    f1_ = f1_score(y_test, y_pred)
        #    auc_ = roc_auc_score(y_test, y_pred_proba)
        #    
        #    print(f"=== {name} on {mix} ===")
        #    print(f"Accuracy: {acc:.3f}, F1: {f1_:.3f}, AUC: {auc_:.3f}")
        #    print(classification_report(y_test, y_pred))
        #    print("---------------------------------------------------")
        #    
        #    # ROC curve
        #    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        #    label_str = f"{name}-{mix} (AUC={auc_:.3f})"
        #    plt.plot(fpr, tpr, label=label_str, color=color, alpha=0.3)
        #? 
        
        # Evaluate on shifted sets
        test_files = sorted([f for f in os.listdir(folder) if f.startswith("mix_")])
        
        colors = get_color_gradient(color, len(test_files))
        
        
        for test_file, color in zip(test_files, colors):
            df_test = pd.read_csv(os.path.join(folder, test_file))
            X_test = df_test.drop(columns=[target])
            y_test = df_test[target]
            
            y_pred = model.predict(X_test)
            y_pred_proba = pred_proba_1d(model, X_test)
            
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

        # one plot per model
        plt.plot([0,1],[0,1],'k--')
        plt.xlim([0,1])
        plt.ylim([0,1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves on Shifted Test Sets")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        plt.close()
        

def compare_adversarial_training(
    folder: str = "data",
    target='Y',
    model_class=None,
    model_params=None,
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
    
    if model_class is None:
        from sklearn.ensemble import GradientBoostingClassifier
        model_class = GradientBoostingClassifier
    if model_params is None:
        model_params = {
            'learning_rate': 0.05,
            'max_depth': 4,
            'max_features': 'log2',
            'min_samples_leaf': 13,
            'n_estimators': 100,
            'subsample': 0.7
        }
        
    # Normal model
    model = model_class(**model_params)
    model.fit(X_train, y_train)
    
    if adv_params is None:
        adv_params = {'epsilon': 0.2, 'max_rounds': 3}
    
    # Use AdversarialTrainer
    adv_trainer = AdversarialTrainer(model_class=model_class, **adv_params)
    adv_trainer.fit(X_train, y_train, **model_params)
    adv_model = adv_trainer.model
    
    # 2) Evaluate on shifted sets
    test_files = [f for f in os.listdir(folder) if f.startswith("mix_")]
    
    max_len = 60
    model_name = model.__class__.__name__
    len_space = (max_len - len(model_name)) // 2
    corrector = ' ' if len(model_name) % 2 == 1 else ''
    print(f" ╔{'═'*(max_len)}╗")
    print(f" ║{' '*len_space}{model_name}{' '*len_space}{corrector}║")
    print(f" ║{' '*((max_len - 38)//2)}Normal vs. Adversarially Trained Model{' '*((max_len - 38)//2)}║")
    print(f" ╠{'═'*(max_len)}╣")
    for test_file in sorted(test_files):
        df_test = pd.read_csv(os.path.join(folder, test_file))
        X_test = df_test[X_train.columns]
        y_test = df_test[target]
        
        # Normal
        y_pred_n = model.predict(X_test)
        y_proba_n = pred_proba_1d(model, X_test)
        acc_n = accuracy_score(y_test, y_pred_n)
        f1_n = f1_score(y_test, y_pred_n)
        auc_n = roc_auc_score(y_test, y_proba_n)
        
        # Adversarial
        y_pred_a = adv_model.predict(X_test)
        y_proba_a = pred_proba_1d(adv_model, X_test)
        acc_a = accuracy_score(y_test, y_pred_a)
        f1_a = f1_score(y_test, y_pred_a)
        auc_a = roc_auc_score(y_test, y_proba_a)
        
        print(f" ║ Shifted file: {test_file}{' '*(max_len - 15 - len(test_file))}║")
        print(f" ║\t Normal Model   => Acc: {acc_n:.3f}, F1: {f1_n:.3f}, AUC: {auc_n:.3f}{(max_len - 58)*' '}║")
        print(f" ║\t AdvTrain Model => Acc: {acc_a:.3f}, F1: {f1_a:.3f}, AUC: {auc_a:.3f}{(max_len - 58)*' '}║")
        if test_file != test_files[-1]:
            print(f" ╟{'─'*(max_len)}╢")
    print(f" ╚{'═'*(max_len)}╝\n")

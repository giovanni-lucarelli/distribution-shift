import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import patsy
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from src.robust_training import adversarial
from src.robust_training.adversarial import AdversarialTrainer

np.random.seed(0)

def pred_proba_1d(model, X: pd.DataFrame) -> pd.DataFrame:
    X_copy = X.copy()
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_copy)
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]                   
    else:
        y_pred_proba = model.predict(X_copy)
    return y_pred_proba

class ModelEvaluator:
    
    def __init__(self, models, test_datasets):
        """
        Initialize the ModelEvaluator with models and test datasets.
        
        Parameters:
        - models: dict
            A dictionary with model names as keys and models as values.
        - test_datasets: list of tuples
            A list of tuples where each tuple contains (dataset_name, X_test, y_test).
        """
        self.models = models
        self.test_datasets = test_datasets
        self.results = {}  # {model_name: {dataset_index: {metrics}}}

    
    def evaluate_models(self, show_metrics=False):
        """
        Evaluate all models on all test datasets and store the metrics.
        
        Parameters:
        - show_metrics: bool
            If True, print the metrics for each model and dataset.
        """
        for model_name, model in self.models.items():
            self.results[model_name] = {}
            for i, (dataset_name,X, y) in enumerate(self.test_datasets):
                X_test, y_test = X.copy(), y.copy()
                if model.__class__.__name__ == "BinaryResultsWrapper":
                    X_tmp = X_test.copy()
                    X_tmp['Y'] = y_test
                    y_test, X_test = patsy.dmatrices('Y ~ X1 * X2 * X3', data=X_tmp, return_type='dataframe')
                
                y_pred_proba = pred_proba_1d(model, X_test)
                
                auc = roc_auc_score(y_test, y_pred_proba)
                
                self.results[model_name][i] = {
                    "dataset_name": dataset_name,
                    "auc": auc,
                    "y_test": y_test,
                    "y_pred_proba": y_pred_proba
                }
                
                if show_metrics:
                    print(f"{model_name}:\tAUC: {auc:.3f}")
                    print("---------------------------------------------------")
        return self.results
    
    def plot_roc_curves(self):
        """
        Plot ROC curves for each model on all test datasets.
        """
        for model_name, datasets in self.results.items():
            plt.figure(figsize=(10, 10))
            for i, metrics in datasets.items():
                fpr, tpr, _ = roc_curve(metrics["y_test"], metrics["y_pred_proba"])
                plt.plot(fpr, tpr, label=f'Dataset {metrics["dataset_name"]} (AUC = {metrics["auc"]:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Random Guess")
            plt.title(f"ROC Curves for {model_name}", fontsize=20)
            plt.xlabel("False Positive Rate", fontsize=16)
            plt.ylabel("True Positive Rate", fontsize=16)
            plt.legend(loc="lower right", fontsize=15)
            plt.grid(alpha=0.5)
            plt.show()
    
    def plot_roc_curves_per_dataset(self):
        """
        Plot ROC curves for each test dataset with all models.
        """
        n_datasets = len(self.test_datasets)
        n_cols = 2
        n_rows = (n_datasets + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        for i, (dataset_name, X_test, y_test) in enumerate(self.test_datasets):
            ax = axes[i]
            for model_name, datasets in self.results.items():
                metrics = datasets[i]
                fpr, tpr, _ = roc_curve(metrics["y_test"], metrics["y_pred_proba"])
                ax.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics["auc"]:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label="Random Guess")
            ax.set_title(f"ROC Curves for Test Dataset {dataset_name}", fontsize=12)
            ax.set_xlabel("False Positive Rate", fontsize=10)
            ax.set_ylabel("True Positive Rate", fontsize=10)
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(alpha=0.5)
        
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.show()

    def plot_auc(self):
        """
        Plot accuracy for each model as a function of the test sets.
        """
        plt.figure(figsize=(12, 6))
        for model_name, datasets in self.results.items():
            dataset_names = [metrics["dataset_name"] for metrics in datasets.values()]
            auc = [metrics["auc"] for metrics in datasets.values()]
            plt.plot(dataset_names, auc, marker='o', label=model_name)
        
        # set x ticks with sensibility 0.1
        xticks = [round(x / 10, 1) for x in range(11)]
        plt.xticks(ticks=xticks, labels=[f"{x:.1f}" for x in xticks])
        
        plt.title("ROC AUC for Each Model on Different Test Sets", fontsize=25)
        plt.xlabel("Test Set", fontsize=16)
        plt.ylabel("AUC", fontsize=16)
        plt.legend(loc="best", fontsize=16)
        plt.grid(alpha=0.5)
        plt.tight_layout()
        plt.show()
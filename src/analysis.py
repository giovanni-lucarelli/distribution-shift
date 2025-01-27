import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import patsy
from sklearn.metrics import roc_auc_score, roc_curve
from multiprocessing import Pool

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

def evaluate_model_on_dataset(args):
    """
    Helper function to evaluate a single model on a single dataset.
    This is necessary for multiprocessing.
    """
    model_name, model, dataset_index, dataset = args
    dataset_name, X, y = dataset
    X_test, y_test = X.copy(), y.copy()
    
    if model.__class__.__name__ == "BinaryResultsWrapper":
        X_tmp = X_test.copy()
        X_tmp['Y'] = y_test
        y_test, X_test = patsy.dmatrices('Y ~ X1 * X2 * X3', data=X_tmp, return_type='dataframe')
    
    y_pred_proba = pred_proba_1d(model, X_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return model_name, dataset_index, {
        "dataset_name": dataset_name,
        "auc": auc,
        "y_test": y_test,
        "y_pred_proba": y_pred_proba
    }

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
    
    def evaluate_models(self, show_metrics=False, n_jobs=1):
        """
        Evaluate all models on all test datasets and store the metrics.
        
        Parameters:
        - show_metrics: bool
            If True, print the metrics for each model and dataset.
        - n_jobs: int
            The number of parallel processes to use. Default is 1 (no parallelism).
        """
        tasks = []
        for model_name, model in self.models.items():
            for i, dataset in enumerate(self.test_datasets):
                tasks.append((model_name, model, i, dataset))
        
        if n_jobs > 1:
            with Pool(processes=n_jobs) as pool:
                results = pool.map(evaluate_model_on_dataset, tasks)
        else:
            results = map(evaluate_model_on_dataset, tasks)
        
        # Organize results into self.results
        for model_name, dataset_index, metrics in results:
            if model_name not in self.results:
                self.results[model_name] = {}
            self.results[model_name][dataset_index] = metrics
            
            if show_metrics:
                print(f"{model_name} on dataset {metrics['dataset_name']}:\tAUC: {metrics['auc']:.3f}")
                print("---------------------------------------------------")
        
        return self.results

    
    def plot_roc_curves(self):
        """
        Plot ROC curves for each model on all test datasets.
        """
        for model_name, datasets in self.results.items():
            plt.figure(figsize=(14, 10))
            for i, metrics in datasets.items():
                fpr, tpr, _ = roc_curve(metrics["y_test"], metrics["y_pred_proba"])
                plt.plot(fpr, tpr, label=f'Dataset {metrics["dataset_name"]} (AUC = {metrics["auc"]:.3f})')
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Random Guess")
            plt.title(f"ROC Curves for {model_name}", fontsize=35)
            plt.xlabel("False Positive Rate", fontsize=20)
            plt.ylabel("True Positive Rate", fontsize=20)
            plt.legend(loc="lower right", fontsize=17)
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
        
def visualize_feature_shifts(
    df_dict: dict,
    original_file: float = 0.0,
    features_to_plot: list = None,
    figsize_base: tuple = (6, 4),
    alpha: float = 0.3
):
    """
    Enhanced visualization of feature distributions and relationships.
    """
    # Make a copy of the dictionary
    shifted_dict = df_dict.copy()

    # Pop the original dataset from the copied dictionary
    df_orig = shifted_dict.pop(original_file, None)

    if df_orig is None:
        raise ValueError(f"Original dataset with key {original_file} not found in df_dict")

    shifted_dict = dict(sorted(shifted_dict.items()))
    
    # Set default features if None
    if features_to_plot is None:
        features_to_plot = df_orig.select_dtypes(include=[np.number]).columns.tolist()
    
    # 1. Scatter Plot Matrix
    n_features = len(features_to_plot)
    n_combinations = (n_features * (n_features - 1)) // 2
    n_cols = min(3, n_combinations)
    n_rows = (n_combinations + n_cols - 1) // n_cols
    
    fig_scatter = plt.figure(figsize=(figsize_base[0] * n_cols, figsize_base[1] * n_rows))
    gs = fig_scatter.add_gridspec(n_rows, n_cols)
    
    plot_idx = 0
    for i in range(n_features):
        for j in range(i+1, n_features):
            if plot_idx >= n_rows * n_cols:
                break
                
            ax = fig_scatter.add_subplot(gs[plot_idx // n_cols, plot_idx % n_cols])
            feat1, feat2 = features_to_plot[i], features_to_plot[j]
            
            ax.scatter(df_orig[feat1], df_orig[feat2], alpha=alpha, label='Original')
            
            for mix, shifted_df in shifted_dict.items():
                ax.scatter(shifted_df[feat1], shifted_df[feat2], alpha=alpha, label=mix)
            
            ax.set_xlabel(feat1)
            ax.set_ylabel(feat2)
            ax.grid(True, alpha=0.3)
            if plot_idx == 0:  # Only show legend for first subplot
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plot_idx += 1
    
    plt.tight_layout()
    plt.show()
    
    # 2. Distribution Plots
    for mix, shifted_df in shifted_dict.items():

        n_features = len(features_to_plot)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig_dist, axes = plt.subplots(n_rows, n_cols, 
                                    figsize=(5*n_cols, 4*n_rows),
                                    squeeze=False)
        axes = axes.flatten()
        
        for idx, feature in enumerate(features_to_plot):
            if idx < len(axes):
                sns.histplot(df_orig[feature], color="blue", alpha=0.4,
                           ax=axes[idx], label=f'Original {feature}', kde=True)
                sns.histplot(shifted_df[feature], color="red", alpha=0.4,
                           ax=axes[idx], label=f'{mix} {feature}', kde=True)
                axes[idx].set_title(f"Distribution of {feature}")
                axes[idx].legend()
        
        # Remove empty subplots
        for idx in range(len(features_to_plot), len(axes)):
            fig_dist.delaxes(axes[idx])
            
        plt.suptitle(f"Feature Distributions Comparison: {mix}")
        plt.tight_layout()
        plt.show()
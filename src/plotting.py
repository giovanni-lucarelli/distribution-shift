import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

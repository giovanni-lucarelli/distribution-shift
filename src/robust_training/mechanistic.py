# src/robust_training/mechanistic.py

from typing import Tuple, Union
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Attempt to import pyGAM; handle gracefully if not installed
try:
    from pygam import LogisticGAM
    PYGAM_AVAILABLE = True
except ImportError:
    PYGAM_AVAILABLE = False

from .base import RobustTrainer  # Ensure this exists in your project

class MechanisticTrainer(RobustTrainer):
    """
    A robust training class that supports multiple model types and uses
    mechanistic interpretability-inspired data augmentation.
    
    Parameters
    ----------
    model_type : str, optional
        One of ['gbc', 'tree', 'gam']. 
        - 'gbc' for GradientBoostingClassifier
        - 'tree' for DecisionTreeClassifier
        - 'gam' for pyGAM's LogisticGAM
    base_shift_factor : float, optional
        Magnitude for feature shifting. Larger values lead to more significant shifts.
    n_rounds : int, optional
        Number of augmentation rounds.
    subset_size_fraction : float, optional
        Fraction of data to select for augmentation each round.
    n_grad_steps : int, optional
        Number of gradient-based steps for each sample (similar to adversarial attacks).
    top_k : int, optional
        Number of top features (by gradient magnitude) to shift for each sample.
    random_state : int, optional
        Seed for reproducibility.
    noise_scale : float, optional
        Standard deviation of Gaussian noise added to shifted samples.
    """
    
    def __init__(
        self,
        model_type: str = 'gbc',
        base_shift_factor: float = 0.1,
        n_rounds: int = 2,
        subset_size_fraction: float = 0.7,
        n_grad_steps: int = 1,
        top_k: int = 3,
        random_state: int = 42,
        noise_scale: float = 0.001
    ):
        super().__init__()
        self.model_type = model_type.lower()
        self.base_shift_factor = base_shift_factor
        self.n_rounds = n_rounds
        self.subset_size_fraction = subset_size_fraction
        self.n_grad_steps = n_grad_steps
        self.top_k = top_k
        self.random_state = random_state
        self.noise_scale = noise_scale

        self.model: Union[BaseEstimator, None] = None
        self.X_final: Union[pd.DataFrame, None] = None
        self.y_final: Union[pd.Series, None] = None
        self.rng = np.random.RandomState(self.random_state)

        if self.model_type == 'gam' and not PYGAM_AVAILABLE:
            raise ImportError("pyGAM is not installed. Install it via `pip install pygam` or choose another model type.")

    def _init_model(self, **kwargs):
        """
        Initialize the chosen model type.
        """
        if self.model_type == 'tree':
            self.model = DecisionTreeClassifier(
                max_depth=5, 
                random_state=self.random_state,
                **kwargs
            )
        elif self.model_type == 'gam':
            # Initialize pyGAM's LogisticGAM
            self.model = LogisticGAM(verbose=False, **kwargs)
        else:  # Default to GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                random_state=self.random_state,
                **kwargs
            )

    def augment_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Perform gradient-based augmentation to shift feature values.
        """
        if self.model is None:
            raise ValueError("Model has not been initialized or fitted yet.")

        # Create copies to avoid modifying original data
        X_aug = X.copy()
        y_aug = y.copy()

        # Determine number of samples to augment
        n_samples = len(X)
        n_to_shift = int(n_samples * self.subset_size_fraction)
        indices_to_shift = self.rng.choice(n_samples, size=n_to_shift, replace=False)

        # Iterate over selected samples
        for idx in indices_to_shift:
            x_i = X_aug.iloc[idx].values.copy()
            label_i = y_aug.iloc[idx]

            # Perform multiple gradient steps if specified
            for _ in range(self.n_grad_steps):
                feature_scores = []
                for feat_idx in range(len(x_i)):
                    x_pos = x_i.copy()
                    x_neg = x_i.copy()
                    eps = 0.1

                    x_pos[feat_idx] += eps
                    x_neg[feat_idx] -= eps

                    # Estimate gradient using predict_proba or predict, considering true label
                    if hasattr(self.model, "predict_proba") and callable(self.model.predict_proba):
                        prob_pos = self.model.predict_proba([x_pos])[0][label_i]
                        prob_neg = self.model.predict_proba([x_neg])[0][label_i]
                        grad_est = (prob_pos - prob_neg) / (2 * eps)
                    else:
                        # For models without predict_proba, use decision function or similar
                        prob_pos = self.model.predict([x_pos])[0]
                        prob_neg = self.model.predict([x_neg])[0]
                        grad_est = (prob_pos - prob_neg) / (2 * eps)

                    feature_scores.append(grad_est)
                print(f"  => Sample {idx + 1}/{n_samples} - Gradient: {feature_scores}")
                # Identify top_k features by absolute gradient magnitude
                top_feats_idx = np.argsort(np.abs(feature_scores))[-self.top_k:]

                # Shift each top feature in the direction that minimizes the predicted probability for the true class
                for feat_idx in top_feats_idx:
                    grad = feature_scores[feat_idx]
                    direction_sign = 1 if grad < 0 else -1
                    x_i[feat_idx] += direction_sign * self.base_shift_factor

            # Add Gaussian noise for variability
            x_i += self.rng.normal(0, self.noise_scale, size=len(x_i))

            # Clip shifted features to original data's min/max to avoid out-of-bound values
            x_i = np.clip(x_i, X_aug.min().values, X_aug.max().values)

            # Update the augmented DataFrame
            X_aug.iloc[idx] = x_i

        return X_aug, y_aug

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        Fit the robust model using augmented data.
        """
        self.rng = np.random.RandomState(self.random_state)

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )

        # Initialize the model
        if self.model is None:
            self._init_model(**kwargs)

        # Initial fit on training data
        self.model.fit(X_train, y_train)

        # Perform augmentation rounds
        X_aug = X_train.copy()
        y_aug = y_train.copy()

        for round_idx in range(self.n_rounds):
            print(f"Augmentation Round {round_idx + 1}/{self.n_rounds}")
            X_shifted, y_shifted = self.augment_data(X_aug, y_aug)
            X_aug = pd.concat([X_aug, X_shifted], ignore_index=True)
            y_aug = pd.concat([y_aug, y_shifted], ignore_index=True)
            print(f"  => Augmented data size: {X_aug.shape[0]} samples")

        # Combine augmented data with validation set
        X_combined = pd.concat([X_aug, X_val], ignore_index=True)
        y_combined = pd.concat([y_aug, y_val], ignore_index=True)

        # Ensure the final dataset size matches the original training set
        original_size = len(X)
        if len(X_combined) > original_size:
            idx_final = self.rng.choice(len(X_combined), size=original_size, replace=False)
            X_final = X_combined.iloc[idx_final].reset_index(drop=True)
            y_final = y_combined.iloc[idx_final].reset_index(drop=True)
            print(f"  => Downsampled combined data to {original_size} samples")
        else:
            X_final = X_combined
            y_final = y_combined
            print(f"  => Combined data size: {X_final.shape[0]} samples")

        # Final fit on the combined dataset
        self.model.fit(X_final, y_final)
        self.X_final = X_final
        self.y_final = y_final

        print("Robust model training completed.\n")

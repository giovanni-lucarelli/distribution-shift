# src/robust_training/mechanistic.py

from typing import Tuple, Union, Optional
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Attempt to import pyGAM; handle gracefully if not installed
try:
    from pygam import LogisticGAM
    PYGAM_AVAILABLE = True
except ImportError:
    PYGAM_AVAILABLE = False

from .base import RobustTrainer  # Make sure this is correct in your project structure

class MechanisticTrainer(RobustTrainer):
    """
    A robust training class that supports multiple model types and
    mechanistic-interpretability-inspired data augmentation.

    This version:
      - Uses smaller numeric gradients to avoid unstable shifts
      - Optionally weights augmented samples that genuinely improve
        classification probability for their correct label
      - Ensures final augmented dataset has the same size as the original training set

    Parameters
    ----------
    model_type : str, optional
        One of {'gbc', 'tree', 'gam'}:
          'gbc' -> GradientBoostingClassifier
          'tree' -> DecisionTreeClassifier
          'gam' -> LogisticGAM from pyGAM
    base_shift_factor : float, optional
        Magnitude for feature shifting each step. Larger => bigger shifts.
    n_rounds : int, optional
        Number of augmentation rounds.
    subset_size_fraction : float, optional
        Fraction of data to augment each round.
    n_grad_steps : int, optional
        Number of gradient-based steps (akin to PGD).
    top_k : int, optional
        Number of top features (by gradient magnitude) to shift each sample.
    random_state : int, optional
        Seed for reproducibility.
    noise_scale : float, optional
        Std dev of Gaussian noise added to shifted samples.
    use_weighting : bool, optional
        If True, assign higher sample_weight to augmented samples that
        improve the model's probability of correct classification.
    weighting_scale : float, optional
        Base scale for weighting newly augmented samples if 'use_weighting=True'.
    val_fraction : float, optional
        Fraction of data used as validation. (default=0.1)
    """

    def __init__(
        self,
        model_type: str = 'gbc',
        base_shift_factor: float = 0.1,
        n_rounds: int = 1,
        subset_size_fraction: float = 0.7,
        n_grad_steps: int = 1,
        top_k: int = 3,
        random_state: int = 42,
        noise_scale: float = 0.001,
        
        val_fraction: float = 0.1,
        eps: float = 0.1,
        ball: bool = False
    ):
        super().__init__()
        if model_type == 'gam' and not PYGAM_AVAILABLE:
            raise ImportError("pyGAM not installed. Please install or pick another model type.")
        self.model_type = model_type.lower()  # 'gbc', 'tree', 'gam'
        self.base_shift_factor = base_shift_factor
        self.n_rounds = n_rounds
        self.subset_size_fraction = subset_size_fraction
        self.n_grad_steps = n_grad_steps
        self.top_k = top_k
        self.random_state = random_state
        self.noise_scale = noise_scale
        self.val_fraction = val_fraction
        self.eps = eps
        self.ball = ball

        self.model: Union[BaseEstimator, None] = None
        self.X_final: Optional[pd.DataFrame] = None
        self.y_final: Optional[pd.Series] = None

        

        #rnaodm state
        self.rng = np.random.RandomState(self.random_state)

    def _init_model(self, **kwargs):
        """Initialize a fresh model depending on model_type."""
        if self.model_type == 'tree':
            self.model = DecisionTreeClassifier(
                max_depth=10,
                random_state=self.random_state,
                **kwargs
            )
        elif self.model_type == 'gam':
            self.model = LogisticGAM(**kwargs)
        elif self.model_type == 'rfc':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                **kwargs
            )
            
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                random_state=self.random_state,
                **kwargs
            )

    def _predict_proba_for_label(self, X_sample: np.ndarray, label: int) -> float:
        """
        Safely predict probability for 'label' for a single sample X_sample.
        If the model doesn't support predict_proba, fallback to a 0-1 guess.
        """
        X_sample_2d = X_sample.reshape(1, -1)
        if hasattr(self.model, "predict_proba") and callable(self.model.predict_proba):
            probs = self.model.predict_proba(X_sample_2d)[0]
            
            # If binary => probs is [p(class=0), p(class=1)].
            # If multiclass => we index by the label integer.
            if len(probs) <= label:
                return 0.0  # fallback
            return probs[label]   # return the probability of the correct label
            
        else:
            # Fallback to predict
            pred = self.model.predict(X_sample_2d)[0]
            
            return float(pred == label)

    def augment_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """
        Perform gradient-based augmentation on a subset of the data.
        Returns (X_aug, y_aug, weights_aug) if use_weighting=True,
        or weights_aug as all ones otherwise.
        """
        if self.model is None:
            raise ValueError("Model has not been initialized or fitted yet.")

        X_aug = X.copy()
        y_aug = y.copy()
        n_samples = len(X)
        print(f"  => Augmenting {n_samples} samples.")
        

        # We'll select a fraction of samples to shift
        n_to_shift = int(n_samples * self.subset_size_fraction)
        if n_to_shift == 0:
            # If fraction is too small or data is too small
            return X_aug, y_aug

        indices_to_shift = self.rng.choice(n_samples, size=n_to_shift, replace=False)

        for idx in indices_to_shift:
            x_i = X_aug.iloc[idx].values
            label_i = y_aug.iloc[idx]

            

            # Perform n_grad_steps small gradient-based shifts
            for _ in range(self.n_grad_steps):
                # Estimate gradient wrt the correct label
                # We'll do a small +/- step on each feature
                gradients = []
                eps = self.eps
                
                if self.ball:
                    print("Ball")
                    TOT = 3
                    # We’ll use a simple distance measure here
                    distances = np.linalg.norm(X_aug.values - x_i, axis=1)
                    nn_indices = np.argsort(distances)[:TOT]
                    neighbors_grad_signs = []

                    # Collect each neighbor’s numeric gradient sign
                    for nn_idx in nn_indices:
                        x_j = X_aug.iloc[nn_idx].values
                        local_grads = []
                        for feat_idx in range(len(x_j)):
                            x_pos = x_j.copy()
                            x_neg = x_j.copy()
                            x_pos[feat_idx] += eps
                            x_neg[feat_idx] -= eps
                            prob_pos = self._predict_proba_for_label(x_pos, label_i)
                            prob_neg = self._predict_proba_for_label(x_neg, label_i)
                            grad_est = (prob_pos - prob_neg) / (2 * eps)
                            # Record sign of neighbor’s gradient
                            local_grads.append(np.sign(grad_est))
                        neighbors_grad_signs.append(local_grads)

                    # Majority vote per feature
                    grad_signs = np.sign(np.sum(neighbors_grad_signs, axis=0))
                    # Shift x_i accordingly
                    top_feats_idx = np.argsort(np.abs(grad_signs))[-self.top_k:]
                    for feat_idx in top_feats_idx:
                        x_i[feat_idx] += grad_signs[feat_idx] * self.base_shift_factor
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                else:
                    for feat_idx in range(len(x_i)):
                        x_pos = x_i.copy()
                        x_neg = x_i.copy()
                        x_pos[feat_idx] += eps
                        x_neg[feat_idx] -= eps

                        prob_pos = self._predict_proba_for_label(x_pos, label_i)
                        prob_neg = self._predict_proba_for_label(x_neg, label_i)
                        grad_est = (prob_pos - prob_neg) / (2 * eps)
                        
                        gradients.append(grad_est)

                    # Pick top_k features by absolute gradient
                    top_feats_idx = np.argsort(np.abs(gradients))[-self.top_k:]

                    # SHIFT: We want to SHIFT in the direction that *increases* the correct label prob
                    # so if grad_est is positive => increasing feature => higher probability => shift up
                    # if grad_est is negative => decreasing feature => higher probability => shift down
                    for feat_idx in top_feats_idx:
                        g = gradients[feat_idx]
                        direction_sign = 1 if g > 0 else -1
                        x_i[feat_idx] += direction_sign * self.base_shift_factor

            # Add random Gaussian noise to the final x_i
            noise = self.rng.normal(0, self.noise_scale, size=len(x_i))
            x_i += noise

            # Clip to global min/max to avoid out-of-bound expansions
            #col_mins = X_aug.min().values
            #col_maxs = X_aug.max().values
            #x_i = np.clip(x_i, col_mins, col_maxs)
            

            # Store the final x_i back
            X_aug.iloc[idx] = x_i

        return X_aug, y_aug
    

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """
        Fit the robust model without train/validation split.
        1) Train initial model on full data X, y 
        2) For n_rounds:
        a) augment data (X_aug, y_aug)
        b) re-append to main pool
        3) Final downsample to original size and fit
        """
        self.rng = np.random.RandomState(self.random_state)
        original_size = len(X)  # Track original dataset size

        # 1) Initialize model if not provided externally
        if self.model is None:
            self._init_model(**kwargs)

        # 2) Initial fit on full dataset
        print("[MechanisticTrainer] Initial fit on full dataset.")
        self.model.fit(X, y)

        # Keep an "augmented" pool starting with full data
        X_aug = X.copy()
        y_aug = y.copy()

        # 3) Repeated augmentation rounds
        for round_idx in range(self.n_rounds):
            print(f"[MechanisticTrainer] Augmentation Round {round_idx+1}/{self.n_rounds}")
            X_shifted, y_shifted = self.augment_data(X_aug, y_aug)

            # Merge new shifted with old
            X_aug = pd.concat([X_aug, X_shifted], ignore_index=True)
            y_aug = pd.concat([y_aug, y_shifted], ignore_index=True)
            print(f"  => Augmented pool size: {X_aug.shape[0]} samples")

        # 4) Downsample to original size, alternatevely take all x, y shifted without merging to old x,y
        if len(X_aug) > original_size:
            idx_final = self.rng.choice(len(X_aug), size=original_size, replace=False)
            X_final = X_aug.iloc[idx_final].reset_index(drop=True)
            y_final = y_aug.iloc[idx_final].reset_index(drop=True)
            print(f"  => Downsampled combined data to {original_size} total samples.")
        else:
            X_final = X_aug
            y_final = y_aug
            print(f"  => Combined data has {X_final.shape[0]} total samples (<= original size).")

        # 5) Final fit on downsampled dataset
        self.model.fit(X_final, y_final)

        # Store final references
        self.X_final = X_final 
        self.y_final = y_final

        print("[MechanisticTrainer] Robust model training completed.\n")
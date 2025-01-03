from typing import Tuple
from .base import RobustTrainer
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

class MechanisticTrainer(RobustTrainer):
    def __init__(self, base_shift_factor: float = 0.1, n_rounds: int = 3):
        self.base_shift_factor = base_shift_factor
        self.n_rounds = n_rounds
    
    def augment_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate mechanistically-motivated examples"""
        rng = np.random.RandomState(42)
        X_aug = X.copy()
        y_aug = y.copy()
        
        for _ in range(self.n_rounds):
            n_samples = len(X_aug)
            subset_size = int(0.3 * n_samples)
            indices_to_shift = rng.choice(n_samples, size=subset_size, replace=False)
            
            for i in indices_to_shift:
                x_i = X_aug.iloc[i].values.copy()
                label_i = y_aug.iloc[i]
                
                prob_original = self.model.predict_proba([x_i])[0][label_i]
                feature_scores = []
                
                for feat_idx in range(len(x_i)):
                    x_pos = x_i.copy()
                    x_neg = x_i.copy()
                    x_pos[feat_idx] += 1e-3
                    x_neg[feat_idx] -= 1e-3
                    prob_pos = self.model.predict_proba([x_pos])[0][label_i]
                    prob_neg = self.model.predict_proba([x_neg])[0][label_i]
                    grad_est = (prob_pos - prob_neg) / (2e-3)
                    feature_scores.append(abs(grad_est))
                
                top_feats = np.argsort(feature_scores)[-1:]
                shift_dir = np.zeros_like(x_i)
                
                for feat_idx in top_feats:
                    x_pos[feat_idx] = x_i[feat_idx] + 1e-3
                    x_neg[feat_idx] = x_i[feat_idx] - 1e-3
                    prob_pos = self.model.predict_proba([x_pos])[0][label_i]
                    prob_neg = self.model.predict_proba([x_neg])[0][label_i]
                    actual_grad = (prob_pos - prob_neg) / (2e-3)
                    direction_sign = -1 if actual_grad > 0 else 1
                    shift_dir[feat_idx] = direction_sign
                
                noise = rng.normal(0, 0.01, size=len(x_i))
                x_shifted = x_i + shift_dir * self.base_shift_factor + noise
                X_aug.iloc[i] = x_shifted
        
        return X_aug, y_aug

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        self.model = GradientBoostingClassifier(**kwargs)
        self.model.fit(X, y)
        X_aug, y_aug = self.augment_data(X, y)
        self.model.fit(pd.concat([X, X_aug]), pd.concat([y, y_aug]))
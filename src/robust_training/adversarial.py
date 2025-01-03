import numpy as np
import pandas as pd
from typing import Tuple
from .base import RobustTrainer
from src.utils import pred_proba_2d

class AdversarialTrainer(RobustTrainer):
    def __init__(self, model_class, epsilon: float = 0.1, max_rounds: int = 3):
        self.epsilon = epsilon
        self.max_rounds = max_rounds
        self.model_class = model_class

    def augment_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate adversarial examples"""
        rng = np.random.RandomState(42)
        X_adv = X.copy()

        for _ in range(self.max_rounds):
            n_samples = len(X_adv)
            subset_size = int(0.5 * n_samples)
            indices_to_perturb = rng.choice(n_samples, size=subset_size, replace=False)

            for i in indices_to_perturb:
                x_i = X_adv.iloc[i].values.copy()
                true_label = y.iloc[i]

                # Determina l'indice della classe vera
                if not hasattr(self.model, 'classes_') or true_label not in self.model.classes_:
                    continue
                true_label_idx = list(self.model.classes_).index(true_label)

                for feat_idx in range(len(x_i)):
                    x_i_pos = x_i.copy()
                    x_i_neg = x_i.copy()
                    x_i_pos[feat_idx] += 1e-3
                    x_i_neg[feat_idx] -= 1e-3
                    df_x_i_pos = pd.DataFrame(x_i_pos.reshape(1, -1), columns=X.columns)
                    df_x_i_neg = pd.DataFrame(x_i_neg.reshape(1, -1), columns=X.columns)

                    # Ottieni le probabilità
                    prob_pos = pred_proba_2d(self.model, df_x_i_pos)
                    prob_neg = pred_proba_2d(self.model, df_x_i_neg)

                    # Gestisci la forma dell'output di predict_proba
                    if prob_pos.shape[1] > true_label_idx:
                        prob_pos = prob_pos[0][true_label_idx]
                    elif prob_pos.shape[1] == 1:
                        # Supponiamo che sia la probabilità della classe positiva
                        if self.model.classes_[0] == true_label:
                            prob_pos = prob_pos[0][0]
                        elif len(self.model.classes_) == 2:
                            prob_pos = 1 - prob_pos[0][0]
                        else:
                            raise ValueError("Unexpected number of classes.")
                    else:
                        raise ValueError("Unexpected predict_proba shape for prob_pos.")

                    if prob_neg.shape[1] > true_label_idx:
                        prob_neg = prob_neg[0][true_label_idx]
                    elif prob_neg.shape[1] == 1:
                        if self.model.classes_[0] == true_label:
                            prob_neg = prob_neg[0][0]
                        elif len(self.model.classes_) == 2:
                            prob_neg = 1 - prob_neg[0][0]
                        else:
                            raise ValueError("Unexpected number of classes.")
                    else:
                        raise ValueError("Unexpected predict_proba shape for prob_neg.")

                    # Stima del gradiente
                    grad_est = (prob_pos - prob_neg) / (2e-3)
                    direction = -1 if grad_est > 0 else 1
                    x_i[feat_idx] += direction * self.epsilon

                X_adv.iloc[i] = x_i

        y_adv = y.copy()
        return X_adv, y_adv

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        self.model = self.model_class(**kwargs)
        self.model.fit(X, y)
        # Fallback se classes_ non è impostato
        if not hasattr(self.model, 'classes_'):
            self.model.classes_ = np.unique(y)
        X_aug, y_aug = self.augment_data(X, y)
        self.model.fit(pd.concat([X, X_aug]), pd.concat([y, y_aug]))

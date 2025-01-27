

from typing import Dict, Tuple, Union, Optional
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator 
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt


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
    fraction_to_shift : float, optional
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
        base_shift_factor: list = [0.1,0.2,0.001,0.01],
        n_rounds: int = 1,
        fraction_to_shift: float = 0.7,
        n_grad_steps: int = 1,
        top_k: int = 3,
        random_state: int = 42,
        noise_scale: float = 0.001,
        
        val_fraction: float = 0.1,
        eps: float = 0.1,
        ball: bool = False, 
        task: str = 'classification'
    ):
        super().__init__()
        self.task = task.lower()
        if self.task not in ['classification', 'regression']:
            raise ValueError("task must be 'classification' or 'regression'")
        
        if model_type == 'gam' and not PYGAM_AVAILABLE:
            raise ImportError("pyGAM not installed. Please install or pick another model type.")
        self.model_type = model_type.lower()  # 'gbc', 'tree', 'gam'
        self.base_shift_factor = base_shift_factor
        self.n_rounds = n_rounds
        self.fraction_to_shift = fraction_to_shift
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
        
        if self.task == 'regression':
            if self.model_type == 'tree':
                from sklearn.tree import DecisionTreeRegressor
                self.model = DecisionTreeRegressor(
                    max_depth=5,
                    random_state=self.random_state,
                    **kwargs
                )
            elif self.model_type == 'gbr':
                from sklearn.ensemble import GradientBoostingRegressor
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=self.random_state,
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported regression model type: {self.model_type}")
        else:
        
        
            if self.model_type == 'tree':
                self.model = DecisionTreeClassifier(
                    max_depth=5,
                    random_state=self.random_state,
                    **kwargs
                )
            elif self.model_type == 'gam':
                self.model = LogisticGAM(**kwargs)
            elif self.model_type == 'rfc':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=self.random_state,
                    **kwargs
                )
                
            else:
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    random_state=self.random_state,
                    **kwargs
                )

    def _predict_proba_for_label(self, X_sample: np.ndarray, label: int) -> float:
        """
        Safely predict probability for 'label' for a single sample X_sample.
        If the model doesn't support predict_proba, fallback to a 0-1 guess.
        """
        X_sample_2d = X_sample.reshape(1, -1)
        
        if self.task == 'regression':
            # For regression, return negative MSE
            pred = self.model.predict(X_sample_2d)[0]
            return -((pred - label) ** 2)
        else:
        
        
        
            if hasattr(self.model, "predict_proba") and callable(self.model.predict_proba):
                probs = self.model.predict_proba(X_sample_2d)[0]
                
                # If binary => probs is [p(class=0), p(class=1)].
                # If multiclass => we index by the label integer.
                if len(probs) <= label:
                    
                    return 0.0  # fallback
                return probs[label]   # return the probability of the correct (old) label
                
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
        

        #selecting proportion of data to shift
        n_to_shift = int(n_samples * self.fraction_to_shift)
        if n_to_shift == 0:
            # If fraction is too small or data is too small
            return X_aug, y_aug

        indices_to_shift = self.rng.choice(n_samples, size=n_to_shift, replace=False)

        for idx in indices_to_shift:
            x_i = X_aug.iloc[idx].values
            label_i = y_aug.iloc[idx]

            

            # Perform n_grad_steps  gradient-based shifts
            for _ in range(self.n_grad_steps):
                # Estimate gradient wrt the correct label
                
                
                
                if self.ball:
                    
                    
                    features_index = np.random.choice(range(len(x_i)), self.top_k, replace=False)
                    rnd_shift = self.rng.choice(self.base_shift_factor)
                    for feat_idx in features_index:
                        direction_sign = self.rng.choice([-1, 1])
                        x_i[feat_idx] += direction_sign * rnd_shift                
                
            
                else:
                    
                    
                    features_index = np.random.choice(range(len(x_i)), self.top_k, replace=False)
                    rnd_shift = self.rng.choice(self.base_shift_factor)
                     
                    for feat_idx in features_index:
                        
                        direction_sign = self.rng.choice([-1, 1])
                        x_i[feat_idx] += direction_sign * rnd_shift 
                    
            
            noise = self.rng.normal(0, self.noise_scale, size=len(x_i))
            x_i += noise

            
            

            
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
        original_size = len(X) # Track original dataset size

        # 1) Initialize model if not provided externally
        if self.model is None:
            self._init_model(**kwargs)

        

        
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



def run_mechanistic_robust_training_and_eval_in_memory(
    df_train: pd.DataFrame,
    df_dict: Optional[Dict[float, pd.DataFrame]],  # make it Optional
    target: str = 'Y',
    n_rounds: int = 2,
    model_type: str = 'gbc',
    base_shift_factor: float = 0.01,
    fraction_to_shift: float = 0.95,
    random_state: int = 42,
    noise_scale: float = 0.,
    n_grad_steps: int = 1,
    top_k: int = 3,
    eps: float = 0.1,
    ball: bool = False
) -> Tuple[object, object]:
    """
    Train a baseline model and a robust model (via MechanisticTrainer),
    then (optionally) evaluate both on each dataset in df_dict if not None.

    Returns
    -------
    baseline_model, robust_model
    """
    if target not in df_train.columns:
        raise ValueError(f"Target column '{target}' not in training DataFrame.")

    # 1) Split training data
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
    print(f"Training set shape = {X_train.shape};  Target distribution:\n{y_train.value_counts()}")

    # 2) Train Baseline Model
    print("\n=== Training Baseline Model ===")
    if model_type == 'gbc':
        baseline_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=random_state
        )
    elif model_type == 'tree':
        baseline_model = DecisionTreeClassifier(
            max_depth=5,
            random_state=random_state
        )
    elif model_type == 'gam':
        if not PYGAM_AVAILABLE:
            raise ImportError("pyGAM is not installed; choose another model.")
        baseline_model = LogisticGAM()
    elif model_type == 'rfc':
        baseline_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'.")

    baseline_model.fit(X_train, y_train)
    print("=> Baseline model trained.")

    # 3) Train Robust Model
    print("\n=== Training Robust Model ===")
    trainer = MechanisticTrainer(
        model_type=model_type,
        base_shift_factor=base_shift_factor,
        n_rounds=n_rounds,
        fraction_to_shift=fraction_to_shift,
        n_grad_steps=n_grad_steps,
        top_k=top_k,
        random_state=random_state,
        noise_scale=noise_scale,
        eps=eps,
        ball=ball,
        task='classification'
    )
    trainer.fit(X_train, y_train)
    robust_model = trainer.model
    print("=> Robust model trained.")

    # 4) Evaluate on shifted test sets ONLY if df_dict is not None
    if df_dict is not None:
        print("\n=== Evaluation on Shifted Datasets ===")
        for shift_prob, df_test in sorted(df_dict.items(), key=lambda x: x[0]):
            if target not in df_test.columns:
                print(f"Skipping shift={shift_prob}: Missing target '{target}' in test DataFrame.")
                continue

            X_test = df_test.drop(columns=[target])
            y_test = df_test[target]

            # Baseline
            y_pred_b = baseline_model.predict(X_test)
            try:
                y_proba_b = baseline_model.predict_proba(X_test)[:, 1]
                auc_b = roc_auc_score(y_test, y_proba_b)
            except:
                auc_b = np.nan

            # Robust
            y_pred_r = robust_model.predict(X_test)
            try:
                y_proba_r = robust_model.predict_proba(X_test)[:, 1]
                auc_r = roc_auc_score(y_test, y_proba_r)
            except:
                auc_r = np.nan

            acc_b = accuracy_score(y_test, y_pred_b)
            f1_b = f1_score(y_test, y_pred_b)
            acc_r = accuracy_score(y_test, y_pred_r)
            f1_r = f1_score(y_test, y_pred_r)

            print(f"\nShift = {shift_prob:.1f}")
            print(f"  Baseline => Accuracy: {acc_b:.3f}, F1: {f1_b:.3f}, AUC: {auc_b:.3f}")
            print(f"  Robust   => Accuracy: {acc_r:.3f}, F1: {f1_r:.3f}, AUC: {auc_r:.3f}")
            if not np.isnan(auc_r) and not np.isnan(auc_b):
                print(f"  Delta AUC (Robust - Baseline) = {auc_r - auc_b:.4f}")
            
    else:
        print("\nNo test dictionary (df_dict=None) provided => skipping evaluation on shifted datasets.")

    return baseline_model, robust_model

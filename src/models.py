import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

#! logic moved to src/robust_training/adversarial.py

#def generate_adversarial_samples(
#    model,
#    X: pd.DataFrame,
#    y: pd.Series,
#    epsilon: float = 0.1,
#    max_rounds: int = 3,
#    fraction_to_perturb: float = 0.5,
#    random_state: int = 42
#) -> pd.DataFrame:
#    """
#    Improved adversarial sample generator.
#    - We do multiple smaller steps (up to max_rounds).
#    - Each round we only perturb a subset (fraction_to_perturb) of samples
#      (e.g., those with the highest model confidence or chosen randomly).
#    - We apply a smaller shift epsilon in each round.
#    
#    Parameters
#    ----------
#    model : fitted classifier with predict_proba
#    X : pd.DataFrame
#        Original features (n_samples, n_features).
#    y : pd.Series
#        True labels.
#    epsilon : float
#        Base step size for each round.
#    max_rounds : int
#        Number of small-shift rounds to perform.
#    fraction_to_perturb : float
#        Fraction of samples to adversarially perturb each round.
#    random_state : int
#        Random seed for reproducibility.
#
#    Returns
#    -------
#    X_adv : pd.DataFrame
#        Adversarially perturbed features, same shape as X but with partial modifications.
#    """
#    rng = np.random.RandomState(random_state)
#    X_adv = X.copy()
#    
#    for _ in range(max_rounds):
#        # Possibly pick the subset of samples to perturb
#        n_samples = len(X_adv)
#        subset_size = int(fraction_to_perturb * n_samples)
#        
#        # Example strategy: pick the samples where the model is most confident
#        # OR randomly pick subset. Here we do random for simplicity:
#        indices_to_perturb = rng.choice(n_samples, size=subset_size, replace=False)
#        
#        for i in indices_to_perturb:
#            x_i = X_adv.iloc[i].values.copy()
#            true_label = y.iloc[i]
#            
#            # For each feature, do small +/- delta checks
#            for feat_idx in range(len(x_i)):
#                x_i_pos = x_i.copy()
#                x_i_neg = x_i.copy()
#                x_i_pos[feat_idx] += 1e-3
#                x_i_neg[feat_idx] -= 1e-3
#                df_x_i_pos = pd.DataFrame(x_i_pos.reshape(1, -1), columns=X.columns)
#                df_x_i_neg = pd.DataFrame(x_i_neg.reshape(1, -1), columns=X.columns)
#                prob_pos = model.predict_proba(df_x_i_pos)[0][true_label]
#                prob_neg = model.predict_proba(df_x_i_neg)[0][true_label]
#                
#                grad_est = (prob_pos - prob_neg) / (2e-3)
#                # Move in direction that *lowers* probability of the correct label
#                direction = -1 if grad_est > 0 else 1
#                # Smaller step
#                x_i[feat_idx] += direction * epsilon
#            
#            X_adv.iloc[i] = x_i
#            
#        # Optional: re-fit the model incrementally, or we can do a single re-fit outside
#        # For demonstration, let's do a single final re-fit outside this function
#    return X_adv
#
#
#def adversarial_training(
#    X_train: pd.DataFrame,
#    y_train: pd.Series,
#    base_model=None,
#    base_model_class=None,
#    base_model_params=None,
#    adv_generation_fn=None,
#    epsilon: float = 0.1,
#    max_rounds: int = 3,
#    fraction_to_perturb: float = 0.5,
#    final_train_size: int = None,
#    random_state: int = 42
#) -> tuple:
#    """
#    Generic adversarial training procedure:
#      1) Fit base_model (or default) on X_train.
#      2) Generate adversarial samples with adv_generation_fn (default: generate_adversarial_samples).
#      3) Augment dataset.
#      4) Optionally downsample.
#      5) Retrain final model.
#    """
#    if adv_generation_fn is None:
#        adv_generation_fn = generate_adversarial_samples
#    if base_model is None:
#        if base_model_class is None:
#            base_model_class = GradientBoostingClassifier
#        if base_model_params is None:
#            base_model_params = {
#                'learning_rate': 0.05,
#                'max_depth': 4,
#                'max_features': 'log2',
#                'min_samples_leaf': 13,
#                'n_estimators': 100,
#                'subsample': 0.7,
#                'random_state': random_state
#            }
#        base_model = base_model_class(**base_model_params)
#    
#    # 1) Fit the initial model
#    base_model.fit(X_train, y_train)
#    
#    # 2) Generate adversarial samples (multiple smaller shifts)
#    X_adv = adv_generation_fn(
#        model=base_model,
#        X=X_train,
#        y=y_train,
#        epsilon=epsilon,
#        max_rounds=max_rounds,
#        fraction_to_perturb=fraction_to_perturb,
#        random_state=random_state
#    )
#    y_adv = y_train.copy()  # same labels
#    
#    # 3) Augment dataset
#    X_aug = pd.concat([X_train, X_adv], ignore_index=True)
#    y_aug = pd.concat([y_train, y_adv], ignore_index=True)
#    
#    # 4) Downsample to final_train_size if needed
#    rng = np.random.RandomState(random_state)
#    if final_train_size is not None and final_train_size < len(X_aug):
#        indices = rng.choice(len(X_aug), size=final_train_size, replace=False)
#        X_aug = X_aug.iloc[indices].reset_index(drop=True)
#        y_aug = y_aug.iloc[indices].reset_index(drop=True)
#    
#    # 5) Retrain final model
#    adv_model = base_model_class(**base_model_params)
#    adv_model.fit(X_aug, y_aug)
#    
#    return adv_model, X_aug, y_aug
#
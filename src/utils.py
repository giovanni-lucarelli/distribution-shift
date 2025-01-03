import pandas as pd
import matplotlib.colors as mcolors

#? ---------------------------------------------------------------------
#?                Custom functions to load and save data
#? ---------------------------------------------------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def save_data(df: pd.DataFrame, csv_path: str):
    df.to_csv(csv_path, index=False)
    
#? ---------------------------------------------------------------------
#?              Custom functions to get features and target
#? ---------------------------------------------------------------------
    
def get_feature_names(df: pd.DataFrame) -> list:
    return df.columns.tolist()

def get_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    return df.drop(target_col, axis=1)

def get_target(df: pd.DataFrame, target_col: str) -> pd.Series:
    return df[target_col]

#? ---------------------------------------------------------------------
#?       Custom function to predict probabilities from a model
#? ---------------------------------------------------------------------

def pred_proba_1d(model, X: pd.DataFrame) -> pd.DataFrame:
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X)
        if y_pred_proba.ndim == 2:
            y_pred_proba = y_pred_proba[:, 1]                   
    else:
        y_pred_proba = model.predict(X)
    return y_pred_proba

def pred_proba_2d(model, X: pd.DataFrame) -> pd.DataFrame:
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X)
    else:
        y_pred_proba = model.predict(X)
        
    if y_pred_proba.ndim == 1:
        y_pred_proba = y_pred_proba.reshape(-1, 1)
        
    return y_pred_proba

#? ---------------------------------------------------------------------
#?                Custom function to get color gradient
#? ---------------------------------------------------------------------

def get_color_gradient(base_color, n_steps, vary = 'both') -> list:
    rgb = mcolors.to_rgb(base_color)
    hsv = mcolors.rgb_to_hsv(rgb)
    colors = []

    for i in range(n_steps):
        if vary == 'saturation':
            # Vary saturation (S) from 0.4 to 1.0
            sat = 0.3 + 0.6 * i / (n_steps - 1)
            new_hsv = [hsv[0], sat, hsv[2]]
        elif vary == 'brightness':
            # Vary brightness (V) from 0.5 to 1.0
            val = 0.3 + 0.6 * i / (n_steps - 1)
            new_hsv = [hsv[0], hsv[1], val]
        elif vary == 'both':
            # Vary both saturation and brightness
            sat = 0.5 + 0.5 * i / (n_steps - 1)
            val = 0.3 + 0.7 * i / (n_steps - 1)
            new_hsv = [hsv[0], sat, val]
        else:
            raise ValueError("`vary` must be one of 'saturation', 'brightness', or 'both'.")

        new_rgb = mcolors.hsv_to_rgb(new_hsv)
        colors.append(new_rgb)

    return colors


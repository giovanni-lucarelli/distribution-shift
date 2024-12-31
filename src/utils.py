import pandas as pd

def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def save_data(df: pd.DataFrame, csv_path: str):
    df.to_csv(csv_path, index=False)
    
def get_feature_names(df: pd.DataFrame) -> list:
    return df.columns.tolist()

def get_target(df: pd.DataFrame, target_col: str) -> pd.Series:
    return df[target_col]

def get_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    return df.drop(target_col, axis=1)


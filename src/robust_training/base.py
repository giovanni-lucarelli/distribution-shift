from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd

class RobustTrainer(ABC):
    """Base class for robust training methods"""
    
    @abstractmethod
    def augment_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        pass

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        pass
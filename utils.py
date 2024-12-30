import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Computes the sigmoid activation function element-wise.
    
    Parameters
    ----------
    x : np.ndarray
        Input array.
    
    Returns
    -------
    np.ndarray
        Sigmoid output, same shape as input.
    """
    return 1 / (1 + np.exp(-x))
import numpy as np
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
import sys

def numpyify(
        input: NUMERICAL_TYPE,
            ) -> np.ndarray:
    """
    Assuring that input is a numpy array by converting it to one.
    Accepts tensors or ndarrays or arbitrary value types.

    Parameters
    ----------
    input : NUMERICAL_TYPE
        The input

    Returns
    -------
    np.ndarray
        The created tensor.
    """
    try:
        import torch
        if isinstance(input, torch.Tensor):
            return input.detach().cpu().numpy()
    except (ModuleNotFoundError, ImportError):
        pass
    if isinstance(input, np.ndarray):
        return input
    return np.array(input)
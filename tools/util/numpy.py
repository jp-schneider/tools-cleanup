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

def numpyify_image(
        input: VEC_TYPE,
        ) -> np.ndarray:
    """
    Assuring that input is a numpy array by converting it to one.
    Accepts tensors or ndarrays or arbitrary value types.
    If the input is a tensor, it is converted to a numpy array and the first dimension is moved to the last dimension if shape is (C, H, W) -> (H, W, C) 
    or second is moved if shape is (B, C, H, W) -> (B, H, W, C).

    Parameters
    ----------
    input : VEC_TYPE
        The input

    Returns
    -------
    np.ndarray
        The created tensor.
    """
    cvt = numpyify(input)
    if not isinstance(input, np.ndarray):
        # Check if shape has len 3 or 4
        if len(cvt.shape) == 3:
            return np.moveaxis(cvt, 0, -1)
        if len(cvt.shape) == 4:
            return np.moveaxis(cvt, 1, -1)
    return cvt
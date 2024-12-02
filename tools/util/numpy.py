from typing import List, Tuple
import numpy as np
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
import sys
from tools.transforms.to_numpy import numpyify
from tools.transforms.to_numpy_image import numpyify_image


def flatten_batch_dims(array: np.ndarray, end_dim: int) -> Tuple[np.ndarray, List[int]]:
    """

    Utility function: flatten multiple batch dimensions into a single one, or add a batch dimension if there is none.

    Parameters
    ----------
    array : np.ndarray
        Tensor to flatten.
    end_dim : int
        Maximum batch dimension to flatten (inclusive).

    Returns
    -------
    Tuple[np.ndarray, Tuple[int]]
        The flattend array and the original batch shape.
    """
    ed = end_dim + 1 if end_dim != -1 else None
    full_batch = False
    if ed is not None:
        batch_shape = array.shape[:ed]
    else:
        batch_shape = array.shape
        full_batch = True

    expected_dim = -1 if end_dim >= 0 else abs(end_dim)

    if len(batch_shape) > 0:
        if not full_batch:
            flattened = array.reshape(np.prod(batch_shape), *array.shape[ed:])
        else:
            flattened = array.reshape(np.prod(batch_shape))
    else:
        flattened = array[np.newaxis, ...]
        if expected_dim > 0:
            missing = expected_dim - len(flattened.shape)
            for _ in range(missing):
                flattened = flattened[np.newaxis, ...]
    return flattened, batch_shape


def unflatten_batch_dims(array: np.ndarray, batch_shape: List[int]) -> np.ndarray:
    """Method to unflatten a numpy array, which was previously flattened using flatten_batch_dims.

    Parameters
    ----------
    array : np.ndarray
        Tensor to unflatten.
    batch_shape : List[int]
        Batch shape to unflatten.

    Returns
    -------
    np.ndarray
        The unflattened array.
    """

    if len(batch_shape) > 0:
        if not isinstance(batch_shape, list):
            batch_shape = list(batch_shape)
        cur_dim = list(array.shape[1:])
        new_dims = batch_shape + cur_dim
        return array.reshape(new_dims)
    else:
        return array.squeeze(0)

from typing import List, Tuple
import numpy as np
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
import sys
from tools.transforms.to_numpy import numpyify
from tools.transforms.to_numpy_image import numpyify_image


def flatten_batch_dims(array: np.ndarray, end_dim: int) -> Tuple[np.ndarray, Tuple[int, ...]]:
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
    Tuple[np.ndarray, Tuple[int, ...]]
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


def index_of_first(values: np.ndarray, search: np.ndarray) -> np.ndarray:
    """Searches for the index of the first occurence of the search array in the values array.

    Tested for 1D array. Returns -1 if the search array is not found in the values array.

    Example:
    values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    search = np.array([5, 6, 7, 12])

    index_of_first(values, search) -> np.array([4, 5, 6, -1])

    Parameters
    ----------
    values : np.ndarray
        Values array to search in. Shape [N, ...]
    search : np.ndarray
        Search array to search for. Shape [M, ...]

    Returns
    -------
    np.ndarray
        Index array of the first occurence of the search value in the values array.
    """
    values = numpyify(values)
    search = numpyify(search)
    E = values.shape
    S = search.shape
    values_rep = values[..., None]
    search_rep = search[None, ...]
    repeats_v = tuple(1 for _ in range(len(E))) + S
    repeats_s = E + tuple(1 for _ in range(len(S)))
    for ax, r in enumerate(repeats_v):
        values_rep = np.repeat(values_rep, r, axis=ax)
    for ax, r in enumerate(repeats_s):
        search_rep = np.repeat(search_rep, r, axis=ax)
    res = values_rep == search_rep
    out = np.zeros(tuple(S), dtype=np.int32)
    out[...] = -1
    aw = np.argwhere(res)
    search_found, where_inverse = np.unique(aw[:, -1], return_inverse=True)
    for i, s in enumerate(search_found):
        widx = aw[where_inverse == i][0]  # Select first match
        out[s] = widx[0]
    return out

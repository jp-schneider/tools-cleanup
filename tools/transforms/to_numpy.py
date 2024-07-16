from typing import Any, Union
from tools.transforms.transform import Transform
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
import numpy as np


class ToNumpy(Transform):
    """Transforms a arbitrary value or tensor to a numpy array."""

    def transform(self, x: Union[NUMERICAL_TYPE, VEC_TYPE], **kwargs) -> np.ndarray:
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except (ModuleNotFoundError, ImportError):
            pass
        if isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)


numpyify = ToNumpy()
"""
Assuring that input is a numpy array by converting it to one.
Accepts tensors or ndarrays or arbitrary value types.

Parameters
----------
input : Union[NUMERICAL_TYPE, VEC_TYPE]
    The input which should be converted to a numpy array.
    Supports tensors, ndarrays or arbitrary value types.

Returns
-------
np.ndarray
    The created numpy array.
"""

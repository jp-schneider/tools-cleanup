from typing import Union, Optional
from tools.logger.logging import logger
from tools.transforms.to_numpy import ToNumpy
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
try:
    import torch
except ImportError:
    torch = None
    pass
import numpy as np

# Hash of np.float types is not correctly implemented so set check does not work and list is used.
FLOAT_SET = [np.float32, np.float64, np.float16]


class ToNumpyImage(ToNumpy):
    """Transforms a arbitrary value or tensor to a numpy array. If the input is a tensor, it will be permuted to (H, W, C) or (B, H, W, C) before conversion.
    So the output will be a numpy array with shape (H, W[, C]) or (B, H, W[, C]).
    """

    output_dtype: np.dtype
    """The output dtype of the numpy array. After transformation."""

    def __init__(self, output_dtype: Optional[np.dtype] = None):
        super().__init__()
        self.output_dtype = output_dtype

    def transform(self, x: Union[NUMERICAL_TYPE, VEC_TYPE], **kwargs) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            # Check if the tensor is a image tensor
            if len(x.shape) in [3, 4]:
                x = x.permute(0, 2, 3, 1) if len(
                    x.shape) == 4 else x.permute(1, 2, 0)
            elif len(x.shape) == 2:
                pass
            else:
                raise ValueError(f"Input tensor has invalid shape: {x.shape}")
        return self.convert_dtype(super().transform(x, **kwargs))

    def convert_dtype(self, x: np.ndarray) -> np.ndarray:
        if self.output_dtype is None:
            return x
        elif self.output_dtype == x.dtype:
            return x
        elif self.output_dtype == np.uint8 and x.dtype in FLOAT_SET:
            eps = 0.1
            if x.min() < (0 - eps) or x.max() > (1 + eps):
                logger.warning(
                    f"Converting float image to uint8, but values are not in [0, 1]: {x.min()}, {x.max()}")
            return (x * 255).astype(np.uint8)
        elif self.output_dtype in FLOAT_SET and x.dtype == np.uint8:
            return x.astype(self.output_dtype) / 255
        elif self.output_dtype in FLOAT_SET and x.dtype in FLOAT_SET:
            return x.astype(self.output_dtype)
        else:
            raise ValueError(
                f"Cannot convert dtype {x.dtype} to {self.output_dtype}")


numpyify_image = ToNumpyImage()
"""
Assuring that input is a numpy array by converting it to one.
Accepts tensors or ndarrays or arbitrary value types.
If the input is a tensor, it is converted to a numpy array and the first dimension is moved to the last dimension if shape is (C, H, W) -> (H, W, C)
or second is moved if shape is (B, C, H, W) -> (B, H, W, C).
If len(shape) == 2, the shape is not changed.

Parameters
----------
input : VEC_TYPE
    The input

Returns
-------
np.ndarray
    The created numpy array / type.
"""

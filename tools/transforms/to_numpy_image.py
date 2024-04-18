from typing import Union
from tools.transforms.to_numpy import ToNumpy
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
try:
    import torch
except ImportError:
    torch = None
    pass
import numpy as np


class ToNumpyImage(ToNumpy):
    """Transforms a arbitrary value or tensor to a numpy array. If the input is a tensor, it will be permuted to (H, W, C) or (B, H, W, C) before conversion.
    So the output will be a numpy array with shape (H, W, C) or (B, H, W, C).
    """

    def transform(self, x: Union[NUMERICAL_TYPE, VEC_TYPE], **kwargs) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            # Check if the tensor is a image tensor
            if len(x.shape) not in [3, 4]:
                raise ValueError("The tensor is not a image tensor. Shape must be (C, H, W) or (B, C, H, W).")
            # Permute the tensor to (H, W, C) or (B, H, W, C)
            x = x.permute(0, 2, 3, 1) if len(x.shape) == 4 else x.permute(1, 2, 0)
        return super().transform(x, **kwargs)
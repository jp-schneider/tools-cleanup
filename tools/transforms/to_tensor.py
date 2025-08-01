from typing import Any, Union, Optional
from tools.transforms.transform import Transform
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
import numpy as np
import torch

try:
    from PIL import Image
except ImportError:
    Image = None  # PIL is optional, handle it gracefully


class ToTensor(Transform):
    """Transforms a arbitrary value or array to a pytorch tensor."""

    def __init__(self, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> None:
        super().__init__()
        if dtype is not None:
            if isinstance(dtype, str):
                if dtype.startswith("torch."):
                    dtype = torch.dtype(dtype[6:])
                else:
                    dtype = torch.dtype(dtype)
            elif not isinstance(dtype, torch.dtype):
                raise TypeError(
                    f"dtype must be a torch.dtype or valid string, got type {type(dtype)} value {dtype}")
        self.dtype = dtype
        """Dtype of the tensor to create. If None, it will not change the dtype of the input tensor / estimate it from the input."""
        if device is not None:
            if isinstance(device, str):
                if device.startswith("torch."):
                    device = torch.device(device[6:])
                else:
                    device = torch.device(device)
            elif not isinstance(device, torch.device):
                raise TypeError(
                    f"device must be a torch.device or valid string, got type {type(device)} value {device}")
        self.device = device
        """Device of the tensor to create. If None, it will not change the device of the input tensor keeps it on cpu if a new tensor is created or keeps it as is."""

    def transform(self,
                  x: Union[NUMERICAL_TYPE, VEC_TYPE],
                  dtype: Optional[torch.dtype] = None,
                  device: Optional[torch.device] = None,
                  requires_grad: bool = False) -> torch.Tensor:
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        if Image is not None and isinstance(x, Image.Image):
            x = np.array(x)
        if isinstance(x, torch.Tensor):
            if (dtype and x.dtype != dtype) or (device and x.device != device):
                x = x.to(dtype=dtype, device=device)
            return x
        try:
            return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)
        except ValueError as e:
            if isinstance(x, np.ndarray) and "At least one stride in the given numpy array is negative" in str(e):
                return torch.tensor(x.copy(), dtype=dtype, device=device, requires_grad=requires_grad)
            else:
                raise e


tensorify = ToTensor()
"""
Assuring that input is a tensor by converting it to one.
Accepts tensors or ndarrays.

Parameters
----------
input : Union[torch.Tensor, np.generic, int, float, complex, Decimal]
    The input

dtype : Optional[torch.dtype], optional
    Dtype where input should be belong to. If it differs it will cast the type.
    By default its None and the dtype wont be changed.

device : Optional[torch.device], optional
    Device where input should be on / send to. If it differs it will move.
    By default its None and the device wont be changed.

requires_grad : bool, optional
    If the created tensor requires gradient, Will be only considered if input is not already a tensor!. Defaults to false.

Returns
-------
torch.Tensor
    The created tensor.
"""

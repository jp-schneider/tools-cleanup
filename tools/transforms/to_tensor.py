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

    def transform(self, x: Union[NUMERICAL_TYPE, VEC_TYPE],
                  dtype: Optional[torch.dtype] = None,
                  device: Optional[torch.device] = None,
                  requires_grad: bool = False) -> torch.Tensor:
        if Image is not None and isinstance(x, Image.Image):
            x = np.array(x)
        if isinstance(x, torch.Tensor):
            if (dtype and x.dtype != dtype) or (device and x.device != device):
                x = x.to(dtype=dtype, device=device)
            return x
        return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


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

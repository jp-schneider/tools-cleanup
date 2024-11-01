from typing import Union, Optional
from tools.logger.logging import logger
from tools.transforms.to_tensor import ToTensor
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
import torch


class ToTensorImage(ToTensor):
    """Transforms a arbitrary value or numpy array to a pytorch tensor. If the input is a numpy array (or not a tensor), it will be permuted to (C, H, W) or (B, C, H, W) before conversion.
    So the output will be a torch tensor with shape ([B, C], H, W).
    """

    output_dtype: torch.dtype
    """The output dtype of the numpy array. After transformation."""

    def __init__(self, output_dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.output_dtype = output_dtype

    def transform(self, x: Union[NUMERICAL_TYPE, VEC_TYPE], **kwargs) -> torch.Tensor:
        is_tensor = isinstance(x, torch.Tensor)
        x = super().transform(x, **kwargs)
        if not is_tensor:
            # Change the shape to ([B], C, H, W)
            if len(x.shape) == 4:
                return x.permute(0, 3, 1, 2)
            # Change the shape to (C, H, W)
            if len(x.shape) == 3:
                return x.permute(2, 0, 1)
        return self.convert_dtype(x)

    def convert_dtype(self, x: torch.Tensor) -> torch.Tensor:
        if self.output_dtype is None:
            return x
        elif self.output_dtype == x.dtype:
            return x
        elif self.output_dtype == torch.uint8 and x.dtype in [torch.float32, torch.float64, torch.float16]:
            eps = 0.1
            if x.min() < (0 - eps) or x.max() > (1 + eps):
                logger.warning(
                    f"Converting float image to uint8, but values are not in [0, 1]: {x.min()}, {x.max()}")
            return (x * 255).to(torch.uint8)
        elif self.output_dtype in [torch.float32, torch.float64, torch.float16] and x.dtype == torch.uint8:
            return x.to(self.output_dtype) / 255
        else:
            raise ValueError(
                f"Cannot convert dtype {x.dtype} to {self.output_dtype}")


tensorify_image = ToTensorImage()
"""Converts an image to a torch tensor.
If its already a tensor, it will be returned as is, possibly with changed dtype, device or requires_grad.

If the image is a numpy array, it will be converted to a tensor with the shape ([B, C,] H, W) depending on the shape of the input, implicitly assuming that the input is in channel last format e.g.
assumes that the image is in the shape (H, W, C) or (B, H, W, C) for numpy arrays.

Parameters
----------
image : VEC_TYPE
    Image to convert to a tensor.
dtype : Optional[torch.dtype], optional
    Dtype for the tensor, by default None
device : Optional[torch.device], optional
    Device for the tensor, by default None
requires_grad : bool, optional
    If tensor should require gradients, by default False

Returns
-------
torch.Tensor
    The converted tensor.
"""

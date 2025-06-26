from typing import Any, Optional, Tuple, Union
import torch

from tools.transforms.transform import Transform
from tools.transforms.invertable_transform import InvertableTransform
from tools.transforms.to_tensor import tensorify
from tools.logger.logging import logger
from tools.transforms.min_max import MinMax, minmax
from tools.util.torch import tensorify


class Quantize(Transform, torch.nn.Module):

    def __init__(self,
                 bits: int = 16,
                 input_min: Optional[torch.Tensor] = None,
                 input_max: Optional[torch.Tensor] = None,
                 output_min: torch.Tensor = 0.,
                 output_max: torch.Tensor = 1.,
                 dim: Optional[Union[int, Tuple[int]]] = None,
                 auto_fit=False,
                 persistent=True):
        super().__init__(auto_fit, persistent)
        self.register_buffer("input_min", tensorify(
            input_min) if input_min is not None else torch.zeros(0))
        self.register_buffer("input_max", tensorify(
            input_max) if input_max is not None else torch.ones(0))
        self.register_buffer("output_min", tensorify(output_min))
        self.register_buffer("output_max", tensorify(output_max))
        self.bits = bits
        self.dim = dim

    def transform(self, x: torch.Tensor, **kwargs):
        input_min = self.input_min
        input_max = self.input_max
        if input_min.numel() == 0 or input_max.numel() == 0:
            input_min = torch.amin(x, dim=self.dim, keepdim=True)
            input_max = torch.amax(x, dim=self.dim, keepdim=True)

        x_norm = minmax(x, input_min, input_max,
                        torch.tensor(0.), torch.tensor(1.))
        quantized_x = torch.round(
            x_norm * (2 ** self.bits - 1)) / (2 ** self.bits - 1)
        x_out = minmax(quantized_x, torch.tensor(0.), torch.tensor(
            1.), self.output_min, self.output_max)
        return x_out

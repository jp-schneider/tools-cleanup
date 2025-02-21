
from typing import Tuple, Union

import torch
from tools.metric.torch.reducible import Reducible, reduce
from tools.metric.torch.se import SE
from tools.util.torch import flatten_batch_dims, unflatten_batch_dims, tensorify
from tools.util.typing import DEFAULT, _DEFAULT


class PSNR(Reducible):
    """Computes the Peak Signal-to-Noise Ratio (PSNR) between the source and the target.
    Expects the source and target to be Image tensors with the same value range.

    Shape should be ([..., B, C, H,] W)
    """

    def __init__(self,
                 dim: Union[int, Tuple[int, ...]] = (-3, -2, -1),
                 max_value: Union[float, _DEFAULT] = DEFAULT,
                 **kwargs
                 ) -> None:
        """Creates a the PSNR metric."""
        super().__init__(
            dim=dim,
            **kwargs)
        self.mse = SE(reduction="mean", dim=self.dim)
        self.max_value = tensorify(
            max_value) if max_value != DEFAULT else max_value

    def __call__(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        source, bs = flatten_batch_dims(source, -4)
        target, _ = flatten_batch_dims(target, -4)
        mse = self.mse(source, target)
        if self.max_value == DEFAULT:
            max_red = reduce(target, torch.amax, dim=self.dim)
        else:
            max_red = self.max_value.to(source.device)
        loss = 20 * torch.log10(max_red / torch.sqrt(mse))
        return unflatten_batch_dims(loss, bs)

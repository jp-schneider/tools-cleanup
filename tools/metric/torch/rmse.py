
from typing import Literal, Tuple
from torch import Tensor
from tools.metric.torch.reducible import Reducible
from tools.metric.torch.se import SE
from tools.metric.torch.metric import register_metric
import torch


@register_metric(("RMSE", {}))
class RMSE(SE):
    """Computes the root mean squared error between the source and the target."""

    def __init__(self,
                 dim: Tuple[int, ...] = (-3, -2, -1),
                 decoding: bool = False, **kwargs) -> None:
        super().__init__(reduction="mean", dim=dim,
                         alter_naming=False, decoding=decoding, **kwargs)

    def __call__(self, source: Tensor, target: Tensor) -> Tensor:
        return torch.sqrt(self.reduce((source - target) ** 2))

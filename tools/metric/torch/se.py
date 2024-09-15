
from typing import Literal, Tuple
from torch import Tensor
from tools.metric.torch.reducible import Reducible


class SE(Reducible):
    """Computes the squared error between the source and the target."""

    def __init__(self, 
                 reduction: str = "mean", 
                 dim: Tuple[int, ...] = (-3, -2, -1),
                 alter_naming: bool = True, 
                 decoding: bool = False, **kwargs) -> None:
        super().__init__(reduction, dim, alter_naming, decoding, **kwargs)

    def __call__(self, source: Tensor, target: Tensor) -> Tensor:
        return self.reduce((source - target) ** 2)
from typing import Any, Union
import torch

from tools.transforms.fittable_transform import FittableTransform
from tools.transforms.invertable_transform import InvertableTransform
from tools.transforms.to_tensor import tensorify
from tools.util.typing import _DEFAULT, DEFAULT


class MeanStd(InvertableTransform, FittableTransform, torch.nn.Module):
    """Mean standard deviation normalization."""

    def __init__(self,
                 mean: Union[torch.Tensor, _DEFAULT] = torch.zeros(1),
                 std: Union[torch.Tensor, _DEFAULT] = torch.ones(1),
                 dim: Any = None):
        super().__init__()
        self._mean_default = False
        self._std_default = False
        if mean == DEFAULT:
            self._mean_default = True
            mean = torch.zeros(1)
        if std == DEFAULT:
            self._std_default = True
            std = torch.ones(1)
        self.register_buffer("mean", tensorify(mean))
        self.register_buffer("std", tensorify(std))
        self.register_buffer("old_mean", torch.zeros(1))
        self.register_buffer("old_std", torch.ones(1))
        self.dim = dim

    def fit(self, x: torch.Tensor):
        super().fit(x)
        if not self._mean_default:
            self.old_mean = x.mean(dim=self.dim, keepdim=True)
        else:
            self.old_mean = torch.zeros_like(
                x.mean(dim=self.dim, keepdim=True))
        if not self._std_default:
            self.old_std = x.std(dim=self.dim, keepdim=True)
        else:
            self.old_std = torch.ones_like(x.std(dim=self.dim, keepdim=True))

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        super().transform(x)
        return (x - self.old_mean) / self.old_std * self.std + self.mean

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / self.std) * self.old_std + self.old_mean

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

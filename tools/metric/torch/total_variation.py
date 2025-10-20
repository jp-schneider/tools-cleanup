
from typing import Literal, Optional, Tuple, Union

import torch
from tools.metric.torch.reducible import Reducible, reduce
from tools.metric.torch.regularizer import Regularizer
from tools.metric.torch.se import SE
from tools.util.torch import flatten_batch_dims, unflatten_batch_dims, tensorify
from tools.util.typing import DEFAULT, _DEFAULT


REDUCTIONS = {
    "sum": torch.sum,
    "mean": torch.mean,
}


class TotalVariation(Regularizer):
    """

    Computes the 2D Total Variation (TV) of the input tensor.
    Expects the input to be a 4D tensor with shape [[...] H, W].

    The TV is computed as the sum of the absolute differences between adjacent pixels in the specified dimensions.

    Parameters
    ----------
    tv_dims : Union[int, Tuple[int, ...]], optional
        The dimensions along which to compute the total variation. Default is (-2, -1)
        E.g., for a 4D tensor with shape (B, C, H, W), using (-2, -1) computes the TV across H and W.

    reduction : Optional[Literal["mean", "sum"]], optional
        The reduction method to apply to the computed TV. Default is "sum".
        If "mean", the TV is averaged over the spatial dimensions, making it invariant to the size of the input.
        If "sum", the TV is summed over the spatial dimensions.

    """

    def __init__(self,
                 tv_dims: Union[int, Tuple[int, ...]] = (-2, -1),
                 reduction: Optional[Literal["mean", "sum"]] = "sum",
                 p: int = 1,
                 **kwargs
                 ) -> None:
        """Creates a the Total Variation metric."""
        super().__init__(**kwargs)
        self.tv_dims = tv_dims
        self.p = p
        self.reduction = reduction

    def __call__(self, source: torch.Tensor) -> torch.Tensor:
        tv_per_dim = [(torch.diff(source, dim=d, n=1) ** self.p)
                      for d in self.tv_dims]
        div = 1
        if self.reduction == "mean":
            div = sum(tv.shape[-1] for tv in tv_per_dim)
        if div != 1:
            tv_per_dim = [tv / div for tv in tv_per_dim]
        # Sum along last dimension
        tv = torch.stack([torch.sum(tv, dim=-1)
                         for tv in tv_per_dim], dim=-1).sum(dim=-1)
        return tv

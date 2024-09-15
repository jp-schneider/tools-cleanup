from tools.metric.torch.metric import Metric
from typing import Any, Callable, Literal, Optional, Tuple, Union
import torch
from tools.util.typing import NOTSET
from tools.util.format import raise_on_none

REDUCTIONS = {
        "sum": torch.sum,
        "mean": torch.mean,
        "max": torch.amax,
        "min": torch.amin,
    }

REDUCTIONS_NAMES = {
    "sum": "S",
    "mean": "M",
    "max": "Max",
    "min": "Min",
    "none": ""
}

def reduce(x: torch.Tensor, reduce_fnc: Optional[Callable[[torch.Tensor], torch.Tensor]], dim: Optional[Union[int, Tuple[int, ...]]] = NOTSET) -> torch.Tensor:
    if reduce_fnc is None:
        return x
    if dim is NOTSET:
        return reduce_fnc(x)
    return reduce_fnc(x, dim=dim)

class Reducible(Metric):
    
    def get_name(self) -> str:
        name = super().get_name()
        if self.alter_naming:
            return f"{self.reduction_name}{name}"
        return name


    def __init__(self, 
                 reduction: Literal["sum", "mean", "none", "max", "min"] = "mean",
                 dim: Optional[Union[int, Tuple[int, ...]]] = None,
                 alter_naming: bool = True,
                 decoding: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        if not decoding:
            reduction = raise_on_none(reduction)
            if reduction not in REDUCTIONS:
                raise ValueError(
                    f"Value {reduction} for reduction is invalid. Supported values are: {','.join(REDUCTIONS.keys())}")
        if reduction == "none":
            self.reduce_fnc = None
        self.reduce_fnc = REDUCTIONS[reduction]
        self.dim = dim
        self.reduction_name = REDUCTIONS_NAMES[reduction]
        self.alter_naming = alter_naming

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Reduces the tensor based on the reduction function and the dimension.

        Parameters
        ----------
        x : torch.Tensor
            The tensor to reduce.

        Returns
        -------
        torch.Tensor
            The reduced tensor.
        """
        return reduce(x, self.reduce_fnc, self.dim)
    
    def reduce_format(self) -> str:
        if self.dim is None:
            return f"{self.reduction_name}"
        return f"{self.reduction_name}({str(self.dim)})"

    def __repr__(self) -> str:
        repr = super().__repr__()
        return repr + self.reduce_format()
    
    def __ignore_on_iter__(self) -> torch.Set[str]:
        args = super().__ignore_on_iter__()
        args.add("reduce_fnc")

    def after_decoding(self):
        self.reduce_fnc = REDUCTIONS.get(self.reduction_name)
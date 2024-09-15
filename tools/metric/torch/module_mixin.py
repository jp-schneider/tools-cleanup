
from typing import Set
import torch
from tools.metric.metric import Metric as BaseMetric
from tools.util.format import raise_on_none


class ModuleMixin(BaseMetric):
    """Metric for torch tensors."""

    module: torch.nn.Module
    """The module reference."""

    def __init__(self,
                 module: torch.nn.Module = None,
                 decoding: bool = False,
                 **kwargs
                 ) -> None:
        super().__init__(decoding=decoding, **kwargs)
        if not decoding:
            self.module = raise_on_none(module)
        else:
            self.module = module

    def __ignore_on_iter__(self) -> Set[str]:
        args = super().__ignore_on_iter__()
        args.add("module")
        return args
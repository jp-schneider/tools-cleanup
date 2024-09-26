
from typing import Any, Optional, Set
import torch
from tools.metric.metric import Metric as BaseMetric
from tools.util.format import raise_on_none


class ModuleMixin(BaseMetric):
    """Metric for torch tensors."""

    _module: torch.nn.Module
    """The module reference."""

    def __init__(self,
                 module: torch.nn.Module = None,
                 decoding: bool = False,
                 **kwargs
                 ) -> None:
        super().__init__(decoding=decoding, **kwargs)
        self.set_module(module)

    def __ignore_on_iter__(self) -> Set[str]:
        args = super().__ignore_on_iter__()
        args.add("module")
        args.add("_module")
        return args
    
    @property
    def module(self) -> torch.nn.Module:
        """Get the module reference."""
        return self._module
    
    @module.setter
    def module(self, module: torch.nn.Module) -> None:
        """Set the module reference."""
        self.set_module(module)

    def set_module(self, module: torch.nn.Module, memo: Optional[Set[int]]=None) -> None:
        """Set the module reference."""
        if memo is None:
            memo = set()
        if id(self) in memo:
            return
        if (module != self._module):
            self._module = module
            memo.add(id(self))
            for key, val in dict(self):
                if isinstance(val, ModuleMixin):
                    val.set_module(module, memo=memo)
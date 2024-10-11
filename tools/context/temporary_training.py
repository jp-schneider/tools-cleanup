from typing import Union
import torch

from tools.util.typing import DEFAULT, _DEFAULT, NOTSET, _NOTSET
from tools.util.format import raise_on_none


class TemporaryTraining():
    """Context manager for changing the training mode of a module temporary and set it back to the original training mode, if desired."""

    _module: torch.nn.Module
    """The module to change the training mode of."""

    _train: bool
    """The training mode to change to."""

    _old_train: Union[bool, _NOTSET]
    """The old training mode of the module."""

    _keep_train: bool
    """If the current train should be kept after the context manager is exited."""

    def __init__(self,
                 module: torch.nn.Module,
                 train: bool,
                 keep_train: bool = False
                 ):
        self._module = raise_on_none(module)
        self._train = train
        self._old_train = NOTSET
        self._keep_train = keep_train

    def __enter__(self):
        self._old_train = self._module.training
        self._module.train(self._train)
        return self

    def __exit__(self, type, value, traceback):
        if self._keep_train:
            return False
        if self._old_train is not NOTSET:
            self._module.train(self._old_train)
            return False
        else:
            raise ValueError("Old train value was not set!")

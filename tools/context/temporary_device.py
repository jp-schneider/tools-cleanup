from typing import Union
import torch

from tools.util.typing import DEFAULT, _DEFAULT, NOTSET, _NOTSET
from tools.util.format import raise_on_none


def parse_device(device: Union[str, torch.device, _DEFAULT], allow_default: bool = False) -> torch.device:
    if device is None:
        raise ValueError("Device cannot be None!")
    if device is DEFAULT:
        if not allow_default:
            raise ValueError("Device cannot be DEFAULT!")
        return DEFAULT
    if isinstance(device, str):
        return torch.device(device)
    if not isinstance(device, torch.device):
        raise ValueError(
            f"Device must be a string or torch.device, but was: {device}")
    return device


class TemporaryDevice():
    """Context manager for changing the device of a module temporary and set it back to the original device or predefined value."""

    _module: torch.nn.Module
    """The module to change the device of."""

    _device: torch.device
    """The device to change to."""

    _old_device: Union[torch.device, _NOTSET]
    """The old device of the module."""

    _output_device: Union[torch.device, _DEFAULT]
    """The device to change to after the context manager is exited. Or default to fallback to the old device."""

    _keep_device: bool
    """If the current device should be kept after the context manager is exited."""

    def __init__(self,
                 module: torch.nn.Module,
                 device: Union[str, torch.device],
                 keep_device: bool = False,
                 output_device: Union[str, torch.device, _DEFAULT] = DEFAULT):
        self._module = raise_on_none(module)
        self._device = parse_device(device, allow_default=False)
        self._old_device = NOTSET
        self._output_device = parse_device(output_device, allow_default=True)
        self._keep_device = keep_device

    def __enter__(self):
        params = next(self._module.parameters(), None)
        if params is None:
            params = next(self._module.buffers(), None)
        if params is None:
            raise ValueError(
                "Module has no parameters or buffers! Cannot change device.")
        self._old_device = params.device
        self._module.to(self._device)
        return self

    def __exit__(self, type, value, traceback):
        if self._keep_device:
            return False
        if self._output_device is not DEFAULT:
            self._module.to(self._output_device)
            return False
        if self._old_device is not NOTSET:
            self._module.to(self._old_device)
            return False
        else:
            raise ValueError("Old device was not set!")

    @property
    def device(self):
        """The device where the module was moved to."""
        return self._device

from typing import Optional, Union
import torch
from tools.util.typing import _DEFAULT, DEFAULT

def _parse_device(device: Union[str, torch.device]) -> torch.device:
    if device is None:
        raise ValueError("Device must be specified.")
    if isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, torch.device):
        return device
    else:
        raise ValueError("Device must be a string or torch.device object. But got: " + str(type(device)))

def parse_device(device: Optional[Union[str, torch.device]], 
                 allow_none: bool = True, 
                 default: Union[str, torch.device, _DEFAULT] = DEFAULT) -> torch.device:
    """Parses a device string or device object to a torch.device object.

    Parameters
    ----------
    device : Optional[Union[str, torch.device]]
        Device string or device object.
        If None, the default device will be used, if allow_none is True, else an error will be raised.

    allow_none : bool, optional
        If None is allowed as device, by default True

    default : Union[str, torch.device, _DEFAULT], optional
        Default device to use if device is None, by default DEFAULT
        If DEFAULT is used, the default device will be the first available GPU or CPU.

    Returns
    -------
    torch.device
        Device object.
    """
    if device is None:
        if allow_none:
            if isinstance(default, _DEFAULT):
                if torch.cuda.is_available():
                    return torch.device("cuda")
                return torch.device("cpu")
            return _parse_device(default)
        else:
            raise ValueError("Device must be specified.")
    else:
        return _parse_device(device)

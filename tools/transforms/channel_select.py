from typing import Union
from tools.transforms.transform import Transform
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
try:
    import torch
except ImportError:
    torch = None
    pass
import numpy as np


class ChannelSelect(Transform):
    """Selects a channel from a tensor or numpy array. 
    If input is tensor, shape must be (B, C, H, W) or (C, H, W). If input is numpy array, shape must be (B, H, W, C) or (H, W, C).

    """

    def __init__(self, channel: int, keepdim: bool = True) -> None:
        """Initializes the channel select transform.
        
        Parameters
        ----------
        channel : int
            Channel index to select.
        keepdim : bool, optional
            If dimension should be kept, by default True
        """
        self.channel = channel
        self.keepdim = keepdim

    def transform(self, x: VEC_TYPE, **kwargs) -> VEC_TYPE:
        if len(x.shape) not in [3, 4]:
            raise ValueError(f"Shape of tensors must be: (B, C, H, W) or (C, H, W), for numpy arrays: (B, H, W, C) or (H, W, C) but is {x.shape}.")
        if isinstance(x, np.ndarray):
            x = x[..., self.channel]
            if self.keepdim:
                x = x[..., np.newaxis]
        elif isinstance(x, torch.Tensor):
            if len(x.shape) == 3:
                x = x[self.channel]
                if self.keepdim:
                    x = x.unsqueeze(0)
            elif len(x.shape) == 4:
                x = x[:, self.channel]
                if self.keepdim:
                    x = x.unsqueeze(1)
            else:
                # Unreachable code
                raise NotImplementedError()
        else:
            raise ValueError(f"Type of x must be either numpy.ndarray or torch.Tensor but is {type(x)}.")
        return x

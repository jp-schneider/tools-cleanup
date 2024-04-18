from typing import Any, Callable, Optional, Union
from tools.transforms.transform import Transform
try:
    import torch
except ImportError:
    torch = None
    pass
import numpy as np


class Conditional(Transform):
    """Transform which applies a transform based on a condition."""

    def __init__(self, condition: Callable[[Any], bool], true_transform: Transform, false_transform: Optional[Transform] = None) -> None:
        if callable(condition) is False:
            raise ValueError("Condition must be a callable.")
        self.condition = condition
        self.true_transform = true_transform
        self.false_transform = false_transform

    def transform(self, *args, **kwargs) -> Any:
        if self.condition(*args, **kwargs):
            return self.true_transform(*args, **kwargs)
        else:
            if self.false_transform is not None:
                return self.false_transform(*args, **kwargs)
            else:
                return args[0]

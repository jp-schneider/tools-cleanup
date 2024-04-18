from typing import Any, Union
from tools.transforms.transform import Transform
from tools.util.typing import NUMERICAL_TYPE, VEC_TYPE
try:
    import torch
except ImportError:
    torch = None
    pass
import numpy as np

class ToNumpy(Transform):
    """Transforms a arbitrary value or tensor to a numpy array."""

    def transform(self, x: Union[NUMERICAL_TYPE, VEC_TYPE], **kwargs) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)
        
    
from typing import Any, Optional, Tuple, Union

from tools.transforms.fittable_transform import FittableTransform
from tools.transforms.invertable_transform import InvertableTransform
import numpy as np


def minmax(v: np.ndarray,
           v_min: Optional[np.ndarray] = None,
           v_max: Optional[np.ndarray] = None,
           new_min: np.ndarray = 0.,
           new_max: np.ndarray = 1.,
           ) -> np.ndarray:
    if v_min is None:
        v_min = np.min(v)
    if v_max is None:
        v_max = np.max(v)
    return (v - v_min)/(v_max - v_min)*(new_max - new_min) + new_min


class MinMax(InvertableTransform, FittableTransform):
    """MinMax normalization."""

    def __init__(self,
                 new_min: np.ndarray = -1,
                 new_max: np.ndarray = 1,
                 axis: Optional[Union[int, Tuple[int]]] = None
                 ):
        super().__init__()
        self.min = np.zeros(1)
        self.max = np.ones(1)
        self.new_min = np.array(new_min)
        self.new_max = np.array(new_max)
        self.axis = axis

    def _reduce(self, x: np.ndarray, op: Any, axis: Any) -> np.ndarray:
        if axis is None or isinstance(axis, int):
            x = op(x, axis=axis)
            if not isinstance(x, np.ndarray) and not isinstance(x, np.generic):
                x = x.values
            return x
        else:
            for d in axis:
                x = op(x, axis=d, keepdims=True)
                if not isinstance(x, np.ndarray) and not isinstance(x, np.generic):
                    x = x.values
            return x

    def fit(self, x: np.ndarray):
        super().fit(x)
        self.min = self._reduce(x, np.min, self.axis)
        self.max = self._reduce(x, np.max, self.axis)

    def transform(self, x: np.ndarray) -> np.ndarray:
        super().transform(x)
        return minmax(x, self.min, self.max, self.new_min, self.new_max)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return minmax(x, self.new_min, self.new_max, self.min, self.max)

    def extra_repr(self) -> str:
        return f"axis={self.axis}"

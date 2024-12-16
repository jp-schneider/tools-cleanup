import inspect
from typing import Any, Union
import torch
from tools.util.format import raise_on_none
from tools.util.reflection import dynamic_import
from tools.util.torch import numpy_to_torch_dtype, torch_to_numpy_dtype, complex_dtype
import numpy as np
from tools.util.reflection import check_fnc_supported_args


def check_dtype(dtype: Any) -> None:
    dtype = raise_on_none(dtype)

    if isinstance(dtype, str):
        supported_prefix = [
            "torch.",
            "numpy."
            "np."
        ]
        if any([dtype.startswith(prefix) for prefix in supported_prefix]):
            # Try import
            dtype = dynamic_import(dtype)

    # Check if dtype is numpy dtype
    if isinstance(dtype, np.dtype):
        dtype = numpy_to_torch_dtype(dtype)

    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"Value {dtype} for dtype is invalid. Must be of type torch.dtype.")
    return dtype


class TorchDtypeMixin:
    """Mixin for storing the dtype within class."""

    _dtype: torch.dtype
    """Data type of tensors."""

    _complex_dtype: torch.dtype
    """Data type of complex tensors."""

    def __init__(self,
                 dtype: Union[torch.dtype, np.dtype, str] = torch.float32,
                 decoding: bool = False,
                 **kwargs) -> None:
        super_type = super()
        supported, left = check_fnc_supported_args(
            super().__init__, kwargs)
        # If left has other parameters than dtype and decoding, raise an error
        if len(left) > 0:
            raise ValueError(
                    f"Unsupported parameters {left.keys()} in {super().__init__}")
        super().__init__(**supported)
        if decoding:
            return
        self._dtype = check_dtype(dtype)
        self._complex_dtype = complex_dtype(self._dtype)

    @property
    def dtype(self) -> torch.dtype:
        """Returns the underlying dtype of the object."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: Union[torch.dtype, np.dtype, str]) -> None:
        """Sets the underlying dtype of the object."""
        self._dtype = check_dtype(dtype)

    @property
    def numpy_dtype(self) -> np.dtype:
        """Returns the undelying dtype as numpy dtpye."""
        return torch_to_numpy_dtype(self._dtype)

    @property
    def complex_dtype(self) -> torch.dtype:
        """Returns the complex dtype of the object."""
        return self._complex_dtype

import sys
import numpy as np
from typing import Any, Callable, Dict, Optional, Set, TypeVar, Union
try:
    import torch
except (ModuleNotFoundError, ImportError):
    torch = None
import decimal

if torch is not None:
    from torch import Tensor
    VEC_TYPE = TypeVar(
        "VEC_TYPE", bound=Union[torch.Tensor, np.ndarray])  # type: ignore
    """Vector type, like torch.Tensor or numpy.ndarray."""

    REAL_TYPE = TypeVar(
        "REAL_TYPE", bound=Union[torch.Tensor, np.generic, int, float, decimal.Decimal])  # type: ignore
    """Real type which can be converted to a tensor."""

    NUMERICAL_TYPE = TypeVar(
        "NUMERICAL_TYPE", bound=Union[torch.Tensor, np.generic, int, float, complex, decimal.Decimal])  # type: ignore
    """Numerical type which can be converted to a tensor."""
else:
    from torch import Tensor
    VEC_TYPE = TypeVar("VEC_TYPE", bound=np.ndarray)
    """Vector type, like torch.Tensor or numpy.ndarray."""

    REAL_TYPE = TypeVar(
        "REAL_TYPE", bound=Union[np.generic, int, float, decimal.Decimal])
    """Real type which can be converted to a numpy.ndarray."""

    NUMERICAL_TYPE = TypeVar(
        "NUMERICAL_TYPE", bound=Union[np.generic, int, float, complex, decimal.Decimal])
    """Numerical type which can be converted to a numpy.ndarray."""


class _DEFAULT():
    """Default value singleton."""
    pass


DEFAULT = _DEFAULT()
"""Default value singleton."""


class _NOCHANGE:
    pass


class _CYCLE:
    pass


class _MISSING:
    pass


NOCHANGE = _NOCHANGE()
CYCLE = _CYCLE()
MISSING = _MISSING()

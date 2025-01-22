import sys
import numpy as np
from typing import Any, Callable, Dict, Optional, Set, TypeVar, Union, List
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

    INDEX_TYPE = TypeVar("INDEX_TYPE", bound=Union[int, np.ndarray, slice, List[int], torch.Tensor])  # type: ignore
    """Index type which can be used to index a certain type like numpy arrays for 1 dimensional indices."""
else:
    VEC_TYPE = TypeVar("VEC_TYPE", bound=np.ndarray)
    """Vector type, like torch.Tensor or numpy.ndarray."""

    REAL_TYPE = TypeVar(
        "REAL_TYPE", bound=Union[np.generic, int, float, decimal.Decimal])
    """Real type which can be converted to a numpy.ndarray."""

    NUMERICAL_TYPE = TypeVar(
        "NUMERICAL_TYPE", bound=Union[np.generic, int, float, complex, decimal.Decimal])
    """Numerical type which can be converted to a numpy.ndarray."""

    INDEX_TYPE = TypeVar("INDEX_TYPE", bound=Union[int, np.ndarray, slice, List[int]])
    """Index type which can be used to index a certain type like numpy arrays for 1 dimensional indices."""

class _DEFAULT():
    """Default value singleton."""
    
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _DEFAULT):
            return True
        return False
    
    def __repr__(self) -> str:
        return "DEFAULT"
    
    def __str__(self) -> str:
        return "DEFAULT"
    
    def __hash__(self) -> int:
        return 0


DEFAULT = _DEFAULT()
"""Default value singleton."""


class _NOCHANGE:

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _NOCHANGE):
            return True
        return False
    
    def __repr__(self) -> str:
        return "NOCHANGE"
    
    def __str__(self) -> str:
        return "NOCHANGE"
    
    def __hash__(self) -> int:
        return 0


class _CYCLE:
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _CYCLE):
            return True
        return False
    
    def __repr__(self) -> str:
        return "CYCLE"
    
    def __str__(self) -> str:
        return "CYCLE"
    
    def __hash__(self) -> int:
        return 0


class _MISSING:
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _MISSING):
            return True
        return False
    
    def __repr__(self) -> str:
        return "MISSING"
    
    def __str__(self) -> str:
        return "MISSING"
    
    def __hash__(self) -> int:
        return 0


NOCHANGE = _NOCHANGE()
CYCLE = _CYCLE()
MISSING = _MISSING()


class _NOTSET():
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _NOTSET):
            return True
        return False
    
    def __repr__(self) -> str:
        return "NOTSET"
    
    def __str__(self) -> str:
        return "NOTSET"
    
    def __hash__(self) -> int:
        return 0
    


class _PATHNONE():
    def __eq__(self, other: Any) -> bool:
        if isinstance(other, _PATHNONE):
            return True
        return False
    
    def __repr__(self) -> str:
        return "PATHNONE"
    
    def __str__(self) -> str:
        return "PATHNONE"
    
    def __hash__(self) -> int:
        return 0


NOTSET = _NOTSET()
"""Constant for a non existing value."""

PATHNONE = _PATHNONE()
"""Constant for a non existing path."""


def is_list_type(lst):
    """Test if the type is a generic list type, including subclasses excluding
    non-generic classes.
    Examples::

        is_list_type(int) == False
        is_list_type(tuple) == False
        is_list_type(List) == True
        is_list_type(List[int]) == True
        class MyClass(List[str]):
            ...
        is_list_type(MyClass) == True

    For more general tests use issubclass(..., list), for more precise test
    (excluding subclasses) use::

        get_origin(tp) is tuple  # Tuple prior to Python 3.7
    """
    from typing_inspect import NEW_TYPING
    
    if NEW_TYPING:
        from typing_inspect import typingGenericAlias, Generic
        from typing import List
        return (lst is List or isinstance(lst, typingGenericAlias) and
                lst.__origin__ is list or
                isinstance(lst, list) and issubclass(lst, Generic) and
                issubclass(lst, list))
    from typing import (
        ListMeta
    )
    return type(lst) is ListMeta
from abc import abstractmethod, ABC
from typing import Any, Union
from pathlib import Path as PathLibPath
from tools.util.format import raise_on_none
import os
from copy import deepcopy

from tools.util.reflection import class_name
from tools.util.typing import MISSING

PATH_TYPE = Union[PathLibPath, "Path"]


class Path(ABC):
    """Custom Path interface for files and directories."""

    _path: PATH_TYPE
    """The underlying path object."""

    def __init__(self, path: Union[str, PATH_TYPE]):
        path = path = raise_on_none(path, "path")
        if isinstance(path, str):
            self._path = PathLibPath(path)
        else:
            self._path = path

    def __fspath__(self):
        if isinstance(self._path, Path):
            return self._path.__fspath__()
        return str(self._path)

    def __str__(self) -> str:
        return str(self._path)

    def __repr__(self):
        return type(self).__name__ + f"({repr(self._path)})"

    def __truediv__(self, other: Union[str, PATH_TYPE]) -> PATH_TYPE:
        if isinstance(self._path, Path):
            new_path = self._path.__truediv__(other)
        elif isinstance(other, Path):
            new_path = self.merge(other)
        else:
            new_path = self._path / other
        cp = deepcopy(self)
        cp._path = new_path
        return cp

    def __getattr__(self, name: str) -> Any:
        if name == "_path":
            # If the attribute _path is not found, it is not initialized yet. Just raise an AttributeError.
            raise AttributeError(
                f"'{name}' not found in '{type(self).__name__}'")
        try:
            return getattr(self._path, name)
        except AttributeError as e:
            raise AttributeError(
                f"{name} not found in '{type(self).__name__}'")

    @abstractmethod
    def merge(self, other: PATH_TYPE) -> PATH_TYPE:
        """Merges the path with another path.

        Parameters
        ----------
        other : Union[str, PATH_TYPE]
            Other path to merge with.

        Returns
        -------
        PATH_TYPE
            The merged path.
        """
        ...

    def __getstate__(self) -> object:
        return super().__getstate__()


os.PathLike.register(Path)

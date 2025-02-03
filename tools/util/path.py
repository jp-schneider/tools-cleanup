from abc import abstractmethod, ABC
from typing import Any, Union
from pathlib import Path as PathLibPath

import os
from copy import deepcopy

from pathlib import Path as PathLibPath

PATH_TYPE = Union[PathLibPath, "Path"]
"""Path type which can be used for paths."""


class Path(ABC):
    """Custom Path interface for files and directories."""

    _path: PATH_TYPE
    """The underlying path object."""

    def __init__(self, path: Union[str, PATH_TYPE]):
        from tools.util.format import raise_on_none  # type: ignore
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
                f"'{name}' not found in '{type(self).__name__}'")  # @IgnoreException
        return getattr(self._path, name)  # @IgnoreException

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

from pathlib import Path as PathLibPath
from typing import List, Union
from tools.util.format import parse_format_string, FormatVariable, raise_on_none
from typing import Any, Dict, List, Tuple, Type, Optional, TypeVar, Union, get_type_hints
from tools.util.reflection import class_name
from tools.util.path import Path, PATH_TYPE


class ContextPath(Path):
    """Context path for a file. Path which depends on a context which can be reevaluated once it has changed."""

    _context: List[FormatVariable]
    """The context for the path."""

    _raw_path: str
    """The raw path string."""

    def __init__(self,
                 path: Union[str, PATH_TYPE],
                 raw_path: str,
                 context: List[FormatVariable]
                 ):
        super().__init__(path)
        self._context = raise_on_none(context, "context")
        self._raw_path = raise_on_none(raw_path, "raw_path")

    def __str__(self) -> str:
        return str(self._path)

    def __truediv__(self, other):
        other_str = None
        if isinstance(other, Path):
            raise NotImplementedError("Not implemented yet.")
        other_str = str(other)
        out: ContextPath = super().__truediv__(other)
        out._raw_path = out._raw_path + "/" + other_str
        return out

    def merge(self, other: Union[str, PATH_TYPE]) -> PATH_TYPE:
        raise NotImplementedError("Not implemented yet.")

    @classmethod
    def from_format_path(cls,
                         path: str,
                         context: Optional[Any] = None,
                         **kwargs: Any
                         ) -> Union["ContextPath", PathLibPath]:
        """Creates a new context path from a path containing format variables.
        If no format variables are found, a normal Path object is returned.

        Parameters
        ----------
        path : str
            The path string.
        context : List[FormatVariable]
            The context for the path.
        """
        fmt = []
        new_path = parse_format_string(
            path, [context], found_variables=fmt, additional_variables=kwargs)[0]
        fmt = fmt[0]
        if len(fmt) == 0:
            return PathLibPath(new_path)
        return cls(new_path, path, fmt)

    def reevaluate(self, context: Dict[str, Any]) -> "ContextPath":
        """Reevaluates the path with the given context.

        Parameters
        ----------
        context : Dict[str, Any]
            The context to reevaluate the path.
        """
        fmt = []
        new_path = parse_format_string(self._raw_path, [context],
                                       additional_variables={
                                           x.variable: x.value for x in self._context},
                                       found_variables=fmt)[0]
        self._path = PathLibPath(new_path)
        self._context = fmt
        return self

from typing import Any, Dict, List, Literal, Type
from tools.error.argument_none_error import ArgumentNoneError
from tools.serialization.json_convertible import JsonConvertible
from tools.util.typing import _DEFAULT, DEFAULT
from .json_serialization_rule import JsonSerializationRule
import decimal
from pathlib import Path
import os
from tools.util.path_tools import relpath, format_os_independent
from tools.util.format import parse_format_string


class PathValueWrapper(JsonConvertible):

    __type_alias__ = "Path"

    def __init__(self,
                 value: Path,
                 decoding: bool = False,
                 convert_path_to_cwd_relative: bool = True,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            self.value = value
            return
        path_str = str(value)
        if convert_path_to_cwd_relative:
            try:
                path_str = relpath(os.getcwd(), path_str,
                                   is_from_file=False, is_to_file=not value.is_dir(), warn_on_different_drives=False)
            except ValueError as e:
                if "path is on mount" in str(e):
                    # Use absolute path if path is on mount
                    path_str = os.path.abspath(path_str)
                else:
                    raise e
        self.value = format_os_independent(path_str)

    def to_python(self) -> Path:
        path = parse_format_string(self.value, [dict()])[0]
        return Path(path)


class JsonPathSerializationRule(JsonSerializationRule):
    """For the default singleton value."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [Path]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [PathValueWrapper]

    def forward(
            self, value: Any,
            name: str,
            object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return PathValueWrapper(value).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: PathValueWrapper, **kwargs) -> _DEFAULT:
        return value.to_python()

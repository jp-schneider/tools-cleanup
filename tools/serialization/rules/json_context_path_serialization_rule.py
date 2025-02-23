from typing import Any, Dict, List, Literal, Type
from tools.error.argument_none_error import ArgumentNoneError
from tools.serialization.json_convertible import JsonConvertible
from tools.util.typing import _DEFAULT, DEFAULT
from .json_serialization_rule import JsonSerializationRule
import decimal
from tools.serialization.files.context_path import ContextPath
import os
from tools.util.path_tools import relpath, format_os_independent
from tools.util.format import parse_format_string


class ContextPathValueWrapper(JsonConvertible):

    __type_alias__ = "ContextPath"

    def __init__(self,
                 value: ContextPath,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            self.raw_path = None
            self.value = value
            self.context = None
            return
        self.raw_path = value._raw_path
        self.value = value._path
        self.context = value._context

    def to_python(self, **kwargs) -> ContextPath:
        pt = ContextPath(self.value, self.raw_path, self.context)
        if len(kwargs) > 0:
            pt = pt.reevaluate(kwargs)
        return pt


class JsonContextPathSerializationRule(JsonSerializationRule):
    """For the default singleton value."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [ContextPath]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [ContextPathValueWrapper]

    def forward(
            self,
            value: ContextPath,
            name: str,
            object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            no_context_paths: bool = False,
            use_raw_context_paths: bool = False,
            **kwargs) -> Any:

        if no_context_paths or use_raw_context_paths:
            if no_context_paths:
                return format_os_independent(str(value))
            else:
                return format_os_independent(value._raw_path)
        else:
            return ContextPathValueWrapper(value).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: ContextPathValueWrapper, **kwargs) -> _DEFAULT:
        return value.to_python(**kwargs)

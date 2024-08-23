from typing import Any, Dict, List, Literal, Type
from tools.error.argument_none_error import ArgumentNoneError
from tools.serialization.json_convertible import JsonConvertible
from tools.util.typing import _DEFAULT, DEFAULT
from .json_serialization_rule import JsonSerializationRule
import decimal


class DefaultValueWrapper(JsonConvertible):

    __type_alias__ = "DEFAULT"

    def __init__(self, decoding: bool = False, **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return

    def to_python(self) -> _DEFAULT:
        return DEFAULT


class JsonDefaultSingletonSerializationRule(JsonSerializationRule):
    """For the default singleton value."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [_DEFAULT]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [DefaultValueWrapper]

    def forward(
            self, value: Any, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return DefaultValueWrapper().to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: DefaultValueWrapper, **kwargs) -> _DEFAULT:
        return value.to_python()

from collections.abc import Set
from typing import Any, Dict, List, Literal, Optional, Type
from uuid import UUID
from tools.error.argument_none_error import ArgumentNoneError
from tools.serialization.json_convertible import JsonConvertible
from tools.util.reflection import class_name
from .json_serialization_rule import JsonSerializationRule
from tools.serialization.json_convertible import convert
from tools.util.format import parse_type


class SetValueWrapper(JsonConvertible):

    __type_alias__ = "set"

    def get_type(self) -> Type[Set]:
        if hasattr(self, "type"):
            return parse_type(self.type, Set, variable_name="type")
        return set

    def __init__(self,
                 value: list = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if value is not None and not decoding and issubclass(type(value), Set) and type(value) != set:
            self.type = class_name(value)
        if decoding:
            return
        if value is None:
            raise ArgumentNoneError("value")
        self.values = value

    def to_python(self) -> complex:
        return self.get_type(self.values)


class JsonSetSerializationRule(JsonSerializationRule):
    """For python sets"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [Set]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [SetValueWrapper]

    def forward(
            self, value: set, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            memo: Optional[Dict[Any, UUID]] = None,
            **kwargs) -> Any:
        if memo is None:
            memo = set()
        return SetValueWrapper(list(value)).to_json_dict(handle_unmatched=handle_unmatched, memo=memo, **kwargs)

    def backward(self, value: SetValueWrapper, **kwargs) -> Any:
        return value.to_python()

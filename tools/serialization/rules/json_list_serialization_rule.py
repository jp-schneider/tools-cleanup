from enum import Enum
from typing import Any, Dict, List, Literal, MutableSequence, Optional, Type, Union
from uuid import UUID

from tools.serialization.json_convertible import convert

from tools.serialization.json_convertible import JsonConvertible
from tools.serialization.rules.json_serialization_rule import JsonSerializationRule
from tools.util.format import parse_type
from tools.util.reflection import class_name
from tools.error.argument_none_error import ArgumentNoneError


class ListValueWrapper(JsonConvertible):

    __type_alias__ = "list"

    def get_type(self) -> Type[List]:
        if hasattr(self, "type"):
            return parse_type(self.type, (MutableSequence, list), variable_name="type")
        return list

    def __init__(self,
                 value: list = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if value is not None and not decoding and issubclass(type(value), (MutableSequence, list)) and type(value) != list:
            self.type = class_name(value)
        if decoding:
            return
        if value is None:
            raise ArgumentNoneError("value")
        self.values = list(value)

    def to_python(self) -> complex:
        return self.get_type()(self.values)


class JsonListSerializationRule(JsonSerializationRule):
    """For lists of objects."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [list, MutableSequence]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [ListValueWrapper]

    def forward(
            self, value: list, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            memo: Optional[Dict[Any, UUID]] = None,
            **kwargs) -> Any:
        if memo is None:
            memo = set()
        a = []
        if type(value) != list:
            # Handle subclasses of list
            return ListValueWrapper(value).to_json_dict(handle_unmatched=handle_unmatched, memo=memo, **kwargs)

        for subval in value:
            a.append(convert(subval, name, object_context,
                     handle_unmatched=handle_unmatched, memo=memo, **kwargs))
        return a

    def backward(self, value: Union[list, ListValueWrapper], **kwargs) -> Any:
        if isinstance(value, ListValueWrapper):
            return value.to_python()
        else:
            return value

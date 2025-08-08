from typing import Any, Dict, List, Literal, Type
from tools.error.argument_none_error import ArgumentNoneError
from tools.serialization.json_convertible import JsonConvertible
from .json_serialization_rule import JsonSerializationRule
from datetime import datetime, timedelta
from tools.util.reflection import class_name, dynamic_import, register_type


@register_type()
class TimeDeltaValueWrapper(JsonConvertible):

    __type_alias__ = "timedelta"

    def __init__(self,
                 value: timedelta = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        self.value = value.total_seconds()

    def to_python(self) -> timedelta:
        return timedelta(seconds=self.value)


class JsonTimedeltaSerializationRule(JsonSerializationRule):
    """For time deltas"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [timedelta]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [TimeDeltaValueWrapper]

    def forward(
            self, value: timedelta, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return TimeDeltaValueWrapper(value=value).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: TimeDeltaValueWrapper, **kwargs) -> Any:
        return value.to_python()

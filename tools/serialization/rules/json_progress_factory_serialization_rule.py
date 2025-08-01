from typing import Any, Dict, List, Literal, Type
from tools.error.argument_none_error import ArgumentNoneError
from tools.serialization.json_convertible import JsonConvertible
from .json_serialization_rule import JsonSerializationRule
from enum import Enum, IntEnum
from tools.util.reflection import class_name, dynamic_import
from tools.util.progress_factory import ProgressFactory


class JsonProgressFactorySerializationRule(JsonSerializationRule):
    """For progress factory types"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [ProgressFactory]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return []

    def forward(
            self, value: Enum, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return None

    def backward(self, value: Dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError()  # Done by others

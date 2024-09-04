from typing import Any, Dict, List, Literal, Type
from tools.error.argument_none_error import ArgumentNoneError
from tools.serialization.files.file_handle import FileHandle
from tools.serialization.json_convertible import JsonConvertible
from .json_serialization_rule import JsonSerializationRule


class JsonFileHandleSerializationRule(JsonSerializationRule):
    """For reading file handles when loading."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return []

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [FileHandle]

    def forward(
            self, value: FileHandle, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return value.to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: FileHandle, **kwargs) -> Any:
        return value.read()

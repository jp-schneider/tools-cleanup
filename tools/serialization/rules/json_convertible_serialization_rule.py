from uuid import UUID
from tools.serialization.object_hook import object_hook
from tools.serialization.object_decoder import ObjectDecoder
from datetime import datetime
from .json_serialization_rule import JsonSerializationRule
from tools.serialization.json_convertible import JsonConvertible
from tools.error.argument_none_error import ArgumentNoneError
mafrom typing import Any, Dict, List, Literal, Optional, Type


class JsonConvertibleSerializationRule(JsonSerializationRule):
    """For Json Convertible instances."""

    def __init__(self, priority: int = 1000) -> None:
        super().__init__(priority=priority)
        self.decoder = ObjectDecoder(object_hook)

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [JsonConvertible]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return []

    def forward(
            self, value: JsonConvertible, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            memo: Optional[Dict[Any, UUID]] = None,
            **kwargs) -> Any:
        return value.to_json_dict(handle_unmatched=handle_unmatched, memo=memo, **kwargs)

    def backward(self, value: Dict[str, Any], **kwargs) -> Any:
        raise NotImplementedError()

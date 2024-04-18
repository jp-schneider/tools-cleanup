from typing import Any, Dict, List, Literal, Type
from tools.error.argument_none_error import ArgumentNoneError
from tools.serialization.json_convertible import JsonConvertible
from .json_serialization_rule import JsonSerializationRule


class ComplexValueWrapper(JsonConvertible):

    __type_alias__ = "complex"

    def __init__(self,
                 value: complex = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        if value is None:
            raise ArgumentNoneError("value")
        self.real = value.real
        self.imag = value.imag

    def to_python(self) -> complex:
        return complex(self.real, self.imag)


class JsonComplexSerializationRule(JsonSerializationRule):
    """For complex numbers"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [complex]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [ComplexValueWrapper]

    def forward(
            self, value: Any, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return ComplexValueWrapper(value=value).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: ComplexValueWrapper, **kwargs) -> Any:
        return value.to_python()

from typing import Any, Dict, List, Literal, Type
from tools.serialization.json_convertible import JsonConvertible
from tools.serialization.rules.json_serialization_rule import JsonSerializationRule
import torch
from tools.util.reflection import dynamic_import
from tools.util.reflection import register_type


@register_type()
class TorchDtypeValueWrapper(JsonConvertible):

    __type_alias__ = "torch/dtype"

    def __init__(self,
                 value: torch.dtype = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        self.value = str(value)

    def to_python(self) -> torch.Tensor:
        return dynamic_import(self.value)


class JsonTorchDtypeSerializationRule(JsonSerializationRule):
    """For Torch dtypes"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [torch.dtype]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [TorchDtypeValueWrapper]

    def forward(
            self, value: Any, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return TorchDtypeValueWrapper(value=value).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: TorchDtypeValueWrapper, **kwargs) -> torch.Tensor:
        return value.to_python()

from typing import Any, Dict, List, Literal, Type
from tools.serialization.json_convertible import JsonConvertible
from tools.serialization.rules.json_serialization_rule import JsonSerializationRule
from tools.util.reflection import dynamic_import
import numpy as np


class NumpyDTypeValueWrapper(JsonConvertible):

    __type_alias__ = "numpy/dtype"

    def __init__(self,
                 value: np.dtype = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        self.value = str(value)

    def to_python(self) -> np.dtype:
        return np.dtype(self.value)


class JsonNumpyDtypeSerializationRule(JsonSerializationRule):
    """For Numpy dtypes"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [np.dtype]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [NumpyDTypeValueWrapper]

    def forward(
            self, value: Any, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return NumpyDTypeValueWrapper(value=value).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: NumpyDTypeValueWrapper, **kwargs) -> np.dtype:
        return value.to_python()

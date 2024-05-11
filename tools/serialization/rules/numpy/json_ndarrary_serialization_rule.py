import base64
import io
from typing import Any, Dict, List, Literal, Type
from tools.error.argument_none_error import ArgumentNoneError
from tools.serialization.json_convertible import JsonConvertible
from tools.serialization.rules.json_serialization_rule import JsonSerializationRule
import decimal
try:
    import numpy as np
except (ModuleNotFoundError, ImportError):
    pass


def _encode_buffer(buf: bytes) -> str:
    return base64.b64encode(buf).decode()


def _decode_buffer(buf: str) -> bytes:
    return base64.b64decode(buf.encode())


class NDArrayValueWrapper(JsonConvertible):

    __type_alias__ = "numpy/ndarray"

    def __init__(self,
                 value: np.ndarray = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        self.dtype = str(value.dtype)
        self.value = NDArrayValueWrapper.to_serialized_string(value)

    @classmethod
    def to_serialized_string(cls, value: np.ndarray) -> str:
        with io.BytesIO() as buf:
            np.save(buf, value)
            buf.seek(0)
            return _encode_buffer(buf.read())

    @classmethod
    def from_serialized_string(cls, value: str) -> np.ndarray:
        with io.BytesIO() as buf:
            buf.write(_decode_buffer(value))
            buf.seek(0)
            return np.load(buf)

    def to_python(self) -> np.ndarray:
        if isinstance(self.value, list):
            return np.array(self.value).astype(np.dtype(self.dtype))
        elif isinstance(self.value, str):
            return NDArrayValueWrapper.from_serialized_string(self.value)
        else:
            raise ArgumentNoneError("value")


class JsonNDArraySerializationRule(JsonSerializationRule):
    """For decimal numbers"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [np.ndarray]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [NDArrayValueWrapper]

    def forward(
            self, value: Any, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return NDArrayValueWrapper(value=value).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: NDArrayValueWrapper, **kwargs) -> np.ndarray:
        return value.to_python()

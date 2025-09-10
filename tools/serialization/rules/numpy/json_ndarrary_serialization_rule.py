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
from tools.serialization.compressable_mixin import CompressableMixin


class NDArrayValueWrapper(CompressableMixin):

    __type_alias__ = "numpy/ndarray"

    def __init__(self,
                 value: np.ndarray = None,
                 decoding: bool = False,
                 compression: bool = False,
                 **kwargs):
        super().__init__(decoding=decoding, compression=compression, **kwargs)
        if decoding:
            return
        self.dtype = str(value.dtype)
        self.value = NDArrayValueWrapper.to_ascii(
            value, compression=self.compression)

    @classmethod
    def from_bytes(cls, buf: bytes) -> np.ndarray:
        with io.BytesIO() as b:
            b.write(buf)
            b.seek(0)
            return np.load(b)

    @classmethod
    def to_bytes(cls, value: np.ndarray) -> bytes:
        with io.BytesIO() as buf:
            np.save(buf, value)
            buf.seek(0)
            return buf.getvalue()

    def to_python(self) -> np.ndarray:
        if isinstance(self.value, list):
            return np.array(self.value).astype(np.dtype(self.dtype))
        elif isinstance(self.value, str):
            return NDArrayValueWrapper.from_ascii(self.value, compression=self.compression)
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

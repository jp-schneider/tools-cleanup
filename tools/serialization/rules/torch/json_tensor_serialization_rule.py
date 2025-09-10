import base64
import logging
from typing import Any, Dict, List, Literal, Optional, Type
from tools.error.argument_none_error import ArgumentNoneError
from tools.serialization.json_convertible import JsonConvertible
from tools.serialization.rules.json_serialization_rule import JsonSerializationRule
import decimal
import torch
import io
import numpy as np
import os
from tools.util.reflection import dynamic_import

from tools.logger.logging import logger, get_messaged
from tools.serialization.compressable_mixin import CompressableMixin


class TensorValueWrapper(CompressableMixin):

    __type_alias__ = "torch/Tensor"

    def __init__(self,
                 value: torch.Tensor = None,
                 decoding: bool = False,
                 max_display_values: int = 100,
                 no_large_data: bool = False,
                 compression: bool = False,
                 **kwargs):
        super().__init__(decoding=decoding, compression=compression, **kwargs)
        if decoding:
            return
        self.dtype = str(value.dtype)
        self.device = str(value.device)
        self.has_data = not no_large_data
        if self.has_data:
            self.data = TensorValueWrapper.to_ascii(
                value.detach().cpu(), compression=self.compression)
        else:
            self.data = None
        inner = ""
        if len(value.shape) > 1:
            inner = ", ".join((str(x) for x in value.shape))
        elif len(value.shape) == 1:
            inner = str(value.shape[0]) + ","
        else:
            inner = "1"

        self.shape = "(" + inner + ")"
        self.preview = repr(value)

    @classmethod
    def to_bytes(cls, value: torch.Tensor) -> bytes:
        with io.BytesIO() as buf:
            torch.save(value, buf)
            buf.seek(0)
            return buf.read()

    @classmethod
    def from_bytes(cls, value: bytes) -> torch.Tensor:
        with io.BytesIO() as buf:
            buf.write(value)
            buf.seek(0)
            try:
                return torch.load(buf, weights_only=True)
            except Exception as e:
                no_cuda_device_err = "Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False"
                if no_cuda_device_err in str(e):
                    # Load to CPU instead
                    if get_messaged("json_tensor_no_cuda_device"):
                        logger.warning(
                            "Tensor was saved with CUDA device, but no CUDA device is available. Loading to CPU instead.")
                    return torch.load(buf, map_location=torch.device('cpu'), weights_only=True)
                else:
                    raise e

    def to_python(self, no_tensor_data_warning: bool = False) -> torch.Tensor:
        if self.has_data:
            return TensorValueWrapper.from_ascii(self.data, compression=self.compression)
        else:
            json_str = self.to_json()
            if not no_tensor_data_warning:
                logging.warning(
                    f"Tensor was saved without data, can not recover! Result will be without data. Wrapper value was: {os.linesep + json_str}")
            shp = self.shape.replace("(", "").replace(")", "")
            shp = tuple([int(x) for x in shp.split(",") if len(x) > 0])
            dtype = dynamic_import(
                self.dtype) if self.dtype.startswith("torch.") else None
            dev = torch.device(
                self.device) if self.device else torch.device("cpu")
            return torch.zeros(shp).to(dtype=dtype, device=dev)


class JsonTensorSerializationRule(JsonSerializationRule):
    """For Tensors"""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [torch.Tensor]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [TensorValueWrapper]

    def forward(
            self, value: Any, name: str, object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            **kwargs) -> Any:
        return TensorValueWrapper(value=value, **kwargs).to_json_dict(handle_unmatched=handle_unmatched, **kwargs)

    def backward(self, value: TensorValueWrapper, **kwargs) -> torch.Tensor:
        return value.to_python(no_tensor_data_warning=kwargs.get("no_tensor_data_warning", False))

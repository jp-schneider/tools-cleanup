from enum import Enum
import logging
from typing import Any, Dict, List, Literal, Optional, Type, Union

from tools.serialization.json_convertible import JsonConvertible, convert
from tools.util.format import parse_type
from tools.util.reflection import class_name, dynamic_import

from .json_serialization_rule import JsonSerializationRule
from uuid import UUID
import inspect
import torch

ALLOWED_KEY_TYPES = set([str, int, float, bool])
NON_STRING_KEY_TYPES = set([int, float, bool])


class KeyValueItem(JsonConvertible):
    
    __type_alias__ = "KeyValue"

    def __init__(self,
                 value: tuple = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        if value is None:
            raise ArgumentNoneError("value")
        self.key = value[0]
        self.value = value[1]

    def to_python(self) -> tuple:
        return (self.key, self.value)

class KeyValueDictWrapper(JsonConvertible):
    """Type for a dictionary which keys are not strings or all from a different kind.
    Converts these into a key-value list, where each key-value pair is serialized independently.
    """

    __type_alias__ = "KVDict"

    def get_type(self) -> Type[dict]:
        if hasattr(self, "dict_type"):
            return parse_type(self.dict_type, dict, variable_name="dict_type")
        return dict

    def __init__(self,
                 value: dict = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        if value is None:
            raise ArgumentNoneError("value")
        self.values = [KeyValueItem((k, v)) for k, v in value.items()]
        self.dict_type = class_name(value)

    def to_python(self) -> dict:
        _type = self.get_type()
        _d = {v.key: v.value for v in self.values}
        return _type(_d)


class KeyTypeDictWrapper(JsonConvertible):
    """Wrapper for a dictionary which keys are not a string, but a different type which can be easily converted to and from string.
    Assumes all keys are from the same type.
    """
 
    __type_alias__ = "KeyTypeDict"

    def get_type(self) -> Type[dict]:
        if hasattr(self, "key_parser"):
            return parse_type(self.key_parser, tuple(ALLOWED_KEY_TYPES), variable_name="key_parser")
        return dict

    def __init__(self,
                 value: dict = None,
                 decoding: bool = False,
                 **kwargs):
        super().__init__(decoding, **kwargs)
        if decoding:
            return
        if value is None:
            raise ArgumentNoneError("value")
        self.key_parser = class_name(next(iter(value.keys())))
        self.values = {str(k): v for k, v in value.items()}

    def to_python(self) -> dict:
        parser_type = self.get_type()
        _type = type(self.values)
        _d = {parser_type(k): v for k, v in self.values.items()}
        return _type(_d)

class JsonDictSerializationRule(JsonSerializationRule):
    """For lists of objects."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def applicable_forward_types(self) -> List[Type]:
        return [dict]

    @classmethod
    def applicable_backward_types(self) -> List[Type]:
        return [KeyValueDictWrapper, KeyTypeDictWrapper]

    def forward(
            self,
            value: Any,
            name: str,
            object_context: Dict[str, Any],
            handle_unmatched: Literal['identity', 'raise', 'jsonpickle'],
            memo: Optional[Dict[Any, UUID]] = None,
            **kwargs) -> Any:
        if memo is None:
            memo = dict()
        if fnc := getattr(value, "to_json_dict", None):
            args = dict()
            sig = inspect.signature(fnc)
            if "memo" in sig.parameters:
                args["memo"] = memo
            if "handle_unmatched" in sig.parameters:
                args["handle_unmatched"] = handle_unmatched
            if "kwargs" in sig.parameters:
                args.update(kwargs)
            if callable(fnc):
                return fnc(**args)
            raise ValueError("to_json_dict is not callable!")
        elif hasattr(value, '__iter__'):
            # Handling iterables which are not lists or tuples => return them as dict.
            # Iterate over items
            
            # Check if all keys are allowed
            key_types = set([type(k) for k in value.keys()])
            mismatch = len(key_types) > 1 or not key_types.issubset(ALLOWED_KEY_TYPES)
            if mismatch:
                return KeyValueDictWrapper(value).to_json_dict(handle_unmatched=handle_unmatched, memo=memo, **kwargs)
            
            # Check if all keys are not strings
            if len(key_types) == 1 and next(iter(key_types)) in NON_STRING_KEY_TYPES:
                return KeyTypeDictWrapper(value).to_json_dict(handle_unmatched=handle_unmatched, memo=memo, **kwargs)

            # All keys are strings use native dict serialization

            ret = {}
            as_dict = dict(value)
            for k, v in as_dict.items():
                ret[k] = convert(
                    v, k, as_dict, handle_unmatched=handle_unmatched, memo=memo, **kwargs)
            return ret
        else:
            raise ValueError(f"Dont know how to handle Type: {type(value)}")

    def backward(self, value: Union[KeyValueDictWrapper, KeyTypeDictWrapper], **kwargs) -> Any:
        return value.to_python()

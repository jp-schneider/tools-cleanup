import decimal
import os
import sys
from datetime import datetime
from typing import Any, Dict, Literal, Optional, Set, Union

from tools.error.argument_none_error import ArgumentNoneError

try:
    import numpy
except (ModuleNotFoundError, ImportError):
    pass
import logging

try:
    import pandas
except (ModuleNotFoundError, ImportError):
    pass
import inspect
import json
import re
import types
from enum import Enum
from tools.serialization.object_reference import ObjectReference

from tools.util.path_tools import numerated_file_name
from tools.util.reflection import class_name, dynamic_import

from tools.error import NoSimpleTypeError, NoIterationTypeError, NoSerializationRuleFoundError
from uuid import UUID, uuid4


class JsonConvertible:
    """Adding functionality to make a object convertable to json 
    format by introducing an iter function over all properties and 
    specific handlers for complex types like numpy or pandas.
    Works together with the ObjectEncoder and object_hook. """

    __to_json_handle_unmatched__: Literal['identity', 'raise', 'jsonpickle']
    """Behavior how to handle types which can not easily be converted to json.
    identity just returns them, 'raise' will raise and Value error, and 'jsonpickle' tries to pickle them."""

    __memo__: Dict[str, UUID]
    """A temporary memo which is only set when using serialization."""

    __serializer_args__: Dict[str, Any]
    """Parameters of kwargs from the to_json_dict to pass in the convert function."""

    TEMP_PROPERTIES = ['__to_json_handle_unmatched__', '__memo__', '__serializer_args__']


    def __init__(self, decoding: bool = False, **kwargs):
        """Constructor of component base class.
        The decoding arguments is used to indicate, 
        that the class is constructed while decoding 
        it and will be filled with values later. 
        Works in combination with the ObjectEncoder and object_hook.
        """
        super(JsonConvertible, self).__init__()
        self.__to_json_handle_unmatched__ = 'identity'

    def __iter__(self):
        as_dict = {}
        if _ := getattr(self, 'to_dict', None):
            as_dict = self.to_dict()
        else:
            as_dict = vars(self)

        ignore = set()
        if hasattr(self, '__ignore_on_iter__'):
            ignore = self.__ignore_on_iter__()

        handle_unmatched = self.__to_json_handle_unmatched__ if hasattr(
            self, '__to_json_handle_unmatched__') else 'identity'

        memo = self.__memo__ if hasattr(self, '__memo__') else None
        serializer_args = self.__serializer_args__ if hasattr(
            self, '__serializer_args__') else dict()

        for attr, value in as_dict.items():
            if attr in ignore:
                continue
            yield attr, convert(value, attr, as_dict, handle_unmatched=handle_unmatched, memo=memo, **serializer_args)

    def __ignore_on_iter__(self) -> Set[str]:
        """Function which should be use to ignore
        properties during the __iter__ function.
        Used to break recursive calls that could lead to infinite loop.
        This is automatically called when calling dict(self) via __iter__.

        Returns
        -------
        Set[str]
            A set of property names to ignore.
        """
        return set(type(self).TEMP_PROPERTIES)

    def to_dict(self) -> Dict[str, Any]:
        """Will convert the current instance to a dict,
        which is by default vars(self)

        Returns
        -------
        Dict[str, Any]
           The dictionary of including all properties and their values.
        """
        return vars(self)

    @classmethod
    def convert_to_json_dict(cls,
                             obj: Any,
                             handle_unmatched: Literal['identity',
                                                       'raise', 'jsonpickle'] = 'identity',
                             memo: Optional[Dict[Any, UUID]] = None,
                             **kwargs
                             ) -> Dict[str, Any]:
        """
        Puplic static function to convert an abitrary object to a json dict
        also if it does not implement JsonConvertible by itself.

        Parameters
        ----------
        obj : Any
           The object to convert into the dict.

        handle_unmatched : Literal[&#39;identity&#39;, &#39;raise&#39;, &#39;jsonpickle&#39;], optional
            How objects should be treated where no conversion strategy is known, by default 'identity'

        Returns
        -------
        Dict[str, Any]
            The serialized dictionary.

        """
        from tools.serialization.rules.json_serialization_rule_registry import JsonSerializationRuleRegistry
        if obj is None:
            return None

        no_uuid = kwargs.get('no_uuid', False)

        res = None

        if memo is None:
            memo = dict()

        fresh_added = False

        # 0. Test if the object was processed before
        uid = None
        try:
            if not (isinstance(obj, type)) and hasattr(obj, "__hash__") and obj.__hash__ is not None and callable(obj.__hash__) and obj.__hash__() is not None:
                if obj in memo:
                    # Returning a object reference as marker for already serialized object.
                    return ObjectReference(uuid=str(memo[obj]), object_type=class_name(obj)).to_dict()
                else:
                    fresh_added = True
                    uid = uuid4()
                    memo[obj] = uid
        except TypeError as err:
            pass # Ignoring error on non hashable objects like dataframes

        # 1. test if method has to_json_dict, then use it
        if not (isinstance(obj, type)) and hasattr(obj, "to_json_dict"):
            # As the method will responsible for checking the memo we removing the key again if fresh added
            if fresh_added:
                memo.pop(obj)
            res = obj.to_json_dict(
                handle_unmatched=handle_unmatched, memo=memo, **kwargs)

        # 2. test if the object has a rule which can be applied for it
        if res is None:
            rule = JsonSerializationRuleRegistry.instance().get_rule_forward(obj)
            if rule is not None:
                res = rule.forward(
                    obj, "", dict(), handle_unmatched, memo=memo, **kwargs)

        # 3. Proceeding to child policy
        if res is None:
            as_dict = {}
            res = {}
            if fnc := getattr(obj, 'to_dict', None):
                as_dict = obj.to_dict()
            elif isinstance(obj, dict):
                as_dict = obj
            elif hasattr(obj, "__dict__"):
                as_dict = vars(obj)
            elif hasattr(obj, "__iter__"):
                as_dict = dict(obj)
            else:
                # If does not match any match child policy Will fallback to the handle_unmatched approach
                res = convert(
                    obj, None, dict(), handle_unmatched=handle_unmatched, memo=memo, **kwargs)
                if uid is not None and res is not None and not no_uuid:
                    res['__uuid__'] = str(uid)
                return res

            ignore = set()
            if hasattr(obj, '__ignore_on_iter__'):
                ignore = obj.__ignore_on_iter__()

            for attr, value in as_dict.items():
                if attr in ignore:
                    continue

                res[attr] = convert(
                    value, attr, as_dict, handle_unmatched=handle_unmatched, memo=memo, **kwargs)

            # Add class information
            if not isinstance(obj, dict):
                res['__class__'] = class_name(obj)

        # Adding __uuid__ to allow for reconstruction of models and references.
        # __uuid__ may be exist in res as it could be filled by child calls (rules) mit the parent should
        # Override it so it corresponds to the value in memo
        if uid is not None and res is not None and not no_uuid:
            res['__uuid__'] = str(uid)

        for _property in cls.TEMP_PROPERTIES:
            res.pop(_property, None)

        return res

    @classmethod
    def convert_to_json_str(cls,
                            obj: Any,
                            handle_unmatched: Literal['identity',
                                                      'raise', 'jsonpickle'] = 'identity',
                            ensure_ascii: bool = False,
                            indent: int = 4,
                            **kwargs) -> str:
        """Converts a arbitrary object to a JsonStr and preserve its class information.

        Parameters
        ----------
        obj : Any
            The object to convert to json.
        handle_unmatched : Literal[&#39;identity&#39;, &#39;raise&#39;, &#39;jsonpickle&#39;], optional
            How objects should be treated where no conversion strategy is known, by default 'identity'
        ensure_ascii : bool, optional
            If the output is restricted to ascii only chars., by default False
        indent : int, optional
            The indent of the output, by default 4

        Returns
        -------
        str
            A json string.
        """
        encoder = cls.get_encoder(
            ensure_ascii=ensure_ascii, indent=indent, handle_unmatched=handle_unmatched, **kwargs)
        return encoder.encode(obj)

    @classmethod
    def convert_to_file(cls, obj: Any, path: str, override: bool = False, ensure_ascii: bool = False, indent: int = 4, **kwargs) -> str:
        """Function to save the given object as json or yaml to a file.
        Supports extensions .json, .yaml, .yml

        Parameters
        ----------
        obj : Any
            The object to save.
        path : str
            The path of the file with a filename. If it does not contain a extension,
            the ".json" will be added.
        override : False
            If the function should override an existing file
        ensure_ascii : bool, optional
            If the serialized file should only contain ascii chars., by default False
        indent : int, optional
            The indent for the json output, by default 4
        kwargs
            Additional arguments for the serialization.

        Returns
        -------
        str
            Returns the actual save path.
        """
        from tools.serialization import ObjectEncoder

        # Check if path has extension
        spl = os.path.basename(path).split(os.path.extsep)
        if len(spl) == 1:
            path = os.path.join(os.path.dirname(
                path), os.path.basename(path) + os.path.extsep + "json")
        if not override and os.path.exists(path):
            path = numerated_file_name(path)
        # Check for file extension
        ext = os.path.basename(path).split(os.path.extsep)[1]
        if ext == "json":
            json_str = cls.convert_to_json_str(obj,
                                    ensure_ascii=ensure_ascii,
                                    indent=indent, **kwargs)
            with open(path, "w") as f:
                f.write(json_str)
        elif ext == "yaml" or ext == "yml":
            yaml_str = cls.convert_to_yaml_str(obj,
                                    ensure_ascii=ensure_ascii,
                                    indent=indent, **kwargs)
            with open(path, "w") as f:
                f.write(yaml_str)
        return path

    @classmethod
    def convert_to_json_file(cls, obj: Any,
                             path: str,
                             handle_unmatched: Literal['identity',
                                                       'raise', 'jsonpickle'] = 'identity',
                             ensure_ascii: bool = False,
                             indent: int = 4,
                             override: bool = False,
                             **kwargs
                             ) -> str:
        """Performs a to json string conversion for any object and saves it at the given path.

        Parameters
        ----------
        obj : Any
            Any object to convert to json.
        path : str
            The path where the object should be saved.
        handle_unmatched : Literal[&#39;identity&#39;, &#39;raise&#39;, &#39;jsonpickle&#39;], optional
            What should happen if there is no existing serialization rule for a type, by default 'identity'
        ensure_ascii : bool, optional
            If the ouput should be ascii only, by default False
        indent : int, optional
            The indent of the file., by default 4
        override : bool, optional
            If existing file should be overriden, by default False

        Returns
        -------
        str
            The path where the object was saved.

        Raises
        ------
        ArgumentNoneError
            If path or object is None
        """
        if path is None:
            raise ArgumentNoneError("path")
        if obj is None:
            raise ArgumentNoneError("obj")
        json_str = cls.convert_to_json_str(obj,
                                           handle_unmatched=handle_unmatched,
                                           ensure_ascii=ensure_ascii,
                                           indent=indent, **kwargs)
        path = os.path.normpath(path)
        if not override:
            path = numerated_file_name(path)
        with open(path, "w") as f:
            f.write(json_str)
        return path

    @classmethod
    def convert_to_yaml_str(cls,
                            obj: Any,
                            handle_unmatched: Literal['identity',
                                                      'raise', 'jsonpickle'] = 'identity',
                            ensure_ascii: bool = False,
                            indent: int = 4,
                            toplevel_wrapping: bool = True,
                            **kwargs) -> str:
        """Converts a arbitrary object to a YAML string and preserve its class information.

        Parameters
        ----------
        obj : Any
            The object to convert to json.
        handle_unmatched : Literal[&#39;identity&#39;, &#39;raise&#39;, &#39;jsonpickle&#39;], optional
            How objects should be treated where no conversion strategy is known, by default 'identity'
        ensure_ascii : bool, optional
            If the output is restricted to ascii only chars., by default False
        indent : int, optional
            The indent of the output, by default 4
        kwargs: Any
            Arguments which will be passed to the serialization structure as kwargs which can
            alter the behavior of the serialization.
        toplevel_wrapping: bool
            If the yaml should be wrapped in a toplevel class name.
        Returns
        -------
        str
            A yaml string.
        """
        yaml = dynamic_import("yaml")

        encoder = cls.get_encoder(
            handle_unmatched=handle_unmatched, ensure_ascii=ensure_ascii, indent=indent, **kwargs)
        encoded_str = encoder.encode(obj)
        json_dict = json.loads(encoded_str)
        # Dumping Dict to yaml
        # Wrapping dict in classname
        if toplevel_wrapping:
            name = type(obj).__name__
            save_dict = {name: json_dict}
        else:
            save_dict = json_dict
        yaml_str = yaml.dump(save_dict)
        return yaml_str

    def to_json_dict(self,
                     handle_unmatched: Literal['identity',
                                               'raise', 'jsonpickle'] = 'identity',
                     memo: Optional[Dict[Any, UUID]] = None,
                     **kwargs
                     ) -> Dict[str, Any]:
        """
        Basic function to convert the current instance into a json dict
        containing also its class information.

        Returns
        -------
        Dict[str, Any]
            The object as dictionary.
        """
        if memo is None:
            memo = dict()
        uid = None

        no_uuid = kwargs.get('no_uuid', False)

        if hasattr(self, "__hash__") and self.__hash__ is not None:
            if self in memo:
                # Returning a object reference as marker for already serialized object.
                return ObjectReference(uuid=str(memo[self]), object_type=class_name(self)).to_dict()
            else:
                uid = uuid4()
                memo[self] = uid

        self.__memo__ = memo
        
        self.__to_json_handle_unmatched__ = handle_unmatched
        # Adding a argument context to allow for parametrization of rules
        self.__serializer_args__ = kwargs

        # Adding the memo to current class to make i available in iter

        as_dict = dict(self)
        as_dict['__class__'] = class_name(self)

        if uid is not None and not no_uuid:
            as_dict['__uuid__'] = str(uid)

        # Purge memo in current object and dict
        def _purge_temp_property(_property: str):
            if hasattr(self, _property):
                delattr(self, _property)
            if _property in as_dict:
                as_dict.pop(_property)
                
        for _property in type(self).TEMP_PROPERTIES:
            _purge_temp_property(_property)

        return as_dict
        

    def after_decoding(self):
        """Special function which will be invoked after decoding of an object.
        Will be invoked by the object_hook after the constructor.
        """
        pass

    # region IO

    def save_to_file(self, 
                     path: str, 
                     override: bool = False, 
                     ensure_ascii: bool = False, 
                     indent: int = 4, 
                     make_dirs: bool = False,
                     **kwargs) -> str:
        """Function to save the current object as json or yaml to a file.
        Supports extensions .json, .yaml, .yml

        Parameters
        ----------
        path : str
            The path of the file with a filename. If it does not contain a extension,
            the ".json" will be added.
        override : False
            If the function should override an existing file
        ensure_ascii : bool, optional
            If the serialized file should only contain ascii chars., by default False
        indent : int, optional
            The indent for the json output, by default 4
        make_dirs : bool, optional
            If the function should create the directories if they do not exist, by default False
        kwargs
            Additional arguments for the serialization.

        Returns
        -------
        str
            Returns the actual save path.
        """
        from tools.serialization import ObjectEncoder

        # Check if path has extension
        spl = os.path.basename(path).split(os.path.extsep)
        if len(spl) == 1:
            path = os.path.join(os.path.dirname(
                path), os.path.basename(path) + os.path.extsep + "json")
        # Check if directory exists
        if make_dirs:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
        if not override and os.path.exists(path):
            path = numerated_file_name(path)
        # Check for file extension
        ext = os.path.basename(path).split(os.path.extsep)[1]
        if ext == "json":
            path = self._save_to_file_json(
                path=path, ensure_ascii=ensure_ascii, indent=indent, **kwargs)
        elif ext == "yaml" or ext == "yml":
            path = self._save_to_file_yaml(
                path=path, ensure_ascii=ensure_ascii, **kwargs)
        return path

    @classmethod
    def get_encoder(cls, ensure_ascii: bool = False, indent: int = 4, **kwargs) -> Any:
        """Gets the default json encoder for json convertibles which is parametrized with the given arguments.
        Arbitrary kwargs are supported and will be passed to the encoder in the json_convertible_kwargs.

        Parameters
        ----------
        ensure_ascii : bool, optional
            If the encode should only use ascii chars, by default False
        indent : int, optional
            The indent for json, by default 4

        Returns
        -------
        Any
            The encoder.
        """
        from tools.serialization import ObjectEncoder
        encoder_kwargs = dict()
        json_convertible_kwargs = dict()
        if len(kwargs) > 0:
            # Split kwargs in supported by object encoder, others go to json_convertible_kwargs based in inspect
            encoder_kwargs = dict()
            json_convertible_kwargs = dict()
            params = inspect.signature(ObjectEncoder).parameters
            for k, v in kwargs.items():
                if k in params:
                    encoder_kwargs[k] = v
                else:
                    json_convertible_kwargs[k] = v

        encoder = ObjectEncoder(ensure_ascii=ensure_ascii, indent=indent,
                                json_convertible_kwargs=json_convertible_kwargs, **encoder_kwargs)
        return encoder

    def to_json(self, ensure_ascii: bool = False, indent: int = 4, **kwargs) -> str:
        """Converts the current instance to json, by calling to_dict and adding class name.

        Returns
        -------
        str
            Serialized instance.
        """

        encoder = type(self).get_encoder(
            ensure_ascii=ensure_ascii, indent=indent, **kwargs)
        json_str = encoder.encode(self)
        return json_str

    def to_yaml(self, ensure_ascii: bool = False, **kwargs) -> str:
        """Converts the current instance to yaml, by calling to_dict and 
        adding class name.

        Parameters
        ----------
        ensure_ascii : bool, optional
            Wether the serializer should ensure only ascii chars, by default False

        Returns
        -------
        str
            A formatted yaml string.
        """
        return self.convert_to_yaml_str(self, ensure_ascii=ensure_ascii, **kwargs)

    @classmethod
    def from_json(cls, json_str: str, on_error: Literal['raise', 'ignore', 'warning'] = 'raise') -> 'JsonConvertible':
        """Tries to recreate the object from a json string.

        Parameters
        ----------
        json_str : str
            The value of the json string.

        on_error : Literal['raise', 'ignore', 'warning']
            How to behave if an error is raised on deserialization. 
            raise will throw an exception, ignore just leaves object as dict and warning leaves them as dict and logs a warning.
            Default 'raise'

        Returns
        -------
        JsonConvertible
            The recreated instance.

        Raises
        ------
        ArgumentNoneError
            If str is none
        """
        from tools.serialization import object_hook, configurable_object_hook
        if json_str is None:
            raise ArgumentNoneError('json_str')
        return json.loads(json_str, object_hook=configurable_object_hook(on_error=on_error))

    def _save_to_file_json(self, path: str, ensure_ascii: bool = False, indent: int = 4, **kwargs) -> str:
        from tools.serialization import ObjectEncoder
        with open(path, "w") as f:
            f.write(self.to_json(ensure_ascii=ensure_ascii,
                    indent=indent, **kwargs))
        return path

    def _save_to_file_yaml(self, path: str, ensure_ascii: bool = False, **kwargs) -> str:
        from tools.serialization import ObjectEncoder
        yaml = dynamic_import("yaml")
        # Encode to json and then yaml
        encoder = ObjectEncoder(ensure_ascii=ensure_ascii,
                                json_convertible_kwargs=kwargs)
        json_str = encoder.encode(self)
        json_dict = json.loads(json_str)
        # Dumping Dict to yaml
        # Wrapping dict in classname
        name = type(self).__name__
        save_dict = {name: json_dict}
        yaml_str = yaml.dump(save_dict)
        with open(path, "w") as f:
            f.write(yaml_str)
        return path

    @classmethod
    def load_from_file(cls, path: str, on_error: Literal['raise', 'ignore', 'warning'] = 'raise') -> Any:
        """Loads a json convertible like object from a file.

        Parameters
        ----------
        path : str
            The path of the file.

        on_error : Literal['raise', 'ignore', 'warning']
            How to behave if an error is raised on deserialization. 
            raise will throw an exception, ignore just leaves object as dict and warning leaves them as dict and logs a warning.
            Default 'raise'

        Returns
        -------
        Any
            The loaded object instance. Will create the original object
            when json has a __class__ attribute, which is automatically created
            when serialized with the json convertible toolset.
        """
        # Check for file extension
        if os.path.extsep not in os.path.basename(path):
            raise ValueError(f"Path: {path} has not file extension!")
        ext = os.path.basename(path).split(os.path.extsep)[1]
        if ext == "json":
            return cls._load_from_file_json(path, on_error)
        elif ext == "yaml" or ext == "yml":
            return cls._load_from_file_yaml(path, on_error)
        else:
            raise ValueError(
                f"Unsupported file extension: {ext} Only json and yaml / yml are supported!")

    @classmethod
    def _load_from_file_yaml(cls, path: str, on_error: Literal['raise', 'ignore', 'warning'] = 'raise') -> Any:
        """Loads a json convertible like object from a file which is in yaml format.

        Parameters
        ----------
        path : str
            The path of the file.

        on_error : Literal['raise', 'ignore', 'warning']
            How to behave if an error is raised on deserialization. 
            raise will throw an exception, ignore just leaves object as dict and warning leaves them as dict and logs a warning.
            Default 'raise'

        Returns
        -------
        Any
            The loaded object instance. Will create the original object
            when entries have a __class__ attribute, which is automatically created
            when serialized with the json convertible toolset.
        """
        yaml = dynamic_import("yaml")
        loader = dynamic_import("yaml.Loader")
        document = None
        with open(path, "r") as f:
            document = f.read()
        object_dict = yaml.load(document, Loader=loader)
        from tools.serialization import ObjectDecoder, configurable_object_hook
        decoder = ObjectDecoder(configurable_object_hook(on_error))
        decoded = decoder.decode(object_dict)
        # Unpack with classname
        if isinstance(decoded, dict):
            # Purge type name if its the only entry
            if len(decoded) == 1:
                name = list(decoded.keys())[0]
                if type(decoded[name]).__name__ == name:
                    return decoded[name]
        return decoded

    @classmethod
    def _load_from_file_json(cls, path: str, on_error: Literal['raise', 'ignore', 'warning'] = 'raise') -> Any:
        """Loads a json convertible like object from a file.

        Parameters
        ----------
        path : str
            The path of the file.

        on_error : Literal['raise', 'ignore', 'warning']
            How to behave if an error is raised on deserialization. 
            raise will throw an exception, ignore just leaves object as dict and warning leaves them as dict and logs a warning.
            Default 'raise'

        Returns
        -------
        Any
            The loaded object instance. Will create the original object
            when json has a __class__ attribute, which is automatically created
            when serialized with the json convertible toolset.
        """
        from tools.serialization import object_hook, configurable_object_hook
        with open(path, "r") as f:
            return json.load(fp=f, object_hook=configurable_object_hook(on_error=on_error))

    @classmethod
    def load_from_string(cls, json_str: str, on_error: Literal['raise', 'ignore', 'warning'] = 'raise') -> Any:
        """Alias for from json.

        Parameters
        ----------
        json_str : str
            The json serialized values to load.

        Returns
        -------
        Any
            Object which was deserialized.
        """
        return cls.from_json(json_str, on_error)

    # endregion


def convert(
        value: Any, name: str,
        context: Dict[str, Any],
        handle_unmatched: Literal['identity',
                                  'raise', 'jsonpickle'] = 'identity',
        memo: Optional[Dict[Any, UUID]] = None,
        **kwargs) -> Any:
    from tools.serialization.rules.json_serialization_rule_registry import JsonSerializationRuleRegistry
    if value is None:
        return None
    # Check if already serialized
    if memo is None:
        memo = dict()
    rule = JsonSerializationRuleRegistry.instance().get_rule_forward(value)
    if rule is not None:
        return rule.forward(value, name, context, handle_unmatched, memo=memo, **kwargs)
    else:
        if hasattr(value, '__dict__'):
            # Try to convert it with class reprensentation
            try:
                return JsonConvertible.convert_to_json_dict(value, memo=memo, **kwargs)
            except Exception as err:
                logging.exception(
                    f"Dont know how to handle: {value} with name: {name} !")
                return str(value)
        if handle_unmatched == 'raise':
            raise NoSerializationRuleFoundError(
                f"Could not convert type: {type(value)} to json!")
        else:
            rule = JsonSerializationRuleRegistry.instance(
            ).get_default_rule_forward(handle_unmatched)
            return rule.forward(value, name, context, handle_unmatched, memo=memo, **kwargs)
